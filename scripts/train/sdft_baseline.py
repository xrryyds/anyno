import os
# 设置 VLLM 环境变量（保持一致）
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import json
import copy
import math
import torch
import logging
import warnings
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from datasets import Dataset
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ==========================================
# Logger 配置
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MAX_SEQ_LENGTH = 2048
DEFAULT_MAX_NEW_TOKENS = 512
SAVE_TOTAL = 2
EMA_ALPHA = 0.02
SYSTEM_PROMPT = "Please reason step by step and put your final answer within \\boxed{}."

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")


@dataclass
class SDFTConfig:
    model_path: str = ""
    data_path: str = ""
    output_base_dir: str = "/root/autodl-tmp/output"
    real_data_epochs: int = 2
    metrics_log_interval: int = 8
    max_seq_length: int = MAX_SEQ_LENGTH
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    ema_alpha: float = EMA_ALPHA
    temperature: float = 1.0


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(output_dir, "train.log"), encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    return (
        os.path.join(output_dir, "step_metrics.jsonl"),
        os.path.join(output_dir, "epoch_metrics.jsonl"),
    )


class SDFTTrainingMetricsTracker:
    def __init__(self):
        self.reset_window()
        self.reset_epoch()

    def reset_window(self):
        self.win_loss = 0.0
        self.win_steps = 0
        self.win_student_logprob = []
        self.win_teacher_logprob = []
        self.win_gen_tokens = []

    def reset_epoch(self):
        self.ep_loss = 0.0
        self.ep_steps = 0
        self.ep_student_logprob = []
        self.ep_teacher_logprob = []
        self.ep_gen_tokens = []

    def update(self, loss: float, debug_info: List[Dict[str, Any]]):
        self.win_loss += loss
        self.win_steps += 1
        self.ep_loss += loss
        self.ep_steps += 1

        for info in debug_info:
            if info is None:
                continue
            self.win_student_logprob.append(float(info.get("student_logprob", 0.0)))
            self.win_teacher_logprob.append(float(info.get("teacher_logprob", 0.0)))
            self.win_gen_tokens.append(float(info.get("gen_tokens", 0.0)))
            self.ep_student_logprob.append(float(info.get("student_logprob", 0.0)))
            self.ep_teacher_logprob.append(float(info.get("teacher_logprob", 0.0)))
            self.ep_gen_tokens.append(float(info.get("gen_tokens", 0.0)))

    def _avg(self, values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _calculate_stats(self, loss_sum, steps, student_logprob, teacher_logprob, gen_tokens):
        return {
            "avg_train_loss": loss_sum / max(steps, 1),
            "avg_student_logprob": self._avg(student_logprob),
            "avg_teacher_logprob": self._avg(teacher_logprob),
            "avg_generated_tokens": self._avg(gen_tokens),
        }

    def get_window_stats(self):
        return self._calculate_stats(
            self.win_loss,
            self.win_steps,
            self.win_student_logprob,
            self.win_teacher_logprob,
            self.win_gen_tokens,
        )

    def get_epoch_stats(self):
        return self._calculate_stats(
            self.ep_loss,
            self.ep_steps,
            self.ep_student_logprob,
            self.ep_teacher_logprob,
            self.ep_gen_tokens,
        )


sdft_tracker = SDFTTrainingMetricsTracker()


def _extract_target_answer(item: Dict[str, Any]) -> str:
    value = item.get("ref_answer")
    if value is not None and str(value).strip():
        return str(value)
    return ""


class SDFTCollator:
    def __init__(self, tokenizer, max_length: int = MAX_SEQ_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, batch):
        prompt_input_ids_batch = []
        prompt_attention_mask_batch = []
        metadata_batch = []

        for item in batch:
            question = str(item["question"])
            answer = _extract_target_answer(item)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
            prompt_str = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = self.tokenizer(prompt_str, add_special_tokens=False).input_ids
            prompt_ids = prompt_ids[: self.max_length]

            prompt_input_ids_batch.append(torch.tensor(prompt_ids, dtype=torch.long))
            prompt_attention_mask_batch.append(torch.ones(len(prompt_ids), dtype=torch.long))
            metadata_batch.append(
                {
                    "mode": "sdft_baseline",
                    "question": question,
                    "answer": answer,
                    "prompt_text": prompt_str,
                }
            )

        return {
            "prompt_input_ids": torch.nn.utils.rnn.pad_sequence(
                prompt_input_ids_batch,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            ),
            "prompt_attention_mask": torch.nn.utils.rnn.pad_sequence(
                prompt_attention_mask_batch,
                batch_first=True,
                padding_value=0,
            ),
            "metadata": metadata_batch,
        }


class SDFTSequentialTrainer(Trainer):
    def __init__(self, sdft_config: SDFTConfig, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sdft_config = sdft_config
        self.tokenizer = tokenizer
        self.teacher_model = None
        self._teacher_requires_sync = True

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            sampler=SequentialSampler(self.train_dataset),
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _build_teacher_prompt(self, question: str, ref_answer: str) -> str:
        teacher_instruction = (
            f"<Question> {question}\n\n"
            f"This is an example for a response to the question:\n\n"
            f"<Demonstration>\n{ref_answer}\n\n"
            f"Now answer with a response of your own, including the thinking process:"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": teacher_instruction},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _ensure_teacher_model(self):
        if self.teacher_model is None:
            self.teacher_model = copy.deepcopy(self.model)
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self._teacher_requires_sync = False
        elif self._teacher_requires_sync:
            self._ema_update_teacher(initial_copy=True)
            self._teacher_requires_sync = False

    def _ema_update_teacher(self, initial_copy: bool = False):
        if self.teacher_model is None:
            return
        alpha = self.sdft_config.ema_alpha
        with torch.no_grad():
            for teacher_param, student_param in zip(self.teacher_model.parameters(), self.model.parameters()):
                student_data = student_param.detach().to(teacher_param.device)
                if initial_copy:
                    teacher_param.data.copy_(student_data)
                else:
                    teacher_param.data.mul_(1.0 - alpha).add_(student_data, alpha=alpha)
        self.teacher_model.eval()

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        self._ema_update_teacher(initial_copy=False)
        return loss

    def _compute_sequence_logprob(self, active_model, input_ids, attention_mask, target_start_positions):
        outputs = active_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        target_mask = torch.zeros_like(labels, dtype=torch.bool)
        seq_len = input_ids.size(1)
        for idx, start_pos in enumerate(target_start_positions):
            shifted_start = max(int(start_pos) - 1, 0)
            if shifted_start < seq_len - 1:
                target_mask[idx, shifted_start:] = attention_mask[idx, 1 + shifted_start :] > 0

        masked_token_log_probs = token_log_probs * target_mask
        token_counts = target_mask.sum(dim=1).clamp(min=1)
        seq_log_probs = masked_token_log_probs.sum(dim=1) / token_counts
        return seq_log_probs, token_counts.float()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self._ensure_teacher_model()

        prompt_input_ids = inputs["prompt_input_ids"].to(model.device)
        prompt_attention_mask = inputs["prompt_attention_mask"].to(model.device)
        metadata = inputs.get("metadata", [])

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                max_new_tokens=self.sdft_config.max_new_tokens,
                do_sample=True,
                temperature=self.sdft_config.temperature,
                top_p=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        generation_attention_mask = (generated_ids != self.tokenizer.pad_token_id).long()
        prompt_lengths = prompt_attention_mask.sum(dim=1).tolist()

        student_seq_log_probs, token_counts = self._compute_sequence_logprob(
            model,
            generated_ids,
            generation_attention_mask,
            prompt_lengths,
        )

        teacher_input_ids_batch = []
        teacher_attention_mask_batch = []
        teacher_target_start_positions = []

        for idx, meta in enumerate(metadata):
            generated_text = self.tokenizer.decode(
                generated_ids[idx, prompt_lengths[idx] :],
                skip_special_tokens=True,
            )
            teacher_prompt = self._build_teacher_prompt(meta.get("question", ""), meta.get("answer", ""))
            teacher_prompt_ids = self.tokenizer(teacher_prompt, add_special_tokens=False).input_ids
            generated_text_ids = self.tokenizer(generated_text, add_special_tokens=False).input_ids
            full_teacher_ids = (teacher_prompt_ids + generated_text_ids)[: self.sdft_config.max_seq_length]
            teacher_target_start = min(len(teacher_prompt_ids), len(full_teacher_ids))
            teacher_input_ids_batch.append(torch.tensor(full_teacher_ids, dtype=torch.long))
            teacher_attention_mask_batch.append(torch.ones(len(full_teacher_ids), dtype=torch.long))
            teacher_target_start_positions.append(teacher_target_start)

        teacher_input_ids = torch.nn.utils.rnn.pad_sequence(
            teacher_input_ids_batch,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).to(model.device)
        teacher_attention_mask = torch.nn.utils.rnn.pad_sequence(
            teacher_attention_mask_batch,
            batch_first=True,
            padding_value=0,
        ).to(model.device)

        with torch.no_grad():
            teacher_seq_log_probs, _ = self._compute_sequence_logprob(
                self.teacher_model,
                teacher_input_ids,
                teacher_attention_mask,
                teacher_target_start_positions,
            )

        reverse_kl = student_seq_log_probs - teacher_seq_log_probs
        loss = reverse_kl.mean()

        if model.training:
            debug_info = []
            for idx in range(len(metadata)):
                debug_info.append(
                    {
                        "student_logprob": float(student_seq_log_probs[idx].detach().cpu().item()),
                        "teacher_logprob": float(teacher_seq_log_probs[idx].detach().cpu().item()),
                        "gen_tokens": float(token_counts[idx].detach().cpu().item()),
                    }
                )
            sdft_tracker.update(loss.item(), debug_info)

        outputs = {
            "student_log_probs": student_seq_log_probs.detach(),
            "teacher_log_probs": teacher_seq_log_probs.detach(),
            "generated_ids": generated_ids.detach(),
        }
        return (loss, outputs) if return_outputs else loss


class SDFTStepLogCallback(TrainerCallback):
    def __init__(self, log_file, log_interval, config: SDFTConfig, output_dir: str):
        self.log_file = log_file
        self.log_interval = log_interval
        self.config = config
        self.output_dir = output_dir

    def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.log_interval == 0:
            stats = sdft_tracker.get_window_stats()
            json_stats = stats.copy()
            json_stats.update(
                {
                    "epoch": state.epoch,
                    "global_step": state.global_step,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(json_stats, ensure_ascii=False) + "\n")

            log_parts = [f"[Step {state.global_step}]"]
            for k, v in stats.items():
                val_str = f"{v:.6f}" if isinstance(v, float) else str(v)
                log_parts.append(f"{k}: {val_str}")
            logger.info(" | ".join(log_parts))

            sdft_tracker.reset_window()


class SDFTLogicalEpochLogCallback(TrainerCallback):
    """逻辑 Epoch 日志打印（不早停，只记录 epoch 级别 loss）"""

    def __init__(self, log_file, steps_per_logical_epoch, config: SDFTConfig, output_dir: str):
        self.log_file = log_file
        self.steps_per_epoch = steps_per_logical_epoch
        self.config = config
        self.output_dir = output_dir
        self.current_logical_epoch = 0

    def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.steps_per_epoch == 0:
            self.current_logical_epoch += 1

            stats = sdft_tracker.get_epoch_stats()
            stats.update(
                {
                    "logical_epoch": self.current_logical_epoch,
                    "global_step": state.global_step,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(stats, ensure_ascii=False) + "\n")

            logger.info("=" * 60)
            logger.info(f"*** LOGICAL EPOCH {self.current_logical_epoch} FINISHED ***")
            logger.info(f"  Avg Train Loss:{stats['avg_train_loss']:.4f}")
            logger.info(f"  Avg Student LogProb:{stats['avg_student_logprob']:.4f}")
            logger.info(f"  Avg Teacher LogProb:{stats['avg_teacher_logprob']:.4f}")
            logger.info(f"  Avg Generated Tokens:{stats['avg_generated_tokens']:.2f}")
            logger.info("=" * 60)
            sdft_tracker.reset_epoch()


def run_sdft_training_baseline(
    model_path: str,
    data_path: str,
    batch_size: int,
    real_data_epochs: int,
    output_base_dir: Optional[str] = None,
    device_num: int = 1,
    lora_path: Optional[str] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_seq_length: int = MAX_SEQ_LENGTH,
    ema_alpha: float = EMA_ALPHA,
):
    """SDFT baseline training API.

    输入：
    - data_path: 数据集地址，数据格式与 adv_hints.json 一致
    - model_path: 基座模型地址
    - batch_size: batch size
    - real_data_epochs: 逻辑 epoch 数
    """
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path))
    project_root = os.path.dirname(os.path.dirname(project_root))

    if output_base_dir is None:
        output_base_dir = os.path.join(project_root, "output")

    set_seed(42)
    sdft_tracker.reset_window()
    sdft_tracker.reset_epoch()

    sdft_config = SDFTConfig(
        model_path=model_path,
        data_path=data_path,
        output_base_dir=output_base_dir,
        real_data_epochs=real_data_epochs,
        metrics_log_interval=batch_size,
        max_seq_length=max_seq_length,
        max_new_tokens=max_new_tokens,
        ema_alpha=ema_alpha,
    )

    output_dir = f"{sdft_config.output_base_dir}/sdft_baseline_{real_data_epochs}ep_{datetime.now().strftime('%m%d_%H%M')}"
    step_log_file, epoch_log_file = setup_logging(output_dir)

    logger.info(f"SDFT Baseline - Model Path: {sdft_config.model_path}")
    logger.info(f"SDFT Baseline - Data Path: {sdft_config.data_path}")
    logger.info(f"SDFT Baseline - Output Dir: {output_dir}")
    logger.info(f"SDFT Baseline - EMA Alpha: {sdft_config.ema_alpha}")
    logger.info(f"SDFT Baseline - Max New Tokens: {sdft_config.max_new_tokens}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            sdft_config.model_path,
            trust_remote_code=True,
            use_fast=False,
        )
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise e

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if not os.path.exists(sdft_config.data_path):
        raise FileNotFoundError(f"Dataset not found at {sdft_config.data_path}")
    dataset = Dataset.from_json(sdft_config.data_path)

    collator = SDFTCollator(tokenizer, max_length=sdft_config.max_seq_length)

    total_samples = len(dataset)
    total_steps = total_samples // batch_size
    steps_per_logical_epoch = total_steps // real_data_epochs

    if steps_per_logical_epoch < 1:
        steps_per_logical_epoch = 1
        logger.warning(
            f"Total steps ({total_steps}) < real epochs ({real_data_epochs}). Set logical step to 1."
        )

    model = AutoModelForCausalLM.from_pretrained(
        sdft_config.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    use_existing_lora = False
    if lora_path is not None and str(lora_path).strip():
        lora_path = str(lora_path).strip()
        if os.path.exists(lora_path) and os.path.isdir(lora_path):
            use_existing_lora = True
            logger.info(f"Using existing LoRA from: {lora_path}")
        else:
            logger.warning(
                f"Provided lora_path '{lora_path}' does not exist or is not a directory. Falling back to fresh LoRA initialization."
            )

    if use_existing_lora:
        try:
            model = PeftModel.from_pretrained(model, lora_path)
            logger.info(f"Successfully loaded existing LoRA weights from '{lora_path}'.")
        except (OSError, ValueError) as e:
            logger.error(
                f"Failed to load LoRA weights from '{lora_path}': {e}. Falling back to fresh LoRA initialization."
            )
            use_existing_lora = False

    if not use_existing_lora:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
            bias="none",
        )
        model = get_peft_model(model, peft_config)

    model.enable_input_require_grads()

    save_steps = max(1, int(steps_per_logical_epoch * (real_data_epochs / SAVE_TOTAL)))
    per_device_batch_size = max(1, batch_size // max(device_num, 1))

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=sdft_config.metrics_log_interval,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=SAVE_TOTAL,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_drop_last=True,
        report_to="none",
    )

    step_callback = SDFTStepLogCallback(
        step_log_file,
        sdft_config.metrics_log_interval,
        sdft_config,
        output_dir,
    )
    epoch_callback = SDFTLogicalEpochLogCallback(
        epoch_log_file,
        steps_per_logical_epoch,
        sdft_config,
        output_dir,
    )

    trainer = SDFTSequentialTrainer(
        sdft_config=sdft_config,
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[step_callback, epoch_callback],
    )

    logger.info("Starting SDFT baseline training...")
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"SDFT baseline training finished. Model saved to {output_dir}")

    return output_dir


if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path))
    project_root = os.path.dirname(os.path.dirname(project_root))

    default_model_dir = os.path.join(project_root, "model", "OREAL")
    default_model_url = os.path.join(default_model_dir, "OREAL-7B")
    default_data_path = os.path.join(project_root, "datasets", "exam", "adv_hints.json")
    default_output_base_dir = os.path.join(project_root, "output")

    run_sdft_training_baseline(
        model_path=default_model_url,
        data_path=default_data_path,
        output_base_dir=default_output_base_dir,
        batch_size=4,
        real_data_epochs=2,
    )

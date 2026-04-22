import os
# 设置 VLLM 环境变量（保持一致）
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import json
import copy
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
ENTROPY_BETA = 0.01
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
    entropy_beta: float = ENTROPY_BETA
    temperature: float = 1.0
    pretrain_epochs: int = 1


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
        self.win_entropy = []
        self.win_phase = []

    def reset_epoch(self):
        self.ep_loss = 0.0
        self.ep_steps = 0
        self.ep_student_logprob = []
        self.ep_teacher_logprob = []
        self.ep_gen_tokens = []
        self.ep_entropy = []
        self.ep_phase = []

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
            self.win_entropy.append(float(info.get("entropy", 0.0)))
            self.win_phase.append(str(info.get("phase", "")))
            self.ep_student_logprob.append(float(info.get("student_logprob", 0.0)))
            self.ep_teacher_logprob.append(float(info.get("teacher_logprob", 0.0)))
            self.ep_gen_tokens.append(float(info.get("gen_tokens", 0.0)))
            self.ep_entropy.append(float(info.get("entropy", 0.0)))
            self.ep_phase.append(str(info.get("phase", "")))

    def _avg(self, values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _phase_name(self, phases: List[str]) -> str:
        if not phases:
            return "unknown"
        counts = {}
        for phase in phases:
            counts[phase] = counts.get(phase, 0) + 1
        return max(counts.items(), key=lambda x: x[1])[0]

    def _calculate_stats(self, loss_sum, steps, student_logprob, teacher_logprob, gen_tokens, entropy_vals, phase_vals):
        return {
            "avg_train_loss": loss_sum / max(steps, 1),
            "avg_student_logprob": self._avg(student_logprob),
            "avg_teacher_logprob": self._avg(teacher_logprob),
            "avg_generated_tokens": self._avg(gen_tokens),
            "avg_entropy": self._avg(entropy_vals),
            "phase": self._phase_name(phase_vals),
        }

    def get_window_stats(self):
        return self._calculate_stats(
            self.win_loss,
            self.win_steps,
            self.win_student_logprob,
            self.win_teacher_logprob,
            self.win_gen_tokens,
            self.win_entropy,
            self.win_phase,
        )

    def get_epoch_stats(self):
        return self._calculate_stats(
            self.ep_loss,
            self.ep_steps,
            self.ep_student_logprob,
            self.ep_teacher_logprob,
            self.ep_gen_tokens,
            self.ep_entropy,
            self.ep_phase,
        )


sdft_tracker = SDFTTrainingMetricsTracker()


def _extract_ref_answer(item: Dict[str, Any]) -> str:
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

    def _build_student_prompt(self, question: str) -> str:
        messages = [{"role": "user", "content": question}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
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

    def __call__(self, batch):
        student_prompt_ids_batch = []
        student_prompt_attention_mask_batch = []
        pretrain_input_ids_batch = []
        pretrain_labels_batch = []
        pretrain_attention_mask_batch = []
        metadata_batch = []

        for item in batch:
            question = str(item["question"])
            ref_answer = _extract_ref_answer(item)

            student_prompt = self._build_student_prompt(question)
            student_prompt_ids = self.tokenizer(student_prompt, add_special_tokens=False).input_ids
            student_prompt_ids = student_prompt_ids[: self.max_length]

            teacher_prompt = self._build_teacher_prompt(question, ref_answer)
            teacher_prompt_ids = self.tokenizer(teacher_prompt, add_special_tokens=False).input_ids
            answer_ids = self.tokenizer(ref_answer, add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]
            full_teacher_ids = (teacher_prompt_ids + answer_ids)[: self.max_length]

            labels = [-100] * min(len(teacher_prompt_ids), len(full_teacher_ids))
            labels += full_teacher_ids[len(labels) :]
            labels = labels[: len(full_teacher_ids)]

            student_prompt_ids_batch.append(torch.tensor(student_prompt_ids, dtype=torch.long))
            student_prompt_attention_mask_batch.append(torch.ones(len(student_prompt_ids), dtype=torch.long))
            pretrain_input_ids_batch.append(torch.tensor(full_teacher_ids, dtype=torch.long))
            pretrain_labels_batch.append(torch.tensor(labels, dtype=torch.long))
            pretrain_attention_mask_batch.append(torch.ones(len(full_teacher_ids), dtype=torch.long))
            metadata_batch.append(
                {
                    "mode": "sdft_baseline",
                    "question": question,
                    "answer": ref_answer,
                    "teacher_prompt": teacher_prompt,
                    "student_prompt": student_prompt,
                }
            )

        return {
            "student_prompt_input_ids": torch.nn.utils.rnn.pad_sequence(
                student_prompt_ids_batch,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            ),
            "student_prompt_attention_mask": torch.nn.utils.rnn.pad_sequence(
                student_prompt_attention_mask_batch,
                batch_first=True,
                padding_value=0,
            ),
            "pretrain_input_ids": torch.nn.utils.rnn.pad_sequence(
                pretrain_input_ids_batch,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            ),
            "pretrain_labels": torch.nn.utils.rnn.pad_sequence(
                pretrain_labels_batch,
                batch_first=True,
                padding_value=-100,
            ),
            "pretrain_attention_mask": torch.nn.utils.rnn.pad_sequence(
                pretrain_attention_mask_batch,
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
        self.current_phase = "pretrain"

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

    def set_phase(self, phase: str):
        self.current_phase = phase
        logger.info(f"Switched training phase to: {phase}")
        if phase == "sdft":
            self._reset_teacher_from_student()

    def _reset_teacher_from_student(self):
        self.teacher_model = copy.deepcopy(self.model)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def _ema_update_teacher(self):
        if self.teacher_model is None:
            return
        alpha = self.sdft_config.ema_alpha
        with torch.no_grad():
            for teacher_param, student_param in zip(self.teacher_model.parameters(), self.model.parameters()):
                student_data = student_param.detach().to(teacher_param.device)
                teacher_param.data.mul_(1.0 - alpha).add_(student_data, alpha=alpha)
        self.teacher_model.eval()

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self.current_phase == "sdft":
            self._ema_update_teacher()
        return loss

    def _compute_masked_logprob_and_entropy(self, active_model, input_ids, attention_mask, target_mask):
        outputs = active_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        shifted_mask = target_mask[:, 1:].bool()

        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        token_log_probs = token_log_probs * shifted_mask

        entropy = -(probs * log_probs).sum(dim=-1)
        entropy = entropy * shifted_mask

        token_counts = shifted_mask.sum(dim=1).clamp(min=1)
        seq_logprob = token_log_probs.sum(dim=1)
        seq_entropy = entropy.sum(dim=1)
        return seq_logprob, seq_entropy, token_counts.float(), outputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        metadata = inputs.get("metadata", [])

        if self.current_phase == "pretrain":
            pretrain_inputs = {
                "input_ids": inputs["pretrain_input_ids"].to(model.device),
                "attention_mask": inputs["pretrain_attention_mask"].to(model.device),
                "labels": inputs["pretrain_labels"].to(model.device),
            }
            outputs = model(**pretrain_inputs)
            loss = outputs.loss

            if model.training:
                debug_info = [
                    {
                        "student_logprob": 0.0,
                        "teacher_logprob": 0.0,
                        "gen_tokens": 0.0,
                        "entropy": 0.0,
                        "phase": "pretrain",
                    }
                    for _ in metadata
                ]
                sdft_tracker.update(loss.item(), debug_info)

            return (loss, outputs) if return_outputs else loss

        student_prompt_input_ids = inputs["student_prompt_input_ids"].to(model.device)
        student_prompt_attention_mask = inputs["student_prompt_attention_mask"].to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=student_prompt_input_ids,
                attention_mask=student_prompt_attention_mask,
                max_new_tokens=self.sdft_config.max_new_tokens,
                do_sample=True,
                temperature=self.sdft_config.temperature,
                top_p=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        generated_attention_mask = (generated_ids != self.tokenizer.pad_token_id).long()
        student_target_mask = torch.zeros_like(generated_ids, dtype=torch.long)
        prompt_lengths = student_prompt_attention_mask.sum(dim=1).tolist()
        for idx, prompt_len in enumerate(prompt_lengths):
            if prompt_len < generated_ids.size(1):
                student_target_mask[idx, prompt_len:] = generated_attention_mask[idx, prompt_len:]

        student_seq_logprob, student_seq_entropy, token_counts, student_outputs = self._compute_masked_logprob_and_entropy(
            model,
            generated_ids,
            generated_attention_mask,
            student_target_mask,
        )

        teacher_input_ids_batch = []
        teacher_attention_mask_batch = []
        teacher_target_mask_batch = []

        for idx, meta in enumerate(metadata):
            generated_text = self.tokenizer.decode(
                generated_ids[idx, prompt_lengths[idx] :],
                skip_special_tokens=True,
            )
            teacher_prompt = meta.get("teacher_prompt", "")
            teacher_prompt_ids = self.tokenizer(teacher_prompt, add_special_tokens=False).input_ids
            generated_text_ids = self.tokenizer(generated_text, add_special_tokens=False).input_ids
            full_teacher_ids = (teacher_prompt_ids + generated_text_ids)[: self.sdft_config.max_seq_length]

            teacher_target_mask = [0] * min(len(teacher_prompt_ids), len(full_teacher_ids))
            teacher_target_mask += [1] * (len(full_teacher_ids) - len(teacher_target_mask))
            teacher_target_mask = teacher_target_mask[: len(full_teacher_ids)]

            teacher_input_ids_batch.append(torch.tensor(full_teacher_ids, dtype=torch.long))
            teacher_attention_mask_batch.append(torch.ones(len(full_teacher_ids), dtype=torch.long))
            teacher_target_mask_batch.append(torch.tensor(teacher_target_mask, dtype=torch.long))

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
        teacher_target_mask = torch.nn.utils.rnn.pad_sequence(
            teacher_target_mask_batch,
            batch_first=True,
            padding_value=0,
        ).to(model.device)

        with torch.no_grad():
            teacher_seq_logprob, _, _, _ = self._compute_masked_logprob_and_entropy(
                self.teacher_model,
                teacher_input_ids,
                teacher_attention_mask,
                teacher_target_mask,
            )

        reverse_kl = student_seq_logprob - teacher_seq_logprob
        entropy_bonus = self.sdft_config.entropy_beta * student_seq_entropy
        loss = (reverse_kl - entropy_bonus).mean()

        if model.training:
            debug_info = []
            for idx in range(len(metadata)):
                debug_info.append(
                    {
                        "student_logprob": float(student_seq_logprob[idx].detach().cpu().item()),
                        "teacher_logprob": float(teacher_seq_logprob[idx].detach().cpu().item()),
                        "gen_tokens": float(token_counts[idx].detach().cpu().item()),
                        "entropy": float(student_seq_entropy[idx].detach().cpu().item()),
                        "phase": "sdft",
                    }
                )
            sdft_tracker.update(loss.item(), debug_info)

        outputs = {
            "student_log_probs": student_seq_logprob.detach(),
            "teacher_log_probs": teacher_seq_logprob.detach(),
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
            logger.info(f"  Phase:{stats['phase']}")
            logger.info(f"  Avg Train Loss:{stats['avg_train_loss']:.4f}")
            logger.info(f"  Avg Student LogProb:{stats['avg_student_logprob']:.4f}")
            logger.info(f"  Avg Teacher LogProb:{stats['avg_teacher_logprob']:.4f}")
            logger.info(f"  Avg Generated Tokens:{stats['avg_generated_tokens']:.2f}")
            logger.info(f"  Avg Entropy:{stats['avg_entropy']:.4f}")
            logger.info("=" * 60)
            sdft_tracker.reset_epoch()


def _build_model(sdft_config: SDFTConfig, lora_path: Optional[str] = None):
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
    return model


def _build_training_args(output_dir: str, batch_size: int, device_num: int, save_steps: int, learning_rate: float):
    per_device_batch_size = max(1, batch_size // max(device_num, 1))
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=batch_size,
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


def _run_phase(
    trainer: SDFTSequentialTrainer,
    phase_name: str,
    step_log_file: str,
    epoch_log_file: str,
    output_dir: str,
    metrics_log_interval: int,
    logical_epochs: int,
):
    trainer.set_phase(phase_name)
    sdft_tracker.reset_window()
    sdft_tracker.reset_epoch()

    total_samples = len(trainer.train_dataset)
    total_steps = max(1, total_samples // max(1, trainer.args.per_device_train_batch_size))
    steps_per_logical_epoch = max(1, total_steps // max(1, logical_epochs))

    trainer.callback_handler.callbacks = [
        cb for cb in trainer.callback_handler.callbacks if not isinstance(cb, (SDFTStepLogCallback, SDFTLogicalEpochLogCallback))
    ]
    trainer.add_callback(SDFTStepLogCallback(step_log_file, metrics_log_interval, trainer.sdft_config, output_dir))
    trainer.add_callback(SDFTLogicalEpochLogCallback(epoch_log_file, steps_per_logical_epoch, trainer.sdft_config, output_dir))

    trainer.train()


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
    - data_path: 数据集地址，数据格式与 adv_hints.json 一致，仅使用 question 和 ref_answer
    - model_path: 基座模型地址
    - batch_size: batch size
    - real_data_epochs: SDFT 自蒸馏 epoch 数
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
        entropy_beta=ENTROPY_BETA,
        pretrain_epochs=1,
    )

    output_dir = f"{sdft_config.output_base_dir}/sdft_baseline_{real_data_epochs}ep_{datetime.now().strftime('%m%d_%H%M')}"
    step_log_file, epoch_log_file = setup_logging(output_dir)

    logger.info(f"SDFT Baseline - Model Path: {sdft_config.model_path}")
    logger.info(f"SDFT Baseline - Data Path: {sdft_config.data_path}")
    logger.info(f"SDFT Baseline - Output Dir: {output_dir}")
    logger.info(f"SDFT Baseline - EMA Alpha: {sdft_config.ema_alpha}")
    logger.info(f"SDFT Baseline - Entropy Beta: {sdft_config.entropy_beta}")
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
    total_steps = max(1, total_samples // max(1, batch_size))
    save_steps = max(1, int(total_steps * max(1, real_data_epochs) / SAVE_TOTAL))

    model = _build_model(sdft_config, lora_path=lora_path)

    training_args = _build_training_args(
        output_dir=output_dir,
        batch_size=batch_size,
        device_num=device_num,
        save_steps=save_steps,
        learning_rate=1e-5,
    )

    trainer = SDFTSequentialTrainer(
        sdft_config=sdft_config,
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[],
    )

    logger.info("Starting SDFT stage 1: teacher demonstration pretraining...")
    _run_phase(
        trainer=trainer,
        phase_name="pretrain",
        step_log_file=step_log_file,
        epoch_log_file=epoch_log_file,
        output_dir=output_dir,
        metrics_log_interval=sdft_config.metrics_log_interval,
        logical_epochs=sdft_config.pretrain_epochs,
    )

    logger.info("Starting SDFT stage 2: self-distillation fine-tuning...")
    _run_phase(
        trainer=trainer,
        phase_name="sdft",
        step_log_file=step_log_file,
        epoch_log_file=epoch_log_file,
        output_dir=output_dir,
        metrics_log_interval=sdft_config.metrics_log_interval,
        logical_epochs=sdft_config.real_data_epochs,
    )

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

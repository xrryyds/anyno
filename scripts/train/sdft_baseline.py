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
from peft import PeftModel, LoraConfig, get_peft_model

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
    # max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
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


def _extract_ref_solution(item: Dict[str, Any]) -> str:
    value = item.get("ref_solution")
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
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _build_teacher_prompt(self, question: str, ref_solution: str) -> str:
        teacher_instruction = (
            f"<Question> {question}\n\n"
            f"This is an example for a response to the question:\n\n"
            f"<Demonstration>\n{ref_solution}\n\n"
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

    @staticmethod
    def _left_pad_sequence(tensors, padding_value):
        """将一组 1D tensor 按左侧对齐进行 padding（左 padding）。
        decoder-only 模型的 generate() 要求 prompt 左对齐 padding，
        否则会触发 'right-padding was detected' 警告并导致生成结果错误。
        """
        max_len = max(t.size(0) for t in tensors)
        padded = []
        for t in tensors:
            pad_len = max_len - t.size(0)
            if pad_len > 0:
                padded.append(torch.cat([torch.full((pad_len,), padding_value, dtype=t.dtype), t]))
            else:
                padded.append(t)
        return torch.stack(padded, dim=0)

    def __call__(self, batch):
        student_prompt_ids_batch = []
        student_prompt_attention_mask_batch = []
        pretrain_input_ids_batch = []
        pretrain_labels_batch = []
        pretrain_attention_mask_batch = []
        metadata_batch = []

        for item in batch:
            question = str(item["question"])
            ref_solution = _extract_ref_solution(item)

            student_prompt = self._build_student_prompt(question)
            student_prompt_ids = self.tokenizer(student_prompt, add_special_tokens=False).input_ids
            student_prompt_ids = student_prompt_ids[: self.max_length]

            teacher_prompt = self._build_teacher_prompt(question, ref_solution)
            teacher_prompt_ids = self.tokenizer(teacher_prompt, add_special_tokens=False).input_ids
            answer_ids = self.tokenizer(ref_solution, add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]
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
                    "answer": ref_solution,
                    "teacher_prompt": teacher_prompt,
                    "student_prompt": student_prompt,
                }
            )

        return {
            # student prompt 用于 generate()，必须左 padding
            "student_prompt_input_ids": self._left_pad_sequence(
                student_prompt_ids_batch,
                padding_value=self.tokenizer.pad_token_id,
            ),
            "student_prompt_attention_mask": self._left_pad_sequence(
                student_prompt_attention_mask_batch,
                padding_value=0,
            ),
            # pretrain 阶段的输入用于常规 forward，右 padding 即可
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
        # 修复: deepcopy 整个 PeftModel（而非 base_model.model），
        # 确保 teacher 和 student 参数结构完全一致，EMA zip 不会错位。
        import copy as _copy
        self.teacher_model = _copy.deepcopy(self.model)
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
                # max_new_tokens=self.sdft_config.max_new_tokens,
                do_sample=True,
                temperature=self.sdft_config.temperature,
                top_p=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        # 修复：当 pad_token_id == eos_token_id 时，不能简单用 != pad_token_id 来构建 mask，
        # 否则生成的 EOS token 会被排除在 loss 计算之外。
        # 改为：基于 prompt 长度和生成长度来构建 attention mask。
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            # 找到每个序列的实际长度（从右边找第一个非 pad/eos，但需要考虑生成末尾的 eos）
            generated_attention_mask = torch.ones_like(generated_ids, dtype=torch.long)
            # 只把左侧 padding 区域设为 0（generate 默认左 padding）
            for idx in range(generated_ids.size(0)):
                for pos in range(generated_ids.size(1)):
                    if generated_ids[idx, pos] != self.tokenizer.pad_token_id:
                        break
                    generated_attention_mask[idx, pos] = 0
        else:
            generated_attention_mask = (generated_ids != self.tokenizer.pad_token_id).long()
        student_target_mask = torch.zeros_like(generated_ids, dtype=torch.long)
        # 修复: 左 padding 场景下，generate() 返回的序列中 prompt 部分的结束位置
        # 是 input_ids 的总长度（含 padding），而不是实际 prompt token 数量。
        # 因为 generated_ids = [左padding + 实际prompt + 生成tokens]，
        # 其中 [左padding + 实际prompt] 的长度 == student_prompt_input_ids.size(1)。
        prompt_end_pos = student_prompt_input_ids.size(1)
        for idx in range(generated_ids.size(0)):
            if prompt_end_pos < generated_ids.size(1):
                student_target_mask[idx, prompt_end_pos:] = generated_attention_mask[idx, prompt_end_pos:]

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
            # 修复 Bug1: 原来先 decode 再 encode（round-trip），由于 tokenizer 的
            # decode→encode 不完全可逆（空格/特殊字符处理差异），会导致 teacher 侧
            # token 数量与 student 侧不一致，KL 散度估计存在系统性偏差。
            # 修复方案：直接复用 generated_ids 中的 token ids，去掉末尾 padding，
            # 避免 round-trip 引入的 token 数量偏差。
            teacher_prompt = meta.get("teacher_prompt", "")
            teacher_prompt_ids = self.tokenizer(teacher_prompt, add_special_tokens=False).input_ids

            # 直接取 student 生成部分的 token ids（去掉左侧 padding+prompt 和右侧 padding）
            # 修复: 左 padding 场景下，生成部分从 prompt_end_pos 开始，而非 prompt_len
            gen_token_ids = generated_ids[idx, prompt_end_pos:].tolist()
            # 去掉末尾的 pad token。
            # 修复: 当 pad_token_id == eos_token_id 时，不能简单地 pop 所有末尾的
            # pad_id，否则会把生成的 EOS 也删掉。改为：只去掉 generate() 因 batch
            # 对齐而追加的多余 padding（即超出实际生成长度的部分）。
            # 利用 generated_attention_mask 来判断哪些位置是真实 token。
            gen_attn = generated_attention_mask[idx, prompt_end_pos:].tolist()
            # 只保留 attention_mask == 1 的 token
            gen_token_ids = [tid for tid, m in zip(gen_token_ids, gen_attn) if m == 1]

            full_teacher_ids = (teacher_prompt_ids + gen_token_ids)[: self.sdft_config.max_seq_length]

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

        # 修复: 按 token 数量归一化，避免长序列对 loss 产生不成比例的影响
        reverse_kl = (student_seq_logprob - teacher_seq_logprob) / token_counts
        entropy_bonus = self.sdft_config.entropy_beta * (student_seq_entropy / token_counts)
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
    # 修复 Bug3: prepare_model_for_kbit_training 专为量化训练设计，
    # 在 bfloat16 全精度场景下语义错误（会对 LayerNorm 做多余的 float32 上转型）。
    # 改为手动冻结 base model 参数并启用 gradient checkpointing。
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for param in model.parameters():
        param.requires_grad = False

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
            model = PeftModel.from_pretrained(model, lora_path, is_trainable=True)
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

    # 修复 Bug 2: 将 num_train_epochs 设置为 logical_epochs，确保数据集被遍历正确的次数
    trainer.args.num_train_epochs = logical_epochs

    # 修复 Bug 3: 重置 optimizer 和 lr_scheduler，避免复用上一阶段已衰减的学习率
    trainer.optimizer = None
    trainer.lr_scheduler = None

    total_samples = len(trainer.train_dataset)
    per_device_bs = max(1, trainer.args.per_device_train_batch_size)
    # 修复 Bug2: 多 GPU 时每个 step 实际消耗 per_device_bs × world_size 个样本，
    # 原来只用 per_device_bs 会高估 world_size 倍，导致 epoch 边界判断提前触发。
    world_size = max(1, trainer.args.world_size)
    effective_bs = per_device_bs * world_size
    # 每个 epoch 的 step 数（单个 epoch 遍历一次数据集）
    steps_per_epoch = max(1, total_samples // effective_bs)
    steps_per_logical_epoch = steps_per_epoch  # 一个 logical epoch = 一次完整遍历

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
    # max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_seq_length: int = MAX_SEQ_LENGTH,
    ema_alpha: float = EMA_ALPHA,
):
    """SDFT baseline training API.

    输入：
    - data_path: 数据集地址，数据格式与 adv_hints.json 一致，仅使用 question 和 ref_solution
    - model_path: 基座模型地址
    - batch_size: batch size
    - real_data_epochs: SDFT 自蒸馏 epoch 数
    """
    current_file_path = os.path.abspath(__file__)
    # 修复 Bug 1: sdft_baseline.py 在 scripts/train/ 下，只需 3 次 dirname 到达项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

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
        # max_new_tokens=max_new_tokens,
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
    # logger.info(f"SDFT Baseline - Max New Tokens: {sdft_config.max_new_tokens}")

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
    # 修复: generate() 需要左 padding 才能正确对齐 prompt，
    # 否则 sdft 阶段构建 student_target_mask 时 prompt_len 偏移会出错。
    tokenizer.padding_side = "left"

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
    # 保存基座模型路径，方便 take_exam 用 use_lora=True 加载
    meta = {"base_model_path": sdft_config.model_path}
    with open(os.path.join(output_dir, "sdft_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info(f"SDFT baseline training finished. Model saved to {output_dir}")
    logger.info(f"To use in take_exam: TakeExam(model_path='{sdft_config.model_path}', use_lora=True, adapter_path='{output_dir}')")

    return output_dir


if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    # 修复 Bug 1: sdft_baseline.py 在 scripts/train/ 下，只需 3 次 dirname 到达项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

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

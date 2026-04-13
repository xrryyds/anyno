import os
# 与 student_train_v2 一致的 vLLM 相关环境变量
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import sys
import json
import math
import random
import torch
import logging
import warnings
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from datasets import Dataset
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed
)
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ==========================================
# Logger 配置（保持一致）
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 全局训练序列长度超参数
MAX_SEQ_LENGTH = 2048

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")

SYSTEM_PROMPT = "Please reason step by step and put your final answer within \\boxed{}."

# ==========================================
# 1. 配置
# ==========================================

@dataclass
class SFTConfig:
    model_path: str = ""
    data_path: str = ""
    output_base_dir: str = "/root/autodl-tmp/output"
    real_data_epochs: int = 50

    # 日志相关
    metrics_log_interval: int = 8

    # LoRA / 训练相关（占位，保持与 SIRA 配置风格相似）
    split_r: float = 0.5
    anchor_loss_weight_k: float = 1
    suppress_max_scale: float = 1.0
    anchor_sigmoid_slope: float = 20.0
    anchor_loss_tolerance: float = 1.01


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(output_dir, "train.log"), encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    return (
        os.path.join(output_dir, "step_metrics.jsonl"),
        os.path.join(output_dir, "epoch_metrics.jsonl"),
    )


# ==========================================
# 2. Metrics Tracker：结构对齐，但指标精简为 SFT
# ==========================================
class SFTTrainingMetricsTracker:
    def __init__(self):
        self.reset_window()
        self.reset_epoch()

    def reset_window(self):
        self.win_loss, self.win_steps = 0.0, 0

    def reset_epoch(self):
        self.ep_loss, self.ep_steps = 0.0, 0

    def update(self, loss):
        self.win_loss += loss
        self.win_steps += 1
        self.ep_loss += loss
        self.ep_steps += 1

    def _calculate_stats(self, loss_sum, steps):
        return {
            "avg_train_loss": loss_sum / max(steps, 1),
        }

    def get_window_stats(self):
        return self._calculate_stats(self.win_loss, self.win_steps)

    def get_epoch_stats(self):
        return self._calculate_stats(self.ep_loss, self.ep_steps)


sft_tracker = SFTTrainingMetricsTracker()


# ==========================================
# 3. Data Collator：与 student_train_v2 数据格式一致
#   - 使用 system prompt + question 作为输入
#   - 仅 answer 部分参与 loss
# ==========================================
class SFTCollator:
    def __init__(self, tokenizer, max_length: int = MAX_SEQ_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, batch):
        input_ids_batch, labels_batch = [], []
        attention_mask_batch, metadata_batch = [], []

        for item in batch:
            q = item["question"]
            a = item.get("answer", "")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(q)},
            ]
            prompt_str = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = self.tokenizer(prompt_str, add_special_tokens=False).input_ids
            len_prompt = len(prompt_ids)

            answer_ids = (
                self.tokenizer(str(a), add_special_tokens=False).input_ids
                + [self.tokenizer.eos_token_id]
            )

            full_ids = prompt_ids + answer_ids

            if len(full_ids) > self.max_length:
                full_ids = full_ids[: self.max_length]

            # 仅让 answer 部分参与 loss
            labels = [-100] * len(full_ids)
            for i in range(len_prompt, len(full_ids)):
                labels[i] = full_ids[i]

            input_ids_batch.append(torch.tensor(full_ids, dtype=torch.long))
            labels_batch.append(torch.tensor(labels, dtype=torch.long))
            attention_mask_batch.append(torch.ones(len(full_ids), dtype=torch.long))
            metadata_batch.append({"mode": "sft_baseline"})

        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                input_ids_batch,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            ),
            "labels": torch.nn.utils.rnn.pad_sequence(
                labels_batch, batch_first=True, padding_value=-100
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                attention_mask_batch, batch_first=True, padding_value=0
            ),
            "metadata": metadata_batch,
        }


# ==========================================
# 4. Baseline SFT Trainer
#    - 使用标准 CE loss
#    - 其余 Trainer 配置对齐 student_train_v2
# ==========================================
class SFTSequentialTrainer(Trainer):
    def __init__(self, sft_config: SFTConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sft_config = sft_config

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

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        metadata = inputs.pop("metadata", None)  # 保留接口对齐但不使用
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")

        # 标准 CrossEntropyLoss，忽略 label=-100 的位置
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        if self.model.training:
            sft_tracker.update(loss.item())

        return (loss, outputs) if return_outputs else loss


# ==========================================
# 5. 回调：日志格式与 student_train_v2 尽量一致
# ==========================================
class SFTStepLogCallback(TrainerCallback):
    def __init__(self, log_file, log_interval, config: SFTConfig, output_dir: str):
        self.log_file = log_file
        self.log_interval = log_interval
        self.config = config
        self.output_dir = output_dir

    def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.log_interval == 0:
            stats = sft_tracker.get_window_stats()
            json_stats = stats.copy()
            json_stats.update(
                {
                    "epoch": state.epoch,
                    "global_step": state.global_step,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(json_stats) + "\n")

            log_parts = [f"[Step {state.global_step}]"]
            for k, v in stats.items():
                val_str = f"{v:.6f}" if isinstance(v, float) else str(v)
                log_parts.append(f"{k}: {val_str}")
            logger.info(" | ".join(log_parts))

            sft_tracker.reset_window()


class SFTLogicalEpochLogCallback(TrainerCallback):
    """
    逻辑 Epoch 日志打印，不做早停，只记录 epoch 级别 loss
    """

    def __init__(self, log_file, steps_per_logical_epoch, config: SFTConfig, output_dir: str):
        self.log_file = log_file
        self.steps_per_epoch = steps_per_logical_epoch
        self.config = config
        self.output_dir = output_dir
        self.current_logical_epoch = 0

    def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.steps_per_epoch == 0:
            self.current_logical_epoch += 1

            stats = sft_tracker.get_epoch_stats()
            stats.update(
                {
                    "logical_epoch": self.current_logical_epoch,
                    "global_step": state.global_step,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(stats) + "\n")

            logger.info("=" * 60)
            logger.info(f"*** LOGICAL EPOCH {self.current_logical_epoch} FINISHED ***")
            logger.info(f"  Avg Train Loss:{stats['avg_train_loss']:.4f}")
            logger.info("=" * 60)
            sft_tracker.reset_epoch()


# ==========================================
# 6. Main Execution Function
#    - 所有路径通过参数传入（model / data / output）
#    - 训练精度、scheduler、logging_steps 等与 student_train_v2 对齐
# ==========================================
def run_sft_training_baseline(
    model_path: str,
    data_path: str,
    output_base_dir: str,
    batch_size: int = 4,
    real_data_epochs: int = 50,
    device_num: int = 1,
    lora_path: Optional[str] = None,
):
    set_seed(42)
    sft_tracker.reset_window()
    sft_tracker.reset_epoch()

    sft_config = SFTConfig(
        model_path=model_path,
        data_path=data_path,
        output_base_dir=output_base_dir,
        real_data_epochs=real_data_epochs,
        metrics_log_interval=batch_size,
    )

    output_dir = f"{sft_config.output_base_dir}/sft_baseline_{real_data_epochs}ep_{datetime.now().strftime('%m%d_%H%M')}"
    step_log_file, epoch_log_file = setup_logging(output_dir)

    logger.info(f"SFT Baseline - Model Path: {sft_config.model_path}")
    logger.info(f"SFT Baseline - Data Path: {sft_config.data_path}")
    logger.info(f"SFT Baseline - Output Dir: {output_dir}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            sft_config.model_path, trust_remote_code=True, use_fast=False
        )
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise e

    if not os.path.exists(sft_config.data_path):
        raise FileNotFoundError(f"Dataset not found at {sft_config.data_path}")
    dataset = Dataset.from_json(sft_config.data_path)

    collator = SFTCollator(tokenizer)

    total_samples = len(dataset)
    total_steps = total_samples // batch_size
    steps_per_logical_epoch = total_steps // real_data_epochs

    if steps_per_logical_epoch < 1:
        steps_per_logical_epoch = 1
        logger.warning(
            f"Total steps ({total_steps}) < real epochs ({real_data_epochs}). "
            f"Set logical step to 1."
        )

    model = AutoModelForCausalLM.from_pretrained(
        sft_config.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # LoRA resume support，与 student_train_v2 对齐
    use_existing_lora = False
    if lora_path is not None and str(lora_path).strip():
        lora_path = str(lora_path).strip()
        if os.path.exists(lora_path) and os.path.isdir(lora_path):
            use_existing_lora = True
            logger.info(f"Using existing LoRA from: {lora_path}")
        else:
            logger.warning(
                f"Provided lora_path '{lora_path}' does not exist or is not a directory. "
                "Falling back to fresh LoRA initialization."
            )

    if use_existing_lora:
        try:
            model = PeftModel.from_pretrained(model, lora_path)
            logger.info(f"Successfully loaded existing LoRA weights from '{lora_path}'.")
        except (OSError, ValueError) as e:
            logger.error(
                f"Failed to load LoRA weights from '{lora_path}': {e}. "
                "Falling back to fresh LoRA initialization."
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

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,                         # 与 student_train_v2 一致：逻辑 epoch 自己算
        per_device_train_batch_size=batch_size // device_num,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,                        # 训练速率对齐
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=sft_config.metrics_log_interval,
        save_strategy="steps",
        save_steps=steps_per_logical_epoch,
        save_total_limit=2,
        fp16=False,
        bf16=True,                                 # 训练精度对齐
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_drop_last=True,
        report_to="none",
    )

    step_callback = SFTStepLogCallback(
        step_log_file, sft_config.metrics_log_interval, sft_config, output_dir
    )
    epoch_callback = SFTLogicalEpochLogCallback(
        epoch_log_file, steps_per_logical_epoch, sft_config, output_dir
    )

    trainer = SFTSequentialTrainer(
        sft_config=sft_config,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[step_callback, epoch_callback],
    )

    logger.info("Starting SFT baseline training...")
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"SFT baseline training finished. Model saved to {output_dir}")


if __name__ == "__main__":
    """
    示例：默认路径从项目结构推断，你可以根据自己工程改掉
    """
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path))
    project_root = os.path.dirname(os.path.dirname(project_root))

    default_model_dir = os.path.join(project_root, "CELPO", "model", "OREAL")
    default_model_url = os.path.join(default_model_dir, "OREAL-7B")
    default_data_path = os.path.join(project_root, "CELPO", "datasets", "exam", "irdcl_data.json")
    default_output_base_dir = os.path.join(project_root, "CELPO", "output")

    run_sft_training_baseline(
        model_path=default_model_url,
        data_path=default_data_path,
        output_base_dir=default_output_base_dir,
        batch_size=8,
        real_data_epochs=50,
    )

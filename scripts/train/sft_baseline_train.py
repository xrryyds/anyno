import os
import json
import torch
import logging
import warnings
from datetime import datetime
from typing import Optional

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

# 全局训练序列长度超参数（SFT collator 截断用）
MAX_SEQ_LENGTH = 2048

SYSTEM_PROMPT = "Please reason step by step and put your final answer within \\boxed{}."


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(output_dir, "train.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    return os.path.join(output_dir, "step_metrics.jsonl")


class SFTCollator:
    def __init__(self, tokenizer, max_length: int = MAX_SEQ_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, batch):
        input_ids_list, labels_list, attn_list = [], [], []
        for item in batch:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(item["question"])},
            ]
            prompt_str = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = self.tokenizer(prompt_str, add_special_tokens=False).input_ids
            answer_ids = (
                self.tokenizer(str(item["answer"]), add_special_tokens=False).input_ids
                + [self.tokenizer.eos_token_id]
            )
            full_ids = (prompt_ids + answer_ids)[: self.max_length]
            labels = ([-100] * len(prompt_ids) + answer_ids)[: self.max_length]
            input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))
            attn_list.append(torch.ones(len(full_ids), dtype=torch.long))

        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id),
            "labels": torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(attn_list, batch_first=True, padding_value=0),
        }


def run_sft_baseline_training(
    model_path: str,
    data_path: Optional[str] = None,
    output_base_dir: Optional[str] = None,
    batch_size: int = 8,
    epochs: int = 3,
    device_num: int = 1,
):
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))

    if data_path is None:
        data_path = os.path.join(project_root, "CELPO", "datasets", "exam", "irdcl_data.json")
    if output_base_dir is None:
        output_base_dir = os.path.join(project_root, "CELPO", "output")

    set_seed(42)

    output_dir = f"{output_base_dir}/sft_baseline_{epochs}ep_{datetime.now().strftime('%m%d_%H%M')}"
    step_log_file = setup_logging(output_dir)

    logger.info(f"Model Path: {model_path}")
    logger.info(f"Data Path: {data_path}")
    logger.info(f"Output Dir: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    dataset = Dataset.from_json(data_path)
    logger.info(f"Loaded {len(dataset)} samples from {data_path}")

    collator = SFTCollator(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM", bias="none"
    )
    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size // device_num,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=batch_size,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=False, bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_drop_last=True,
        group_by_length=False,
        report_to="none",
    )

    from transformers import TrainerCallback

    class _StepLogCB(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                record = {"step": state.global_step, **{k: v for k, v in logs.items() if isinstance(v, (int, float))}}
                with open(step_log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[_StepLogCB()],
    )

    logger.info("Starting SFT baseline training...")
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"SFT baseline training finished. Model saved to {output_dir}")


if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
    model_path = os.path.join(project_root, "CELPO", "model", "OREAL", "OREAL-7B")

    run_sft_baseline_training(
        model_path=model_path,
        batch_size=8,
        epochs=10,
    )

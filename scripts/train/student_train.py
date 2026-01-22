import os
import sys
import json
import random
import torch
import torch.nn.functional as F
import transformers
import logging
from datetime import datetime
from collections import OrderedDict
from dataclasses import dataclass
import peft

from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments,
    TrainerCallback,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 假设 prompt 模块已存在
try:
    from prompt import GEN_PROMPT, GEN_HINTS_WIH_ANSWER, GEN_ENHANCE_PROMPT
except ImportError:
    # 占位符，防止代码无法运行
    GEN_PROMPT = "Question: {question}\nAnswer:"
    GEN_ENHANCE_PROMPT = "Question: {question}\nHints: {hints}\nAnswer:"
    GEN_HINTS_WIH_ANSWER = "# known:\n{hints}\n{answer}"

# ==========================================
# 1. 配置与工具类
# ==========================================

@dataclass
class HintSFTConfig:
    p_hint_start: float = 0.95    
    p_hint_end: float = 0.10      
    hint_fixed_weight: float = 1.0 
    gate_threshold: float = 2.5    
    gate_slope: float = 3.0       
    debug_sample_steps: int = 50

logger = logging.getLogger(__name__)

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_format = "[%(asctime)s][%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(os.path.join(output_dir, "train.log"), encoding='utf-8')
        ]
    )
    return os.path.join(output_dir, "metrics.jsonl")

# ==========================================
# 2. Data Collator (课程学习逻辑)
# ==========================================

class HintDropoutCollator:
    def __init__(self, tokenizer, hint_config: HintSFTConfig, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.config = hint_config
        self.max_length = max_length
        self.current_step = 0
        self.total_steps = 1
        
        if self.tokenizer.pad_token_id is None:
             self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def set_progress(self, step, total):
        self.current_step = step
        self.total_steps = max(total, 1)

    def get_current_p_hint(self):
        progress = min(self.current_step / self.total_steps, 1.0)
        return self.config.p_hint_start + progress * (self.config.p_hint_end - self.config.p_hint_start)

    def __call__(self, batch):
        p_hint = self.get_current_p_hint()
        input_ids_batch, labels_batch = [], []
        hint_masks_batch, answer_masks_batch = [], []
        attention_mask_batch, metadata_batch = [], []

        for item in batch:
            q, b, c = item['question'], item.get('hints', ""), item['answer']
            has_hint_data = (b is not None and len(b.strip()) > 0)

            if not has_hint_data:
                # Scenario 1: Pure SFT
                full_text = GEN_PROMPT.format(question=q) + c 
                mode = "pure_sft_anchor"
                full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]
                prompt_text = GEN_PROMPT.format(question=q)
                len_prompt = len(self.tokenizer(prompt_text, add_special_tokens=False).input_ids)
                
                h_mask, a_mask = [0]*len(full_ids), [0]*len(full_ids)
                for i in range(min(len_prompt, len(full_ids)), len(full_ids)): a_mask[i] = 1
            else:
                # Scenario 2: IRDCL Modes
                if random.random() < p_hint:
                    # Mode A: q + h -> a
                    full_text = GEN_ENHANCE_PROMPT.format(question=q, hints=b) + c
                    mode = "mode_a_utilization"
                    full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]
                    len_prompt = len(self.tokenizer(GEN_ENHANCE_PROMPT.format(question=q, hints=b), add_special_tokens=False).input_ids)
                    h_mask, a_mask = [0]*len(full_ids), [0]*len(full_ids)
                    for i in range(min(len_prompt, len(full_ids)), len(full_ids)): a_mask[i] = 1
                else:
                    # Mode B: q -> h + a
                    full_text = GEN_PROMPT.format(question=q) + GEN_HINTS_WIH_ANSWER.format(hints=b, answer=c)
                    mode = "mode_b_generation"
                    full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]
                    len_prompt = len(self.tokenizer(GEN_PROMPT.format(question=q), add_special_tokens=False).input_ids)
                    len_hint_end = len(self.tokenizer(GEN_PROMPT.format(question=q) + "# known:\n" + b, add_special_tokens=False).input_ids)
                    h_mask, a_mask = [0]*len(full_ids), [0]*len(full_ids)
                    for i in range(min(len_prompt, len(full_ids)), min(len_hint_end, len(full_ids))): h_mask[i] = 1
                    for i in range(min(len_hint_end, len(full_ids)), len(full_ids)): a_mask[i] = 1

            if len(full_ids) > self.max_length:
                full_ids, h_mask, a_mask = full_ids[:self.max_length], h_mask[:self.max_length], a_mask[:self.max_length]

            labels = [full_ids[i] if (h_mask[i] or a_mask[i]) else -100 for i in range(len(full_ids))]
            
            input_ids_batch.append(torch.tensor(full_ids))
            labels_batch.append(torch.tensor(labels))
            hint_masks_batch.append(torch.tensor(h_mask, dtype=torch.float32))
            answer_masks_batch.append(torch.tensor(a_mask, dtype=torch.float32))
            attention_mask_batch.append(torch.ones(len(full_ids)))
            metadata_batch.append({"mode": mode, "p_hint": p_hint, "raw_text": full_text[:50]})

        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(input_ids_batch, batch_first=True, padding_value=self.tokenizer.pad_token_id),
            "labels": torch.nn.utils.rnn.pad_sequence(labels_batch, batch_first=True, padding_value=-100),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(attention_mask_batch, batch_first=True, padding_value=0),
            "hint_masks": torch.nn.utils.rnn.pad_sequence(hint_masks_batch, batch_first=True, padding_value=0.0),
            "answer_masks": torch.nn.utils.rnn.pad_sequence(answer_masks_batch, batch_first=True, padding_value=0.0),
            "metadata": metadata_batch
        }

# ==========================================
# 3. Trainer (Adaptive Gating 修复)
# ==========================================

class HintSFTTrainer(Trainer):
    def __init__(self, hint_config: HintSFTConfig, snapshot_file: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hint_config = hint_config
        self.snapshot_file = snapshot_file

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        hint_masks = inputs.pop("hint_masks")
        answer_masks = inputs.pop("answer_masks")
        metadata = inputs.pop("metadata")
        
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss, debug_info = self.adaptive_gating_loss(logits, labels, hint_masks, answer_masks)

        if self.state.global_step % self.hint_config.debug_sample_steps == 0 and self.state.is_local_process_zero:
            self.save_debug_snapshot(metadata, loss.item(), debug_info)
            
        return (loss, outputs) if return_outputs else loss

    def adaptive_gating_loss(self, logits, labels, hint_masks, answer_masks):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_h_masks = hint_masks[..., 1:].contiguous()
        shift_a_masks = answer_masks[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        token_losses = token_losses.view(shift_labels.shape)

        batch_losses = []
        weights_log = []

        for i in range(token_losses.size(0)):
            h_m, a_m = shift_h_masks[i], shift_a_masks[i]
            h_count, a_count = h_m.sum(), a_m.sum()
            
            if h_count > 0:
                # Mode B
                avg_h_loss = (token_losses[i] * h_m).sum() / h_count
                gate = torch.sigmoid(self.hint_config.gate_slope * (self.hint_config.gate_threshold - avg_h_loss.detach()))
                avg_a_loss = (token_losses[i] * a_m).sum() / a_count if a_count > 0 else 0.0
                batch_losses.append(self.hint_config.hint_fixed_weight * avg_h_loss + gate * avg_a_loss)
                weights_log.append(f"ModeB(Gate={gate:.2f})")
            else:
                # Mode A / Anchor
                avg_a_loss = (token_losses[i] * a_m).sum() / a_count if a_count > 0 else torch.tensor(0.0, device=logits.device)
                batch_losses.append(avg_a_loss)
                weights_log.append("Anchor")

        return torch.stack(batch_losses).mean(), weights_log

    def save_debug_snapshot(self, metadata, current_loss, debug_info):
        with open(self.snapshot_file, "a", encoding="utf-8") as f:
            entry = {"step": self.state.global_step, "loss": current_loss, "mode": metadata[0]["mode"], "gate": debug_info[0]}
            f.write(json.dumps(entry) + "\n")

# ==========================================
# 4. Main Execution
# ==========================================

class CurriculumCallback(TrainerCallback):
    def __init__(self, collator): self.collator = collator
    def on_step_begin(self, args, state, control, **kwargs):
        self.collator.set_progress(state.global_step, state.max_steps)

def main():
    set_seed(42)
    model_path = "/root/autodl-tmp/model/Qwen/Qwen/Qwen2.5-Math-7B-Instruct"
    output_dir = f"/root/autodl-tmp/output/hint_sft_{datetime.now().strftime('%m%d_%H%M')}"
    data_path = "./datasets/exam/irdcl_data.json"

    setup_logging(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dataset = Dataset.from_json(data_path)

    # 模型加载
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    
    # --- 关键修复 1: 准备训练 ---
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    peft_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM", bias="none"
    )
    model = get_peft_model(model, peft_config)
    
    # --- 关键修复 2: 开启输入梯度追踪 ---
    model.enable_input_require_grads() 

    hint_config = HintSFTConfig(hint_fixed_weight=2.0)
    collator = HintDropoutCollator(tokenizer, hint_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        logging_steps=10,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, # 消除警告
        save_total_limit=2,
        remove_unused_columns=False
    )

    trainer = HintSFTTrainer(
        hint_config=hint_config,
        snapshot_file=os.path.join(output_dir, "debug.jsonl"),
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[CurriculumCallback(collator)]
    )

    logger.info("Starting training...")
    trainer.train()
    trainer.save_model(output_dir)

if __name__ == "__main__":
    main()
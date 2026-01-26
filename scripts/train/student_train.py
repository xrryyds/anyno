import os
import sys
import json
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 过滤烦人的警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")

# 假设 prompt 模块
try:
    from prompt import GEN_PROMPT, GEN_HINTS_WIH_ANSWER, GEN_ENHANCE_PROMPT
except ImportError:
    GEN_PROMPT = "Question: {question}\nAnswer:"
    GEN_HINTS_WIH_ANSWER = "# known:\n{hints}\n{answer}"

# ==========================================
# 1. 配置与工具类
# ==========================================

@dataclass
class HintSFTConfig:
    # --- Mode B (Generation) 配置 ---
    # Mode B 权重固定为 1.0
    hint_fixed_weight: float = 1.0 
    gate_threshold: float = 2.5    
    gate_slope: float = 3.0       
    
    # --- Mode A (Anchor) 动态权重衰减配置 ---
    # 随着训练进行，Anchor 的权重从 start 线性衰减到 end
    # 实验证明：即使降到 0.1，Anchor Loss 依然保持稳定，这是极好的结果
    anchor_weight_start: float = 1.0
    anchor_weight_end: float = 0.01
    
    # 路径配置
    model_path: str = "/root/autodl-tmp/model/Qwen/Qwen/Qwen2.5-Math-7B-Instruct"
    data_path: str = "/xrr/CELPO/datasets/exam/irdcl_data.json" 
    output_base_dir: str = "/root/autodl-tmp/output"

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
    return os.path.join(output_dir, "epoch_metrics.jsonl")

# ==========================================
# 2. Metrics Tracker (统计追踪器)
# ==========================================

class TrainingMetricsTracker:
    """用于在内存中累积统计数据，以便按 Epoch 记录"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.total_loss = 0.0
        self.steps = 0
        self.gate_values = []
        self.anchor_losses = []
        self.mode_b_losses = []
        self.current_anchor_weight = 0.0 # 记录当前的 anchor 权重
        self.mode_b_counts = 0
        self.anchor_counts = 0

    def update(self, loss, metadata, debug_info, current_anchor_w):
        self.total_loss += loss
        self.steps += 1
        self.current_anchor_weight = current_anchor_w # 更新当前权重
        
        for meta, info in zip(metadata, debug_info):
            mode = meta.get("mode")
            if mode == "mode_b_generation":
                self.mode_b_counts += 1
                self.mode_b_losses.append(info["loss_contrib"])
                self.gate_values.append(info["gate"])
            elif mode == "pure_sft_anchor":
                self.anchor_counts += 1
                self.anchor_losses.append(info["loss_contrib"])

    def get_epoch_stats(self):
        avg_gate = sum(self.gate_values) / len(self.gate_values) if self.gate_values else 0.0
        avg_anchor_loss = sum(self.anchor_losses) / len(self.anchor_losses) if self.anchor_losses else 0.0
        avg_mode_b_loss = sum(self.mode_b_losses) / len(self.mode_b_losses) if self.mode_b_losses else 0.0
        
        return {
            "avg_train_loss": self.total_loss / max(self.steps, 1),
            "avg_gate_value": avg_gate,
            "avg_anchor_loss": avg_anchor_loss,
            "avg_mode_b_loss": avg_mode_b_loss,
            "anchor_weight": self.current_anchor_weight, # 记录这一轮结束时的权重
            "sample_counts": {
                "anchor": self.anchor_counts,
                "mode_b": self.mode_b_counts
            }
        }

tracker = TrainingMetricsTracker()

# ==========================================
# 3. Data Collator (固定模式逻辑)
# ==========================================

class FixedModeCollator:
    def __init__(self, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token_id is None:
             self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, batch):
        input_ids_batch, labels_batch = [], []
        hint_masks_batch, answer_masks_batch = [], []
        attention_mask_batch, metadata_batch = [], []

        for item in batch:
            q = item['question']
            b = item.get('hints', "")
            c = item.get('answer') 
            data_type = item.get('type', 'anchor_data') 

            # --- Logic Branching based on Data Type ---
            
            if data_type == 'anchor_data':
                # === Anchor Mode (Pure SFT) ===
                full_text = GEN_PROMPT.format(question=q) + c 
                mode_str = "pure_sft_anchor"
                
                full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]
                prompt_ids = self.tokenizer(GEN_PROMPT.format(question=q), add_special_tokens=False).input_ids
                len_prompt = len(prompt_ids)
                
                h_mask = [0] * len(full_ids) 
                a_mask = [0] * len(full_ids)
                for i in range(min(len_prompt, len(full_ids)), len(full_ids)): 
                    a_mask[i] = 1

            else: 
                # === Hint Mode (Mode B: Generation) ===
                full_text = GEN_PROMPT.format(question=q) + GEN_HINTS_WIH_ANSWER.format(hints=b, answer=c)
                mode_str = "mode_b_generation"
                
                full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]
                prompt_ids = self.tokenizer(GEN_PROMPT.format(question=q), add_special_tokens=False).input_ids
                len_prompt = len(prompt_ids)
                
                hint_part = GEN_PROMPT.format(question=q) + "# known:\n" + b
                len_hint_end = len(self.tokenizer(hint_part, add_special_tokens=False).input_ids)
                
                h_mask = [0] * len(full_ids)
                a_mask = [0] * len(full_ids)
                
                for i in range(min(len_prompt, len(full_ids)), min(len_hint_end, len(full_ids))):
                    h_mask[i] = 1 # Hint loss
                for i in range(min(len_hint_end, len(full_ids)), len(full_ids)):
                    a_mask[i] = 1 # Answer loss

            # --- Padding & Truncation ---
            if len(full_ids) > self.max_length:
                full_ids = full_ids[:self.max_length]
                h_mask = h_mask[:self.max_length]
                a_mask = a_mask[:self.max_length]

            labels = [full_ids[i] if (h_mask[i] or a_mask[i]) else -100 for i in range(len(full_ids))]
            
            input_ids_batch.append(torch.tensor(full_ids))
            labels_batch.append(torch.tensor(labels))
            hint_masks_batch.append(torch.tensor(h_mask, dtype=torch.float32))
            answer_masks_batch.append(torch.tensor(a_mask, dtype=torch.float32))
            attention_mask_batch.append(torch.ones(len(full_ids)))
            metadata_batch.append({"mode": mode_str})

        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(input_ids_batch, batch_first=True, padding_value=self.tokenizer.pad_token_id),
            "labels": torch.nn.utils.rnn.pad_sequence(labels_batch, batch_first=True, padding_value=-100),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(attention_mask_batch, batch_first=True, padding_value=0),
            "hint_masks": torch.nn.utils.rnn.pad_sequence(hint_masks_batch, batch_first=True, padding_value=0.0),
            "answer_masks": torch.nn.utils.rnn.pad_sequence(answer_masks_batch, batch_first=True, padding_value=0.0),
            "metadata": metadata_batch
        }

# ==========================================
# 4. Sequential Trainer (线性衰减)
# ==========================================

class SequentialTrainer(Trainer):
    def __init__(self, hint_config: HintSFTConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hint_config = hint_config

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        
        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=SequentialSampler(train_dataset), 
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        hint_masks = inputs.pop("hint_masks")
        answer_masks = inputs.pop("answer_masks")
        metadata = inputs.pop("metadata")
        
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss, debug_info_list, current_anchor_w = self.adaptive_gating_loss(logits, labels, hint_masks, answer_masks)
        
        # 更新全局统计器
        if self.model.training:
            tracker.update(loss.item(), metadata, debug_info_list, current_anchor_w)
            
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
        debug_info_list = [] 
        
        # --- 计算当前的动态 Anchor 权重 ---
        # 公式: w(t) = start - (t / T) * (start - end)
        current_step = self.state.global_step
        max_steps = self.state.max_steps
        
        # 防止除零错误 (刚开始训练时 max_steps 可能是 0)
        if max_steps > 0:
            progress = min(current_step / max_steps, 1.0)
        else:
            progress = 0.0
            
        # 线性衰减计算
        anchor_w = self.hint_config.anchor_weight_start - progress * (self.hint_config.anchor_weight_start - self.hint_config.anchor_weight_end)

        for i in range(token_losses.size(0)):
            h_m, a_m = shift_h_masks[i], shift_a_masks[i]
            h_count, a_count = h_m.sum(), a_m.sum()
            
            if h_count > 0:
                # === Mode B (Generation) ===
                avg_h_loss = (token_losses[i] * h_m).sum() / h_count
                
                # Gating Calculation
                gate_input = self.hint_config.gate_slope * (self.hint_config.gate_threshold - avg_h_loss.detach())
                gate = torch.sigmoid(gate_input)
                
                avg_a_loss = (token_losses[i] * a_m).sum() / a_count if a_count > 0 else 0.0
                
                # Mode B 权重固定为 1.0 (即 hint_fixed_weight)
                total_sample_loss = self.hint_config.hint_fixed_weight * avg_h_loss + gate * avg_a_loss
                
                batch_losses.append(total_sample_loss)
                debug_info_list.append({"gate": gate.item(), "loss_contrib": total_sample_loss.item()})
            else:
                # === Anchor Mode (Pure SFT) ===
                avg_a_loss = (token_losses[i] * a_m).sum() / a_count if a_count > 0 else torch.tensor(0.0, device=logits.device)
                
                # 应用动态衰减的权重
                weighted_anchor_loss = anchor_w * avg_a_loss
                
                batch_losses.append(weighted_anchor_loss)
                
                # 记录原始 loss 供观察，验证它是否保持平稳
                debug_info_list.append({"gate": 0.0, "loss_contrib": avg_a_loss.item()}) 

        return torch.stack(batch_losses).mean(), debug_info_list, anchor_w

# ==========================================
# 5. Callbacks
# ==========================================

class EpochLogCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file

    def on_epoch_begin(self, args, state, control, **kwargs):
        tracker.reset()

    def on_epoch_end(self, args, state, control, **kwargs):
        stats = tracker.get_epoch_stats()
        stats["epoch"] = state.epoch
        stats["global_step"] = state.global_step
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(stats) + "\n")
        
        print(f"\n[Epoch {state.epoch:.2f} Stats] Loss: {stats['avg_train_loss']:.4f} | "
              f"Gate: {stats['avg_gate_value']:.4f} | "
              f"AnchorW: {stats['anchor_weight']:.4f} | " 
              f"AnchorLoss(Raw): {stats['avg_anchor_loss']:.4f} | "
              f"ModeBLoss: {stats['avg_mode_b_loss']:.4f}")

# ==========================================
# 6. Main Execution
# ==========================================

def main():
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path)) 
    project_root = os.path.dirname(os.path.dirname(project_root))
    model_dir = os.path.join(project_root, "CELPO", "model", "Qwen")
    model_url = os.path.join(model_dir, "Qwen2.5-Math-7B-Instruct")

    data_path =  os.path.join(project_root, "CELPO", "datasets", "exam", "irdcl_data.json")

    output_base_dir = os.path.join(project_root, "CELPO", "output")

    set_seed(42)
    
    # 初始化配置
    hint_config = HintSFTConfig(model_path=model_url, data_path=data_path, output_base_dir=output_base_dir)
    
    output_dir = f"{hint_config.output_base_dir}/hint_sft_{datetime.now().strftime('%m%d_%H%M')}"
    metric_log_file = setup_logging(output_dir)
    
    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hint_config.model_path, trust_remote_code=True)
    
    # 2. 加载预处理好的 Dataset
    dataset = Dataset.from_json(hint_config.data_path)
    logger.info(f"Loaded dataset from {hint_config.data_path}, size: {len(dataset)}")

    # 3. 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        hint_config.model_path, 
        torch_dtype=torch.float16, 
        device_map="auto", 
        trust_remote_code=True
    )
    
    # --- 修复警告的关键代码 ---
    model.config.use_cache = False  # 禁用 cache 以兼容 gradient checkpointing
    
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM", 
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads() 
    model.print_trainable_parameters()

    # 4. 准备 Collator
    collator = FixedModeCollator(tokenizer)

    # 5. 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=20,             
        per_device_train_batch_size=2,   
        gradient_accumulation_steps=4,   
        learning_rate=5e-5,
        warmup_ratio=0.1,
        logging_steps=10,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="epoch",           
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_drop_last=True,       
        group_by_length=False,           
        report_to="none"                 
    )

    # 6. 初始化自定义 Trainer
    trainer = SequentialTrainer(
        hint_config=hint_config,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[EpochLogCallback(metric_log_file)]
    )

    logger.info("Starting sequential training...")
    trainer.train()
    trainer.save_model(output_dir)

if __name__ == "__main__":
    main()

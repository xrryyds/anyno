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

# ==========================================
# Logger 配置
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")
warnings.filterwarnings("ignore", message="Could not find a config file in")

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
    hint_fixed_weight: float = 1.0 
    gate_threshold: float = 0.3   
    gate_slope: float = 3.0       

    # Volume Balance Term: (1-r)/r
    split_r: float = 0.5
    
    # alpha = ((1-r)/r) * (L_gen / beta)
    beta: float = 0.3 
    
    # 路径配置
    model_path: str = "/mnt/petrelfs/wanhaiyuan/xrr/CELPO/model/OREAL/OREAL-7B"
    data_path: str = "/xrr/CELPO/datasets/exam/irdcl_data.json" 
    output_base_dir: str = "/root/autodl-tmp/output"

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(output_dir, "train.log"), encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    return os.path.join(output_dir, "epoch_metrics.jsonl")

# ==========================================
# 2. Metrics Tracker (统计追踪器)
# ==========================================

class TrainingMetricsTracker:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.total_loss = 0.0
        self.steps = 0
        self.gate_values = []
        self.anchor_losses = [] # 记录加权后的 Anchor Loss
        self.mode_b_losses = []
        self.current_alpha = 0.0 
        self.mode_b_counts = 0
        self.anchor_counts = 0

    def update(self, loss, metadata, debug_info, current_alpha):
        self.total_loss += loss
        self.steps += 1
        self.current_alpha = current_alpha
        
        for meta, info in zip(metadata, debug_info):
            if info is None: continue # 应该不会发生，但为了安全
            
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
            "avg_anchor_loss_weighted": avg_anchor_loss, # 注意：这是乘了 alpha 后的 loss
            "avg_mode_b_loss": avg_mode_b_loss,
            "final_alpha": self.current_alpha,
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
# 4. SIRA Trainer (复现论文算法)
# ==========================================

class SequentialTrainer(Trainer):
    def __init__(self, hint_config: HintSFTConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hint_config = hint_config
        # 用于记录平滑的 L_gen，防止某个 micro-batch 没有生成数据导致 Alpha 计算不稳
        # 初始值设为 beta，保证初始 Alpha = VolumeBalance
        self.running_gen_loss = self.hint_config.beta 

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_dataset = self.train_dataset
        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=SequentialSampler(train_dataset), # 数据已按比例预混
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
        
        # 核心算法调用
        loss, debug_info_list, current_alpha = self.adaptive_gating_loss(logits, labels, hint_masks, answer_masks)
        
        if self.model.training:
            tracker.update(loss.item(), metadata, debug_info_list, current_alpha)
            
        return (loss, outputs) if return_outputs else loss

    def adaptive_gating_loss(self, logits, labels, hint_masks, answer_masks):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_h_masks = hint_masks[..., 1:].contiguous()
        shift_a_masks = answer_masks[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        token_losses = token_losses.view(shift_labels.shape)

        # 临时存储
        gen_losses = []     # 存放 Mode B 的 loss Tensor
        anchor_indices = [] # 存放 Mode A 在 batch 中的下标
        gen_indices = []    # 存放 Mode B 在 batch 中的下标
        debug_map = {}      # 存放 debug info，最后按 index 组装

        # --- Step 1: 遍历 Batch，计算 Mode B (Generation) Loss ---
        for i in range(token_losses.size(0)):
            h_m = shift_h_masks[i]
            h_count = h_m.sum()
            
            if h_count > 0:
                # === Mode B: Knowledge Injection ===
                gen_indices.append(i)
                avg_h_loss = (token_losses[i] * h_m).sum() / h_count
                
                # Gate Calculation: SG[H(h)] (Detach)
                gate_input = self.hint_config.gate_slope * (self.hint_config.gate_threshold - avg_h_loss.detach())
                gate = torch.sigmoid(gate_input)
                
                # Answer Loss
                a_m = shift_a_masks[i]
                a_count = a_m.sum()
                avg_a_loss = (token_losses[i] * a_m).sum() / a_count if a_count > 0 else 0.0
                
                # L_gen = L_hint + gate * L_ans
                l_gen = self.hint_config.hint_fixed_weight * avg_h_loss + gate * avg_a_loss
                gen_losses.append(l_gen)
                
                debug_map[i] = {"gate": gate.item(), "loss_contrib": l_gen.item()}
            else:
                # === Mode A: Stability Anchor (暂存 Index) ===
                anchor_indices.append(i)

        # --- Step 2: 计算动态 Alpha (Eq. 2) ---
        if len(gen_losses) > 0:
            current_gen_loss_tensor = torch.stack(gen_losses).mean()
            # EMA 更新 running loss (0.9 history + 0.1 current)
            self.running_gen_loss = 0.9 * self.running_gen_loss + 0.1 * current_gen_loss_tensor.item()
            scaling_loss = current_gen_loss_tensor.detach() # Detach! Alpha 不回传梯度
        else:
            # 极端情况：micro-batch 全是 Anchor，使用历史均值
            scaling_loss = torch.tensor(self.running_gen_loss, device=logits.device)

        # Alpha = ((1-r)/r) * (L_gen / beta)
        r = self.hint_config.split_r
        vol_balance = (1 - r) / r
        alpha = vol_balance * (scaling_loss / self.hint_config.beta)

        # --- Step 3: 计算 Mode A Loss 并加权，组装最终 Loss ---
        
        # 将 Mode B Loss 放入 Map
        final_losses_map = {}
        for idx, l_val in zip(gen_indices, gen_losses):
            final_losses_map[idx] = l_val

        # 处理 Mode A
        for idx in anchor_indices:
            a_m = shift_a_masks[idx]
            a_count = a_m.sum()
            
            # L_anchor (Raw)
            raw_anchor_loss = (token_losses[idx] * a_m).sum() / a_count if a_count > 0 else torch.tensor(0.0, device=logits.device)
            
            # Weighted: alpha * L_anchor
            weighted_anchor_loss = alpha * raw_anchor_loss
            final_losses_map[idx] = weighted_anchor_loss
            
            # Debug info 记录的是加权后的值，还是原始值？通常记录原始值用于观察，记录加权值用于 Loss
            debug_map[idx] = {"gate": 0.0, "loss_contrib": weighted_anchor_loss.item()}

        # 按 Batch 顺序还原 Loss 列表
        batch_loss_list = [final_losses_map[i] for i in range(token_losses.size(0))]
        debug_info_list = [debug_map[i] for i in range(token_losses.size(0))]
        
        return torch.stack(batch_loss_list).mean(), debug_info_list, alpha.item()

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
        
        logger.info(f"[Epoch {state.epoch:.2f}] Loss: {stats['avg_train_loss']:.4f} | "
                    f"Alpha: {stats['final_alpha']:.4f} | " 
                    f"Gate: {stats['avg_gate_value']:.4f} | "
                    f"W_AnchorLoss: {stats['avg_anchor_loss_weighted']:.4f} | "
                    f"ModeBLoss: {stats['avg_mode_b_loss']:.4f}")

# ==========================================
# 6. Main Execution
# ==========================================

def main():
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path)) 
    project_root = os.path.dirname(os.path.dirname(project_root))
    model_dir = os.path.join(project_root, "CELPO", "model", "OREAL")
    model_url = os.path.join(model_dir, "OREAL-7B")

    data_path =  os.path.join(project_root, "CELPO", "datasets", "exam", "irdcl_data.json")
    output_base_dir = os.path.join(project_root, "CELPO", "output")

    set_seed(42)
    
    # 初始化配置
    hint_config = HintSFTConfig(
        model_path=model_url, 
        data_path=data_path, 
        output_base_dir=output_base_dir,
        split_r=0.5, # 你的数据混合比例
        beta=2.0     # 建议值
    )
    
    output_dir = f"{hint_config.output_base_dir}/sira_sft_{datetime.now().strftime('%m%d_%H%M')}"
    metric_log_file = setup_logging(output_dir)
    
    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hint_config.model_path, trust_remote_code=True, use_fast=False)
    
    # 2. Dataset
    dataset = Dataset.from_json(hint_config.data_path)
    logger.info(f"Loaded dataset from {hint_config.data_path}, size: {len(dataset)}")

    # 3. Model
    model = AutoModelForCausalLM.from_pretrained(
        hint_config.model_path, 
        torch_dtype=torch.float16, 
        device_map="auto", 
        trust_remote_code=True
    )
    model.config.use_cache = False  
    
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
    
    trainable_params, all_param = model.get_nb_trainable_parameters()
    logger.info(f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}")

    # 4. Collator
    collator = FixedModeCollator(tokenizer)

    # 5. Training Args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,             
        per_device_train_batch_size=4, # micro batch size   
        gradient_accumulation_steps=2, # total batch size per step = 4*2 = 8 (满足混合要求)
        learning_rate=5e-5,
        warmup_ratio=0.1,
        logging_steps=5,
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

    # 6. Trainer
    trainer = SequentialTrainer(
        hint_config=hint_config,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[EpochLogCallback(metric_log_file)]
    )

    logger.info("Starting SIRA training with dynamic Alpha...")
    trainer.train()
    trainer.save_model(output_dir)

if __name__ == "__main__":
    main()

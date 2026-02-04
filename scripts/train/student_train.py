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
    hint_fixed_weight: float = 1.0 
    gate_threshold: float = 0.3   
    gate_slope: float = 3.0       
    split_r: float = 0.5
    beta: float = 2  # 保持你设置的 0.8
    metrics_log_interval: int = 8
    
    model_path: str = "/mnt/petrelfs/wanhaiyuan/xrr/CELPO/model/OREAL/OREAL-7B"
    data_path: str = "/xrr/CELPO/datasets/exam/irdcl_data.json" 
    output_base_dir: str = "/root/autodl-tmp/output"

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(output_dir, "train.log"), encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    
    # 返回两个日志路径
    step_log_path = os.path.join(output_dir, "step_metrics.jsonl")
    epoch_log_path = os.path.join(output_dir, "epoch_metrics.jsonl")
    return step_log_path, epoch_log_path

# ==========================================
# 2. Metrics Tracker (修改版：支持 Raw Anchor Loss)
# ==========================================

class TrainingMetricsTracker:
    def __init__(self):
        self.reset_window()
        self.reset_epoch()
        
    def reset_window(self):
        """重置 Step 窗口统计数据"""
        self.win_loss = 0.0
        self.win_steps = 0
        self.win_gate_values = []
        self.win_anchor_losses_weighted = [] # 记录加权后的
        self.win_anchor_losses_raw = []      # [NEW] 记录原始的
        self.win_mode_b_losses = []
        self.current_alpha = 0.0 
        self.win_counts = {"anchor": 0, "mode_b": 0}

    def reset_epoch(self):
        """重置 Epoch 全局统计数据"""
        self.ep_loss = 0.0
        self.ep_steps = 0
        self.ep_gate_values = []
        self.ep_anchor_losses_weighted = []
        self.ep_anchor_losses_raw = []       # [NEW] 记录原始的
        self.ep_mode_b_losses = []
        self.ep_counts = {"anchor": 0, "mode_b": 0}

    def update(self, loss, metadata, debug_info, current_alpha):
        self.current_alpha = current_alpha
        
        # 1. 更新 Window 数据
        self.win_loss += loss
        self.win_steps += 1
        
        # 2. 更新 Epoch 数据
        self.ep_loss += loss
        self.ep_steps += 1
        
        for meta, info in zip(metadata, debug_info):
            if info is None: continue 
            
            mode = meta.get("mode")
            gate_val = info.get("gate", 0.0)
            loss_contrib = info["loss_contrib"] # 这是加权后的 Loss，用于反向传播的那个

            if mode == "mode_b_generation":
                # Window
                self.win_counts["mode_b"] += 1
                self.win_mode_b_losses.append(loss_contrib)
                self.win_gate_values.append(gate_val)
                # Epoch
                self.ep_counts["mode_b"] += 1
                self.ep_mode_b_losses.append(loss_contrib)
                self.ep_gate_values.append(gate_val)
                
            elif mode == "pure_sft_anchor":
                # 获取 Raw Loss (未加权的)
                raw_loss = info.get("raw_loss", 0.0) 

                # Window
                self.win_counts["anchor"] += 1
                self.win_anchor_losses_weighted.append(loss_contrib)
                self.win_anchor_losses_raw.append(raw_loss) # [NEW]
                # Epoch
                self.ep_counts["anchor"] += 1
                self.ep_anchor_losses_weighted.append(loss_contrib)
                self.ep_anchor_losses_raw.append(raw_loss) # [NEW]

    def _calculate_stats(self, loss_sum, steps, gate_vals, anchor_w_losses, anchor_raw_losses, mode_b_losses, counts, alpha):
        avg_gate = sum(gate_vals) / len(gate_vals) if gate_vals else 0.0
        avg_anchor_w = sum(anchor_w_losses) / len(anchor_w_losses) if anchor_w_losses else 0.0
        # [NEW] 计算原始 Loss 的平均值
        avg_anchor_raw = sum(anchor_raw_losses) / len(anchor_raw_losses) if anchor_raw_losses else 0.0
        
        avg_mode_b = sum(mode_b_losses) / len(mode_b_losses) if mode_b_losses else 0.0
        
        return {
            "avg_train_loss": loss_sum / max(steps, 1),
            "avg_gate_value": avg_gate,
            "avg_anchor_loss_weighted": avg_anchor_w, # 依然保留加权 Loss，便于观察梯度贡献
            "avg_anchor_loss_raw": avg_anchor_raw,    # [NEW] 这是真实的拟合程度
            "avg_mode_b_loss": avg_mode_b,
            "final_alpha": alpha, 
            "sample_counts": counts
        }

    def get_window_stats(self):
        return self._calculate_stats(
            self.win_loss, self.win_steps, self.win_gate_values, 
            self.win_anchor_losses_weighted, self.win_anchor_losses_raw, 
            self.win_mode_b_losses, self.win_counts, self.current_alpha
        )

    def get_epoch_stats(self):
        return self._calculate_stats(
            self.ep_loss, self.ep_steps, self.ep_gate_values, 
            self.ep_anchor_losses_weighted, self.ep_anchor_losses_raw, 
            self.ep_mode_b_losses, self.ep_counts, self.current_alpha
        )

tracker = TrainingMetricsTracker()

# ==========================================
# 3. Data Collator (不变)
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
                    h_mask[i] = 1 
                for i in range(min(len_hint_end, len(full_ids)), len(full_ids)):
                    a_mask[i] = 1 

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
# 4. SIRA Trainer (修改 compute_loss 传递 raw loss)
# ==========================================

class SequentialTrainer(Trainer):
    def __init__(self, hint_config: HintSFTConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hint_config = hint_config
        self.running_gen_loss = self.hint_config.beta 

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

        gen_losses = []     
        anchor_indices = [] 
        gen_indices = []    
        debug_map = {}      

        for i in range(token_losses.size(0)):
            h_m = shift_h_masks[i]
            h_count = h_m.sum()
            
            if h_count > 0:
                gen_indices.append(i)
                avg_h_loss = (token_losses[i] * h_m).sum() / h_count
                
                gate_input = self.hint_config.gate_slope * (self.hint_config.gate_threshold - avg_h_loss.detach())
                gate = torch.sigmoid(gate_input)
                
                a_m = shift_a_masks[i]
                a_count = a_m.sum()
                avg_a_loss = (token_losses[i] * a_m).sum() / a_count if a_count > 0 else 0.0
                
                l_gen = self.hint_config.hint_fixed_weight * avg_h_loss + gate * avg_a_loss
                gen_losses.append(l_gen)
                
                debug_map[i] = {"gate": gate.item(), "loss_contrib": l_gen.item()}
            else:
                anchor_indices.append(i)

        if len(gen_losses) > 0:
            current_gen_loss_tensor = torch.stack(gen_losses).mean()
            self.running_gen_loss = 0.9 * self.running_gen_loss + 0.1 * current_gen_loss_tensor.item()
            scaling_loss = current_gen_loss_tensor.detach()
        else:
            scaling_loss = torch.tensor(self.running_gen_loss, device=logits.device)

        r = self.hint_config.split_r
        vol_balance = (1 - r) / r
        alpha = vol_balance * (scaling_loss / self.hint_config.beta)
        
        # 可选：如果想在这里加 alpha 的 Hard Clamp，可以取消下面这行的注释
        # alpha = max(alpha.item(), 0.1) # 强制最小 0.1

        final_losses_map = {}
        for idx, l_val in zip(gen_indices, gen_losses):
            final_losses_map[idx] = l_val

        for idx in anchor_indices:
            a_m = shift_a_masks[idx]
            a_count = a_m.sum()
            raw_anchor_loss = (token_losses[idx] * a_m).sum() / a_count if a_count > 0 else torch.tensor(0.0, device=logits.device)
            weighted_anchor_loss = alpha * raw_anchor_loss
            final_losses_map[idx] = weighted_anchor_loss
            
            # [Modify] 记录 raw_loss 供 tracker 使用，loss_contrib 依然是 weighted 的保持梯度计算逻辑不变
            debug_map[idx] = {
                "gate": 0.0, 
                "loss_contrib": weighted_anchor_loss.item(),
                "raw_loss": raw_anchor_loss.item() 
            }

        batch_loss_list = [final_losses_map[i] for i in range(token_losses.size(0))]
        debug_info_list = [debug_map[i] for i in range(token_losses.size(0))]
        
        return torch.stack(batch_loss_list).mean(), debug_info_list, (alpha.item() if isinstance(alpha, torch.Tensor) else alpha)

# ==========================================
# 5. Callbacks
# ==========================================

class StepLogCallback(TrainerCallback):
    """
    按 step 记录日志的 Callback
    """
    def __init__(self, log_file, log_interval):
        self.log_file = log_file
        self.log_interval = log_interval

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.log_interval == 0:
            stats = tracker.get_window_stats() 
            stats["epoch"] = state.epoch
            stats["global_step"] = state.global_step
            stats["timestamp"] = datetime.now().isoformat()
            
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(stats) + "\n")
            
            # 日志中增加 Raw Anchor Loss 的显示
            logger.info(f"[Step {state.global_step}] Loss: {stats['avg_train_loss']:.4f} | "
                        f"RawAnchor: {stats['avg_anchor_loss_raw']:.4f} | " # <--- 看这里
                        f"ModeB: {stats['avg_mode_b_loss']:.4f}")
            
            tracker.reset_window()

class EpochLogCallback(TrainerCallback):
    """
    按 epoch 记录日志的 Callback
    """
    def __init__(self, log_file):
        self.log_file = log_file

    def on_epoch_end(self, args, state, control, **kwargs):
        stats = tracker.get_epoch_stats()
        stats["epoch"] = state.epoch 
        stats["global_step"] = state.global_step
        stats["timestamp"] = datetime.now().isoformat()
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(stats) + "\n")
            
        logger.info(f"="*60)
        logger.info(f"*** EPOCH {state.epoch} FINISHED ***")
        logger.info(f"  Avg Epoch Loss: {stats['avg_train_loss']:.4f}")
        logger.info(f"  Avg Gate:       {stats['avg_gate_value']:.4f}")
        logger.info(f"  Avg ModeB Loss: {stats['avg_mode_b_loss']:.4f}")
        logger.info(f"  Avg Raw Anchor: {stats['avg_anchor_loss_raw']:.4f}") # <--- 重点关注这个
        logger.info(f"  Samples: ModeB={stats['sample_counts']['mode_b']}, Anchor={stats['sample_counts']['anchor']}")
        logger.info(f"="*60)
        
        tracker.reset_epoch()

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
        split_r=0.5, 
        beta=0.8, # 保持较小的 Beta
        metrics_log_interval=8 
    )
    
    output_dir = f"{hint_config.output_base_dir}/sira_sft_{datetime.now().strftime('%m%d_%H%M')}"
    
    step_log_file, epoch_log_file = setup_logging(output_dir)
    
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
        num_train_epochs=1, # [建议] 降低 Epoch 数，例如 3-5
        per_device_train_batch_size=8,   
        gradient_accumulation_steps=2, 
        learning_rate=5e-5,
        warmup_ratio=0.1,
        logging_steps=hint_config.metrics_log_interval, 
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
        callbacks=[
            StepLogCallback(step_log_file, hint_config.metrics_log_interval),
            EpochLogCallback(epoch_log_file)
        ]
    )

    logger.info(f"Starting SIRA training with dynamic Alpha, Logging steps every {hint_config.metrics_log_interval} and every epoch...")
    trainer.train()
    trainer.save_model(output_dir)

if __name__ == "__main__":
    main()

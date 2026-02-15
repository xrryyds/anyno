import os
# 设置 VLLM 环境变量
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

# 尝试导入 prompt 模板，如果没有则使用默认
try:
    from prompt import GEN_HINTS_WIH_ANSWER
except ImportError:
    GEN_HINTS_WIH_ANSWER = "# known:\n{hints}\n{answer}"

# ==========================================
# 1. 配置与工具类
# ==========================================

@dataclass
class HintSFTConfig:
    hint_fixed_weight: float = 3.0
    gate_threshold: float = 0.3
    gate_slope: float = 3.0
    split_r: float = 0.5
    
    # 核心超参
    anchor_loss_weight_k: float = 1
    suppress_max_scale: float = 5.0
    anchor_sigmoid_slope: float = 1000.0 
    anchor_loss_tolerance: float = 1.00
    
    # 早停目标倍率
    target_loss_ratio: float = 1.01
    
    metrics_log_interval: int = 8   
    model_path: str = "" 
    data_path: str = ""  
    output_base_dir: str = "/root/autodl-tmp/output"
    
    # 真实数据包含的 Epoch 数
    real_data_epochs: int = 50

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(output_dir, "train.log"), encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    return os.path.join(output_dir, "step_metrics.jsonl"), os.path.join(output_dir, "epoch_metrics.jsonl")

# ==========================================
# 2. Metrics Tracker
# ==========================================
class TrainingMetricsTracker:
    def __init__(self):
        self.reset_window()
        self.reset_epoch()

    def reset_window(self):
        """重置短期窗口统计 (用于Step Log)"""
        self.win_loss, self.win_steps = 0.0, 0
        self.win_gate_values, self.win_anchor_losses_weighted = [], []
        self.win_anchor_losses_raw = []
        self.win_mode_b_losses, self.win_mode_b_raw = [], [] 
        self.win_counts = {"anchor": 0, "mode_b": 0}
        self.win_beta_values = [] 
        self.win_alpha_values = []

    def reset_epoch(self):
        """重置 Epoch 统计 (用于 Logical Epoch Log)"""
        self.ep_loss, self.ep_steps = 0.0, 0
        self.ep_gate_values, self.ep_anchor_losses_weighted = [], []
        self.ep_anchor_losses_raw = []
        self.ep_mode_b_losses, self.ep_mode_b_raw = [], []
        self.ep_counts = {"anchor": 0, "mode_b": 0}
        self.ep_beta_values = []
        self.ep_alpha_values = []

    def update(self, loss, metadata, debug_info, current_alpha, current_beta):
        self.win_loss += loss
        self.win_steps += 1
        self.ep_loss += loss
        self.ep_steps += 1
        
        if current_beta is not None:
             self.win_beta_values.append(current_beta)
             self.ep_beta_values.append(current_beta)
        
        if current_alpha is not None:
             self.win_alpha_values.append(current_alpha)
             self.ep_alpha_values.append(current_alpha)
        
        for meta, info in zip(metadata, debug_info):
            if info is None: 
                continue 
            mode = meta.get("mode")
            gate_val = info.get("gate", 0.0)
            loss_contrib = info["loss_contrib"]
            raw_loss = info.get("raw_loss", 0.0)

            if mode == "mode_b_generation":
                self.win_counts["mode_b"] += 1
                self.ep_counts["mode_b"] += 1
                self.win_mode_b_losses.append(loss_contrib)
                self.ep_mode_b_losses.append(loss_contrib)
                self.win_gate_values.append(gate_val)
                self.ep_gate_values.append(gate_val)
                self.win_mode_b_raw.append(raw_loss)
                self.ep_mode_b_raw.append(raw_loss)
            
            elif mode == "pure_sft_anchor":
                self.win_counts["anchor"] += 1
                self.ep_counts["anchor"] += 1
                self.win_anchor_losses_weighted.append(loss_contrib)
                self.ep_anchor_losses_weighted.append(loss_contrib)
                self.win_anchor_losses_raw.append(raw_loss)
                self.ep_anchor_losses_raw.append(raw_loss) 

    def _calculate_stats(self, loss_sum, steps, gate_vals, anchor_w, anchor_raw, mode_b, mode_b_raw, counts, alpha_vals, beta_vals):
        avg = lambda l: sum(l)/len(l) if l else 0.0
        return {
            "avg_train_loss": loss_sum / max(steps, 1), 
            "avg_gate_value": avg(gate_vals),
            "avg_anchor_loss_weighted": avg(anchor_w), 
            "avg_anchor_loss_raw": avg(anchor_raw),    
            "avg_mode_b_loss": avg(mode_b), 
            "avg_mode_b_loss_raw": avg(mode_b_raw), 
            "avg_final_alpha": avg(alpha_vals), 
            "avg_dynamic_beta": avg(beta_vals),
            "sample_counts": counts
        }

    def get_window_stats(self):
        return self._calculate_stats(
            self.win_loss, self.win_steps, self.win_gate_values, 
            self.win_anchor_losses_weighted, self.win_anchor_losses_raw, 
            self.win_mode_b_losses, self.win_mode_b_raw, 
            self.win_counts, self.win_alpha_values, self.win_beta_values
        )

    def get_epoch_stats(self):
        return self._calculate_stats(
            self.ep_loss, self.ep_steps, self.ep_gate_values, 
            self.ep_anchor_losses_weighted, self.ep_anchor_losses_raw, 
            self.ep_mode_b_losses, self.ep_mode_b_raw, 
            self.ep_counts, self.ep_alpha_values, self.ep_beta_values
        )

# 全局 Tracker 实例
tracker = TrainingMetricsTracker()

# ==========================================
# 3. Data Collator
# ==========================================
class FixedModeCollator:
    def __init__(self, tokenizer, max_length: int = 2048): 
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

            # 1. 构造 Prompt
            prompt_str = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": str(q)}],
                tokenize=False,
                add_generation_prompt=True 
            )
            prompt_ids = self.tokenizer(prompt_str, add_special_tokens=False).input_ids
            len_prompt = len(prompt_ids)

            # 2. 构造 Output
            if data_type == 'anchor_data':
                mode_str = "pure_sft_anchor"
                answer_ids = self.tokenizer(str(c), add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]
                full_ids = prompt_ids + answer_ids
                h_mask = [0] * len(full_ids)
                a_mask = [0] * len(full_ids)
                for i in range(len_prompt, len(full_ids)):
                    a_mask[i] = 1
            else:
                mode_str = "mode_b_generation"
                target_text = GEN_HINTS_WIH_ANSWER.format(hints=b, answer=c)
                target_ids = self.tokenizer(target_text, add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]
                full_ids = prompt_ids + target_ids
                
                hint_only_text = f"# known:\n{b}\n" 
                hint_ids_only = self.tokenizer(hint_only_text, add_special_tokens=False).input_ids
                len_hint_part = len(hint_ids_only)
                
                h_mask = [0] * len(full_ids)
                a_mask = [0] * len(full_ids)
                hint_end_idx = min(len_prompt + len_hint_part, len(full_ids))
                
                for i in range(len_prompt, hint_end_idx):
                    h_mask[i] = 1
                for i in range(hint_end_idx, len(full_ids)):
                    a_mask[i] = 1

            # 3. 截断
            if len(full_ids) > self.max_length:
                full_ids = full_ids[:self.max_length]
                h_mask = h_mask[:self.max_length]
                a_mask = a_mask[:self.max_length]

            # 4. Labels
            labels = [full_ids[i] if (h_mask[i] or a_mask[i]) else -100 for i in range(len(full_ids))]
            
            input_ids_batch.append(torch.tensor(full_ids, dtype=torch.long))
            labels_batch.append(torch.tensor(labels, dtype=torch.long))
            hint_masks_batch.append(torch.tensor(h_mask, dtype=torch.float32))
            answer_masks_batch.append(torch.tensor(a_mask, dtype=torch.float32))
            attention_mask_batch.append(torch.ones(len(full_ids), dtype=torch.long))
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
# 4. SIRA Trainer (Modified)
# ==========================================
class SequentialTrainer(Trainer):
    def __init__(self, hint_config: HintSFTConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hint_config = hint_config
        self.running_gen_loss = 1.0 

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
        
        # Step 1: Base Model (Ref) Logits
        with torch.no_grad():
            with self.model.disable_adapter():
                ref_outputs = model(**inputs)
                ref_logits = ref_outputs.get("logits")
        
        # Step 2: Current Model Logits
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss, debug_info_list, current_alpha, dynamic_beta = self.adaptive_gating_loss(
            logits, ref_logits, labels, hint_masks, answer_masks
        )
        
        if self.model.training:
            tracker.update(loss.item(), metadata, debug_info_list, current_alpha, dynamic_beta)
            
        return (loss, outputs) if return_outputs else loss

    def adaptive_gating_loss(self, logits, ref_logits, labels, hint_masks, answer_masks):
        """
        Modified Loss Calculation:
        1. Instance-level Suppression (comparing current sample loss vs ref sample loss).
        2. Batch-level Balance (comparing gen mean vs ref anchor mean).
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_ref_logits = ref_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_h_masks = hint_masks[..., 1:].contiguous()
        shift_a_masks = answer_masks[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        
        # (Batch, Seq_Len)
        token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        token_losses = token_losses.view(shift_labels.shape)

        # (Batch, Seq_Len) - Reference Model Losses
        ref_token_losses = loss_fct(shift_ref_logits.view(-1, shift_ref_logits.size(-1)), shift_labels.view(-1))
        ref_token_losses = ref_token_losses.view(shift_labels.shape)

        gen_losses = []      
        anchor_indices = [] 
        gen_indices = []    
        debug_map = {}      

        # ======================================================
        # Part 1: Mode B (Generation) Loss Collection
        # ======================================================
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
                avg_a_loss = (token_losses[i] * a_m).sum() / a_count if a_count > 0 else torch.tensor(0.0, device=logits.device)
                
                l_gen = self.hint_config.hint_fixed_weight * avg_h_loss + gate * avg_a_loss
                gen_losses.append(l_gen)

                total_valid_tokens = h_count + a_count
                raw_b_loss = ((token_losses[i] * h_m).sum() + (token_losses[i] * a_m).sum()) / total_valid_tokens if total_valid_tokens > 0 else torch.tensor(0.0)
                debug_map[i] = {"gate": gate.item(), "loss_contrib": l_gen.item(), "raw_loss": raw_b_loss.item()}
            else:
                anchor_indices.append(i)

        # Calculate Gen Loss Mean for Balance Ratio
        if len(gen_losses) > 0:
            current_gen_loss_mean = torch.stack(gen_losses).mean()
            self.running_gen_loss = 0.9 * self.running_gen_loss + 0.1 * current_gen_loss_mean.item()
            scaling_gen_loss = current_gen_loss_mean.detach()
        else:
            scaling_gen_loss = torch.tensor(self.running_gen_loss, device=logits.device)

        # ======================================================
        # Part 2: Anchor Loss (Instance-level Suppression)
        # ======================================================
        
        final_losses_map = {}
        batch_ref_anchor_loss_sum = torch.tensor(0.0, device=logits.device)
        suppressed_anchor_losses = [] # List to store individual suppressed losses
        valid_anchor_count = 0

        # 2.1 Calculate Suppression per Instance
        for idx in anchor_indices:
            a_m = shift_a_masks[idx]
            a_count = a_m.sum()
            
            if a_count > 0:
                # A. Raw Losses (Current vs Ref) for this specific sample
                raw_curr = (token_losses[idx] * a_m).sum() / a_count
                raw_ref = (ref_token_losses[idx] * a_m).sum() / a_count
                
                # Accumulate for Batch Stats (used later for Balance)
                batch_ref_anchor_loss_sum += raw_ref
                valid_anchor_count += 1
                
                # B. Instance-level Suppression Calculation
                # Beta is now specific to this sample (raw_ref)
                instance_beta = raw_ref.detach() + 1e-6
                target_threshold = instance_beta * self.hint_config.anchor_loss_tolerance
                
                # Ratio of this sample's current loss to its own ref loss
                ratio = raw_curr.detach() / target_threshold
                sigmoid_input = (ratio - 1.0) * self.hint_config.anchor_sigmoid_slope
                
                # Suppression factor for this sample
                instance_suppress = torch.sigmoid(sigmoid_input) * self.hint_config.suppress_max_scale
                
                # Apply suppression immediately
                weighted_item_loss = instance_suppress * raw_curr
                suppressed_anchor_losses.append(weighted_item_loss)
                
                # Store intermediate for logging (will update with balance later)
                debug_map[idx] = {
                    "gate": 0.0, 
                    "raw_loss": raw_curr.item(),
                    "instance_beta": raw_ref.item(),
                    "instance_suppress": instance_suppress.item() 
                }

        # ======================================================
        # Part 3: Batch-level Balance (Log Smoothing)
        # ======================================================
        
        alpha_balance = 0.0
        batch_beta_val = 0.0 # Log-ready batch beta
        
        # Only calculate balance if we have anchors
        if valid_anchor_count > 0:
            # A. Calculate Batch Ref Mean (Batch Beta)
            batch_ref_mean = batch_ref_anchor_loss_sum / valid_anchor_count
            batch_beta_val = batch_ref_mean.item()
            safe_batch_beta = batch_ref_mean.detach() + 1e-6
            
            # B. Calculate Ratio (Batch Gen Mean / Batch Ref Mean)
            # "Ratio is this batch's GenLoss mean / this batch's anchor mean (ref)"
            r = self.hint_config.split_r
            vol_balance = (1 - r) / r
            
            raw_loss_ratio = scaling_gen_loss / safe_batch_beta
            
            if raw_loss_ratio > 1.0:
                smooth_scaler = 1.0 + 0.8 * torch.log(raw_loss_ratio)
            else:
                smooth_scaler = raw_loss_ratio
            
            # This is the global balance factor applied to the aggregated suppressed anchors
            k = self.hint_config.anchor_loss_weight_k
            alpha_balance = (k * vol_balance * smooth_scaler).item()

        # ======================================================
        # Part 4: Apply Balance and Combine
        # ======================================================
        
        # Apply global balance to each anchor instance and store in map
        anchor_idx_ptr = 0
        for idx in anchor_indices:
            if idx in debug_map: # Valid anchor with tokens
                suppressed_loss = suppressed_anchor_losses[anchor_idx_ptr]
                
                # Final Loss = (Instance Suppressed) * (Batch Balance)
                final_anchor_loss = suppressed_loss * alpha_balance
                final_losses_map[idx] = final_anchor_loss
                
                # Update debug info with final contribution
                debug_map[idx]["loss_contrib"] = final_anchor_loss.item()
                debug_map[idx]["final_balance_alpha"] = alpha_balance
                
                anchor_idx_ptr += 1

        debug_info_list = [debug_map.get(i) for i in range(token_losses.size(0))]
        
        # ======================================================
        # Part 5: Task Reweighting & Norm (Standard SIRA)
        # ======================================================
        batch_size = token_losses.size(0)
        # Default 0.0 for samples that were masked out entirely
        raw_loss_vector = torch.stack([
            final_losses_map.get(i, torch.tensor(0.0, device=logits.device)) 
            for i in range(batch_size)
        ])

        num_gen = len(gen_indices)
        num_anchor = len(anchor_indices)
        task_weights = torch.zeros(batch_size, device=raw_loss_vector.device)
        
        if num_gen > 0:
            w_gen = 0.5 / num_gen
            for idx in gen_indices: 
                task_weights[idx] = w_gen
        if num_anchor > 0:
            w_anchor = 0.5 / num_anchor
            for idx in anchor_indices: 
                task_weights[idx] = w_anchor

        weighted_loss_vec = raw_loss_vector * task_weights * 2.0 

        if num_gen > 0:
            is_mode_b = torch.zeros(batch_size, device=logits.device)
            for idx in gen_indices: 
                is_mode_b[idx] = 1.0
            loss_b_vec = weighted_loss_vec * is_mode_b

            norm_b = torch.norm(loss_b_vec, p=2) 
            norm_total = torch.norm(weighted_loss_vec, p=2)
            
            if norm_total > 1e-6:
                scale_factor = (norm_b / norm_total).detach()
                final_loss_tensor = weighted_loss_vec * scale_factor
                final_loss = final_loss_tensor.sum()
            else:
                final_loss = weighted_loss_vec.sum()
        else:
            final_loss = weighted_loss_vec.sum()

        return final_loss, debug_info_list, alpha_balance, batch_beta_val

# ==========================================
# 5. Callbacks
# ==========================================

class StepLogCallback(TrainerCallback):
    """
    Step 级别的日志打印和早停策略
    """
    def __init__(self, log_file, log_interval, config: HintSFTConfig, output_dir: str):
        self.log_file = log_file
        self.log_interval = log_interval
        self.config = config
        self.output_dir = output_dir
        self.early_stopped = False

    def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.log_interval == 0:
            stats = tracker.get_window_stats() 
            json_stats = stats.copy()
            json_stats.update({
                "epoch": state.epoch,
                "global_step": state.global_step, 
                "timestamp": datetime.now().isoformat()
            })
            with open(self.log_file, "a", encoding="utf-8") as f: 
                f.write(json.dumps(json_stats) + "\n")
            
            log_parts = [f"[Step {state.global_step}]"]
            for k, v in stats.items():
                if isinstance(v, float):
                    val_str = f"{v:.6f}"
                else:
                    val_str = str(v)
                log_parts.append(f"{k}: {val_str}")
            logger.info(" | ".join(log_parts))
            
            # Early Stopping Logic
            avg_mode_b = stats.get("avg_mode_b_loss_raw", 999.0)
            avg_beta = stats.get("avg_dynamic_beta", 0.0)
            
            if avg_beta > 0 and avg_mode_b <= avg_beta * self.config.target_loss_ratio:
                logger.info(f"="*60)
                logger.info(f"*** EARLY STOPPING TRIGGERED (Step {state.global_step}) ***")
                logger.info(f"Mode B Loss ({avg_mode_b:.4f}) reached {self.config.target_loss_ratio}x Beta ({avg_beta:.4f})")
                
                # 保存早停模型
                early_stop_dir = os.path.join(
                    self.output_dir, 
                    f"checkpoint-early-stop-step-{state.global_step}"
                )
                logger.info(f"Saving early-stopped model to {early_stop_dir}...")
                
                try:
                    if model is not None:
                        model.save_pretrained(early_stop_dir)
                        if tokenizer is not None:
                            tokenizer.save_pretrained(early_stop_dir)
                        logger.info(f"✓ Model successfully saved to {early_stop_dir}")
                    else:
                        logger.warning("Model is None, cannot save!")
                except Exception as e:
                    logger.error(f"Failed to save early-stopped model: {e}")
                
                self.early_stopped = True
                control.should_training_stop = True
                logger.info(f"="*60)
            
            tracker.reset_window()

class LogicalEpochLogCallback(TrainerCallback):
    """
    逻辑 Epoch 的日志打印
    """
    def __init__(self, log_file, steps_per_logical_epoch): 
        self.log_file = log_file
        self.steps_per_epoch = steps_per_logical_epoch
        self.current_logical_epoch = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.steps_per_epoch == 0:
            self.current_logical_epoch += 1
            
            stats = tracker.get_epoch_stats()
            stats.update({
                "logical_epoch": self.current_logical_epoch,
                "global_step": state.global_step, 
                "timestamp": datetime.now().isoformat()
            })
            with open(self.log_file, "a", encoding="utf-8") as f: 
                f.write(json.dumps(stats) + "\n")
                
            logger.info(f"="*60)
            logger.info(f"*** LOGICAL EPOCH {self.current_logical_epoch} FINISHED (Step {state.global_step}) ***")
            logger.info(f"  Avg Loss:       {stats['avg_train_loss']:.4f}")
            logger.info(f"  Avg Gate:       {stats['avg_gate_value']:.4f}")
            logger.info(f"  Avg Beta (Batch): {stats['avg_dynamic_beta']:.4f}")
            logger.info(f"  Avg ModeB (W):  {stats['avg_mode_b_loss']:.4f}")
            logger.info(f"  Avg ModeB (Raw):{stats['avg_mode_b_loss_raw']:.4f}")
            logger.info(f"  Avg Anc Raw:    {stats['avg_anchor_loss_raw']:.4f}")
            logger.info(f"  Avg Alpha(Bal): {stats['avg_final_alpha']:.4f}")
            logger.info(f"="*60)
            
            tracker.reset_epoch()

# ==========================================
# 6. Main Execution Function
# ==========================================

def run_sira_training_v2(
    model_path: str,
    data_path: Optional[str] = None,
    output_base_dir: Optional[str] = None,
    batch_size: int = 16,
    real_data_epochs: int = 50,
    device_num: int = 1,
    spilt: float = 0.5
):
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path)) 
    project_root = os.path.dirname(os.path.dirname(project_root)) 

    if data_path is None:
        data_path = os.path.join(project_root, "CELPO", "datasets", "exam", "irdcl_data.json")
    if output_base_dir is None:
        output_base_dir = os.path.join(project_root, "CELPO", "output")
        
    set_seed(42)
    tracker.reset_window()
    tracker.reset_epoch()

    hint_config = HintSFTConfig(
        model_path=model_path, 
        data_path=data_path, 
        output_base_dir=output_base_dir,
        split_r=spilt,
        anchor_loss_weight_k=1, 
        suppress_max_scale=1.0,
        anchor_sigmoid_slope=50.0, 
        anchor_loss_tolerance=1.01,
        target_loss_ratio=1.01,
        metrics_log_interval=batch_size,
        real_data_epochs=real_data_epochs
    )
    
    output_dir = f"{hint_config.output_base_dir}/sira_sft_{real_data_epochs}ep_{datetime.now().strftime('%m%d_%H%M')}"
    step_log_file, epoch_log_file = setup_logging(output_dir)
    
    logger.info(f"Model Path: {hint_config.model_path}")
    logger.info(f"Output Dir: {output_dir}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(hint_config.model_path, trust_remote_code=True, use_fast=False)
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise e

    if not os.path.exists(hint_config.data_path): 
        raise FileNotFoundError(f"Dataset not found at {hint_config.data_path}")
    dataset = Dataset.from_json(hint_config.data_path)
    
    total_samples = len(dataset)
    total_steps = total_samples // batch_size
    steps_per_logical_epoch = total_steps // real_data_epochs
    
    if steps_per_logical_epoch < 1:
        steps_per_logical_epoch = 1
        logger.warning(f"Total steps ({total_steps}) < real epochs ({real_data_epochs}). Set logical step to 1.")

    logger.info(f"Loaded dataset size: {total_samples}")
    logger.info(f"Simulating {real_data_epochs} Epochs.")
    logger.info(f"Total Steps: {total_steps} | Steps per Logical Epoch: {steps_per_logical_epoch}")

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

    collator = FixedModeCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=int(batch_size/2/device_num),
        learning_rate=1e-4, 
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=hint_config.metrics_log_interval, 
        save_strategy="steps",           
        save_steps=steps_per_logical_epoch,
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_drop_last=True,       
        group_by_length=False,           
        report_to="none"                 
    )

    # 创建 Callback 实例并保持引用
    step_callback = StepLogCallback(
        step_log_file, 
        hint_config.metrics_log_interval, 
        hint_config,
        output_dir
    )
    epoch_callback = LogicalEpochLogCallback(epoch_log_file, steps_per_logical_epoch)

    trainer = SequentialTrainer(
        hint_config=hint_config,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[step_callback, epoch_callback]
    )

    logger.info(f"Starting SIRA training with Instance-Level Suppression and Batch Balance...")
    trainer.train()
    
    # 通过保持的引用检查早停状态
    if not step_callback.early_stopped:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Training finished normally. Model saved to {output_dir}")
    else:
        logger.info(f"Training finished with early stopping. Model already saved.")

if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path)) 
    project_root = os.path.dirname(os.path.dirname(project_root))
    default_model_dir = os.path.join(project_root, "CELPO", "model", "OREAL")
    default_model_url = os.path.join(default_model_dir, "OREAL-7B")

    run_sira_training_v2(
        model_path=default_model_url,
        batch_size=16,
        real_data_epochs=50
    )

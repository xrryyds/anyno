import os
# 设置 VLLM 环境变量
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import sys
import json
import random
import torch
import logging
import warnings
import numpy as np 
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

# ==========================================
# System Prompt
# ==========================================
SYSTEM_PROMPT = "Please reason step by step and put your final answer within \\boxed{}."
try:
    from prompt import GEN_HINTS_WIH_ANSWER
except ImportError:
    GEN_HINTS_WIH_ANSWER = "{hints}{answer}"

# ==========================================
# 1. 配置与工具类
# ==========================================

@dataclass
class HintSFTConfig:
    hint_fixed_weight: float = 3.0
    gate_threshold: float = 0.3
    gate_slope: float = 3.0
    split_r: float = 0.5
    anchor_loss_weight_k: float = 1
    suppress_max_scale: float = 5.0
    anchor_sigmoid_slope: float = 1000.0 
    anchor_loss_tolerance: float = 1.00
    target_loss_ratio: float = 1.01
    metrics_log_interval: int = 8   
    model_path: str = "" 
    data_path: str = ""  
    output_base_dir: str = "/root/autodl-tmp/output"

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
        self.win_loss, self.win_steps = 0.0, 0
        self.win_gate_values, self.win_anchor_losses_weighted = [], []
        self.win_anchor_losses_raw = []
        self.win_mode_b_losses, self.win_mode_b_raw = [], [] 
        self.win_counts = {"anchor": 0, "mode_b": 0}
        self.win_beta_values = [] 
        self.win_alpha_values = []

    def reset_epoch(self):
        self.ep_loss, self.ep_steps = 0.0, 0
        self.ep_gate_values, self.ep_anchor_losses_weighted = [], []
        self.ep_anchor_losses_raw = []
        self.ep_mode_b_losses, self.ep_mode_b_raw = [], []
        self.ep_counts = {"anchor": 0, "mode_b": 0}
        self.ep_beta_values = []
        self.ep_alpha_values = []

    def update(self, loss, metadata, debug_info, current_alpha, current_beta):
        self.win_loss += loss; self.win_steps += 1
        self.ep_loss += loss; self.ep_steps += 1
        
        if current_beta is not None:
             self.win_beta_values.append(current_beta)
             self.ep_beta_values.append(current_beta)
        if current_alpha is not None:
             self.win_alpha_values.append(current_alpha)
             self.ep_alpha_values.append(current_alpha)
        
        for meta, info in zip(metadata, debug_info):
            if info is None: continue 
            mode = meta.get("mode")
            gate_val = info.get("gate", 0.0)
            loss_contrib = info["loss_contrib"]
            raw_loss = info.get("raw_loss", 0.0)

            if mode == "mode_b_generation":
                self.win_counts["mode_b"] += 1; self.ep_counts["mode_b"] += 1
                self.win_mode_b_losses.append(loss_contrib); self.ep_mode_b_losses.append(loss_contrib)
                self.win_gate_values.append(gate_val); self.ep_gate_values.append(gate_val)
                self.win_mode_b_raw.append(raw_loss); self.ep_mode_b_raw.append(raw_loss)
            elif mode == "pure_sft_anchor":
                self.win_counts["anchor"] += 1; self.ep_counts["anchor"] += 1
                self.win_anchor_losses_weighted.append(loss_contrib); self.ep_anchor_losses_weighted.append(loss_contrib)
                self.win_anchor_losses_raw.append(raw_loss); self.ep_anchor_losses_raw.append(raw_loss) 

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
        return self._calculate_stats(self.win_loss, self.win_steps, self.win_gate_values, self.win_anchor_losses_weighted, self.win_anchor_losses_raw, self.win_mode_b_losses, self.win_mode_b_raw, self.win_counts, self.win_alpha_values, self.win_beta_values)

    def get_epoch_stats(self):
        return self._calculate_stats(self.ep_loss, self.ep_steps, self.ep_gate_values, self.ep_anchor_losses_weighted, self.ep_anchor_losses_raw, self.ep_mode_b_losses, self.ep_mode_b_raw, self.ep_counts, self.ep_alpha_values, self.ep_beta_values)

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

            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": str(q)}]
            prompt_str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_ids = self.tokenizer(prompt_str, add_special_tokens=False).input_ids
            len_prompt = len(prompt_ids)

            if data_type == 'anchor_data':
                mode_str = "pure_sft_anchor"
                answer_ids = self.tokenizer(str(c), add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]
                full_ids = prompt_ids + answer_ids
                h_mask = [0] * len(full_ids)
                a_mask = [0] * len(full_ids)
                for i in range(len_prompt, len(full_ids)): a_mask[i] = 1
            else:
                mode_str = "mode_b_generation"
                target_text = GEN_HINTS_WIH_ANSWER.format(hints=b, answer=c)
                target_ids = self.tokenizer(target_text, add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]
                full_ids = prompt_ids + target_ids
                hint_only_text = f"{b}" 
                len_hint_part = len(self.tokenizer(hint_only_text, add_special_tokens=False).input_ids)
                h_mask = [0] * len(full_ids)
                a_mask = [0] * len(full_ids)
                hint_end_idx = min(len_prompt + len_hint_part, len(full_ids))
                for i in range(len_prompt, hint_end_idx): h_mask[i] = 1
                for i in range(hint_end_idx, len(full_ids)): a_mask[i] = 1

            if len(full_ids) > self.max_length:
                full_ids = full_ids[:self.max_length]; h_mask = h_mask[:self.max_length]; a_mask = a_mask[:self.max_length]

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
# 4. SIRA Trainer (MODIFIED)
# ==========================================

class SequentialTrainer(Trainer):
    def __init__(self, hint_config: HintSFTConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hint_config = hint_config
        self.running_gen_loss = 1.0 

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        hint_masks = inputs.pop("hint_masks")
        answer_masks = inputs.pop("answer_masks")
        metadata = inputs.pop("metadata")
        
        with torch.no_grad():
            with self.model.disable_adapter():
                ref_outputs = model(**inputs)
                ref_logits = ref_outputs.get("logits")
        
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss, debug_info_list, current_alpha, dynamic_beta = self.adaptive_gating_loss(
            logits, ref_logits, labels, hint_masks, answer_masks
        )
        
        if self.model.training:
            tracker.update(loss.item(), metadata, debug_info_list, current_alpha, dynamic_beta)
            
        return (loss, outputs) if return_outputs else loss

    def adaptive_gating_loss(self, logits, ref_logits, labels, hint_masks, answer_masks):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_ref_logits = ref_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_h_masks = hint_masks[..., 1:].contiguous()
        shift_a_masks = answer_masks[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        
        token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.shape)
        ref_token_losses = loss_fct(shift_ref_logits.view(-1, shift_ref_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.shape)

        gen_losses = []      
        anchor_indices = [] 
        gen_indices = []    
        debug_map = {}      

        # Part 1: Mode B Loss
        for i in range(token_losses.size(0)):
            h_m = shift_h_masks[i]
            h_count = h_m.sum()
            
            if h_count > 0:
                gen_indices.append(i)
                avg_h_loss = (token_losses[i] * h_m).sum() / h_count
                gate = torch.sigmoid(self.hint_config.gate_slope * (self.hint_config.gate_threshold - avg_h_loss.detach()))
                a_m = shift_a_masks[i]
                a_count = a_m.sum()
                avg_a_loss = (token_losses[i] * a_m).sum() / a_count if a_count > 0 else torch.tensor(0.0, device=logits.device)
                
                l_gen = self.hint_config.hint_fixed_weight * avg_h_loss + gate * avg_a_loss
                gen_losses.append(l_gen)

                total_valid_tokens = h_count + a_count
                raw_b_loss = ((token_losses[i] * h_m).sum() + (token_losses[i] * a_m).sum()) / total_valid_tokens
                debug_map[i] = {"gate": gate.item(), "loss_contrib": l_gen.item(), "raw_loss": raw_b_loss.item()}
            else:
                anchor_indices.append(i)

        if len(gen_losses) > 0:
            current_gen_loss_mean = torch.stack(gen_losses).mean()
            self.running_gen_loss = 0.9 * self.running_gen_loss + 0.1 * current_gen_loss_mean.item()
            scaling_gen_loss = current_gen_loss_mean.detach()
        else:
            scaling_gen_loss = torch.tensor(self.running_gen_loss, device=logits.device)

        # Part 2: Alpha 计算
        final_losses_map = {}
        for idx, l_val in zip(gen_indices, gen_losses):
            final_losses_map[idx] = l_val

        final_alpha_val = 0.0
        dynamic_beta_val = 0.0
        
        if len(anchor_indices) > 0:
            current_raw_sum = torch.tensor(0.0, device=logits.device)
            ref_raw_sum = torch.tensor(0.0, device=logits.device)
            valid_anchor_count = 0
            for idx in anchor_indices:
                a_m = shift_a_masks[idx]
                a_count = a_m.sum()
                if a_count > 0:
                    current_raw_sum += (token_losses[idx] * a_m).sum() / a_count
                    ref_raw_sum += (ref_token_losses[idx] * a_m).sum() / a_count
                    valid_anchor_count += 1
            
            if valid_anchor_count > 0:
                batch_current_mean = current_raw_sum / valid_anchor_count
                batch_ref_mean = ref_raw_sum / valid_anchor_count 
                dynamic_beta_val = batch_ref_mean.item()
                safe_beta = batch_ref_mean.detach() + 1e-6
                
                raw_loss_ratio = scaling_gen_loss / safe_beta
                smooth_scaler = 1.0 + 0.8 * torch.log(raw_loss_ratio) if raw_loss_ratio > 1.0 else raw_loss_ratio
                
                alpha_balance = ((1 - self.hint_config.split_r) / self.hint_config.split_r) * smooth_scaler
                
                ratio = batch_current_mean.detach() / (safe_beta * self.hint_config.anchor_loss_tolerance)
                alpha_suppress = torch.sigmoid((ratio - 1.0) * self.hint_config.anchor_sigmoid_slope) * self.hint_config.suppress_max_scale 
                final_alpha_val = (self.hint_config.anchor_loss_weight_k * alpha_balance * alpha_suppress).item()
        
        # Part 3: 组合
        for idx in anchor_indices:
            a_m = shift_a_masks[idx]
            a_count = a_m.sum()
            raw_anchor_loss = (token_losses[idx] * a_m).sum() / a_count if a_count > 0 else torch.tensor(0.0, device=logits.device)
            applied_alpha = max(final_alpha_val, 0.0001)
            weighted_anchor_loss = applied_alpha * raw_anchor_loss
            final_losses_map[idx] = weighted_anchor_loss
            debug_map[idx] = {"gate": 0.0, "loss_contrib": weighted_anchor_loss.item(), "raw_loss": raw_anchor_loss.item(), "scale_factor": applied_alpha}

        debug_info_list = [debug_map[i] for i in range(token_losses.size(0))]
        
        # =========================================================================
        # Part 4: Task Reweighting & Norm [MODIFIED / FIXED]
        # =========================================================================
        batch_size = token_losses.size(0)
        # 确保按顺序堆叠
        raw_loss_vector = torch.stack([final_losses_map[i] for i in range(batch_size)])

        num_gen = len(gen_indices)
        num_anchor = len(anchor_indices)
        task_weights = torch.zeros(batch_size, device=raw_loss_vector.device)
        
        if num_gen > 0:
            for idx in gen_indices: task_weights[idx] = 0.5 / num_gen
        if num_anchor > 0:
            for idx in anchor_indices: task_weights[idx] = 0.5 / num_anchor

        weighted_loss_vec = raw_loss_vector * task_weights * 2.0 

        # Norm Logic
        scale_factor_debug = 1.0 # for printing
        norm_b_debug = 0.0
        norm_total_debug = 0.0
        
        if num_gen > 0:
            # 1. FIX DEVICE MISMATCH
            is_mode_b = torch.zeros(batch_size, device=weighted_loss_vec.device)
            for idx in gen_indices: is_mode_b[idx] = 1.0
            
            loss_b_vec = weighted_loss_vec * is_mode_b
            norm_b = torch.norm(loss_b_vec, p=2) 
            norm_total = torch.norm(weighted_loss_vec, p=2) + 1e-9 # 防止除0
            
            # 使用 item() 进行安全除法，避免 detach 问题
            ratio = norm_b / norm_total
            scale_factor = ratio.detach()
            
            # 2. FIX SCALE FACTOR ZERO (Gradient Collapse)
            if scale_factor < 1e-6:
                scale_factor = torch.tensor(1.0, device=scale_factor.device)

            final_loss_tensor = weighted_loss_vec * scale_factor
            final_loss = final_loss_tensor.sum()
            
            # 记录 debug 变量
            scale_factor_debug = scale_factor.item()
            norm_b_debug = norm_b.item()
            norm_total_debug = norm_total.item()
        else:
            final_loss = weighted_loss_vec.sum()

        # =========================================================================
        # >>>>>> [CRITICAL DEBUG] DIAGNOSTIC PRINT <<<<<<
        # =========================================================================
        # 随机打印或在前几个 step 打印，用于排查 Loss 为 0 的原因
        if self.model.training and random.random() < 0.2: # 20% 概率打印
            # 只有当 Loss 异常小时才强制打印
            if final_loss.item() < 1e-6:
                print("\n" + "!"*20 + " [CRITICAL DEBUG] LOSS=0 INVESTIGATION " + "!"*20)
                print(f"  Batch Size: {batch_size}")
                print(f"  Gen Indices: {gen_indices}")
                print(f"  Anchor Indices: {anchor_indices}")
                print(f"  Raw Loss Vector Sum: {raw_loss_vector.sum().item()}")
                print(f"  Weighted Vector Sum: {weighted_loss_vec.sum().item()}")
                if num_gen > 0:
                    print(f"  Norm B: {norm_b_debug:.6f}")
                    print(f"  Norm Total: {norm_total_debug:.6f}")
                    print(f"  Scale Factor: {scale_factor_debug:.6f}")
                print(f"  >>> FINAL LOSS RETURNED: {final_loss.item():.8f}")
                print("!"*70 + "\n")
        # =========================================================================

        return final_loss, debug_info_list, final_alpha_val, dynamic_beta_val

class StepLogCallback(TrainerCallback):
    def __init__(self, log_file, log_interval, config: HintSFTConfig):
        self.log_file = log_file; self.log_interval = log_interval; self.config = config
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.log_interval == 0:
            stats = tracker.get_window_stats() 
            stats.update({"epoch": state.epoch, "global_step": state.global_step, "timestamp": datetime.now().isoformat()})
            with open(self.log_file, "a", encoding="utf-8") as f: f.write(json.dumps(stats) + "\n")
            log_parts = [f"[Step {state.global_step}]"]
            for k, v in stats.items():
                val_str = f"{v:.6f}" if isinstance(v, float) else str(v)
                log_parts.append(f"{k}: {val_str}")
            logger.info(" | ".join(log_parts))
            avg_mode_b = stats.get("avg_mode_b_loss_raw", 999.0); avg_beta = stats.get("avg_dynamic_beta", 0.0)
            if avg_beta > 0 and avg_mode_b <= avg_beta * self.config.target_loss_ratio:
                logger.info(f"*** EARLY STOPPING *** Mode B Loss ({avg_mode_b:.4f}) reached target ratio.")
                control.should_training_stop = True
            tracker.reset_window()

class EpochLogCallback(TrainerCallback):
    def __init__(self, log_file): self.log_file = log_file
    def on_epoch_end(self, args, state, control, **kwargs):
        stats = tracker.get_epoch_stats()
        stats.update({"epoch": state.epoch, "global_step": state.global_step})
        with open(self.log_file, "a", encoding="utf-8") as f: f.write(json.dumps(stats) + "\n")
        logger.info(f"*** EPOCH {state.epoch} FINISHED *** Avg Loss: {stats['avg_train_loss']:.4f}")
        tracker.reset_epoch()

def run_sira_training(model_path: str, data_path: Optional[str] = None, output_base_dir: Optional[str] = None, batch_size: int = 16, epoch: int = 20, device_num: int = 1, spilt: float = 0.5):
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))) 
    if data_path is None: data_path = os.path.join(project_root, "CELPO", "datasets", "exam", "irdcl_data.json")
    if output_base_dir is None: output_base_dir = os.path.join(project_root, "CELPO", "output")
    set_seed(42); tracker.reset_window(); tracker.reset_epoch()

    hint_config = HintSFTConfig(model_path=model_path, data_path=data_path, output_base_dir=output_base_dir, split_r=spilt, metrics_log_interval=batch_size)
    output_dir = f"{hint_config.output_base_dir}/sira_sft_DEBUG_{datetime.now().strftime('%m%d_%H%M')}"
    step_log_file, epoch_log_file = setup_logging(output_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(hint_config.model_path, trust_remote_code=True, use_fast=False)
    dataset = Dataset.from_json(hint_config.data_path)
    model = AutoModelForCausalLM.from_pretrained(hint_config.model_path, torch_dtype=torch.float32, device_map="auto", trust_remote_code=True)
    model.config.use_cache = False  
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM", bias="none")
    model = get_peft_model(model, peft_config); model.enable_input_require_grads() 
    
    training_args = TrainingArguments(output_dir=output_dir, num_train_epochs=epoch, per_device_train_batch_size=2, gradient_accumulation_steps=int(batch_size/2/device_num), learning_rate=1e-4, warmup_ratio=0, logging_steps=hint_config.metrics_log_interval, fp16=False, bf16=False, gradient_checkpointing=True, save_strategy="epoch", remove_unused_columns=False, dataloader_drop_last=True, report_to="none")

    trainer = SequentialTrainer(hint_config=hint_config, model=model, args=training_args, train_dataset=dataset, data_collator=FixedModeCollator(tokenizer), callbacks=[StepLogCallback(step_log_file, hint_config.metrics_log_interval, hint_config), EpochLogCallback(epoch_log_file)])
    logger.info("Starting Debug Training..."); trainer.train(); trainer.save_model(output_dir)

if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
    default_model_url = os.path.join(project_root, "CELPO", "model", "OREAL", "OREAL-7B")
    run_sira_training(model_path=default_model_url)

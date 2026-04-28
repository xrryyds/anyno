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
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ==========================================
# Logger 配置
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 全局训练序列长度超参数（collator 截断用）
MAX_SEQ_LENGTH = 2048
SAVE_TOTAL = 2

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")

try:
    from prompt import GEN_HINTS_WIH_ANSWER
except ImportError:
    GEN_HINTS_WIH_ANSWER = "{hints}{answer}"

SYSTEM_PROMPT = "Please reason step by step and put your final answer within \\boxed{}."

# ==========================================
# 1. 配置与工具类
# ==========================================

@dataclass
class HintSFTConfig:
    hint_fixed_weight: float = 1.0
    gate_threshold: float = 0.2
    gate_slope: float = 100.0
    split_r: float = 0.5
    
    # 核心超参
    anchor_loss_weight_k: float = 0.2
    suppress_max_scale: float = 1.0
    anchor_sigmoid_slope: float = 100.0
    anchor_loss_tolerance: float = 1.00
    
    target_mode_b: Optional[float] = None
    
    metrics_log_interval: int = 8
    model_path: str = ""
    data_path: str = ""
    output_base_dir: str = "/root/autodl-tmp/output"
    real_data_epochs: int = 50
    kl_lambda: float = 0.5   # 弃用（不再使用旧的前向KL组合公式）
    kl_beta: float = 1    # KL(ref||student) weight for anchor

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
            if info is None: continue 
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
            "avg_mode_b_loss": avg(mode_b), 
            "avg_final_alpha": avg(alpha_vals),   
            "avg_mode_b_loss_raw": avg(mode_b_raw),
            "avg_anchor_loss_raw": avg(anchor_raw),   
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
    def __init__(self, tokenizer, max_length: int = MAX_SEQ_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token_id is None:
             self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, batch):
        input_ids_batch, labels_batch = [], []
        hint_masks_batch, answer_masks_batch = [], []
        attention_mask_batch, metadata_batch = [], []
        ref_betas_batch = []

        for item in batch:
            q = item['question']
            b = item.get('hints', "")
            c = item.get('answer')
            data_type = item.get('type', 'anchor_data')
            ref_betas_batch.append(item.get('ref_beta'))

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
                for i in range(len_prompt, len(full_ids)):
                    a_mask[i] = 1
            else:
                mode_str = "mode_b_generation"
                target_text = GEN_HINTS_WIH_ANSWER.format(hints=b, answer=c)
                target_ids = self.tokenizer(target_text, add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]
                full_ids = prompt_ids + target_ids
                hint_only_text = f"{b}" 
                hint_ids_only = self.tokenizer(hint_only_text, add_special_tokens=False).input_ids
                len_hint_part = len(hint_ids_only)
                h_mask, a_mask = [0] * len(full_ids), [0] * len(full_ids)
                hint_end_idx = min(len_prompt + len_hint_part, len(full_ids))
                for i in range(len_prompt, hint_end_idx): h_mask[i] = 1
                for i in range(hint_end_idx, len(full_ids)): a_mask[i] = 1

            if len(full_ids) > self.max_length:
                full_ids = full_ids[:self.max_length]
                h_mask = h_mask[:self.max_length]
                a_mask = a_mask[:self.max_length]

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
            "metadata": metadata_batch,
            "ref_betas": ref_betas_batch,
        }

# ==========================================
# 4. SIRA Trainer
# ==========================================
class SequentialTrainer(Trainer):
    def __init__(self, hint_config: HintSFTConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hint_config = hint_config
        self.running_gen_loss = 1.0

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
        hint_masks = inputs.pop("hint_masks")
        answer_masks = inputs.pop("answer_masks")
        metadata = inputs.pop("metadata")
        ref_betas = inputs.pop("ref_betas")

        with torch.no_grad():
            with self.model.disable_adapter():
                ref_logits = model(**inputs).get("logits")[..., :-1, :].contiguous()

        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")

        loss, debug_info_list, current_alpha, dynamic_beta = self.adaptive_gating_loss(
            logits, ref_logits, labels, hint_masks, answer_masks, ref_betas
        )

        if self.model.training:
            tracker.update(loss.item(), metadata, debug_info_list, current_alpha, dynamic_beta)

        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def _per_token_forward_kl(logits_s, logits_t, chunk_size: int = 8192):
        """KL(p_t || p_s) per token in bfloat16. Inputs: raw logits [T, V].
        计算正向 KL (Forward KL)，与 SDFT 默认蒸馏方向一致。
        """
        T, V = logits_s.shape

        logsumexp_s = torch.logsumexp(logits_s, dim=-1, keepdim=True)  # [T, 1]
        logsumexp_t = torch.logsumexp(logits_t, dim=-1, keepdim=True)  # [T, 1]

        kl = torch.zeros(T, device=logits_s.device, dtype=logits_s.dtype)

        for start in range(0, V, chunk_size):
            end = min(start + chunk_size, V)

            s_chunk = logits_s[:, start:end]  # [T, C]
            t_chunk = logits_t[:, start:end]  # [T, C]

            log_p_s_chunk = s_chunk - logsumexp_s      # [T, C]
            log_p_t_chunk = t_chunk - logsumexp_t      # [T, C]
            
            p_t_chunk = log_p_t_chunk.exp()            # [T, C]

            # p_t * (log_p_t - log_p_s)
            kl = kl + (p_t_chunk * (log_p_t_chunk - log_p_s_chunk)).sum(dim=-1)

            del s_chunk, t_chunk, log_p_s_chunk, log_p_t_chunk, p_t_chunk

        return kl

    def adaptive_gating_loss(self, logits, ref_logits, labels, hint_masks, answer_masks, cached_betas):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_ref_logits = ref_logits  
        shift_labels = labels[..., 1:].contiguous()
        shift_h_masks = hint_masks[..., 1:].contiguous()
        shift_a_masks = answer_masks[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_losses_list = []
        kl_list = []
        for j in range(shift_logits.size(0)):
            token_losses_list.append(loss_fct(shift_logits[j], shift_labels[j]))
            # 使用 SDFT 风格的正向 KL: KL(ref || student)
            kl_list.append(self._per_token_forward_kl(shift_logits[j], shift_ref_logits[j]))
            
        token_losses = torch.stack(token_losses_list, dim=0)  # (batch, seq_len)
        kl_ts = torch.stack(kl_list)  # [B, T]

        eps = 1e-8
        gen_losses = []
        anchor_indices, gen_indices = [], []
        debug_map, final_losses_map = {}, {}

        for i in range(token_losses.size(0)):
            h_m = shift_h_masks[i]
            h_count = h_m.sum()
            if h_count > 0:
                gen_indices.append(i)
                avg_h_loss = (token_losses[i] * h_m).sum() / h_count
                gate = 2.0 * torch.sigmoid(
                    self.hint_config.gate_slope * (self.hint_config.gate_threshold - avg_h_loss.detach())
                )
                a_m = shift_a_masks[i]
                a_count = a_m.sum()
                
                # Answer 部分使用 forward KL
                kl_answer = (kl_ts[i] * a_m).sum() / (a_count + eps) if a_count > 0 else torch.tensor(0.0, device=logits.device)

                # Mode B Loss: Hint CE + Gate * Answer KL
                l_gen = self.hint_config.hint_fixed_weight * avg_h_loss + gate * kl_answer

                gen_losses.append(l_gen)
                final_losses_map[i] = l_gen
                total_valid_tokens = h_count + a_count
                raw_b_loss = ((token_losses[i] * h_m).sum() + (token_losses[i] * a_m).sum()) / total_valid_tokens if total_valid_tokens > 0 else torch.tensor(0.0)
                debug_map[i] = {"gate": gate.item(), "loss_contrib": l_gen.item(), "raw_loss": raw_b_loss.item()}
            else:
                anchor_indices.append(i)

        if len(gen_losses) > 0:
            current_gen_loss_mean = torch.stack(gen_losses).mean()
            self.running_gen_loss = 0.9 * self.running_gen_loss + 0.1 * current_gen_loss_mean.item()
            scaling_gen_loss = current_gen_loss_mean.detach()
        else:
            scaling_gen_loss = torch.tensor(self.running_gen_loss, device=logits.device)

        batch_ref_anchor_loss_sum = torch.tensor(0.0, device=logits.device)
        suppressed_anchor_losses = [] 
        valid_anchor_count = 0

        for idx in anchor_indices:
            a_m = shift_a_masks[idx]
            a_count = a_m.sum()
            if a_count > 0:
                raw_curr = (token_losses[idx] * a_m).sum() / a_count
                cached = cached_betas[idx]
                raw_ref = torch.tensor(cached, device=logits.device, dtype=logits.dtype) if cached is not None else raw_curr.detach()
                batch_ref_anchor_loss_sum += raw_ref
                valid_anchor_count += 1

                # Forward KL(ref||student) on answer only
                kl_anchor = (kl_ts[idx] * a_m).sum() / (a_count + eps)

                # Anchor 简化为纯 KL 乘以 kl_beta
                suppressed_anchor_losses.append(self.hint_config.kl_beta * kl_anchor)
                
                # 记录调试信息 (Instance suppress 等不再用于计算，仅占位)
                debug_map[idx] = {"gate": 0.0, "raw_loss": raw_curr.item(), "instance_beta": raw_ref.item(), "instance_suppress": 0.0}

        alpha_balance, batch_beta_val = 0.0, 0.0
        if valid_anchor_count > 0:
            batch_ref_mean = batch_ref_anchor_loss_sum / valid_anchor_count
            batch_beta_val = batch_ref_mean.item()
            
            # 不再应用 Alpha Balance
            alpha_balance = 1.0

        anchor_idx_ptr = 0
        for idx in anchor_indices:
            if idx in debug_map: 
                final_anchor_loss = suppressed_anchor_losses[anchor_idx_ptr]
                final_losses_map[idx] = final_anchor_loss
                debug_map[idx]["loss_contrib"] = final_anchor_loss.item()
                debug_map[idx]["final_balance_alpha"] = alpha_balance
                anchor_idx_ptr += 1

        debug_info_list = [debug_map.get(i) for i in range(token_losses.size(0))]
        batch_size = token_losses.size(0)
        raw_loss_vector = torch.stack([final_losses_map.get(i, torch.tensor(0.0, device=logits.device)) for i in range(batch_size)])
        
        task_weights = torch.zeros(batch_size, device=raw_loss_vector.device)
        if len(gen_indices) > 0: 
            task_weights[gen_indices] = 0.5 / len(gen_indices)
        if len(anchor_indices) > 0:
            task_weights[anchor_indices] = 0.5 / len(anchor_indices)

        weighted_loss_vec = raw_loss_vector * task_weights * 2.0
        final_loss = weighted_loss_vec.sum()

        return final_loss, debug_info_list, alpha_balance, batch_beta_val

# ==========================================
# 5. Callbacks
# ==========================================

class StepLogCallback(TrainerCallback):
    def __init__(self, log_file, log_interval, config: HintSFTConfig, output_dir: str):
        self.log_file = log_file
        self.log_interval = log_interval
        self.config = config
        self.output_dir = output_dir

    def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.log_interval == 0:
            stats = tracker.get_window_stats() 
            json_stats = stats.copy()
            json_stats.update({"epoch": state.epoch, "global_step": state.global_step, "timestamp": datetime.now().isoformat()})
            with open(self.log_file, "a", encoding="utf-8") as f: f.write(json.dumps(json_stats) + "\n")
            
            log_parts = [f"[Step {state.global_step}]"]
            for k, v in stats.items():
                val_str = f"{v:.6f}" if isinstance(v, float) else str(v)
                log_parts.append(f"{k}: {val_str}")
            logger.info(" | ".join(log_parts))
            
            tracker.reset_window()

class LogicalEpochLogCallback(TrainerCallback):
    def __init__(self, log_file, steps_per_logical_epoch, config: HintSFTConfig, output_dir: str): 
        self.log_file = log_file
        self.steps_per_epoch = steps_per_logical_epoch
        self.config = config
        self.output_dir = output_dir
        self.current_logical_epoch = 0
        self.early_stopped = False

    def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.steps_per_epoch == 0:
            self.current_logical_epoch += 1
            
            stats = tracker.get_epoch_stats()
            stats.update({"logical_epoch": self.current_logical_epoch, "global_step": state.global_step, "timestamp": datetime.now().isoformat()})
            with open(self.log_file, "a", encoding="utf-8") as f: f.write(json.dumps(stats) + "\n")
                
            logger.info(f"="*60)
            logger.info(f"*** LOGICAL EPOCH {self.current_logical_epoch} FINISHED ***")
            logger.info(f"  Avg ModeB (Raw):{stats['avg_mode_b_loss_raw']:.4f}")
            
            epoch_mode_b_raw = stats.get('avg_mode_b_loss_raw', 999.0)
            target = self.config.target_mode_b
            
            if target is not None and epoch_mode_b_raw <= target:
                logger.info(f"!!! TARGET REACHED: Epoch Mode B Loss ({epoch_mode_b_raw:.4f}) <= Target ({target}) !!!")
                
                early_stop_dir = os.path.join(self.output_dir, f"checkpoint-target-reached-epoch-{self.current_logical_epoch}")
                logger.info(f"Saving model to {early_stop_dir}...")
                
                try:
                    if model is not None:
                        model.save_pretrained(early_stop_dir)
                        if tokenizer is not None: tokenizer.save_pretrained(early_stop_dir)
                except Exception as e:
                    logger.error(f"Failed to save model: {e}")
                
                self.early_stopped = True
                control.should_training_stop = True
            
            logger.info(f"="*60)
            tracker.reset_epoch()

# ==========================================
# 6. Main Execution Function
# ==========================================

def run_sira_training_v3(
    model_path: str,
    data_path: Optional[str] = None,
    output_base_dir: Optional[str] = None,
    batch_size: int = 8,
    real_data_epochs: int = 10,
    device_num: int = 1,
    spilt: float = 0.5,
    target_mode_b: float = 0.13,
    lora_path: Optional[str] = None,
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
        anchor_loss_weight_k=0.1, 
        suppress_max_scale=1.0,
        anchor_sigmoid_slope=50, 
        anchor_loss_tolerance=1.01,
        metrics_log_interval=batch_size,
        real_data_epochs=real_data_epochs,
        target_mode_b=target_mode_b 
    )
    
    output_dir = f"{hint_config.output_base_dir}/sira_sft_{real_data_epochs}ep_{datetime.now().strftime('%m%d_%H%M')}"
    step_log_file, epoch_log_file = setup_logging(output_dir)
    
    logger.info(f"Model Path: {hint_config.model_path}")
    logger.info(f"Target Mode B Loss (Stop Condition): {target_mode_b}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(hint_config.model_path, trust_remote_code=True, use_fast=False)
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise e

    if not os.path.exists(hint_config.data_path):
        raise FileNotFoundError(f"Dataset not found at {hint_config.data_path}")
    dataset = Dataset.from_json(hint_config.data_path)

    collator = FixedModeCollator(tokenizer)

    total_samples = len(dataset)
    total_steps = total_samples // batch_size
    steps_per_logical_epoch = total_steps // real_data_epochs
    
    if steps_per_logical_epoch < 1:
        steps_per_logical_epoch = 1
        logger.warning(f"Total steps ({total_steps}) < real epochs ({real_data_epochs}). Set logical step to 1.")

    model = AutoModelForCausalLM.from_pretrained(
        hint_config.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
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
            r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM", bias="none"
        )
        model = get_peft_model(model, peft_config)

    model.enable_input_require_grads()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size // device_num,
        gradient_accumulation_steps=1,
        learning_rate=5e-5, 
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=hint_config.metrics_log_interval, 
        save_strategy="steps",           
        save_steps=steps_per_logical_epoch * (real_data_epochs/SAVE_TOTAL),
        save_total_limit=SAVE_TOTAL,
        fp16=False, bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_drop_last=True, report_to="none"                 
    )

    step_callback = StepLogCallback(step_log_file, hint_config.metrics_log_interval, hint_config, output_dir)
    epoch_callback = LogicalEpochLogCallback(epoch_log_file, steps_per_logical_epoch, hint_config, output_dir)

    trainer = SequentialTrainer(
        hint_config=hint_config,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[step_callback, epoch_callback]
    )

    logger.info(f"Starting SIRA training...")
    trainer.train()
    
    if not epoch_callback.early_stopped:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Training finished normally (max epochs reached). Model saved to {output_dir}")
    else:
        logger.info(f"Training finished with early stopping (Target Loss reached).")

if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path)) 
    project_root = os.path.dirname(os.path.dirname(project_root))
    default_model_dir = os.path.join(project_root, "CELPO", "model", "OREAL")
    default_model_url = os.path.join(default_model_dir, "OREAL-7B")

    run_sira_training_v2(
        model_path=default_model_url,
        batch_size=8,
        real_data_epochs=50,
        target_mode_b=1.2 
    )

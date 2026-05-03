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
    split_r: float = 0.5
    
    # KL 正则权重
    hint_kl_beta: float = 0.3          # Hint 部分的 Forward KL 正则权重
    answer_kl_beta: float = 1.0        # Mode B answer 部分的 Reverse KL 权重
    anchor_kl_beta: float = 1.0        # Anchor 部分的 Reverse KL 权重
    
    # Early stop 目标
    target_mode_b: Optional[float] = None
    
    # 训练配置
    metrics_log_interval: int = 8
    model_path: str = ""
    data_path: str = ""
    output_base_dir: str = "/root/autodl-tmp/output"
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
    """跟踪训练指标：hint_ce, hint_kl, hint_loss, answer_raw/weighted, anchor_raw/weighted, total_loss"""
    
    _FIELDS = [
        "hint_ce", "hint_kl", "hint_loss",
        "answer_raw", "answer_weighted",
        "anchor_raw", "anchor_weighted",
    ]

    def __init__(self):
        self.reset_window()
        self.reset_epoch()

    def _empty_buckets(self):
        return {f: [] for f in self._FIELDS}

    def reset_window(self):
        self.win_loss, self.win_steps = 0.0, 0
        self.win = self._empty_buckets()
        self.win_counts = {"anchor": 0, "mode_b": 0}

    def reset_epoch(self):
        self.ep_loss, self.ep_steps = 0.0, 0
        self.ep = self._empty_buckets()
        self.ep_counts = {"anchor": 0, "mode_b": 0}

    def update(self, loss, metadata, debug_info):
        self.win_loss += loss
        self.win_steps += 1
        self.ep_loss += loss
        self.ep_steps += 1

        for meta, info in zip(metadata, debug_info):
            if info is None:
                continue
            mode = meta.get("mode")

            if mode == "mode_b_generation":
                self.win_counts["mode_b"] += 1
                self.ep_counts["mode_b"] += 1
                for k in ("hint_ce", "hint_kl", "hint_loss", "answer_raw", "answer_weighted"):
                    self.win[k].append(info.get(k, 0.0))
                    self.ep[k].append(info.get(k, 0.0))

            elif mode == "pure_sft_anchor":
                self.win_counts["anchor"] += 1
                self.ep_counts["anchor"] += 1
                for k in ("anchor_raw", "anchor_weighted"):
                    self.win[k].append(info.get(k, 0.0))
                    self.ep[k].append(info.get(k, 0.0))

    @staticmethod
    def _avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    def _calculate_stats(self, loss_sum, steps, buckets, counts):
        stats = {"total_loss": loss_sum / max(steps, 1)}
        for f in self._FIELDS:
            stats[f"avg_{f}"] = self._avg(buckets[f])
        stats["sample_counts"] = counts
        return stats

    def get_window_stats(self):
        return self._calculate_stats(self.win_loss, self.win_steps, self.win, self.win_counts)

    def get_epoch_stats(self):
        return self._calculate_stats(self.ep_loss, self.ep_steps, self.ep, self.ep_counts)

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
        }

# ==========================================
# 4. SIRA Trainer
# ==========================================
class SequentialTrainer(Trainer):
    def __init__(self, hint_config: HintSFTConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hint_config = hint_config

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

        # --- 极致显存优化：逐样本计算 ref logits，避免 batch 级别的 [B,T,V] 张量 ---
        # Step 1: student 前向（需要梯度）
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")

        # Step 2: 逐样本计算 loss（ref forward 在循环内部按需计算）
        loss, debug_info_list = self._ultra_memory_efficient_loss(
            model, inputs, logits, labels, hint_masks, answer_masks
        )

        if self.model.training:
            tracker.update(loss.item(), metadata, debug_info_list)

        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def _compute_masked_kl(logits_s, logits_t, mask, forward=True):
        """显存优化版 KL 计算：只对 mask=1 的 token 计算 KL
        
        Args:
            logits_s: student logits [T, V]
            logits_t: ref logits [T, V]
            mask: [T] 0/1 mask
            forward: True=Forward KL(ref||student), False=Reverse KL(student||ref)
        
        Returns:
            masked_kl_sum: 只对 mask=1 的 token 的 KL 总和（标量）
        """
        valid_indices = mask.nonzero(as_tuple=True)[0]
        if len(valid_indices) == 0:
            return torch.tensor(0.0, device=logits_s.device)
        
        # 只取有效 token 的 logits
        s_valid = logits_s[valid_indices]  # [N, V]
        t_valid = logits_t[valid_indices]  # [N, V]
        
        # 转为 log_prob
        log_p_s = torch.log_softmax(s_valid, dim=-1)  # [N, V]
        log_p_t = torch.log_softmax(t_valid, dim=-1)  # [N, V]
        
        if forward:
            # Forward KL: KL(ref || student) = sum_v p_t(v) * (log p_t(v) - log p_s(v))
            p_t = torch.exp(log_p_t)
            kl = (p_t * (log_p_t - log_p_s)).sum(dim=-1)  # [N]
        else:
            # Reverse KL: KL(student || ref) = sum_v p_s(v) * (log p_s(v) - log p_t(v))
            p_s = torch.exp(log_p_s)
            kl = (p_s * (log_p_s - log_p_t)).sum(dim=-1)  # [N]
        
        return kl.sum()  # 返回总和

    def _ultra_memory_efficient_loss(self, model, inputs, logits, labels, hint_masks, answer_masks):
        """极致显存优化版 loss 计算：
        - ref forward 逐样本计算，每次只保留 1 个样本的 ref logits [1, T, V]
        - 只对有 mask 的 token 计算 KL
        - 峰值显存 = 模型权重 + 1份student logits[B,T,V] + 1份ref logits[1,T,V]
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_h_masks = hint_masks[..., 1:].contiguous()
        shift_a_masks = answer_masks[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        eps = 1e-8
        batch_size = shift_logits.size(0)
        device = logits.device

        gen_indices, anchor_indices = [], []
        final_losses_map, debug_map = {}, {}

        for i in range(batch_size):
            s_logits_i = shift_logits[i]   # [T, V]
            labels_i = shift_labels[i]     # [T]
            h_m = shift_h_masks[i]         # [T]
            a_m = shift_a_masks[i]         # [T]

            # 判断是否需要 KL（如果 hint 或 answer mask 有值才需要 ref）
            need_kl = (h_m.sum() > 0) or (a_m.sum() > 0)

            # 逐样本计算 ref logits（只保留当前 1 个样本的 ref logits）
            if need_kl:
                single_inputs = {
                    k: v[i:i+1] for k, v in inputs.items()
                    if isinstance(v, torch.Tensor)
                }
                with torch.no_grad():
                    with self.model.disable_adapter():
                        ref_out = model(**single_inputs)
                        r_logits_i = ref_out.logits[0, :-1, :].contiguous()  # [T, V]
                        del ref_out
            else:
                r_logits_i = None

            # CE loss
            token_ce = loss_fct(s_logits_i, labels_i)  # [T]

            h_count = h_m.sum()

            if h_count > 0:
                # === Mode B 样本 ===
                gen_indices.append(i)

                # Hint CE
                hint_ce = (token_ce * h_m).sum() / h_count

                # Hint Forward KL
                hint_fwd_kl_sum = self._compute_masked_kl(s_logits_i, r_logits_i, h_m, forward=True)
                hint_kl_raw = hint_fwd_kl_sum / (h_count + eps)
                hint_loss = hint_ce + self.hint_config.hint_kl_beta * hint_kl_raw

                a_count = a_m.sum()
                if a_count > 0:
                    answer_ce = (token_ce * a_m).sum() / a_count
                    answer_rev_kl_sum = self._compute_masked_kl(s_logits_i, r_logits_i, a_m, forward=False)
                    answer_kl_raw = answer_rev_kl_sum / (a_count + eps)
                    answer_weighted = self.hint_config.answer_kl_beta * answer_kl_raw
                else:
                    answer_ce = torch.tensor(0.0, device=device)
                    answer_kl_raw = torch.tensor(0.0, device=device)
                    answer_weighted = torch.tensor(0.0, device=device)

                l_gen = hint_loss + answer_weighted
                final_losses_map[i] = l_gen
                debug_map[i] = {
                    "hint_ce": hint_ce.item(),
                    "hint_kl": hint_kl_raw.item(),
                    "hint_loss": hint_loss.item(),
                    "answer_raw": answer_ce.item(),
                    "answer_kl_raw": answer_kl_raw.item(),
                    "answer_weighted": answer_weighted.item(),
                }
            else:
                # === Anchor 样本 ===
                anchor_indices.append(i)
                a_count = a_m.sum()
                if a_count > 0:
                    anchor_ce = (token_ce * a_m).sum() / a_count
                    anchor_rev_kl_sum = self._compute_masked_kl(s_logits_i, r_logits_i, a_m, forward=False)
                    anchor_kl_raw = anchor_rev_kl_sum / (a_count + eps)
                    anchor_weighted = self.hint_config.anchor_kl_beta * anchor_kl_raw
                    final_losses_map[i] = anchor_weighted
                    debug_map[i] = {
                        "anchor_raw": anchor_ce.item(),
                        "anchor_kl_raw": anchor_kl_raw.item(),
                        "anchor_weighted": anchor_weighted.item(),
                    }

            # 立即释放当前样本的 ref logits
            del token_ce, r_logits_i

        debug_info_list = [debug_map.get(i) for i in range(batch_size)]

        # 50/50 任务加权
        raw_loss_vector = torch.stack([
            final_losses_map.get(i, torch.tensor(0.0, device=device))
            for i in range(batch_size)
        ])
        task_weights = torch.zeros(batch_size, device=device)
        if len(gen_indices) > 0:
            task_weights[gen_indices] = 0.5 / len(gen_indices)
        if len(anchor_indices) > 0:
            task_weights[anchor_indices] = 0.5 / len(anchor_indices)

        weighted_loss_vec = raw_loss_vector * task_weights * 2.0
        final_loss = weighted_loss_vec.sum()

        return final_loss, debug_info_list

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
        self.saved_model_dir = None

    def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.steps_per_epoch == 0:
            self.current_logical_epoch += 1
            
            stats = tracker.get_epoch_stats()
            stats.update({"logical_epoch": self.current_logical_epoch, "global_step": state.global_step, "timestamp": datetime.now().isoformat()})
            with open(self.log_file, "a", encoding="utf-8") as f: f.write(json.dumps(stats) + "\n")
                
            logger.info(f"="*60)
            logger.info(f"*** LOGICAL EPOCH {self.current_logical_epoch} FINISHED ***")
            logger.info(f"  Total Loss: {stats['total_loss']:.4f}")
            logger.info(f"  Hint CE: {stats['avg_hint_ce']:.4f} | Hint KL: {stats['avg_hint_kl']:.4f} | Hint Loss: {stats['avg_hint_loss']:.4f}")
            logger.info(f"  Answer Raw(CE): {stats['avg_answer_raw']:.4f} | Answer Weighted(βKL): {stats['avg_answer_weighted']:.4f}")
            logger.info(f"  Anchor Raw(CE): {stats['avg_anchor_raw']:.4f} | Anchor Weighted(βKL): {stats['avg_anchor_weighted']:.4f}")
            
            epoch_hint_ce = stats.get('avg_hint_ce', 999.0)
            target = self.config.target_mode_b
            
            if target is not None and epoch_hint_ce <= target:
                logger.info(f"!!! TARGET REACHED: Epoch Hint CE ({epoch_hint_ce:.4f}) <= Target ({target}) !!!")
                
                early_stop_dir = os.path.join(self.output_dir, f"checkpoint-target-reached-epoch-{self.current_logical_epoch}")
                logger.info(f"Saving model to {early_stop_dir}...")
                
                try:
                    if model is not None:
                        model.save_pretrained(early_stop_dir)
                        if tokenizer is not None: tokenizer.save_pretrained(early_stop_dir)
                        self.saved_model_dir = early_stop_dir
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
    target_mode_b: float = 0.16,
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
        final_lora_path = output_dir
    else:
        logger.info(f"Training finished with early stopping (Target Loss reached).")
        final_lora_path = epoch_callback.saved_model_dir or output_dir

    logger.info(f"Final LoRA path: {final_lora_path}")
    return final_lora_path

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

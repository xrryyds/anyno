import os
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
from typing import Dict, List, Optional, Tuple

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
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

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
# ==========================================

@dataclass
class HintSFTConfig:
    split_r: float = 0.5
    
    hint_kl_beta: float = 0.3

    # ── answer_kl_beta / anchor_kl_beta ──────────────────────────────────
    # 两种设置方式（二选一）：
    #
    # 方式 A：固定值（简单基线）
    #   经验推荐范围：
    #     answer_kl_beta  ∈ [0.05, 0.5]
    #       - answer 部分用 Reverse KL 防止答案区域坍缩，但不能压制 hint_loss
    #       - 目标：answer_kl_beta * answer_KL_raw ≈ 0.05~0.15 × hint_loss
    #       - 若 answer_KL_raw 典型值 ≈ 2~5，hint_loss ≈ 1~3，则 beta ≈ 0.05~0.2
    #     anchor_kl_beta  ∈ [0.5, 3.0]
    #       - anchor 样本无 CE，只有 KL；需使 anchor loss 量级 ≈ mode_b loss 量级
    #       - 若 anchor_KL_raw 典型值 ≈ 2~5，mode_b loss ≈ 1~3，则 beta ≈ 0.3~1.5
    #
    # 方式 B：自适应（adaptive=True，推荐）
    #   用 EMA 追踪各 KL 项的滑动均值，动态计算 beta 使加权 KL 保持在
    #   目标比例 answer_kl_target_ratio / anchor_kl_target_ratio 处。
    #   公式：beta_t = target_ratio * ema_hint_loss / (ema_kl + eps)
    # ─────────────────────────────────────────────────────────────────────
    answer_kl_beta: float = 0.1          # 方式 A 固定值（adaptive=False 时生效）
    anchor_kl_beta: float = 1.0          # 方式 A 固定值（adaptive=False 时生效）

    # 自适应开关与超参
    adaptive_kl_beta: bool = True        # True = 方式 B 自适应
    answer_kl_target_ratio: float = 0.1  # 目标：answer_weighted ≈ ratio × hint_loss
    anchor_kl_target_ratio: float = 1.0  # 目标：anchor_weighted ≈ ratio × hint_loss
    adaptive_ema_alpha: float = 0.05     # EMA 平滑系数（越小越稳定，越大越敏感）
    adaptive_beta_min: float = 0.01      # beta 下界，防止过小
    adaptive_beta_max: float = 20.0      # beta 上界，防止梯度爆炸

    top_entropy_quantile: float = 0.2
    
    target_mode_b: Optional[float] = None
    
    metrics_log_interval: int = 8
    model_path: str = ""
    data_path: str = ""
    output_base_dir: str = "/root/autodl-tmp/output"
    real_data_epochs: int = 50

    # ── 验证集早停（LiveMathBench）────────────────────────────────────────
    # use_val_early_stop: 总开关，True = 每个逻辑 epoch 结束后在 LiveMathBench
    #   上做 greedy decode 评估，以验证集准确率作为早停依据。
    # val_max_size: 每次评估使用的题目数量（None = 全量，建议 100~300 加速）
    # val_max_new_tokens: 生成时最大 token 数
    # val_patience: 连续多少个逻辑 epoch 准确率不提升则触发早停
    # val_min_epochs: 至少跑多少个逻辑 epoch 后才允许早停（冷启动保护）
    # ─────────────────────────────────────────────────────────────────────
    use_val_early_stop: bool = True
    val_max_size: Optional[int] = 200
    val_max_new_tokens: int = 2048   # 与训练 MAX_SEQ_LENGTH 对齐（prompt ≤ 1024 + gen ≤ 2048）
    val_patience: int = 3
    val_min_epochs: int = 3

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
    """hint_ce, hint_kl, hint_loss, answer_raw/weighted, anchor_raw/weighted, total_loss"""
    
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

        # ── 自适应 beta 的 EMA 状态 ──────────────────────────────────────
        # 追踪三个量的指数移动平均：
        #   _ema_hint_loss    : hint_loss（CE + hint_kl）的均值，作为"参考量级"
        #   _ema_answer_kl    : answer 部分 raw KL 的均值
        #   _ema_anchor_kl    : anchor 部分 raw KL 的均值
        # 初始值设为 1.0，避免冷启动时 beta 爆炸
        self._ema_hint_loss: float = 1.0
        self._ema_answer_kl: float = 1.0
        self._ema_anchor_kl: float = 1.0
        self._ema_initialized: bool = False   # 第一个 batch 后用真实值覆盖

    def _update_adaptive_betas(self, hint_loss_val: float, answer_kl_val: float, anchor_kl_val: float):
        """用 EMA 追踪各项均值，动态计算 answer_kl_beta / anchor_kl_beta。

        公式（每个 batch 更新一次）：
            ema_x  ← α * x + (1-α) * ema_x
            beta   = clamp(target_ratio * ema_hint_loss / (ema_kl + ε),
                           beta_min, beta_max)

        目标语义：
            answer_weighted  ≈ answer_kl_target_ratio  × hint_loss
            anchor_weighted  ≈ anchor_kl_target_ratio  × hint_loss
        """
        cfg = self.hint_config
        α = cfg.adaptive_ema_alpha
        eps = 1e-8

        if not self._ema_initialized:
            # 冷启动：直接用第一个 batch 的真实值初始化，避免 1.0 偏差
            self._ema_hint_loss = hint_loss_val if hint_loss_val > 0 else 1.0
            self._ema_answer_kl = answer_kl_val if answer_kl_val > 0 else 1.0
            self._ema_anchor_kl = anchor_kl_val if anchor_kl_val > 0 else 1.0
            self._ema_initialized = True
        else:
            if hint_loss_val > 0:
                self._ema_hint_loss = α * hint_loss_val + (1 - α) * self._ema_hint_loss
            if answer_kl_val > 0:
                self._ema_answer_kl = α * answer_kl_val + (1 - α) * self._ema_answer_kl
            if anchor_kl_val > 0:
                self._ema_anchor_kl = α * anchor_kl_val + (1 - α) * self._ema_anchor_kl

        new_answer_beta = cfg.answer_kl_target_ratio * self._ema_hint_loss / (self._ema_answer_kl + eps)
        new_anchor_beta = cfg.anchor_kl_target_ratio * self._ema_hint_loss / (self._ema_anchor_kl + eps)

        cfg.answer_kl_beta = float(max(cfg.adaptive_beta_min, min(cfg.adaptive_beta_max, new_answer_beta)))
        cfg.anchor_kl_beta = float(max(cfg.adaptive_beta_min, min(cfg.adaptive_beta_max, new_anchor_beta)))

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

        with torch.no_grad():
            with self.model.disable_adapter():
                ref_outputs = model(**inputs)
                ref_logits = ref_outputs.logits[..., :-1, :].contiguous()
                del ref_outputs
                torch.cuda.empty_cache()

        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")

        loss, debug_info_list = self._memory_efficient_loss(
            logits, ref_logits, labels, hint_masks, answer_masks
        )
        del ref_logits

        if self.model.training:
            tracker.update(loss.item(), metadata, debug_info_list)

            # ── 自适应 beta 更新 ──────────────────────────────────────────
            if self.hint_config.adaptive_kl_beta:
                hint_loss_vals  = [d["hint_loss"]      for d in debug_info_list if d and "hint_loss"      in d]
                answer_kl_vals  = [d["answer_kl_raw"]  for d in debug_info_list if d and "answer_kl_raw"  in d]
                anchor_kl_vals  = [d["anchor_kl_raw"]  for d in debug_info_list if d and "anchor_kl_raw"  in d]
                avg_hint_loss   = sum(hint_loss_vals)  / len(hint_loss_vals)  if hint_loss_vals  else 0.0
                avg_answer_kl   = sum(answer_kl_vals)  / len(answer_kl_vals)  if answer_kl_vals  else 0.0
                avg_anchor_kl   = sum(anchor_kl_vals)  / len(anchor_kl_vals)  if anchor_kl_vals  else 0.0
                self._update_adaptive_betas(avg_hint_loss, avg_answer_kl, avg_anchor_kl)

        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def _compute_masked_kl(logits_s, logits_t, mask, forward=True, top_entropy_quantile=1.0):
        """ KL  mask=1  token  KL
        
        Args:
            logits_s: student logits [T, V]
            logits_t: ref logits [T, V]
            mask: [T] 0/1 mask
            forward: True=Forward KL(ref||student), False=Reverse KL(student||ref)
            top_entropy_quantile:  student  top-ρ  token  KL
                                  1.0 = 0.2 =  top 20%
        
        Returns:
            masked_kl_sum: KL 
            effective_count:  token 
        """
        valid_indices = mask.nonzero(as_tuple=True)[0]
        if len(valid_indices) == 0:
            return torch.tensor(0.0, device=logits_s.device), torch.tensor(0.0, device=logits_s.device)
        
        s_valid = logits_s[valid_indices]  # [N, V]
        t_valid = logits_t[valid_indices]  # [N, V]
        
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
        
        if top_entropy_quantile < 1.0 and len(kl) > 1:
            p_s_for_entropy = torch.exp(log_p_s)
            entropy = -(p_s_for_entropy * log_p_s).sum(dim=-1)  # [N]
            
            threshold = torch.quantile(entropy, 1.0 - top_entropy_quantile)
            high_entropy_mask = entropy >= threshold
            kl = kl[high_entropy_mask]
            
            del p_s_for_entropy, entropy
        
        effective_count = torch.tensor(float(len(kl)), device=logits_s.device)
        return kl.sum(), effective_count

    def _memory_efficient_loss(self, logits, ref_logits, labels, hint_masks, answer_masks):
        """ loss 
        -  mask  token  KL prompt 
        - 
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_ref_logits = ref_logits
        shift_labels = labels[..., 1:].contiguous()
        shift_h_masks = hint_masks[..., 1:].contiguous()
        shift_a_masks = answer_masks[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        eps = 1e-8
        batch_size = shift_logits.size(0)
        device = logits.device
        rho = self.hint_config.top_entropy_quantile

        gen_indices, anchor_indices = [], []
        final_losses_map, debug_map = {}, {}

        for i in range(batch_size):
            s_logits_i = shift_logits[i]   # [T, V]
            r_logits_i = shift_ref_logits[i]  # [T, V]
            labels_i = shift_labels[i]     # [T]
            h_m = shift_h_masks[i]         # [T]
            a_m = shift_a_masks[i]         # [T]

            token_ce = loss_fct(s_logits_i, labels_i)  # [T]

            h_count = h_m.sum()

            if h_count > 0:
                gen_indices.append(i)

                # Hint CE
                hint_ce = (token_ce * h_m).sum() / h_count

                hint_fwd_kl_sum, hint_kl_eff_count = self._compute_masked_kl(
                    s_logits_i, r_logits_i, h_m, forward=True, top_entropy_quantile=rho
                )
                hint_kl_raw = hint_fwd_kl_sum / (hint_kl_eff_count + eps)

                hint_loss = hint_ce + self.hint_config.hint_kl_beta * hint_kl_raw

                a_count = a_m.sum()
                if a_count > 0:
                    answer_ce = (token_ce * a_m).sum() / a_count
                    answer_rev_kl_sum, answer_kl_eff_count = self._compute_masked_kl(
                        s_logits_i, r_logits_i, a_m, forward=False, top_entropy_quantile=rho
                    )
                    answer_kl_raw = answer_rev_kl_sum / (answer_kl_eff_count + eps)
                    # answer_kl_beta 在 adaptive 模式下由 _update_adaptive_betas() 动态更新
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
                anchor_indices.append(i)
                a_count = a_m.sum()
                if a_count > 0:
                    anchor_ce = (token_ce * a_m).sum() / a_count
                    anchor_rev_kl_sum, anchor_kl_eff_count = self._compute_masked_kl(
                        s_logits_i, r_logits_i, a_m, forward=False, top_entropy_quantile=rho
                    )
                    anchor_kl_raw = anchor_rev_kl_sum / (anchor_kl_eff_count + eps)
                    # anchor_kl_beta 在 adaptive 模式下由 _update_adaptive_betas() 动态更新
                    anchor_weighted = self.hint_config.anchor_kl_beta * anchor_kl_raw
                    final_losses_map[i] = anchor_weighted
                    debug_map[i] = {
                        "anchor_raw": anchor_ce.item(),
                        "anchor_kl_raw": anchor_kl_raw.item(),
                        "anchor_weighted": anchor_weighted.item(),
                    }

            del token_ce

        debug_info_list = [debug_map.get(i) for i in range(batch_size)]

        raw_loss_vector = torch.stack([
            final_losses_map.get(i, torch.tensor(0.0, device=device))
            for i in range(batch_size)
        ])
        final_loss = raw_loss_vector.mean()

        return final_loss, debug_info_list

# ==========================================
# 5. LiveMathBench 验证集评估
# ==========================================

def evaluate_on_livemathbench(
    model,
    tokenizer,
    val_problems: List[str],
    val_answers: List[str],
    max_new_tokens: int = 1024,
    system_prompt: str = SYSTEM_PROMPT,
) -> Tuple[float, Dict]:
    """在 LiveMathBench 验证集上做 greedy decode，返回答题准确率。

    Args:
        model: 当前训练中的 PeftModel（已在 GPU 上）。
        tokenizer: 对应的 tokenizer。
        val_problems: 题目列表。
        val_answers: 标准答案列表（纯数字/表达式字符串）。
        max_new_tokens: 生成最大 token 数。
        system_prompt: 系统提示词。

    Returns:
        accuracy: 正确率 ∈ [0, 1]。
        detail: 包含 correct/total/accuracy 的字典。
    """
    import re as _re

    def _extract_boxed(text: str) -> str:
        """提取 \\boxed{...} 中的内容，支持嵌套花括号。"""
        if not text:
            return ""
        idx = text.rfind("\\boxed{")
        if idx == -1:
            return ""
        i = idx + 7
        content_start = i
        depth = 0
        while i < len(text):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                if depth == 0:
                    return text[content_start:i].strip()
                depth -= 1
            i += 1
        return ""

    def _normalize(s: str) -> str:
        if not s:
            return ""
        s = s.strip().lower().replace(" ", "")
        # 去掉 LaTeX 命令
        s = _re.sub(r'\\[a-zA-Z]+', '', s)
        # 只保留数字、字母、基本运算符
        s = _re.sub(r'[^0-9a-zA-Z\+\-\*/=\.\,]', '', s)
        return s

    model.eval()
    correct = 0
    total = len(val_problems)

    with torch.no_grad():
        for question, ref_ans in zip(val_problems, val_answers):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": str(question)},
            ]
            prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # prompt 截断到 1024 tokens，与训练侧 MAX_SEQ_LENGTH=2048 中 prompt 占位对齐
            inputs = tokenizer(
                prompt_str,
                return_tensors="pt",
                add_special_tokens=False,
                max_length=1024,
                truncation=True,
            ).to(model.device)

            # stop token ids 与 take_exam.py 对齐：eos + <|endoftext|>(151643) + <|im_end|>(151645)
            stop_ids = list({tokenizer.eos_token_id, 151643, 151645} - {None})
            try:
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,          # greedy，与 take_exam temperature=0.0 对齐
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=stop_ids,
                )
            except Exception as e:
                logger.warning(f"[Val] generate failed: {e}")
                continue

            # 只解码新生成的部分
            new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
            generated = tokenizer.decode(new_ids, skip_special_tokens=True)

            pred = _extract_boxed(generated)
            if _normalize(pred) == _normalize(str(ref_ans)) and ref_ans:
                correct += 1

    model.train()
    accuracy = correct / total if total > 0 else 0.0
    detail = {"correct": correct, "total": total, "accuracy": accuracy}
    return accuracy, detail


# ==========================================
# 6. Callbacks
# ==========================================

class StepLogCallback(TrainerCallback):
    def __init__(self, log_file, log_interval, config: HintSFTConfig, output_dir: str, trainer_ref=None):
        self.log_file = log_file
        self.log_interval = log_interval
        self.config = config
        self.output_dir = output_dir
        self.trainer_ref = trainer_ref  # 用于读取自适应 beta 的当前值

    def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.log_interval == 0:
            stats = tracker.get_window_stats()
            # 若开启自适应，把当前 beta 值也写入日志
            if self.config.adaptive_kl_beta and self.trainer_ref is not None:
                stats["answer_kl_beta"] = self.config.answer_kl_beta
                stats["anchor_kl_beta"] = self.config.anchor_kl_beta
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
    def __init__(
        self,
        log_file,
        steps_per_logical_epoch,
        config: HintSFTConfig,
        output_dir: str,
        val_problems: Optional[List[str]] = None,
        val_answers: Optional[List[str]] = None,
        tokenizer=None,                            # Bug1 fix: 存入 tokenizer 实例
    ):
        self.log_file = log_file
        self.steps_per_epoch = steps_per_logical_epoch
        self.config = config
        self.output_dir = output_dir
        self.current_logical_epoch = 0
        self.early_stopped = False
        self.saved_model_dir = None
        self.tokenizer = tokenizer                 # Bug1 fix: 持久化 tokenizer

        # ── 验证集早停状态 ────────────────────────────────────────────────
        self.val_problems = val_problems or []
        self.val_answers  = val_answers  or []
        self._best_val_acc: float = -1.0          # 历史最优验证集准确率
        self._no_improve_count: int = 0            # 连续未提升计数

    def _try_val_early_stop(self, model, tokenizer_kwarg, control) -> Optional[float]:
        """在验证集上评估，根据 patience 决定是否早停。

        tokenizer 优先使用构造时传入的 self.tokenizer，
        其次才用 HF Trainer callback 传来的 tokenizer_kwarg（可能为 None）。

        Returns:
            当前验证集准确率（若未评估则返回 None）。
        """
        cfg = self.config
        if not cfg.use_val_early_stop or not self.val_problems:
            return None

        # Bug1 fix: 优先用实例持有的 tokenizer，避免 HF Trainer 不传 tokenizer 的问题
        tok = self.tokenizer or tokenizer_kwarg
        if tok is None:
            logger.error("  [Val] tokenizer is None, skipping validation evaluation.")
            return None

        # 冷启动保护：至少跑 val_min_epochs 个逻辑 epoch 后才开始评估
        if self.current_logical_epoch < cfg.val_min_epochs:
            logger.info(
                f"  [Val] Skipping evaluation (epoch {self.current_logical_epoch} < "
                f"val_min_epochs {cfg.val_min_epochs})"
            )
            return None

        logger.info(f"  [Val] Evaluating on LiveMathBench ({len(self.val_problems)} samples)...")
        try:
            acc, detail = evaluate_on_livemathbench(
                model=model,
                tokenizer=tok,
                val_problems=self.val_problems,
                val_answers=self.val_answers,
                max_new_tokens=cfg.val_max_new_tokens,
            )
        except Exception as e:
            logger.error(f"  [Val] Evaluation failed: {e}")
            return None

        logger.info(
            f"  [Val] Accuracy: {acc:.4f} ({detail['correct']}/{detail['total']}) | "
            f"Best so far: {self._best_val_acc:.4f} | "
            f"No-improve streak: {self._no_improve_count}/{cfg.val_patience}"
        )

        if acc > self._best_val_acc:
            self._best_val_acc = acc
            self._no_improve_count = 0
            # 保存当前最优 checkpoint
            best_dir = os.path.join(self.output_dir, "checkpoint-best-val")
            logger.info(f"  [Val] New best! Saving model to {best_dir}...")
            try:
                if model is not None:
                    model.save_pretrained(best_dir)
                    if tok is not None:
                        tok.save_pretrained(best_dir)
                    self.saved_model_dir = best_dir
            except Exception as e:
                logger.error(f"  [Val] Failed to save best model: {e}")
        else:
            self._no_improve_count += 1
            logger.info(
                f"  [Val] No improvement for {self._no_improve_count} epoch(s). "
                f"Patience={cfg.val_patience}."
            )
            if self._no_improve_count >= cfg.val_patience:
                logger.info(
                    f"!!! VAL EARLY STOP: accuracy did not improve for "
                    f"{cfg.val_patience} consecutive epochs. "
                    f"Best val acc = {self._best_val_acc:.4f} !!!"
                )
                self.early_stopped = True
                control.should_training_stop = True

        return acc

    def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.steps_per_epoch == 0:
            self.current_logical_epoch += 1

            # Bug1 fix: 若 HF Trainer 传来了 tokenizer，更新实例持有的引用
            if tokenizer is not None:
                self.tokenizer = tokenizer

            stats = tracker.get_epoch_stats()
            stats.update({
                "logical_epoch": self.current_logical_epoch,
                "global_step": state.global_step,
                "timestamp": datetime.now().isoformat(),
            })

            logger.info("=" * 60)
            logger.info(f"*** LOGICAL EPOCH {self.current_logical_epoch} FINISHED ***")
            logger.info(f"  Total Loss: {stats['total_loss']:.4f}")
            logger.info(f"  Hint CE: {stats['avg_hint_ce']:.4f} | Hint KL: {stats['avg_hint_kl']:.4f} | Hint Loss: {stats['avg_hint_loss']:.4f}")
            logger.info(f"  Answer Raw(CE): {stats['avg_answer_raw']:.4f} | Answer Weighted(βKL): {stats['avg_answer_weighted']:.4f}")
            logger.info(f"  Anchor Raw(CE): {stats['avg_anchor_raw']:.4f} | Anchor Weighted(βKL): {stats['avg_anchor_weighted']:.4f}")
            if self.config.adaptive_kl_beta:
                logger.info(f"  [Adaptive β] answer_kl_beta={self.config.answer_kl_beta:.4f} | anchor_kl_beta={self.config.anchor_kl_beta:.4f}")

            # ── Bug2 fix: 先做验证集评估，再把 val_accuracy 写入 stats，最后写文件 ──
            val_acc = self._try_val_early_stop(model, tokenizer, control)
            if val_acc is not None:
                stats["val_accuracy"] = val_acc

            # 写 epoch 日志文件（包含 val_accuracy）
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(stats) + "\n")

            # 若验证集早停已触发，跳过训练集损失阈值判断
            if self.early_stopped:
                logger.info("=" * 60)
                tracker.reset_epoch()
                return

            # ── 训练集损失阈值早停（兜底，use_val_early_stop=False 时生效）────
            epoch_hint_ce = stats.get('avg_hint_ce', 999.0)
            target = self.config.target_mode_b

            if target is not None and epoch_hint_ce <= target:
                # Bug5 fix: 明确标注是 target-loss 早停
                logger.info(
                    f"!!! TRAIN LOSS EARLY STOP: Epoch Hint CE ({epoch_hint_ce:.4f}) "
                    f"<= Target ({target}) !!!"
                )
                early_stop_dir = os.path.join(
                    self.output_dir,
                    f"checkpoint-target-reached-epoch-{self.current_logical_epoch}"
                )
                logger.info(f"Saving model to {early_stop_dir}...")
                # Bug1 fix: 用 self.tokenizer 保存，不依赖 callback kwarg
                tok_to_save = self.tokenizer
                try:
                    if model is not None:
                        model.save_pretrained(early_stop_dir)
                        if tok_to_save is not None:
                            tok_to_save.save_pretrained(early_stop_dir)
                        self.saved_model_dir = early_stop_dir
                except Exception as e:
                    logger.error(f"Failed to save model: {e}")

                self.early_stopped = True
                control.should_training_stop = True

            logger.info("=" * 60)
            tracker.reset_epoch()

# ==========================================
# 7. Main Execution Function
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
    use_val_early_stop: bool = True,
    val_max_size: Optional[int] = 200,
    val_max_new_tokens: int = 2048,
    val_patience: int = 3,
    val_min_epochs: int = 3,
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
        target_mode_b=target_mode_b,
        use_val_early_stop=use_val_early_stop,
        val_max_size=val_max_size,
        val_max_new_tokens=val_max_new_tokens,
        val_patience=val_patience,
        val_min_epochs=val_min_epochs,
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

    # ── 加载 LiveMathBench 验证集 ─────────────────────────────────────────
    val_problems: List[str] = []
    val_answers:  List[str] = []
    if hint_config.use_val_early_stop:
        try:
            import sys as _sys
            _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if _project_root not in _sys.path:
                _sys.path.insert(0, _project_root)
            from data_math.LiveMath_data_util import LiveMathBench
            _lmb = LiveMathBench(split="test", max_size=hint_config.val_max_size)
            val_problems = _lmb.problems
            val_answers  = _lmb.answers
            logger.info(
                f"[Val] LiveMathBench loaded: {len(val_problems)} samples "
                f"(max_size={hint_config.val_max_size})"
            )
        except Exception as e:
            logger.warning(
                f"[Val] Failed to load LiveMathBench: {e}. "
                "Validation early stopping will be disabled."
            )
            hint_config.use_val_early_stop = False

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
        learning_rate=1e-6,
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
    epoch_callback = LogicalEpochLogCallback(
        log_file=epoch_log_file,
        steps_per_logical_epoch=steps_per_logical_epoch,
        config=hint_config,
        output_dir=output_dir,
        val_problems=val_problems,
        val_answers=val_answers,
        tokenizer=tokenizer,               # Bug1 fix: 确保 callback 持有 tokenizer 引用
    )

    trainer = SequentialTrainer(
        hint_config=hint_config,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[step_callback, epoch_callback]
    )
    # 把 trainer 引用回传给 step_callback，用于读取自适应 beta 当前值
    step_callback.trainer_ref = trainer

    logger.info(f"Starting SIRA training...")
    trainer.train()
    
    if not epoch_callback.early_stopped:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Training finished normally (max epochs reached). Model saved to {output_dir}")
        final_lora_path = output_dir
    else:
        logger.info(f"Training finished with early stopping.")
        final_lora_path = epoch_callback.saved_model_dir or output_dir

    logger.info(f"Final LoRA path: {final_lora_path}")
    return final_lora_path

if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path)) 
    project_root = os.path.dirname(os.path.dirname(project_root))
    default_model_dir = os.path.join(project_root, "CELPO", "model", "OREAL")
    default_model_url = os.path.join(default_model_dir, "OREAL-7B")

    run_sira_training_v3(
        model_path=default_model_url,
        batch_size=8,
        real_data_epochs=50,
        target_mode_b=1.2
    )

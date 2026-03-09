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
    set_seed,
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

try:
    from prompt import GEN_HINTS_WIH_ANSWER
except ImportError:
    GEN_HINTS_WIH_ANSWER = "{hints}{answer}"

SYSTEM_PROMPT = "Please reason step by step and put your final answer within \\boxed{}."

# ==========================================
# 1. 配置与工具类（沿用 v2）
# ==========================================


@dataclass
class HintSFTConfig:
    hint_fixed_weight: float = 1.0
    gate_threshold: float = 0.3
    gate_slope: float = 3.0
    split_r: float = 0.5

    # 核心超参（v3 中 anchor 相关的不会参与 loss，仅保留字段以兼容日志与配置）
    anchor_loss_weight_k: float = 1
    suppress_max_scale: float = 1.0
    anchor_sigmoid_slope: float = 1000.0
    anchor_loss_tolerance: float = 1.00

    # 仅保留 target_mode_b 作为早停标准
    target_mode_b: Optional[float] = None

    metrics_log_interval: int = 8
    model_path: str = ""
    data_path: str = ""
    output_base_dir: str = "/root/autodl-tmp/output"
    real_data_epochs: int = 50
    kl_lambda: float = 0.5  # KL(teacher||student) weight for mode_b
    kl_beta: float = 0.05   # v3 中不再使用 anchor KL，但字段保留


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(output_dir, "train.log"), encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    return (
        os.path.join(output_dir, "step_metrics.jsonl"),
        os.path.join(output_dir, "epoch_metrics.jsonl"),
    )


# ==========================================
# 2. Metrics Tracker（沿用 v2，anchor 统计在 v3 中会为 0）
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
                # v3 中 loss 不再显式使用 anchor，但数据与日志结构保持兼容
                self.win_counts["anchor"] += 1
                self.ep_counts["anchor"] += 1
                self.win_anchor_losses_weighted.append(loss_contrib)
                self.ep_anchor_losses_weighted.append(loss_contrib)
                self.win_anchor_losses_raw.append(raw_loss)
                self.ep_anchor_losses_raw.append(raw_loss)

    def _calculate_stats(
        self,
        loss_sum,
        steps,
        gate_vals,
        anchor_w,
        anchor_raw,
        mode_b,
        mode_b_raw,
        counts,
        alpha_vals,
        beta_vals,
    ):
        avg = lambda l: sum(l) / len(l) if l else 0.0
        return {
            "avg_train_loss": loss_sum / max(steps, 1),
            "avg_gate_value": avg(gate_vals),
            "avg_anchor_loss_weighted": avg(anchor_w),
            "avg_mode_b_loss": avg(mode_b),
            "avg_final_alpha": avg(alpha_vals),
            "avg_mode_b_loss_raw": avg(mode_b_raw),
            "avg_anchor_loss_raw": avg(anchor_raw),
            "avg_dynamic_beta": avg(beta_vals),
            "sample_counts": counts,
        }

    def get_window_stats(self):
        return self._calculate_stats(
            self.win_loss,
            self.win_steps,
            self.win_gate_values,
            self.win_anchor_losses_weighted,
            self.win_anchor_losses_raw,
            self.win_mode_b_losses,
            self.win_mode_b_raw,
            self.win_counts,
            self.win_alpha_values,
            self.win_beta_values,
        )

    def get_epoch_stats(self):
        return self._calculate_stats(
            self.ep_loss,
            self.ep_steps,
            self.ep_gate_values,
            self.ep_anchor_losses_weighted,
            self.ep_anchor_losses_raw,
            self.ep_mode_b_losses,
            self.ep_mode_b_raw,
            self.ep_counts,
            self.ep_alpha_values,
            self.ep_beta_values,
        )


tracker = TrainingMetricsTracker()


# ==========================================
# 3. Data Collator（沿用 v2）
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
        ref_betas_batch = []

        for item in batch:
            q = item["question"]
            b = item.get("hints", "")
            c = item.get("answer")
            data_type = item.get("type", "anchor_data")
            ref_betas_batch.append(item.get("ref_beta"))

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(q)},
            ]
            prompt_str = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = self.tokenizer(prompt_str, add_special_tokens=False).input_ids
            len_prompt = len(prompt_ids)

            if data_type == "anchor_data":
                mode_str = "pure_sft_anchor"
                answer_ids = (
                    self.tokenizer(str(c), add_special_tokens=False).input_ids
                    + [self.tokenizer.eos_token_id]
                )
                full_ids = prompt_ids + answer_ids
                h_mask = [0] * len(full_ids)
                a_mask = [0] * len(full_ids)
                for i in range(len_prompt, len(full_ids)):
                    a_mask[i] = 1
            else:
                mode_str = "mode_b_generation"
                target_text = GEN_HINTS_WIH_ANSWER.format(hints=b, answer=c)
                target_ids = (
                    self.tokenizer(target_text, add_special_tokens=False).input_ids
                    + [self.tokenizer.eos_token_id]
                )
                full_ids = prompt_ids + target_ids
                hint_only_text = f"{b}"
                hint_ids_only = self.tokenizer(
                    hint_only_text, add_special_tokens=False
                ).input_ids
                len_hint_part = len(hint_ids_only)
                h_mask, a_mask = [0] * len(full_ids), [0] * len(full_ids)
                hint_end_idx = min(len_prompt + len_hint_part, len(full_ids))
                for i in range(len_prompt, hint_end_idx):
                    h_mask[i] = 1
                for i in range(hint_end_idx, len(full_ids)):
                    a_mask[i] = 1

            if len(full_ids) > self.max_length:
                full_ids = full_ids[: self.max_length]
                h_mask = h_mask[: self.max_length]
                a_mask = a_mask[: self.max_length]

            labels = [
                full_ids[i] if (h_mask[i] or a_mask[i]) else -100
                for i in range(len(full_ids))
            ]
            input_ids_batch.append(torch.tensor(full_ids, dtype=torch.long))
            labels_batch.append(torch.tensor(labels, dtype=torch.long))
            hint_masks_batch.append(torch.tensor(h_mask, dtype=torch.float32))
            answer_masks_batch.append(torch.tensor(a_mask, dtype=torch.float32))
            attention_mask_batch.append(torch.ones(len(full_ids), dtype=torch.long))
            metadata_batch.append({"mode": mode_str})

        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                input_ids_batch,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            ),
            "labels": torch.nn.utils.rnn.pad_sequence(
                labels_batch, batch_first=True, padding_value=-100
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                attention_mask_batch, batch_first=True, padding_value=0
            ),
            "hint_masks": torch.nn.utils.rnn.pad_sequence(
                hint_masks_batch, batch_first=True, padding_value=0.0
            ),
            "answer_masks": torch.nn.utils.rnn.pad_sequence(
                answer_masks_batch, batch_first=True, padding_value=0.0
            ),
            "metadata": metadata_batch,
            "ref_betas": ref_betas_batch,
        }


# ==========================================
# 4. SIRA Trainer v3（无 anchor，加入 vocab 级 anchor 向量）
# ==========================================


class SequentialTrainerV3(Trainer):
    """v3 版本：

    - Mode B 的 token 级 loss + KL 计算方式与 v2 完全一致；
    - 不再显式使用样本级 anchor loss；
    - 在每个梯度更新中，将 "mode b 的平均 loss" 升维为 vocab 向量，
      与预先计算好的 avg_loss_per_vocab 做矢量和；
    - 对合成向量做 L2 归一化，使其模长与原始 mode b 向量的模长一致，
      从而保持整体梯度尺度不变，仅改变“方向”（由 anchor 引导）。
    """

    def __init__(self, hint_config: HintSFTConfig, anchor_vocab_loss: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hint_config = hint_config
        self.running_gen_loss = 1.0
        # anchor_vocab_loss: 预先计算好的 avg_loss_per_vocab, shape [V]
        # 在 compute_loss 中会搬到当前设备/精度
        if not isinstance(anchor_vocab_loss, torch.Tensor):
            anchor_vocab_loss = torch.tensor(anchor_vocab_loss, dtype=torch.float32)
        self.anchor_vocab_loss = anchor_vocab_loss.detach().clone()

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

        # Teacher logits（同 v2）
        with torch.no_grad():
            with self.model.disable_adapter():
                ref_logits = model(**inputs).get("logits")

        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")

        # v3：只返回 mode b 的平均 loss 和 debug 信息
        mode_b_mean_loss, debug_info_list = self.adaptive_gating_loss(
            logits, ref_logits, labels, hint_masks, answer_masks, ref_betas
        )

        # =========================
        # vocab 级矢量和 + 归一化
        # =========================
        device = logits.device
        dtype = logits.dtype
        anchor_vec = self.anchor_vocab_loss.to(device=device, dtype=dtype)

        vocab_size = anchor_vec.size(0)
        # 将标量的 mode_b_mean_loss 升维为 vocab 向量
        mode_b_vec = mode_b_mean_loss * torch.ones(
            vocab_size, device=device, dtype=dtype
        )

        combined_vec = mode_b_vec + anchor_vec

        mode_b_norm = torch.norm(mode_b_vec, p=2)
        combined_norm = torch.norm(combined_vec, p=2)
        eps = 1e-8

        if combined_norm.item() < eps or mode_b_norm.item() < eps:
            # 极端情况下，保持原始 mode_b 向量
            final_vec = mode_b_vec
        else:
            # 归一化，使模长保持为 ||mode_b_vec||
            final_vec = combined_vec / combined_norm * mode_b_norm

        # 将最终矢量 reduce 成标量 loss
        loss = final_vec.mean()

        if self.model.training:
            # v3 中不再显式使用 alpha/beta，这里传 None
            tracker.update(loss.item(), metadata, debug_info_list, None, None)

        return (loss, outputs) if return_outputs else loss

    def adaptive_gating_loss(
        self, logits, ref_logits, labels, hint_masks, answer_masks, cached_betas
    ):
        """v3 版 gating loss"""

        shift_logits = logits[..., :-1, :].contiguous()
        shift_ref_logits = ref_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_h_masks = hint_masks[..., 1:].contiguous()
        shift_a_masks = answer_masks[..., 1:].contiguous()
        del logits, ref_logits, labels, hint_masks, answer_masks

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        B = shift_logits.size(0)
        token_losses_list = []
        kl_list = []
        for j in range(B):
            ls_j = shift_logits[j]   # [T, V]
            lr_j = shift_ref_logits[j]  # [T, V]
            token_losses_list.append(loss_fct(ls_j, shift_labels[j]))
            # KL only on masked positions to avoid [T,V] intermediate
            kl_j = torch.zeros(ls_j.size(0), device=ls_j.device, dtype=ls_j.dtype)
            mask_j = (shift_h_masks[j] + shift_a_masks[j]).bool()
            if mask_j.any():
                idx = mask_j.nonzero(as_tuple=True)[0]
                lps = torch.nn.functional.log_softmax(ls_j[idx], dim=-1)
                lpt = torch.nn.functional.log_softmax(lr_j[idx], dim=-1).detach()
                kl_j[idx] = (lpt.exp() * (lpt - lps)).sum(-1)
            kl_list.append(kl_j)
        token_losses = torch.stack(token_losses_list, dim=0)  # [B, T]
        kl_ts = torch.stack(kl_list)                          # [B, T]
        del shift_logits, shift_ref_logits

        eps = 1e-8
        gen_losses = []
        debug_map = {}

        for i in range(token_losses.size(0)):
            h_m = shift_h_masks[i]
            h_count = h_m.sum()
            if h_count > 0:
                # Mode B 样本（与 v2 完全一致的定义）
                avg_h_loss = (token_losses[i] * h_m).sum() / h_count
                gate = torch.sigmoid(
                    self.hint_config.gate_slope
                    * (self.hint_config.gate_threshold - avg_h_loss.detach())
                )
                a_m = shift_a_masks[i]
                a_count = a_m.sum()
                avg_a_loss = (
                    (token_losses[i] * a_m).sum() / a_count
                    if a_count > 0
                    else torch.tensor(0.0, device=logits.device)
                )
                l_ce = (
                    self.hint_config.hint_fixed_weight * avg_h_loss
                    + gate * avg_a_loss
                )

                # KL(teacher||student): hint always, answer weighted by gate
                kl_w = h_m + gate.detach() * a_m
                l_kl = (kl_ts[i] * kl_w).sum() / (kl_w.sum() + eps)
                l_gen = l_ce + self.hint_config.kl_lambda * l_kl

                gen_losses.append(l_gen)

                total_valid_tokens = h_count + a_count
                raw_b_loss = (
                    ((token_losses[i] * h_m).sum() + (token_losses[i] * a_m).sum())
                    / total_valid_tokens
                    if total_valid_tokens > 0
                    else torch.tensor(0.0, device=logits.device)
                )
                debug_map[i] = {
                    "gate": gate.item(),
                    "loss_contrib": l_gen.item(),
                    "raw_loss": raw_b_loss.item(),
                }
            else:
                # v3 中：无 hint 样本不参与此 loss，debug 中标记为 None
                debug_map[i] = None

        if len(gen_losses) > 0:
            mode_b_mean_loss = torch.stack(gen_losses).mean()
            self.running_gen_loss = (
                0.9 * self.running_gen_loss + 0.1 * mode_b_mean_loss.item()
            )
        else:
            # 若本 batch 没有 mode B 样本，则退化为使用 running_gen_loss
            mode_b_mean_loss = torch.tensor(
                self.running_gen_loss, device=logits.device
            )

        debug_info_list = [debug_map.get(i) for i in range(token_losses.size(0))]
        return mode_b_mean_loss, debug_info_list


# ==========================================
# 5. Callbacks（沿用 v2，统计逻辑保持兼容）
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
            json_stats.update(
                {
                    "epoch": state.epoch,
                    "global_step": state.global_step,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(json_stats) + "\n")

            log_parts = [f"[Step {state.global_step}]"]
            for k, v in stats.items():
                val_str = f"{v:.6f}" if isinstance(v, float) else str(v)
                log_parts.append(f"{k}: {val_str}")
            logger.info(" | ".join(log_parts))

            tracker.reset_window()


class LogicalEpochLogCallback(TrainerCallback):
    """逻辑 Epoch 的日志打印 + 绝对值早停检查 (target_mode_b)"""

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
            stats.update(
                {
                    "logical_epoch": self.current_logical_epoch,
                    "global_step": state.global_step,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(stats) + "\n")

            logger.info("=" * 60)
            logger.info(
                f"*** LOGICAL EPOCH {self.current_logical_epoch} FINISHED ***"
            )
            logger.info(
                f"  Avg ModeB (Raw):{stats['avg_mode_b_loss_raw']:.4f}"
            )

            # === 仅检查 Epoch 级别的 Mode B Loss 是否达标 (target_mode_b) ===
            epoch_mode_b_raw = stats.get("avg_mode_b_loss_raw", 999.0)
            target = self.config.target_mode_b

            if target is not None and epoch_mode_b_raw <= target:
                logger.info(
                    f"!!! TARGET REACHED: Epoch Mode B Loss ({epoch_mode_b_raw:.4f}) <= Target ({target}) !!!"
                )

                early_stop_dir = os.path.join(
                    self.output_dir,
                    f"checkpoint-target-reached-epoch-{self.current_logical_epoch}",
                )
                logger.info(f"Saving model to {early_stop_dir}...")

                try:
                    if model is not None:
                        model.save_pretrained(early_stop_dir)
                        if tokenizer is not None:
                            tokenizer.save_pretrained(early_stop_dir)
                except Exception as e:
                    logger.error(f"Failed to save model: {e}")

                self.early_stopped = True
                control.should_training_stop = True

            logger.info("=" * 60)
            tracker.reset_epoch()


# ==========================================
# 6. Main Execution Function (v3)
# ==========================================


def run_sira_training_v3(
    model_path: str,
    anchor_vocab_loss_path: Optional[str] = None,
    data_path: Optional[str] = None,
    output_base_dir: Optional[str] = None,
    batch_size: int = 8,
    real_data_epochs: int = 50,
    device_num: int = 1,
    spilt: float = 0.5,
    target_mode_b: float = 0.03,
):
    """运行 v3 版 SIRA 训练：

    - mode b 的损失形态与 v2 一致；
    - 通过预计算的 avg_loss_per_vocab 作为 anchor 向量来引导梯度方向。
    """

    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path))
    project_root = os.path.dirname(os.path.dirname(project_root))

    if data_path is None:
        data_path = os.path.join(
            project_root, "CELPO", "datasets", "exam", "irdcl_data.json"
        )
    if output_base_dir is None:
        output_base_dir = os.path.join(project_root, "CELPO", "output")

    # 默认的 avg_loss_per_vocab 存放路径
    if anchor_vocab_loss_path is None:
        anchor_vocab_loss_path = os.path.join(
            project_root, "CELPO", "datasets", "exam", "avg_loss_per_vocab.pt"
        )

    if not os.path.exists(anchor_vocab_loss_path):
        raise FileNotFoundError(
            f"Anchor vocab loss file not found at {anchor_vocab_loss_path}"
        )

    anchor_vocab_loss = torch.load(anchor_vocab_loss_path, map_location="cpu")

    set_seed(42)
    tracker.reset_window()
    tracker.reset_epoch()

    hint_config = HintSFTConfig(
        model_path=model_path,
        data_path=data_path,
        output_base_dir=output_base_dir,
        split_r=spilt,
        anchor_loss_weight_k=1,
        suppress_max_scale=1.2,
        anchor_sigmoid_slope=50.0,
        anchor_loss_tolerance=1.001,
        metrics_log_interval=batch_size,
        real_data_epochs=real_data_epochs,
        target_mode_b=target_mode_b,
    )

    output_dir = f"{hint_config.output_base_dir}/sira_sft_v3_{real_data_epochs}ep_{datetime.now().strftime('%m%d_%H%M')}"
    step_log_file, epoch_log_file = setup_logging(output_dir)

    logger.info(f"Model Path: {hint_config.model_path}")
    logger.info(f"Data Path: {hint_config.data_path}")
    logger.info(f"Anchor Vocab Loss Path: {anchor_vocab_loss_path}")
    logger.info(f"Target Mode B Loss (Stop Condition): {target_mode_b}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            hint_config.model_path, trust_remote_code=True, use_fast=False
        )
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
        logger.warning(
            f"Total steps ({total_steps}) < real epochs ({real_data_epochs}). Set logical step to 1."
        )

    model = AutoModelForCausalLM.from_pretrained(
        hint_config.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
        bias="none",
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
        save_steps=steps_per_logical_epoch,
        save_total_limit=2,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_drop_last=True,
        group_by_length=False,
        report_to="none",
    )

    step_callback = StepLogCallback(
        step_log_file, hint_config.metrics_log_interval, hint_config, output_dir
    )
    epoch_callback = LogicalEpochLogCallback(
        epoch_log_file, steps_per_logical_epoch, hint_config, output_dir
    )

    trainer = SequentialTrainerV3(
        hint_config=hint_config,
        anchor_vocab_loss=anchor_vocab_loss,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[step_callback, epoch_callback],
    )

    logger.info("Starting SIRA v3 training...")
    trainer.train()

    if not epoch_callback.early_stopped:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(
            f"Training finished normally (max epochs reached). Model saved to {output_dir}"
        )
    else:
        logger.info(
            "Training finished with early stopping (Target Mode B Loss reached)."
        )


if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path))
    project_root = os.path.dirname(os.path.dirname(project_root))
    default_model_dir = os.path.join(project_root, "CELPO", "model", "OREAL")
    default_model_url = os.path.join(default_model_dir, "OREAL-7B")

    default_anchor_vocab_path = os.path.join(
        project_root, "CELPO", "datasets", "exam", "avg_loss_per_vocab.pt"
    )

    run_sira_training_v3(
        model_path=default_model_url,
        anchor_vocab_loss_path=default_anchor_vocab_path,
        batch_size=2,
        real_data_epochs=50,
        target_mode_b=1.2,
    )

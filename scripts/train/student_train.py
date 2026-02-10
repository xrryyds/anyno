import os
# 设置 VLLM 环境变量
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

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

# 尝试导入 Prompt 模板，失败则使用默认
try:
    from prompt import GEN_HINTS_WIH_ANSWER
except ImportError:
    GEN_HINTS_WIH_ANSWER = "# known:\n{hints}\n{answer}"

# ==========================================
# 1. 配置与工具类
# ==========================================

@dataclass
class HintSFTConfig:
    hint_fixed_weight: float = 1    # Hint Loss 的固定权重
    gate_threshold: float = 0.3     # Gate 开启的阈值
    gate_slope: float = 3.0         # Gate Sigmoid 的斜率
    split_r: float = 0.5            # Anchor 数据占比调节参数
    beta: float = 0.08              # 【修改】Mode B 期望的基准 Loss，改为 0.08
    metrics_log_interval: int = 8   # 日志打印间隔步数
    
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
        self.current_alpha = 0.0 
        self.win_counts = {"anchor": 0, "mode_b": 0}

    def reset_epoch(self):
        self.ep_loss, self.ep_steps = 0.0, 0
        self.ep_gate_values, self.ep_anchor_losses_weighted = [], []
        self.ep_anchor_losses_raw = []
        self.ep_mode_b_losses, self.ep_mode_b_raw = [], []
        self.ep_counts = {"anchor": 0, "mode_b": 0}

    def update(self, loss, metadata, debug_info, current_alpha):
        self.current_alpha = current_alpha
        self.win_loss += loss; self.win_steps += 1
        self.ep_loss += loss; self.ep_steps += 1
        
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

    def _calculate_stats(self, loss_sum, steps, gate_vals, anchor_w, anchor_raw, mode_b, mode_b_raw, counts, alpha):
        avg = lambda l: sum(l)/len(l) if l else 0.0
        return {
            "avg_train_loss": loss_sum / max(steps, 1), 
            "avg_gate_value": avg(gate_vals),
            "avg_anchor_loss_weighted": avg(anchor_w), 
            "avg_anchor_loss_raw": avg(anchor_raw),    
            "avg_mode_b_loss": avg(mode_b), 
            "avg_mode_b_loss_raw": avg(mode_b_raw), 
            "final_alpha": alpha, 
            "sample_counts": counts
        }

    def get_window_stats(self):
        return self._calculate_stats(
            self.win_loss, self.win_steps, self.win_gate_values, 
            self.win_anchor_losses_weighted, self.win_anchor_losses_raw, 
            self.win_mode_b_losses, self.win_mode_b_raw, 
            self.win_counts, self.current_alpha
        )

    def get_epoch_stats(self):
        return self._calculate_stats(
            self.ep_loss, self.ep_steps, self.ep_gate_values, 
            self.ep_anchor_losses_weighted, self.ep_anchor_losses_raw, 
            self.ep_mode_b_losses, self.ep_mode_b_raw, 
            self.ep_counts, self.current_alpha
        )

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
                
                # 计算 Hint 长度
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
# 4. SIRA Trainer (含模长归一化修正)
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
        # 计算每个token的loss
        token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        token_losses = token_losses.view(shift_labels.shape)

        gen_losses = []      # 用于构建向量和计算Alpha (优化后的实际梯度来源)
        gen_raw_losses = []  # 仅用于日志记录 (Raw Loss)
        
        anchor_indices = [] 
        gen_indices = []    
        debug_map = {}      

        # ------------------------------------------------------------------
        # Part 1: 计算每个样本的 Loss
        # ------------------------------------------------------------------
        for i in range(token_losses.size(0)):
            h_m = shift_h_masks[i]
            h_count = h_m.sum()
            
            # --- Mode B (Hint) ---
            if h_count > 0:
                gen_indices.append(i)
                avg_h_loss = (token_losses[i] * h_m).sum() / h_count
                
                # Gate 机制
                gate_input = self.hint_config.gate_slope * (self.hint_config.gate_threshold - avg_h_loss.detach())
                gate = torch.sigmoid(gate_input)
                
                a_m = shift_a_masks[i]
                a_count = a_m.sum()
                avg_a_loss = (token_losses[i] * a_m).sum() / a_count if a_count > 0 else torch.tensor(0.0, device=logits.device)
                
                # 加权生成 Loss (真实的优化目标)
                l_gen = self.hint_config.hint_fixed_weight * avg_h_loss + gate * avg_a_loss
                gen_losses.append(l_gen)

                # Raw Loss (仅做记录)
                total_valid_tokens = h_count + a_count
                if total_valid_tokens > 0:
                    raw_b_loss = ((token_losses[i] * h_m).sum() + (token_losses[i] * a_m).sum()) / total_valid_tokens
                else:
                    raw_b_loss = torch.tensor(0.0, device=logits.device)
                gen_raw_losses.append(raw_b_loss)

                debug_map[i] = {"gate": gate.item(), "loss_contrib": l_gen.item(), "raw_loss": raw_b_loss.item()}
            
            # --- Mode A (Anchor) ---
            else:
                anchor_indices.append(i)

        # ------------------------------------------------------------------
        # Part 2: 计算 Alpha 
        # 【修改】使用 gen_losses (真实梯度 Loss) 代替 raw_losses
        # ------------------------------------------------------------------
        if len(gen_losses) > 0:
            # 这里取的是 gen_losses 的平均值 (即 avg_mode_b_loss)
            current_optim_loss_tensor = torch.stack(gen_losses).mean()
            self.running_gen_loss = 0.9 * self.running_gen_loss + 0.1 * current_optim_loss_tensor.item()
            scaling_loss = current_optim_loss_tensor.detach()
        else:
            scaling_loss = torch.tensor(self.running_gen_loss, device=logits.device)

        r = self.hint_config.split_r
        vol_balance = (1 - r) / r
        # alpha = (真实ModeB Loss / Anchor基准RawLoss) * 平衡因子
        # 这样 Alpha * AnchorRawLoss 就会被拉升到与 ModeB Loss 同一个数量级
        alpha = vol_balance * (scaling_loss / self.hint_config.beta)
        
        final_losses_map = {}
        for idx, l_val in zip(gen_indices, gen_losses):
            final_losses_map[idx] = l_val

        for idx in anchor_indices:
            a_m = shift_a_masks[idx]
            a_count = a_m.sum()
            raw_anchor_loss = (token_losses[idx] * a_m).sum() / a_count if a_count > 0 else torch.tensor(0.0, device=logits.device)
            # Alpha 加权
            weighted_anchor_loss = alpha * raw_anchor_loss
            final_losses_map[idx] = weighted_anchor_loss
            debug_map[idx] = {"gate": 0.0, "loss_contrib": weighted_anchor_loss.item(), "raw_loss": raw_anchor_loss.item()}

        debug_info_list = [debug_map[i] for i in range(token_losses.size(0))]
        
        # ------------------------------------------------------------------
        # Part 3: 向量模长归一化
        # ------------------------------------------------------------------
        batch_size = token_losses.size(0)
        loss_vector = torch.stack([final_losses_map[i] for i in range(batch_size)])

        if len(gen_indices) > 0:
            is_mode_b = torch.zeros(batch_size, device=loss_vector.device)
            for idx in gen_indices:
                is_mode_b[idx] = 1.0
            
            loss_b_vec = loss_vector * is_mode_b
            
            norm_b = torch.norm(loss_b_vec, p=2)
            norm_total = torch.norm(loss_vector, p=2)
            
            if norm_total > 1e-6:
                scale_factor = (norm_b / norm_total).detach()
                final_loss_tensor = loss_vector * scale_factor
                final_loss = final_loss_tensor.mean()
            else:
                final_loss = loss_vector.mean()
        else:
            final_loss = loss_vector.mean()

        return final_loss, debug_info_list, (alpha.item() if isinstance(alpha, torch.Tensor) else alpha)

# ==========================================
# 5. Callbacks
# ==========================================

class StepLogCallback(TrainerCallback):
    def __init__(self, log_file, log_interval):
        self.log_file = log_file
        self.log_interval = log_interval
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.log_interval == 0:
            stats = tracker.get_window_stats() 
            stats.update({"epoch": state.epoch, "global_step": state.global_step, "timestamp": datetime.now().isoformat()})
            with open(self.log_file, "a", encoding="utf-8") as f: f.write(json.dumps(stats) + "\n")
            logger.info(f"[Step {state.global_step}] Loss: {stats['avg_train_loss']:.4f} | "
                        f"AncRaw: {stats['avg_anchor_loss_raw']:.4f} | "
                        f"ModeB(W): {stats['avg_mode_b_loss']:.4f} | "
                        f"ModeB(Raw): {stats['avg_mode_b_loss_raw']:.4f} | "
                        f"Alpha: {stats['final_alpha']:.4f}")
            tracker.reset_window()

class EpochLogCallback(TrainerCallback):
    def __init__(self, log_file): self.log_file = log_file
    def on_epoch_end(self, args, state, control, **kwargs):
        stats = tracker.get_epoch_stats()
        stats.update({"epoch": state.epoch, "global_step": state.global_step, "timestamp": datetime.now().isoformat()})
        with open(self.log_file, "a", encoding="utf-8") as f: f.write(json.dumps(stats) + "\n")
        logger.info(f"="*60)
        logger.info(f"*** EPOCH {state.epoch} FINISHED ***")
        logger.info(f"  Avg Epoch Loss: {stats['avg_train_loss']:.4f}")
        logger.info(f"  Avg Gate:       {stats['avg_gate_value']:.4f}")
        logger.info(f"  Avg ModeB (W):  {stats['avg_mode_b_loss']:.4f}")
        logger.info(f"  Avg ModeB Raw:  {stats['avg_mode_b_loss_raw']:.4f}")
        logger.info(f"  Avg Anc Raw:    {stats['avg_anchor_loss_raw']:.4f}")
        logger.info(f"="*60)
        tracker.reset_epoch()

# ==========================================
# 6. Main Execution Function
# ==========================================

def run_sira_training(
    model_path: str,
    data_path: Optional[str] = None,
    output_base_dir: Optional[str] = None,
    batch_size: int = 16,
    epoch: int = 20,
    device_num: int = 1
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
        split_r=0.5, 
        beta=0.08,  # 【修改】设置为 Anchor 的初始 Raw Loss 水平
        metrics_log_interval=8 
    )
    
    output_dir = f"{hint_config.output_base_dir}/sira_sft_{datetime.now().strftime('%m%d_%H%M')}"
    step_log_file, epoch_log_file = setup_logging(output_dir)
    
    logger.info(f"Model Path: {hint_config.model_path}")
    logger.info(f"Output Dir: {output_dir}")

    # --- Tokenizer ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(hint_config.model_path, trust_remote_code=True, use_fast=False)
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}"); raise e

    # --- Dataset ---
    if not os.path.exists(hint_config.data_path): raise FileNotFoundError(f"Dataset not found at {hint_config.data_path}")
    dataset = Dataset.from_json(hint_config.data_path)
    logger.info(f"Loaded dataset size: {len(dataset)}")

    # --- Model ---
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

    # --- Collator ---
    collator = FixedModeCollator(tokenizer)

    # --- Training Args ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epoch, 
        per_device_train_batch_size=2,   
        gradient_accumulation_steps=int(batch_size/2/device_num), # 【修改】强制转换为 int
        learning_rate=2e-5,
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

    logger.info(f"Starting SIRA training...")
    trainer.train()
    trainer.save_model(output_dir)
    logger.info(f"Training finished. Model saved to {output_dir}")

if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path)) 
    project_root = os.path.dirname(os.path.dirname(project_root))
    default_model_dir = os.path.join(project_root, "CELPO", "model", "OREAL")
    default_model_url = os.path.join(default_model_dir, "OREAL-7B")

    run_sira_training(model_path=default_model_url)

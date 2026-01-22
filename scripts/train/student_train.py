import os
import sys
import json
import time
import random
import torch
import torch.nn.functional as F
import transformers
import logging
from datetime import datetime, timedelta
from collections import OrderedDict
from dataclasses import dataclass, field
import peft

from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments,
    TrainerCallback,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from prompt import GEN_PROMPT, GEN_HINTS_WIH_ANSWER, GEN_ENHANCE_PROMPT

# ==========================================
# 2. 配置与工具类
# ==========================================

@dataclass
class HintSFTConfig:
    p_hint_start: float = 0.95    
    p_hint_end: float = 0.10      
    hint_fixed_weight: float = 1.0 
    gate_threshold: float = 2.5    
    gate_slope: float = 3.0       
    debug_sample_steps: int = 50

logger = logging.getLogger(__name__)

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_format = "[%(asctime)s][%(levelname)s][Rank %(process)d] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(os.path.join(output_dir, "train.log"), encoding='utf-8')
        ]
    )
    return os.path.join(output_dir, "metrics.jsonl")

def log_environment(args, output_dir):
    env_info = OrderedDict()
    env_info["Python"] = sys.version.split()[0]
    env_info["PyTorch"] = torch.__version__
    env_info["Transformers"] = transformers.__version__
    env_info["PEFT"] = peft.__version__
    env_info["CUDA"] = torch.version.cuda if torch.cuda.is_available() else "N/A"
    env_info["GPUs"] = torch.cuda.device_count()
    
    logger.info("*" * 40)
    logger.info("Runtime Environment:")
    for k, v in env_info.items():
        logger.info(f"{k}: {v}")
    logger.info("*" * 40)
    
    with open(os.path.join(output_dir, "training_args.json"), "w", encoding='utf-8') as f:
        json.dump(args.to_dict(), f, indent=4)

class HintDropoutCollator:
    def __init__(self, tokenizer, hint_config: HintSFTConfig, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.config = hint_config
        self.max_length = max_length
        self.current_step = 0
        self.total_steps = 1
        
        if self.tokenizer.pad_token_id is None:
             self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def set_progress(self, step, total):
        self.current_step = step
        self.total_steps = max(total, 1)

    def get_current_p_hint(self):
        # 论文 Eq.5: 线性衰减 p(t)
        if self.total_steps == 0: return self.config.p_hint_start
        progress = min(self.current_step / self.total_steps, 1.0)
        return self.config.p_hint_start + progress * (self.config.p_hint_end - self.config.p_hint_start)

    def __call__(self, batch):
        p_hint = self.get_current_p_hint()
        
        input_ids_batch = []
        labels_batch = []
        hint_masks_batch = []   
        answer_masks_batch = [] 
        attention_mask_batch = []
        metadata_batch = []

        for item in batch:
            q = item['question']
            b = item.get('hints', "")
            c = item['answer']

            # ====================================================
            # 论文对齐点: Data Replay Strategy
            # 判断样本属于 D_err (有Hint) 还是 D_corr (无Hint)
            # ====================================================
            has_hint_data = (b is not None and len(b.strip()) > 0)

            if not has_hint_data:
                # --- Scenario 1: D_corr (Correct Set) ---
                # 纯 SFT 模式，作为 "Stability Anchor"
                # 只有 Q -> A
                # 假设 GEN_PROMPT 只要 Q
                full_text = GEN_PROMPT.format(question=q) + c 
                mode = "pure_sft_anchor"
                
                # 编码
                full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]
                
                # 找到 Answer 开始位置
                prompt_text = GEN_PROMPT.format(question=q)
                len_prompt = len(self.tokenizer(prompt_text, add_special_tokens=False).input_ids)
                
                # Mask 构建: Hint全0, Answer全1
                current_hint_mask = [0] * len(full_ids)
                current_answer_mask = [0] * len(full_ids)
                
                safe_start = min(len_prompt, len(full_ids))
                for i in range(safe_start, len(full_ids)):
                    current_answer_mask[i] = 1

            else:
                # --- Scenario 2: D_err (Error Set) ---
                # 应用 IRDCL 课程学习: Mode A vs Mode B
                use_mode_a = random.random() < p_hint

                if use_mode_a:
                    # --- Mode A: Imitation Utilization (q + h -> a) ---
                    # 论文理论: "Stability Anchor" (High Prob initially)
                    full_text = GEN_ENHANCE_PROMPT.format(question=q, hints=b) + c
                    mode = "mode_a_utilization"
                    
                    full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]
                    
                    prompt_text = GEN_ENHANCE_PROMPT.format(question=q, hints=b)
                    len_prompt = len(self.tokenizer(prompt_text, add_special_tokens=False).input_ids)
                    
                    current_hint_mask = [0] * len(full_ids)
                    current_answer_mask = [0] * len(full_ids)
                    
                    safe_start = min(len_prompt, len(full_ids))
                    for i in range(safe_start, len(full_ids)):
                        current_answer_mask[i] = 1
                
                else:
                    # --- Mode B: Spontaneous Generation (q -> h + a) ---
                    # 论文理论: Capability Expansion
                    full_text = GEN_PROMPT.format(question=q) + GEN_HINTS_WIH_ANSWER.format(hints=b, answer=c)
                    mode = "mode_b_generation"
                    
                    full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]
                    
                    prompt_text = GEN_PROMPT.format(question=q)
                    len_prompt = len(self.tokenizer(prompt_text, add_special_tokens=False).input_ids)
                    
                    # 确定 Hint 边界
                    hint_header = "# known:\n"
                    prefix_upto_hint_end = prompt_text + hint_header + b
                    len_hint_end = len(self.tokenizer(prefix_upto_hint_end, add_special_tokens=False).input_ids)
                    
                    current_hint_mask = [0] * len(full_ids)
                    current_answer_mask = [0] * len(full_ids)
                    
                    # Hint Mask
                    safe_start_hint = min(len_prompt, len(full_ids))
                    safe_end_hint = min(len_hint_end, len(full_ids))
                    for i in range(safe_start_hint, safe_end_hint):
                        current_hint_mask[i] = 1
                    
                    # Answer Mask
                    for i in range(safe_end_hint, len(full_ids)):
                        current_answer_mask[i] = 1

            # --- 通用处理 (截断 & Labeling) ---
            if len(full_ids) > self.max_length:
                full_ids = full_ids[:self.max_length]
                current_hint_mask = current_hint_mask[:self.max_length]
                current_answer_mask = current_answer_mask[:self.max_length]

            labels = [-100] * len(full_ids)
            for i in range(len(full_ids)):
                # 只有被 mask 标记为 1 的部分才计算 loss
                if current_hint_mask[i] == 1 or current_answer_mask[i] == 1:
                    labels[i] = full_ids[i]

            input_ids_batch.append(torch.tensor(full_ids, dtype=torch.long))
            labels_batch.append(torch.tensor(labels, dtype=torch.long))
            hint_masks_batch.append(torch.tensor(current_hint_mask, dtype=torch.float))
            answer_masks_batch.append(torch.tensor(current_answer_mask, dtype=torch.float))
            attention_mask_batch.append(torch.ones(len(full_ids), dtype=torch.long))
            
            metadata_batch.append({
                "mode": mode, 
                "p_hint": p_hint,
                "raw_text": full_text
            })

        # Padding
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_batch, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels_batch, batch_first=True, padding_value=-100)
        hint_masks = torch.nn.utils.rnn.pad_sequence(hint_masks_batch, batch_first=True, padding_value=0.0)
        answer_masks = torch.nn.utils.rnn.pad_sequence(answer_masks_batch, batch_first=True, padding_value=0.0)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_batch, batch_first=True, padding_value=0)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "hint_masks": hint_masks,     
            "answer_masks": answer_masks, 
            "metadata": metadata_batch
        }

# ==========================================
# 4. Trainer (对齐 Adaptive Gating)
# ==========================================
class HintSFTTrainer(Trainer):
    def __init__(self, hint_config: HintSFTConfig, snapshot_file: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hint_config = hint_config
        self.snapshot_file = snapshot_file
        os.makedirs(os.path.dirname(snapshot_file), exist_ok=True)
        with open(snapshot_file, "w", encoding="utf-8") as f:
            f.write("")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        hint_masks = inputs.pop("hint_masks", None)
        answer_masks = inputs.pop("answer_masks", None)
        metadata = inputs.pop("metadata", None)
        
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss, debug_info = self.adaptive_gating_loss(logits, labels, hint_masks, answer_masks)

        if self.state.global_step % self.hint_config.debug_sample_steps == 0 and self.state.is_local_process_zero:
            self.save_debug_snapshot(metadata, loss.item(), debug_info)
            
        return (loss, outputs) if return_outputs else loss

    def adaptive_gating_loss(self, logits, labels, hint_masks, answer_masks):
        """
        Implementation of Confidence-Aware Adaptive Gating (Eq. 6, 7, 8 in Paper)
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_hint_masks = hint_masks[..., 1:].contiguous()
        shift_answer_masks = answer_masks[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        token_losses = token_losses.view(shift_labels.shape)

        batch_size = token_losses.size(0)
        total_loss = 0.0
        weights_log = [] 

        for i in range(batch_size):
            sample_losses = token_losses[i]
            h_mask = shift_hint_masks[i]
            a_mask = shift_answer_masks[i]
            
            h_count = h_mask.sum()
            
            if h_count > 0:
                # === Mode B: Spontaneous Generation ===
                avg_h_loss = (sample_losses * h_mask).sum() / h_count
                
                # 论文 Eq.7: Adaptive Gating with Stop-Gradient
                # w_gate = sigmoid( kappa * (mu - SG[L_hint]) )
                gate_input = self.hint_config.gate_slope * (self.hint_config.gate_threshold - avg_h_loss.detach())
                w_gate = torch.sigmoid(gate_input)
                
                w_h = self.hint_config.hint_fixed_weight
                
                a_count = a_mask.sum()
                avg_a_loss = (sample_losses * a_mask).sum() / a_count if a_count > 0 else 0.0
                
                # 论文 Eq.8: L_ModeB = L_hint + w_gate * L_answer
                # 注意: 代码中实际上对Hint也加了权重以强调生成，这在工程上是合理的
                sample_final_loss = (w_h * avg_h_loss) + (w_gate * avg_a_loss)
                
                weights_log.append(f"ModeB(Gate={w_gate:.2f}, HL={avg_h_loss.item():.2f})")
                
            else:
                # === Mode A (Utilization) OR Pure SFT (Anchor) ===
                # 论文: Stability Anchor. 
                # 此时没有Hint Loss，只有Answer Loss，权重为1.0
                a_count = a_mask.sum()
                avg_a_loss = (sample_losses * a_mask).sum() / a_count if a_count > 0 else 0.0
                
                sample_final_loss = avg_a_loss
                weights_log.append("Anchor")

            total_loss += sample_final_loss

        return total_loss / batch_size, weights_log

    def save_debug_snapshot(self, metadata, current_loss, debug_info):
        if not metadata: return
        sample = metadata[0]
        entry = {
            "step": self.state.global_step,
            "timestamp": datetime.now().isoformat(),
            "loss": current_loss,
            "p_hint": sample["p_hint"],
            "mode": sample["mode"],
            "gate_info": debug_info[0] if debug_info else "N/A",
            "text_preview": sample["raw_text"][:100].replace("\n", "\\n")
        }
        with open(self.snapshot_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ... (CurriculumCallback 保持不变) ...
class CurriculumCallback(TrainerCallback):
    def __init__(self, collator, log_file_path):
        self.collator = collator
        self.log_file_path = log_file_path
    def on_step_begin(self, args, state, control, **kwargs):
        self.collator.set_progress(state.global_step, state.max_steps)
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and logs is not None:
            logs["p_hint"] = self.collator.get_current_p_hint()
            log_entry = {"step": state.global_step, "timestamp": datetime.now().isoformat(), "epoch": state.epoch, **logs}
            with open(self.log_file_path, "a", encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + "\n")

# ==========================================
# 5. 验证函数 (Updated)
# ==========================================
def verify_collator(collator, dataset, tokenizer):
    print("\n" + "="*40)
    print(">>> Running Collator Verification")
    print("="*40)
    
    # 强制 p_hint=0 测试 Mode B, 但也要测试 Anchor 数据
    original_start = collator.config.p_hint_start
    collator.config.p_hint_start = 0.5 # 50% 概率 Mode A/B
    
    # 取前3个样本 (假设混合了有Hint和无Hint的数据)
    batch_data = [dataset[0], dataset[1]] 
    # 确保dataset[1]是无hint的以测试anchor逻辑，如果是随机生成需要注意
    
    output = collator(batch_data)
    
    for idx, meta in enumerate(output['metadata']):
        print(f"\n[Sample {idx} Mode]: {meta['mode']}")
        print(f"[Text Preview]: {meta['raw_text'][:50]}...")
        
        h_mask = output['hint_masks'][idx]
        a_mask = output['answer_masks'][idx]
        
        print(f"Hint Mask Sum: {h_mask.sum().item()}")
        print(f"Answer Mask Sum: {a_mask.sum().item()}")

    print("="*40 + "\n")
    collator.config.p_hint_start = original_start

# ==========================================
# 6. Main Execution
# ==========================================
def main():
    SEED = 42
    set_seed(SEED)

    ###################################### set model path
    model_name_or_path = "/root/project/data/xrr/OREAL-7B" 
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = os.getcwd()
    output_dir = os.path.join(project_root, "outputs", "hint_sft", timestamp)
    data_path= os.path.join(project_root, "datasets", "exam", "irdcl_data.json")

    metrics_log_path = setup_logging(output_dir)
    snapshot_log_path = os.path.join(output_dir, "debug_snapshots.jsonl")
    
    logger.info(f"Model: {model_name_or_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token



    if not os.path.exists(data_path):
        raise TypeError(f"Data file {data_path} not found. Creating mixed dummy data.")
    else:
        dataset = Dataset.from_json(data_path)

    hint_config = HintSFTConfig(
        p_hint_start=0.95, p_hint_end=0.1, 
        hint_fixed_weight=2.0, gate_threshold=2.5, gate_slope=3.0
    )

    # 验证
    debug_collator = HintDropoutCollator(tokenizer, hint_config, max_length=512)
    verify_collator(debug_collator, dataset, tokenizer)

    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if tokenizer.pad_token is None:
        model.config.pad_token_id = model.config.eos_token_id

    peft_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM", bias="none", modules_to_save=["embed_tokens", "lm_head"] 
    )
    model = get_peft_model(model, peft_config)
    for name, param in model.named_parameters():
        if "lora" in name or param.requires_grad:
            param.data = param.data.to(torch.float32)
    model.print_trainable_parameters()

    collator = HintDropoutCollator(tokenizer, hint_config, max_length=1024)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=f"hint_sft_{timestamp}",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        report_to=None
    )
    
    log_environment(training_args, output_dir)

    trainer = HintSFTTrainer(
        hint_config=hint_config,
        snapshot_file=snapshot_log_path,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[CurriculumCallback(collator, metrics_log_path)]
    )

    logger.info("Starting training...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()

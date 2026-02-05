import os
# 建议开启 vLLM 以加速 GRPO 采样，显存不足可设为 False
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import sys
import json
import torch
import logging
import warnings
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional

from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainerCallback,
    set_seed
)
from peft import PeftModel, LoraConfig
from trl import GRPOTrainer, GRPOConfig


from data_math import Math_All 

# ==========================================
# Logger 配置
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 1. 核心工具函数 (复用你的代码)
# ==========================================

def extract_boxed_content(text: str) -> Optional[str]:
    """
    提取 LaTeX 字符串中最后一个 \boxed{...} 里的内容。
    支持嵌套括号，例如 \boxed{\frac{1}{2}}。
    """
    if not text: return ""
    
    # 找到最后一个 \boxed{ 的位置
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""

    # 移动索引到 \boxed{ 之后
    i = idx + 7 
    content_start = i
    brace_balance = 0 
    
    # 开始遍历字符
    while i < len(text):
        char = text[i]
        
        if char == '{':
            brace_balance += 1
        elif char == '}':
            if brace_balance == 0:
                return text[content_start:i].strip()
            else:
                brace_balance -= 1
        
        i += 1
    return ""

# ==========================================
# 2. Metrics Tracker (复刻日志风格)
# ==========================================

class GRPOMetricsTracker:
    def __init__(self):
        self.reset_window()
        self.reset_epoch()
        
    def reset_window(self):
        """重置 Step 窗口统计数据"""
        self.win_steps = 0
        self.win_data = {
            "reward": [],
            "kl": [],
            "completion_length": [],
            "loss": 0.0
        }

    def reset_epoch(self):
        """重置 Epoch 全局统计数据"""
        self.ep_steps = 0
        self.ep_data = {
            "reward": [],
            "kl": [],
            "completion_length": [],
            "loss": 0.0
        }

    def update(self, logs: Dict):
        # TRL 的日志通常包含: 'loss', 'reward', 'kl', 'completion_length'
        self.win_steps += 1
        self.ep_steps += 1
        
        for key in ["reward", "kl", "completion_length"]:
            if key in logs:
                val = logs[key]
                # 有时是 tensor，有时是 float
                if isinstance(val, torch.Tensor):
                    val = val.item()
                self.win_data[key].append(val)
                self.ep_data[key].append(val)
        
        if "loss" in logs:
            val = logs["loss"]
            if isinstance(val, torch.Tensor):
                val = val.item()
            self.win_data["loss"] += val
            self.ep_data["loss"] += val

    def _calculate_stats(self, steps, data_dict):
        stats = {}
        # 计算平均值
        for key in ["reward", "kl", "completion_length"]:
            values = data_dict[key]
            stats[f"avg_{key}"] = sum(values) / len(values) if values else 0.0
        
        stats["avg_loss"] = data_dict["loss"] / max(steps, 1)
        stats["sample_count"] = len(data_dict["reward"])
        return stats

    def get_window_stats(self):
        return self._calculate_stats(self.win_steps, self.win_data)

    def get_epoch_stats(self):
        return self._calculate_stats(self.ep_steps, self.ep_data)

tracker = GRPOMetricsTracker()

# ==========================================
# 3. Callbacks (日志写入)
# ==========================================

def setup_logging_paths(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(output_dir, "train.log"), encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    return os.path.join(output_dir, "step_metrics.jsonl"), os.path.join(output_dir, "epoch_metrics.jsonl")

class GRPOStepLogCallback(TrainerCallback):
    def __init__(self, log_file, log_interval):
        self.log_file = log_file
        self.log_interval = log_interval

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # 过滤掉不相关的系统 log，只处理包含 RL 指标的 log
            if "reward" in logs or "loss" in logs:
                tracker.update(logs)

        if state.global_step > 0 and state.global_step % self.log_interval == 0:
            if tracker.win_steps > 0:
                stats = tracker.get_window_stats() 
                stats["epoch"] = state.epoch
                stats["global_step"] = state.global_step
                stats["timestamp"] = datetime.now().isoformat()
                
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(stats) + "\n")
                
                logger.info(f"[Step {state.global_step}] Loss: {stats['avg_loss']:.4f} | "
                            f"Reward: {stats['avg_reward']:.4f} | " 
                            f"KL: {stats['avg_kl']:.4f} | "
                            f"Len: {stats['avg_completion_length']:.1f}")
                
                tracker.reset_window()

class GRPOEpochLogCallback(TrainerCallback):
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
        logger.info(f"  Avg Epoch Loss:   {stats['avg_loss']:.4f}")
        logger.info(f"  Avg Reward:       {stats['avg_reward']:.4f}")
        logger.info(f"  Avg KL Diverg:    {stats['avg_kl']:.4f}")
        logger.info(f"  Avg Gen Length:   {stats['avg_completion_length']:.1f}")
        logger.info(f"="*60)
        
        tracker.reset_epoch()

# ==========================================
# 4. 奖励函数 (直接结果导向)
# ==========================================

def correctness_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    """
    answer: 数据集中的 ground truth answer (list of strings)
    """
    rewards = []
    for completion, gt in zip(completions, answer):
        # 1. 提取模型生成内容
        pred_content = extract_boxed_content(completion)
        
        # 2. 清洗数据 (去除首尾空白)
        clean_pred = pred_content.strip()
        clean_gt = gt.strip()
        
        # 3. 如果提取为空，直接0分
        if not clean_pred:
            rewards.append(0.0)
            continue
            
        # 4. 直接字符串比对
        if clean_pred == clean_gt:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            
    return rewards

# ==========================================
# 5. 主训练流程
# ==========================================

def run_grpo_training(
    model_path_base: str,
    model_path_lora: str,
    output_base_dir: str = "./output",
    subset_name: str = "algebra"
):
    set_seed(42)
    tracker.reset_window()
    tracker.reset_epoch()

    # --- 配置参数 ---
    output_dir = f"{output_base_dir}/grpo_{subset_name}_{datetime.now().strftime('%m%d_%H%M')}"
    step_log_file, epoch_log_file = setup_logging_paths(output_dir)
    
    logger.info(f"Base Model: {model_path_base}")
    logger.info(f"SFT LoRA:   {model_path_lora}")
    
    # --- 1. 加载 Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_path_base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 2. 加载模型并合并 SFT 权重 ---
    logger.info("Loading Base Model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path_base,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if os.path.exists(model_path_lora):
        logger.info(f"Merging SFT LoRA from {model_path_lora}...")
        model = PeftModel.from_pretrained(model, model_path_lora)
        model = model.merge_and_unload() # 合并，成为新的 Base Model 用于 RL
    else:
        logger.warning("SFT LoRA path not found. Starting GRPO from raw Base Model.")

    # --- 3. 准备数据 ---
    logger.info(f"Loading Dataset ({subset_name})...")
    # 使用你的 Math_All 类
    math_obj = Math_All(train=True, subset_name=subset_name, shuffle=True)
    
    data_dict = {"prompt": [], "answer": []}
    for prob, ans in zip(math_obj.problems, math_obj.answers):
        # 关键修改：直接使用 problem 作为 prompt，不添加任何 system prompt
        data_dict["prompt"].append(prob)
        # 假设 math_obj.answers 已经是清洗好的答案
        data_dict["answer"].append(ans)
        
    dataset = Dataset.from_dict(data_dict)
    logger.info(f"Dataset Size: {len(dataset)}")

    # --- 4. 配置 GRPO 新的 LoRA ---
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        bias="none"
    )

    # --- 5. 训练参数 ---
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-6,               # RL 学习率
        num_train_epochs=1,
        per_device_train_batch_size=2,    # Batch Size
        gradient_accumulation_steps=4,
        logging_steps=1,                  # 这里的 logging_steps 配合我们的 Callback 使用
        bf16=True,
        save_strategy="steps",
        save_steps=100,
        max_prompt_length=512,
        max_completion_length=512,
        num_generations=8,                # G: Group Size
        beta=0.04,                        # KL 惩罚
        report_to="none",                 # 关闭默认 wandb/tensorboard
        use_vllm=True,                    # 开启 vLLM 加速
        vllm_gpu_memory_utilization=0.5,
    )

    # --- 6. Trainer ---
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[correctness_reward_func], # 只有这一个奖励函数
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[
            GRPOStepLogCallback(step_log_file, log_interval=1),
            GRPOEpochLogCallback(epoch_log_file)
        ]
    )

    logger.info("Starting GRPO Training...")
    trainer.train()
    trainer.save_model(output_dir)
    logger.info(f"Training finished. Model saved to {output_dir}")

if __name__ == "__main__":
    # 配置你的路径
    BASE_MODEL = "deepseek-ai/deepseek-math-7b-base" # 替换实际路径
    SFT_LORA = "/root/autodl-tmp/output/sira_sft_output" # 替换实际路径
    
    try:
        run_grpo_training(
            model_path_base=BASE_MODEL,
            model_path_lora=SFT_LORA,
            subset_name="algebra" 
        )
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()

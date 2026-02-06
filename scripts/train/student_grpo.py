import os
import sys
import json
import torch
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    set_seed
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from trl import GRPOTrainer, GRPOConfig

# ==========================================
# 1. Logger 配置
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ==========================================
# 2. 答案提取工具 (使用您提供的函数)
# ==========================================

def extract_boxed_content(text: str) -> Optional[str]:
    """
    提取 LaTeX 字符串中最后一个 \\boxed{...} 里的内容。
    支持嵌套括号，例如 \\boxed{\\frac{1}{2}}。
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

def normalize_math_str(s: str) -> str:
    """简单归一化：去除空白字符"""
    if not s: return ""
    return s.replace(" ", "").replace("\n", "").strip()

# ==========================================
# 3. 结果导向奖励函数 (Result-Oriented Reward)
# ==========================================

def correctness_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    """
    GRPO 核心奖励函数。
    Logic: Extract(Pred) == Extract(Gold) -> 1.0, else 0.0
    """
    rewards = []
    for completion, gold_ans in zip(completions, answer):
        # 1. 提取模型生成的答案
        pred_val = extract_boxed_content(completion)
        
        # 2. 处理标准答案
        # 如果标准答案本身包含 \boxed，也提取一下；否则直接视为纯文本
        gold_str = str(gold_ans)
        if "\\boxed{" in gold_str:
            gold_val = extract_boxed_content(gold_str)
        else:
            gold_val = gold_str
            
        # 3. 对比 (归一化后)
        if pred_val and gold_val:
            if normalize_math_str(pred_val) == normalize_math_str(gold_val):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            # 提取失败或为空
            rewards.append(0.0)
            
    return rewards

# ==========================================
# 4. GRPO 训练主程序
# ==========================================

def run_result_oriented_grpo(
    base_model_path: str,
    sft_lora_path: str,
    questions: List[str],
    answers: List[str],
    output_base_dir: str = "/root/autodl-tmp/output",
    num_generations: int = 8,  # Group Size (G)
    max_steps: int = 500,      # 训练步数
    learning_rate: float = 1e-6
):
    """
    结果导向 GRPO 训练入口
    """
    # --- 0. 基础设置 ---
    set_seed(42)
    timestamp = datetime.now().strftime('%m%d_%H%M')
    output_dir = os.path.join(output_base_dir, f"grpo_result_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    file_handler = logging.FileHandler(os.path.join(output_dir, "train.log"), encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting GRPO (Result-Oriented)")
    logger.info(f"Base Model: {base_model_path}")
    logger.info(f"SFT Path:   {sft_lora_path}")
    logger.info(f"Output Dir: {output_dir}")
    logger.info(f"Dataset Size: {len(questions)}")
    logger.info(f"Group Size: {num_generations}")

    # --- 1. 数据集构建 ---
    logger.info("Preparing Dataset and Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise e
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    formatted_data = []
    for q, a in zip(questions, answers):
        # 使用 apply_chat_template 保证与 SFT/Inference 一致
        # 这里只生成 User 输入部分
        messages = [{"role": "user", "content": str(q)}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        formatted_data.append({
            "prompt": prompt_text,
            "answer": str(a) # 传递给 reward function
        })
    
    dataset = Dataset.from_list(formatted_data)

    # --- 2. 模型准备 (Base + SFT -> Merge -> New LoRA) ---
    logger.info("Loading Base Model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if os.path.exists(sft_lora_path):
        logger.info(f"Merging SFT Adapter from {sft_lora_path}...")
        model = PeftModel.from_pretrained(model, sft_lora_path)
        model = model.merge_and_unload() # 关键：合并权重
    else:
        logger.warning(f"SFT path {sft_lora_path} not found! Training on Base Model only.")

    model.gradient_checkpointing_enable() # 节省显存

    # GRPO 训练的新 LoRA 配置
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    # --- 3. GRPO 参数配置 ---
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=1,     # 显存敏感
        gradient_accumulation_steps=8,     # 累计梯度
        num_generations=num_generations,   # G: 每个Prompt采样多少个
        max_prompt_length=1024,
        max_completion_length=1024,
        max_steps=max_steps,
        save_steps=100,
        save_total_limit=1,
        report_to="none",
        use_vllm=False, # 设为True需环境支持，设为False更稳定
        beta=0.04,      # KL惩罚系数
    )

    # --- 4. 初始化 Trainer ---
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=correctness_reward_func, # 核心：使用自定义的提取+对比函数
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    # --- 5. 执行训练 ---
    logger.info("Starting Training Loop...")
    trainer.train()
    
    # --- 6. 保存 ---
    logger.info(f"Saving final model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("GRPO Training Finished.")

# ==========================================
# 5. 调用示例
# ==========================================
if __name__ == "__main__":
    # 模拟数据加载 (请替换为您实际的数据加载逻辑)
    # from data_math import Math_500 ...
    
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path)) 
    project_root = os.path.dirname(os.path.dirname(project_root))

    # 配置路径
    BASE_MODEL = os.path.join(project_root, "CELPO", "model", "OREAL", "OREAL-7B")
    SFT_LORA = os.path.join(project_root, "CELPO", "output", "hint_sft_XXXX_XXXX") 
    
    # 示例数据
    sample_questions = [
        "What is 2+2?", 
        "Calculate \\int x dx."
    ]
    sample_answers = [
        "\\boxed{4}", 
        "\\boxed{\\frac{x^2}{2} + C}"
    ]

    # 运行
    try:
        run_result_oriented_grpo(
            base_model_path=BASE_MODEL,
            sft_lora_path=SFT_LORA,
            questions=sample_questions, # 替换为您的 questions list
            answers=sample_answers,     # 替换为您的 answers list
            num_generations=8,
            max_steps=200
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise e

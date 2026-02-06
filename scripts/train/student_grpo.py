import os
import re
import torch
import logging
from typing import Optional, List, Dict
from dataclasses import dataclass, field

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer

# 设置并行方式，防止 vLLM 和 PyTorch 冲突
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# =====================================================
# Logger Setup
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =====================================================
# 1. 核心工具函数：提取与奖励计算
# =====================================================

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

def result_oriented_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    """
    结果导向奖励函数。
    :param completions: 模型生成的完整文本列表
    :param answer: 参考答案列表 (Ground Truth)
    :return: 奖励值列表 [1.0 或 0.0]
    """
    rewards = []
    for content, ref_ans in zip(completions, answer):
        # 1. 提取模型输出的答案
        extracted_ans = extract_boxed_content(content)
        
        # 2. 对比参考答案 (这里假设 ref_ans 已经是纯净的答案字符串，或者也包含在 boxed 中)
        # 如果 dataset 中的 answer 是纯数字/字符串，直接对比
        # 如果 dataset 中的 answer 也是 latex 格式，建议也做一次清洗
        
        # 简单清洗：去除空格
        clean_extracted = extracted_ans.replace(" ", "")
        clean_ref = str(ref_ans).strip().replace(" ", "")
        
        # 3. 判定奖励
        if clean_extracted == clean_ref and clean_extracted != "":
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            
    return rewards

# =====================================================
# 2. GRPO 训练主流程
# =====================================================

def run_grpo_training(
    base_model_path: str,
    sft_lora_path: str,
    questions: List[str],
    answers: List[str],
    output_dir: str = "output/grpo_result",
    num_generations: int = 8  # 对应 Roll 8
):
    logger.info(f"Loading Base Model from: {base_model_path}")
    logger.info(f"Loading SFT LoRA from: {sft_lora_path}")

    # -----------------------------------------------------
    # Step 1: 准备数据
    # GRPO Trainer 需要 Dataset 格式，包含 'prompt' 和 'answer' (用于奖励计算)
    # -----------------------------------------------------
    # 构造 Prompt，这里需要跟 SFT 训练时的 Template 保持一致
    # 假设使用 Qwen 的 Chat Template
    
    # 简单的 Chat Template 构造器，实际使用建议用 tokenizer.apply_chat_template
    # 但 dataset 预处理需要纯文本，我们在 Trainer 内部处理
    data_dict = {
        "prompt": [], 
        "answer": answers
    }
    
    # 临时加载 Tokenizer 以处理 prompt 格式
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Formatting prompts...")
    for q in questions:
        # 构造对话格式，GRPO 会基于此生成后续内容
        messages = [{"role": "user", "content": str(q)}]
        # add_generation_prompt=True 会添加 <|im_start|>assistant\n
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        data_dict["prompt"].append(prompt_text)

    dataset = Dataset.from_dict(data_dict)
    logger.info(f"Dataset prepared: {len(dataset)} samples.")

    # -----------------------------------------------------
    # Step 2: 模型加载与合并 (Merge SFT Adapter)
    # -----------------------------------------------------
    logger.info("Loading model and merging SFT weights...")
    # 加载 Base Model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" # A100 必备
    )

    # 加载并合并 SFT LoRA
    # 注意：GRPO 需要在此基础上训练新的 Policy，所以我们把 SFT 视为新的 Base
    model = PeftModel.from_pretrained(model, sft_lora_path)
    model = model.merge_and_unload()
    logger.info("SFT LoRA merged into Base Model.")

    # -----------------------------------------------------
    # Step 3: 配置 GRPO Trainer
    # -----------------------------------------------------
    # 定义 GRPO 训练的新 LoRA 配置
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        bias="none",
    )

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-6,           # RL 阶段 LR 通常比 SFT 低
        per_device_train_batch_size=1, # 显存优化
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        max_steps=500,                # 演示用，实际根据数据集调整
        logging_steps=10,
        save_steps=100,
        
        # GRPO 核心参数
        num_generations=num_generations, # Roll 8
        max_completion_length=2048,      # 最大生成长度
        beta=0.04,                       # KL 散度惩罚系数
        
        # vLLM 加速生成配置 (8x A100 强力推荐开启)
        use_vllm=True,
        vllm_device="cuda:0",            # vLLM 主设备，TRL 会自动处理多卡
        vllm_gpu_memory_utilization=0.3, # 留给 vLLM 的显存比例，训练需要占用大量显存，这里不能太高
        
        # 混合精度
        bf16=True,
    )

    # -----------------------------------------------------
    # Step 4: 启动训练
    # -----------------------------------------------------
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer, # 传入 tokenizer
        reward_funcs=[result_oriented_reward_func], # 你的结果导向奖励函数
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    logger.info("Starting GRPO Training...")
    trainer.train()
    
    # 保存最终模型
    logger.info(f"Saving final model to {output_dir}")
    trainer.save_model(output_dir)
    # 保存 tokenizer
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    # =====================================================
    # 模拟数据输入 (替换为你真实的加载逻辑)
    # =====================================================
    # 假设这是从 Data_Math_500 加载的数据
    # question = [...]
    # answer = [...]
    
    # 示例数据
    dummy_questions = [
        "Calculate 1 + 1 and put the answer in a box.",
        "Solve for x: 2x = 10."
    ]
    dummy_answers = [
        "2",
        "5"
    ]
    
    # 路径配置
    BASE_MODEL_URL = "/root/autodl-tmp/model/Qwen/Qwen2.5-Math-7B-Instruct"
    # 这里假设 SFT LoRA 路径存在，实际使用请修改
    SFT_LORA_URL = "/root/project/CELPO/output/hint_sft_XXXX_XXXX"
    
    # 为了演示代码能跑，做个简单的路径检查
    if not os.path.exists(BASE_MODEL_URL):
        logger.warning("Base model path not found, using a dummy path for structure demo.")
        
    try:
        # 这里传入你的真实数据集合
        run_grpo_training(
            base_model_path=BASE_MODEL_URL,
            sft_lora_path=SFT_LORA_URL,
            questions=dummy_questions, # 你的 question 集合
            answers=dummy_answers,     # 你的 answer 集合
            num_generations=8          # Roll 8
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")

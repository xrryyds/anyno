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

# 即使禁用了 vLLM，保留此行无害
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =====================================================
# 1. 工具函数
# =====================================================

def extract_boxed_content(text: str) -> Optional[str]:
    if not text: return ""
    idx = text.rfind("\\boxed{")
    if idx == -1: return ""
    i = idx + 7 
    content_start = i
    brace_balance = 0 
    while i < len(text):
        char = text[i]
        if char == '{': brace_balance += 1
        elif char == '}':
            if brace_balance == 0: return text[content_start:i].strip()
            else: brace_balance -= 1
        i += 1
    return ""

def result_oriented_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    rewards = []
    for content, ref_ans in zip(completions, answer):
        extracted_ans = extract_boxed_content(content)
        clean_extracted = extracted_ans.replace(" ", "") if extracted_ans else ""
        clean_ref = str(ref_ans).strip().replace(" ", "")
        rewards.append(1.0 if clean_extracted == clean_ref and clean_extracted != "" else 0.0)
    return rewards

# =====================================================
# 2. GRPO 训练主流程 (Max Performance)
# =====================================================

def run_grpo_training(
    base_model_path: str,
    sft_lora_path: str,
    questions: List[str],
    answers: List[str],
    num_generations: int = 8 
):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if local_rank == 0:
        logger.info(f"Loading Base Model from: {base_model_path}")
        logger.info(f"Loading SFT LoRA from: {sft_lora_path}")

    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path))
    project_root = os.path.dirname(os.path.dirname(project_root))
    output_dir = os.path.join(project_root, "CELPO", "output", "grpo_max")
    
    # --- 1. 准备数据 ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}"); return

    if local_rank == 0: logger.info("Formatting prompts...")
    
    data_dict = {"prompt": [], "answer": answers}
    for q in questions:
        messages = [{"role": "user", "content": str(q)}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        data_dict["prompt"].append(prompt_text)

    dataset = Dataset.from_dict(data_dict)

    # --- 2. 加载模型 ---
    if local_rank == 0: logger.info("Loading model...")
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        # device_map="auto", # DDP 必须禁用
        trust_remote_code=True,
        attn_implementation="sdpa" 
    )

    try:
        model = PeftModel.from_pretrained(model, sft_lora_path)
        model = model.merge_and_unload()
        if local_rank == 0: logger.info("SFT LoRA merged.")
    except Exception as e:
        logger.warning(f"Failed to merge SFT LoRA: {e}")

    # --- 3. 配置训练参数 (极致优化) ---
    peft_config = LoraConfig(
        r=64, lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM", lora_dropout=0.05, bias="none",
    )

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-6,
        
        # === 【极致优化核心】 ===
        # 1. 压榨显存：从 8 提到 16，利用 A800 巨大的显存优势
        per_device_train_batch_size=16, 
        
        # 2. 减少累积：因为 Batch 翻倍了，累积步数减半，减少开销
        gradient_accumulation_steps=2, 
        
        # 3. 开启编译：PyTorch 2.0 图编译，加速计算 (第一步会慢，后面起飞)
        torch_compile=True, 
        
        # 4. 数据加载：拉满 CPU
        dataloader_num_workers=8,
        
        # 其他保持不变
        num_generations=num_generations, 
        tf32=True,                  
        bf16=True,                  
        use_vllm=False,
        max_completion_length=1024,
        num_train_epochs=1,
        logging_steps=5,
        save_steps=100,
        beta=0.04,
        report_to="none"
    )

    # --- 4. 启动训练 ---
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[result_oriented_reward_func],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    if local_rank == 0: logger.info("🚀 Starting ULTIMATE GRPO Training...")
    
    trainer.train()
    
    if local_rank == 0:
        logger.info(f"Saving final model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

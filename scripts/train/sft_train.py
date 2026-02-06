import os
import json
import logging
import torch
from typing import List
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    set_seed
)
from peft import LoraConfig, TaskType
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

# =====================================================
# 1. 环境与多进程设置
# =====================================================
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn" 

# =====================================================
# 2. Logger 设置
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def run_sft_training(
    model_url: str, 
    question_list: List[str], 
    answer_list: List[str], 
    output_dir: str = None,
    num_train_epochs: int = 2,
    learning_rate: float = 2e-4
):
    """
    SFT 训练主函数 (修复 Formatting 逻辑与参数兼容性)
    """
    set_seed(42)
    
    if output_dir is None:
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file_path)) 
        project_root = os.path.dirname(project_root) 
        output_dir = os.path.join(project_root, "output", "sft_lora_checkpoints")
    
    logger.info(f"Starting SFT Training...")
    logger.info(f"Model Path: {model_url}")
    logger.info(f"Data Size: {len(question_list)} pairs")
    logger.info(f"Output Dir: {output_dir}")

    # ================== 3. 加载 Tokenizer ==================
    logger.info(f"Loading tokenizer from {model_url}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_url,
        trust_remote_code=True,
        use_fast=False, 
    )
    tokenizer.padding_side = "right" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ================== 4. 数据处理 ==================
    logger.info("Processing dataset...")
    
    data_entries = []
    for q, a in zip(question_list, answer_list):
        conversation = [
            {"role": "user", "content": str(q)},
            {"role": "assistant", "content": str(a)}
        ]
        data_entries.append({"messages": conversation})
    
    dataset = Dataset.from_list(data_entries)

    # ⭐⭐⭐ 核心修复：Formatting 函数 ⭐⭐⭐
    def formatting_prompts_func(example):
        # TRICK: TRL 在处理 dataset 时，有时传入单个样本(Dict)，有时传入Batch(List)。
        # 根据之前的报错，这里的 example 是单个样本的 Dict。
        # example['messages'] 已经是我们要的 List[Dict] 结构了。
        
        conversation = example['messages']
        
        # 直接转换整个对话列表
        text = tokenizer.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=False 
        )
        
        # SFTTrainer 要求返回 list[str]
        return [text]

    # ================== 5. 加载模型 ==================
    logger.info(f"Loading base model from {model_url}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_url,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    model.gradient_checkpointing_enable()

    # ================== 6. LoRA 配置 ==================
    logger.info("Configuring LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,                   
        lora_alpha=128,         
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        bias="none",
    )

    # ================== 7. 训练参数 ==================
    logger.info("Configuring Training Arguments...")
    
    # 兼容性写法：不在 init 中传 max_seq_length
    training_args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        per_device_train_batch_size=4,  
        gradient_accumulation_steps=4,  
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
        bf16=True,                      
        optim="adamw_torch",
        report_to="none",               
        run_name="sft_run",
        remove_unused_columns=True,     
        gradient_checkpointing=True,
    )
    
    # 手动赋值 max_seq_length
    training_args.max_seq_length = 4096

    # ================== 8. 初始化 Trainer ==================
    logger.info("Initializing SFTTrainer...")
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer, 
        formatting_func=formatting_prompts_func, 
    )

    # ================== 9. 开始训练 ==================
    logger.info("Starting training execution...")
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise e

    # ================== 10. 保存结果 ==================
    final_output_path = os.path.join(output_dir, "final_adapter")
    logger.info(f"Saving final LoRA adapter to {final_output_path}")
    
    trainer.save_model(final_output_path)
    tokenizer.save_pretrained(final_output_path)
    
    logger.info("SFT Training completed successfully.")
    return final_output_path

if __name__ == "__main__":
    pass

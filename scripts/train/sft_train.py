import os
import json
import logging
import torch
from typing import List
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    set_seed
)
from peft import LoraConfig, TaskType
from datasets import Dataset
from trl import SFTTrainer

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
    num_train_epochs: int = 1,
    learning_rate: float = 2e-4
):
    """
    SFT 训练主函数 (优化版)
    """
    # 设置随机种子，保证可复现性
    set_seed(42)
    
    if output_dir is None:
        # 自动推导输出路径
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file_path))
        project_root = os.path.dirname(os.path.dirname(project_root))
        output_dir = os.path.join(project_root,"CELPO", "output", "sft_lora_checkpoints")
    
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
    # Qwen/Llama 等模型通常 padding 在右侧，但某些旧版本 SFT 需要检查
    # 这里保持默认即可，SFTTrainer 会自动处理
    tokenizer.padding_side = "right" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ================== 4. 数据处理 (保持与推理一致的 Chat 格式) ==================
    logger.info("Processing dataset...")
    
    data_entries = []
    for q, a in zip(question_list, answer_list):
        # 构造标准的对话列表
        conversation = [
            {"role": "user", "content": str(q)},
            {"role": "assistant", "content": str(a)}
        ]
        data_entries.append({"messages": conversation})
    
    dataset = Dataset.from_list(data_entries)

    # 格式化函数：将 list[dict] 转为训练所需的文本字符串
    def formatting_prompts_func(examples):
        output_texts = []
        for conversation in examples['messages']:
            # apply_chat_template 会自动处理 user/assistant 标签和特殊 token
            # 这与推理时的 add_generation_prompt=True 是完美对应的
            text = tokenizer.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=False 
            )
            output_texts.append(text)
        return output_texts

    # ================== 5. 加载模型 ==================
    logger.info(f"Loading base model from {model_url}")
    
    # 既然你有 A800，这里保留 bf16
    model = AutoModelForCausalLM.from_pretrained(
        model_url,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    # 开启梯度检查点 (Gradient Checkpointing)，大幅节省显存
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
    training_args = TrainingArguments(
        output_dir=output_dir,
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
        # ⭐ [关键修复] 必须为 True，确保过滤掉原始 messages 字典，避免 DataCollator 报错
        remove_unused_columns=True     
    )

    # ================== 8. 初始化 Trainer ==================
    logger.info("Initializing SFTTrainer...")
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        # ⭐ [优化点] 提升到 4096，与你推理代码的 MAX_MODEL_LEN 一致
        max_seq_length=4096, 
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

# =====================================================
# Main Execution (测试入口)
# =====================================================
if __name__ == "__main__":
    # 模拟数据 (实际运行时这里会被你的业务逻辑替换)
    mock_questions = [
        "1+1等于几？", 
        "求解方程 2x + 4 = 10"
    ]
    mock_answers = [
        "1+1等于2。",
        "移项得 2x=6，解得 x=3。"
    ]
    
    # 请修改为你的实际模型路径
    MODEL_PATH = "/root/autodl-tmp/CELPO/model/OREAL/OREAL-7B"
    
    if os.path.exists(MODEL_PATH) or MODEL_PATH.startswith("Qwen/"):
        try:
            saved_adapter_path = run_sft_training(
                model_url=MODEL_PATH,
                question_list=mock_questions,
                answer_list=mock_answers,
                num_train_epochs=1
            )
            print(f"\n✅ 训练完成，Adapter保存在: {saved_adapter_path}")
        except Exception as e:
            logger.error(f"❌ Execution Failed: {e}")
            # 打印完整的错误堆栈以便调试
            import traceback
            traceback.print_exc()
    else:
        logger.error(f"❌ 模型路径 {MODEL_PATH} 不存在，请修改代码中的 MODEL_PATH 变量。")

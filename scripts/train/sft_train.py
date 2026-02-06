import os
# 设置多进程启动方式，保持和你代码一致
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn" 

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
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# =====================================================
# Logger (保持一致)
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
    num_train_epochs: int = 3,
    learning_rate: float = 2e-4
):
    """
    封装的 SFT 方法
    :param model_url: 模型路径 (如 /root/autodl-tmp/...)
    :param question_list: 问题列表
    :param answer_list: 答案列表 (与问题一一对应)
    :param output_dir: LoRA 权重保存路径，若为 None 则自动生成
    :param num_train_epochs: 训练轮数
    :param learning_rate: 学习率
    """
    
    # ================== Config & Seed ==================
    set_seed(42)
    
    if output_dir is None:
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file_path))
        output_dir = os.path.join(project_root, "CELPO", "output", "sft_lora_checkpoints")
    
    logger.info(f"Starting SFT Training...")
    logger.info(f"Model Path: {model_url}")
    logger.info(f"Data Size: {len(question_list)} pairs")
    logger.info(f"Output Dir: {output_dir}")

    # ================== 1. Data Processing ==================
    logger.info("Processing dataset...")
    
    # 构造符合 Chat 模板的数据集
    # 格式: {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}
    data_entries = []
    for q, a in zip(question_list, answer_list):
        conversation = [
            {"role": "user", "content": str(q)},
            {"role": "assistant", "content": str(a)}
        ]
        data_entries.append({"messages": conversation})
    
    dataset = Dataset.from_list(data_entries)

    # ================== 2. Load Tokenizer ==================
    logger.info(f"Loading tokenizer from {model_url}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_url,
        trust_remote_code=True,
        use_fast=False, # 保持和你推理代码一致
    )
    # Qwen 等模型通常需要设置 padding_side 和 pad_token
    tokenizer.padding_side = "right" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ================== 3. Load Model ==================
    logger.info(f"Loading base model from {model_url}")
    # 注意：训练时我们通常使用 transformers 原生加载，而不是 vLLM
    # 为了显存优化，默认使用 bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_url,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # 如果显存不够，可以解开下面这行的注释使用 4bit 量化加载 (需要 bitsandbytes)
        # quantization_config=BitsAndBytesConfig(load_in_4bit=True) 
    )
    
    # 开启梯度检查点以节省显存
    model.gradient_checkpointing_enable()

    # ================== 4. LoRA Config ==================
    logger.info("Configuring LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,                   # Rank，和你推理代码中的 max_lora_rank 对应
        lora_alpha=128,         # 通常是 rank 的 2 倍
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Qwen 全参覆盖
        bias="none",
    )

    # ================== 5. Training Arguments ==================
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,  # 根据显存调整
        gradient_accumulation_steps=4,  # 显存不够时调大这个
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
        bf16=True,                      # 推荐使用 bf16
        optim="adamw_torch",
        report_to="none",               # 不上传 wandb
        run_name="sft_run",
        remove_unused_columns=False     # 防止 dataset 中的 messages 列被删除
    )

    # ================== 6. SFT Trainer ==================
    logger.info("Initializing SFTTrainer...")
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        # TRL 库会自动处理 'messages' 格式的数据并应用 chat_template
        # max_seq_length 需要和你推理时的 MAX_MODEL_LEN 匹配或略小
        max_seq_length=2048, 
        dataset_text_field="messages", 
    )

    # ================== 7. Start Training ==================
    logger.info("Starting training execution...")
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise e

    # ================== 8. Save Adapter ==================
    final_output_path = os.path.join(output_dir, "final_adapter")
    logger.info(f"Saving final LoRA adapter to {final_output_path}")
    
    trainer.save_model(final_output_path)
    tokenizer.save_pretrained(final_output_path)
    
    logger.info("SFT Training completed successfully.")
    return final_output_path

# =====================================================
# 调用示例 (放在 __main__ 块中)
# =====================================================
if __name__ == "__main__":
    # 模拟数据
    mock_questions = [
        "1+1等于几？", 
        "求解方程 2x + 4 = 10"
    ]
    mock_answers = [
        "1+1等于2。",
        "移项得 2x=6，解得 x=3。"
    ]
    
    # 假设的模型路径
    MODEL_PATH = "/root/autodl-tmp/model/Qwen/Qwen2.5-Math-7B-Instruct"
    
    # 确保路径存在，否则 demo 无法运行
    if os.path.exists(MODEL_PATH):
        saved_adapter_path = run_sft_training(
            model_url=MODEL_PATH,
            question_list=mock_questions,
            answer_list=mock_answers,
            num_train_epochs=1 # 演示用1个epoch
        )
        print(f"训练完成，Adapter保存在: {saved_adapter_path}")
    else:
        logger.warning(f"模型路径 {MODEL_PATH} 不存在，请修改后运行。")

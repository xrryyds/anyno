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

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MAX_SEQ_LENGTH = 1024

# =====================================================
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
    output_dir = os.path.join(project_root, "CELPO", "output", "grpo_stable")
    
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

    if local_rank == 0: logger.info("Loading model...")
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa" 
    )

    try:
        model = PeftModel.from_pretrained(model, sft_lora_path)
        model = model.merge_and_unload()
        if local_rank == 0: logger.info("SFT LoRA merged.")
    except Exception as e:
        logger.warning(f"Failed to merge SFT LoRA: {e}")

    peft_config = LoraConfig(
        r=64, lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM", lora_dropout=0.05, bias="none",
    )

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-6,
        
        per_device_train_batch_size=8, 
        gradient_accumulation_steps=4, 
        
        torch_compile=False, 
        
        ddp_find_unused_parameters=False,
        
        num_generations=num_generations, 
        tf32=True,                  
        bf16=True,                  
        dataloader_num_workers=4,
        use_vllm=False,
        max_completion_length=MAX_SEQ_LENGTH,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=100,
        beta=0.04,
        report_to="none"
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[result_oriented_reward_func],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    if local_rank == 0: logger.info("🚀 Starting STABLE GRPO Training...")
    
    trainer.train()
    
    if local_rank == 0:
        logger.info(f"Saving final model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

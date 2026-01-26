import os
import sys
import json
import re
import torch
import torch.nn.functional as F
import logging
from dataclasses import dataclass
from typing import List
from tqdm import tqdm  # <--- 新增引入

from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    set_seed,
    get_scheduler
)
from peft import PeftModel

# ==========================================
# 0. 基础配置
# ==========================================

try:
    from prompt import GEN_PROMPT
except ImportError:
    GEN_PROMPT = "Question: {question}\nAnswer:"

@dataclass
class GRPOConfig:
    # 路径配置
    base_model_path: str = "/root/autodl-tmp/model/Qwen/Qwen/Qwen2.5-Math-7B-Instruct"
    sft_adapter_path: str = "/root/autodl-tmp/output/hint_sft_0126_0945" 
    data_path: str = "/xrr/CELPO/datasets/exam/irdcl_data.json"
    output_dir: str = "/root/autodl-tmp/output/hint_grpo"
    
    # GRPO 参数
    group_size: int = 8          
    num_train_epochs: int = 1
    learning_rate: float = 1e-6  
    beta: float = 0.04           
    
    # 生成配置
    max_gen_length: int = 4096   
    gradient_accumulation_steps: int = 4
    temperature: float = 0.9     
    top_p: float = 0.95

logger = logging.getLogger(__name__)

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(os.path.join(output_dir, "grpo.log"))]
    )

# ==========================================
# 1. 数据集与工具
# ==========================================

class MathProblemDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
            for item in raw_data:
                if item.get('ref_answer'):
                    self.data.append(item)
        logger.info(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def extract_answer(text):
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if matches:
        return matches[-1].strip()
    parts = text.split("Answer:")
    if len(parts) > 1:
        ans_part = parts[-1].strip()
        match = re.search(r"(\-?\d+\.?\d*)", ans_part) 
        if match:
            return match.group(1)
    return ""

def check_correctness(generated_text, ref_answer):
    pred = extract_answer(generated_text)
    ref = str(ref_answer).strip()
    if not pred: return 0.0
    if pred == ref: return 1.0
    if ref in pred: return 1.0
    return 0.0

# ==========================================
# 2. GRPOTrainer
# ==========================================

class GRPOTrainer:
    def __init__(self, config: GRPOConfig, model, ref_model, tokenizer, dataset):
        self.config = config
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        
        num_steps = len(self.dataloader) * config.num_train_epochs // config.gradient_accumulation_steps
        self.scheduler = get_scheduler("cosine", optimizer=self.optimizer, num_warmup_steps=10, num_training_steps=num_steps)

    def get_per_token_logps(self, model, input_ids, attention_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        per_token_logps = -F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction='none'
        )
        return per_token_logps.view(labels.shape)

    def train(self):
        logger.info("Starting GRPO Training...")
        
        global_step = 0
        total_loss = 0
        
        # 计算总步数用于进度条
        total_steps = len(self.dataloader) * self.config.num_train_epochs
        progress_bar = tqdm(total=total_steps, desc="Training", unit="batch")
        
        for epoch in range(self.config.num_train_epochs):
            for step, batch in enumerate(self.dataloader):
                q_text = batch[0]['question']
                ref_ans = batch[0]['ref_answer']
                
                # 1. 生成 (Eval Mode)
                self.model.eval()
                
                prompt = GEN_PROMPT.format(question=q_text)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                input_ids = inputs.input_ids.repeat(self.config.group_size, 1)
                attention_mask = inputs.attention_mask.repeat(self.config.group_size, 1)
                prompt_length = input_ids.shape[1]
                
                with torch.no_grad():
                    generated_output = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.config.max_gen_length,
                        do_sample=True,
                        temperature=self.config.temperature, 
                        top_p=self.config.top_p,
                        pad_token_id=self.tokenizer.pad_token_id,
                        use_cache=True 
                    )
                
                # 2. 训练 (Train Mode)
                self.model.train()
                full_attention_mask = torch.ones_like(generated_output)
                
                # Policy LogProbs
                per_token_logps = self.get_per_token_logps(self.model, generated_output, full_attention_mask)
                
                # Ref LogProbs
                with torch.no_grad():
                    ref_logps = self.get_per_token_logps(self.ref_model, generated_output, full_attention_mask)

                # Mask
                mask = torch.zeros_like(per_token_logps)
                mask[:, prompt_length-1:] = 1.0 
                for i in range(self.config.group_size):
                    eos_idx = (generated_output[i] == self.tokenizer.eos_token_id).nonzero()
                    if len(eos_idx) > 0:
                        first_eos = eos_idx[0].item()
                        if first_eos > prompt_length:
                             mask[i, first_eos-1:] = 0 
                
                # Rewards
                decoded_texts = self.tokenizer.batch_decode(generated_output, skip_special_tokens=True)
                rewards = []
                for text in decoded_texts:
                    r = 0.2 if "# known:" in text else 0.0
                    r += check_correctness(text, ref_ans)
                    rewards.append(r)
                
                rewards_tensor = torch.tensor(rewards).to(self.model.device)
                
                # KL & Loss
                per_token_kl = per_token_logps.detach() - ref_logps
                kl_penalty = -self.config.beta * (per_token_kl * mask).sum(dim=-1)
                total_rewards = rewards_tensor + kl_penalty
                
                mean_r = total_rewards.mean()
                std_r = total_rewards.std() + 1e-8
                advantages = (total_rewards - mean_r) / std_r
                
                adv_expanded = advantages.view(-1, 1).expand_as(per_token_logps)
                loss = -(per_token_logps * adv_expanded * mask).sum() / mask.sum()
                
                # Backward
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                total_loss += loss.item()
                
                # 更新进度条
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "Loss": f"{loss.item():.4f}", 
                    "Rw": f"{rewards_tensor.mean().item():.2f}",
                    "KL": f"{-kl_penalty.mean().item():.2f}"
                })
                
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    
                    if global_step % 10 == 0:
                        sample_text = decoded_texts[0].split("Answer:")[-1].strip() if "Answer:" in decoded_texts[0] else decoded_texts[0][-50:]
                        logger.info(f"Step {global_step} | Gen: {sample_text}")
                        total_loss = 0
            
            # Save Checkpoint
            save_path = os.path.join(self.config.output_dir, f"epoch_{epoch+1}")
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"Saved checkpoint to {save_path}")
            
        progress_bar.close()

# ==========================================
# 3. Main
# ==========================================

def main():
    set_seed(42)
    config = GRPOConfig()
    setup_logging(config.output_dir)
    
    # 1. Load Tokenizer & Base Model
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_path, trust_remote_code=True)
    logger.info("Loading Base Model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    
    # 2. Load Policy Adapter
    logger.info(f"Loading SFT Adapter: {config.sft_adapter_path}")
    model = PeftModel.from_pretrained(model, config.sft_adapter_path, is_trainable=True)
    
    # ⚠️ 关键修复：开启梯度检查点支持
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.get_input_embeddings().weight.requires_grad_(True)
    model.config.use_cache = False
    
    model.print_trainable_parameters()
    
    # 3. Load Ref Model
    logger.info("Creating Reference Model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    ref_model = PeftModel.from_pretrained(ref_model, config.sft_adapter_path)
    ref_model.eval()
    ref_model.config.use_cache = False
    
    dataset = MathProblemDataset(config.data_path)
    trainer = GRPOTrainer(config, model, ref_model, tokenizer, dataset)
    trainer.train()

if __name__ == "__main__":
    main()

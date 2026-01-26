import os
import sys
import json
import torch
import torch.nn.functional as F
import logging
import re
import copy
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional

from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    set_seed,
    get_scheduler
)
from peft import PeftModel, LoraConfig, get_peft_model

# 假设 prompt 模块 (保持一致)
try:
    from prompt import GEN_PROMPT, GEN_HINTS_WIH_ANSWER
except ImportError:
    GEN_PROMPT = "Question: {question}\nAnswer:"
    GEN_HINTS_WIH_ANSWER = "# known:\n{hints}\n{answer}"

# ==========================================
# 1. 配置类
# ==========================================

@dataclass
class GRPOConfig:
    # 路径
    base_model_path: str = "/root/autodl-tmp/model/Qwen/Qwen/Qwen2.5-Math-7B-Instruct"
    sft_adapter_path: str = "/root/autodl-tmp/output/hint_sft_XXXX_XXXX" # <--- 修改为你刚刚SFT训练好的 output 目录
    data_path: str = "/xrr/CELPO/datasets/exam/irdcl_data.json"
    output_dir: str = "/root/autodl-tmp/output/hint_grpo"
    
    # GRPO 参数
    group_size: int = 4          # 每个问题采样的数量 (G)
    num_train_epochs: int = 1
    learning_rate: float = 1e-6  # RL 阶段学习率通常比 SFT 低
    beta: float = 0.04           # KL 惩罚系数
    max_gen_length: int = 512    # 生成长度
    max_prompt_length: int = 512
    batch_size: int = 1          # 实际 batch size = batch_size * group_size
    gradient_accumulation_steps: int = 4
    
    # 生成参数 (用于探索)
    temperature: float = 0.9     # 稍微高一点，鼓励探索不同的 Hint 路径
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
# 2. 数据与奖励函数
# ==========================================

class MathProblemDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.data = []
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
            # 我们只需要 Question 和 Reference Answer
            # 过滤掉那些本来就没有答案的坏数据
            for item in raw_data:
                if item.get('ref_answer'):
                    self.data.append(item)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "question": item['question'],
            "ref_answer": item['ref_answer']
        }

def extract_answer_number(text):
    """简单的答案提取逻辑，根据你的数据格式调整"""
    # 假设答案可能是 \boxed{...} 或者直接是数字
    # 这里写一个通用的提取最后数字/选项的正则
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if matches:
        return matches[-1].strip()
    
    # 备用：提取最后一个连续数字/字母
    matches = re.findall(r"Answer:\s*([A-Za-z0-9\.\-]+)", text)
    if matches:
        return matches[-1].strip()
    return ""

def compute_rewards(generated_texts: List[str], ref_answers: List[str]) -> torch.Tensor:
    rewards = []
    for gen, ref in zip(generated_texts, ref_answers):
        reward = 0.0
        
        # 1. 格式奖励 (Format Reward)
        # 检查是否包含 Hint 标记，鼓励模型坚持使用 SIRA 训练出的推理模式
        if "# known:" in gen:
            reward += 0.1
        
        # 2. 正确性奖励 (Correctness Reward)
        # 提取模型生成的答案
        # 注意：gen 包含了 prompt，我们需要截取
        pred_ans = extract_answer_number(gen)
        clean_ref = str(ref).strip()
        
        if pred_ans == clean_ref:
            reward += 1.0
        elif pred_ans and clean_ref in pred_ans: # 宽松匹配
            reward += 1.0
            
        rewards.append(reward)
    
    return torch.tensor(rewards)

# ==========================================
# 3. 自定义 GRPO Trainer
# ==========================================

class GRPOTrainer:
    def __init__(self, config: GRPOConfig, model, ref_model, tokenizer, dataset):
        self.config = config
        self.model = model
        self.ref_model = ref_model # 冻结的参考模型 (SFT后的模型)
        self.tokenizer = tokenizer
        self.dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=self.collate_fn)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = get_scheduler(
            "cosine", optimizer=self.optimizer, 
            num_warmup_steps=10, 
            num_training_steps=len(self.dataloader) * config.num_train_epochs // config.gradient_accumulation_steps
        )
    
    def collate_fn(self, batch):
        # 简单的 batch 收集
        return batch

    def get_log_probs(self, model, input_ids, attention_mask):
        """计算生成的 token 的 log probabilities"""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :] # Shift right
        labels = input_ids[:, 1:].clone()
        
        # CrossEntropy with reduction='none' returns loss per token
        # Loss = -log(p), so log(p) = -Loss
        per_token_logps = -F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), 
            labels.reshape(-1), 
            reduction='none'
        ).view(labels.size())
        
        return per_token_logps

    def train(self):
        logger.info("Starting GRPO Training...")
        self.model.train()
        self.ref_model.eval()
        
        global_step = 0
        total_loss = 0
        
        for epoch in range(self.config.num_train_epochs):
            for step, batch in enumerate(self.dataloader):
                # 目前 batch_size = 1，便于逻辑处理
                # 如果 batch_size > 1，需要更复杂的 padding 处理
                q_text = batch[0]['question']
                ref_ans = batch[0]['ref_answer']
                
                # 构建 Prompt
                prompt = GEN_PROMPT.format(question=q_text)
                prompt_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                
                # ==================================
                # 1. Rollout (生成 G 个样本)
                # ==================================
                # 复制 input_ids G 份
                rollout_input_ids = prompt_inputs.input_ids.repeat(self.config.group_size, 1)
                rollout_attention_mask = prompt_inputs.attention_mask.repeat(self.config.group_size, 1)
                
                with torch.no_grad():
                    # 使用 Policy Model 进行生成
                    # 开启 sampling 以探索不同路径
                    generated_ids = self.model.generate(
                        input_ids=rollout_input_ids,
                        attention_mask=rollout_attention_mask,
                        max_new_tokens=self.config.max_gen_length,
                        do_sample=True,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                # 解码生成的文本用于计算 Reward
                generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                # ==================================
                # 2. Compute Rewards & Advantages
                # ==================================
                rewards = compute_rewards(generated_texts, [ref_ans] * self.config.group_size).to(self.model.device)
                
                # 计算组内 Advantage: (r - mean) / (std + eps)
                mean_reward = rewards.mean()
                std_reward = rewards.std() + 1e-8
                advantages = (rewards - mean_reward) / std_reward
                
                # ==================================
                # 3. Forward Pass for Log Probs
                # ==================================
                # 我们需要在生成的完整序列上计算 Log Probs
                # 创建 Mask，只计算 Answer 部分的 Loss，忽略 Prompt 部分
                prompt_length = rollout_input_ids.shape[1]
                full_seq_length = generated_ids.shape[1]
                
                # 计算 Policy Model 的 Log Probs
                policy_log_probs = self.get_log_probs(self.model, generated_ids, torch.ones_like(generated_ids))
                
                # 计算 Reference Model 的 Log Probs (无需梯度)
                with torch.no_grad():
                    ref_log_probs = self.get_log_probs(self.ref_model, generated_ids, torch.ones_like(generated_ids))
                
                # ==================================
                # 4. Compute GRPO Loss
                # ==================================
                # Loss = E [ - (policy_logp / ref_logp) * A + beta * KL ]
                # 简化版 GRPO Loss (Approximation): 
                # L = - (policy_logp * A) + beta * KL(policy || ref)
                # KL = policy_logp - ref_logp
                
                # Mask: 只关心生成的 token
                mask = torch.zeros_like(policy_log_probs)
                mask[:, prompt_length-1:] = 1.0 
                
                # Per token KL
                kl_div = policy_log_probs - ref_log_probs
                
                # Per token Loss
                # 注意：advantages 是 per sample 的，需要广播到 per token
                # advantages shape: [G], policy_log_probs shape: [G, seq_len]
                adv_expanded = advantages.view(-1, 1).expand_as(policy_log_probs)
                
                # 核心 GRPO Loss 公式
                # 我们最大化 (Advantage * log_p)，即最小化 -(Advantage * log_p)
                # 同时最小化 KL
                pg_loss = -(policy_log_probs * adv_expanded) 
                kl_loss = self.config.beta * kl_div
                
                loss_tensor = (pg_loss + kl_loss) * mask
                loss = loss_tensor.sum() / mask.sum() # Average over valid tokens
                
                # ==================================
                # 5. Backward & Update
                # ==================================
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                total_loss += loss.item()
                
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    
                    if global_step % 10 == 0:
                        logger.info(f"Step {global_step} | Loss: {total_loss:.4f} | Avg Reward: {mean_reward:.4f}")
                        total_loss = 0

            # 每个 Epoch 保存一次
            save_path = os.path.join(self.config.output_dir, f"checkpoint-epoch-{epoch+1}")
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"Saved checkpoint to {save_path}")

# ==========================================
# 4. 主函数
# ==========================================

def main():
    set_seed(42)
    config = GRPOConfig()
    setup_logging(config.output_dir)
    
    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. 加载基础模型
    logger.info("Loading Base Model...")
    # 注意：这里加载的是 Base 模型，后面会挂载 SFT 的 Adapter
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_path, 
        torch_dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True
    )
    
    # 3. 加载 SFT Adapter (Policy Model)
    logger.info(f"Loading SFT Adapter from {config.sft_adapter_path}...")
    model = PeftModel.from_pretrained(model, config.sft_adapter_path, is_trainable=True)
    model.print_trainable_parameters() # 确保 LoRA 是可训练的
    
    # 4. 创建 Reference Model (冻结)
    # Ref Model = Base + SFT Adapter (Fixed)
    logger.info("Creating Reference Model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_path, 
        torch_dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True
    )
    ref_model = PeftModel.from_pretrained(ref_model, config.sft_adapter_path)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
        
    # 5. 加载数据
    dataset = MathProblemDataset(config.data_path, tokenizer)
    logger.info(f"Dataset Loaded: {len(dataset)} examples")
    
    # 6. 开始训练
    trainer = GRPOTrainer(config, model, ref_model, tokenizer, dataset)
    trainer.train()

if __name__ == "__main__":
    main()

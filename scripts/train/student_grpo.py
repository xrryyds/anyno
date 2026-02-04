import os
import sys
import json
import re
import torch
import torch.nn.functional as F
import logging
import shutil
import argparse
import gc
from dataclasses import dataclass
from typing import List, Optional
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    set_seed,
    get_scheduler,
    BitsAndBytesConfig
)
from peft import PeftModel

# vLLM Imports
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError:
    raise ImportError("请先安装 vllm: pip install vllm")

# ==========================================
# 0. 基础配置与Prompt
# ==========================================

# 假设 prompt 模块
try:
    from prompt import GEN_PROMPT
except ImportError:
    GEN_PROMPT = "Question: {question}\nAnswer:"

logger = logging.getLogger(__name__)

@dataclass
class GRPOConfig:
    base_model_path: str
    sft_adapter_path: str
    data_path: str
    output_dir: str
    
    # GRPO 参数
    group_size: int = 8         # 每条Prompt生成的样本数 (G)
    num_train_epochs: int = 1
    learning_rate: float = 1e-6 # GRPO 学习率通常比 SFT 低
    beta: float = 0.04          # KL 惩罚系数
    
    # 生成配置
    max_gen_length: int = 1024  
    temperature: float = 0.9     
    top_p: float = 0.95
    
    # 训练配置
    gradient_accumulation_steps: int = 4
    # [关键显存参数] 
    # vLLM 显存占用比例。RefModel(4bit)约5G, PolicyModel(16bit)约15G。
    # 给 vLLM 留 0.3 (约7G) 用于 KV Cache 和推理权重是比较安全的平衡点。
    vllm_gpu_memory_utilization: float = 0.3 
    vllm_max_model_len: int = 4096

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout), 
            logging.FileHandler(os.path.join(output_dir, "grpo.log"), encoding='utf-8')
        ]
    )

# ==========================================
# 1. 数据处理工具
# ==========================================

class MathProblemDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            for item in raw_data:
                # 过滤掉没有参考答案的数据
                if item.get('ref_answer') or item.get('solution'):
                    self.data.append(item)
        logger.info(f"Loaded {len(self.data)} samples from {data_path}.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def extract_answer(text):
    """
    增强版答案提取：
    1. 优先提取 \boxed{...} 中的内容
    2. 其次尝试提取 Answer: 后的数字
    """
    if not text: return ""
    
    # 策略 1: LaTeX boxed
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if matches:
        return matches[-1].strip()
    
    # 策略 2: 最后的数字
    parts = text.split("Answer:")
    if len(parts) > 1:
        ans_part = parts[-1].strip()
        # 匹配整数、小数、负数
        match = re.search(r"(\-?\d+\.?\d*)", ans_part) 
        if match:
            return match.group(1)
            
    return ""

def compute_rewards(generated_texts, ref_answer):
    rewards = []
    ref_str = str(ref_answer).strip()
    
    for text in generated_texts:
        pred_ans = extract_answer(text)
        score = 0.0
        
        # 1. 格式奖励 (Format Reward)
        if "\\boxed{" in text or "Answer:" in text:
            score += 0.1
            
        # 2. 正确性奖励 (Correctness Reward)
        # 简单字符串匹配，实际工程中可能需要 sympy 做数学等价性判断
        if pred_ans == ref_str:
            score += 1.0
        elif ref_str in text and len(ref_str) > 2: # 宽松匹配
            score += 0.5
            
        rewards.append(score)
    return rewards

# ==========================================
# 2. VLLM GRPO Trainer
# ==========================================

class VLLMGRPOTrainer:
    def __init__(self, config: GRPOConfig, model, ref_model, tokenizer, dataset):
        self.config = config
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        
        # DataLoader (BS=1, shuffle=True)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)
        
        # Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        num_steps = len(self.dataloader) * config.num_train_epochs // config.gradient_accumulation_steps
        self.scheduler = get_scheduler("cosine", optimizer=self.optimizer, num_warmup_steps=min(20, num_steps//10), num_training_steps=num_steps)

        # 初始化 vLLM
        logger.info(f"Initializing vLLM (GPU Util: {config.vllm_gpu_memory_utilization})...")
        self.llm = LLM(
            model=config.base_model_path,
            enable_lora=True,
            max_lora_rank=64, 
            gpu_memory_utilization=config.vllm_gpu_memory_utilization,
            max_model_len=config.vllm_max_model_len,
            trust_remote_code=True,
            tensor_parallel_size=1, 
            disable_log_stats=True 
        )
        
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_gen_length,
            n=config.group_size, 
        )
        
        # 临时目录用于交换 LoRA 权重
        self.temp_lora_path = os.path.join(config.output_dir, "temp_lora_adapter")
        os.makedirs(self.temp_lora_path, exist_ok=True)

    def get_per_token_logps(self, model, input_ids, attention_mask):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        logits = logits[:, :-1, :] 
        labels = input_ids[:, 1:]
        
        return -F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction='none'
        ).view(labels.shape)

    def train(self):
        logger.info("🚀 Starting GRPO Training...")
        global_step = 0
        
        progress_bar = tqdm(total=len(self.dataloader) * self.config.num_train_epochs)
        
        for epoch in range(self.config.num_train_epochs):
            for step, batch in enumerate(self.dataloader):
                q_text = batch[0]['question']
                ref_ans = batch[0].get('ref_answer', batch[0].get('solution')) # 兼容字段名
                
                # --- Step 1: 保存当前 Policy Adapter ---
                self.model.save_pretrained(self.temp_lora_path)
                
                # --- Step 2: vLLM 采样 (Rollout) ---
                prompt = GEN_PROMPT.format(question=q_text)
                try:
                    lora_req = LoRARequest("adapter", 1, self.temp_lora_path)
                    outputs = self.llm.generate(
                        prompts=[prompt], 
                        sampling_params=self.sampling_params,
                        lora_request=lora_req,
                        use_tqdm=False
                    )
                    generated_texts = [o.text for o in outputs[0].outputs]
                except Exception as e:
                    logger.error(f"vLLM Generation failed: {e}")
                    continue # 跳过此 Batch

                # --- Step 3: 计算 Rewards ---
                rewards_list = compute_rewards(generated_texts, ref_ans)
                rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32, device=self.model.device)
                
                # --- Step 4: 准备数据 ---
                full_texts = [prompt + gen for gen in generated_texts]
                inputs = self.tokenizer(
                    full_texts, return_tensors="pt", padding=True, truncation=True, 
                    max_length=self.config.vllm_max_model_len
                ).to(self.model.device)
                
                input_ids = inputs.input_ids
                attention_mask = inputs.attention_mask
                
                # 确定 Prompt 长度用于 Mask
                prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
                prompt_len = len(prompt_ids)
                
                # --- Step 5: 计算 Loss (PyTorch) ---
                self.model.train()
                
                # Policy Logprobs
                policy_logps = self.get_per_token_logps(self.model, input_ids, attention_mask)
                
                # Ref Logprobs (No Grad)
                with torch.no_grad():
                    ref_logps = self.get_per_token_logps(self.ref_model, input_ids, attention_mask)
                
                # Mask (只保留回答部分)
                mask = attention_mask[:, 1:].clone()
                # 简单 Mask 策略：前 prompt_len 个 token 设为 0
                safe_prompt_len = min(prompt_len, mask.size(1))
                mask[:, :safe_prompt_len] = 0
                
                # KL Divergence (Token-level)
                kl = policy_logps.detach() - ref_logps 
                
                # KL Penalty (Sample-level sum for reward calc)
                # GRPO 论文通常将 -beta * KL 加入 Reward
                kl_penalty = -self.config.beta * (kl * mask).sum(dim=-1)
                
                # Total Reward & Advantage
                total_rewards = rewards_tensor + kl_penalty
                mean_r = total_rewards.mean()
                std_r = total_rewards.std() + 1e-8
                advantages = (total_rewards - mean_r) / std_r
                
                # Final Loss: - (Advantage * Policy_Logprobs)
                # 将 Advantage 广播到 Token 维度
                adv_expanded = advantages.view(-1, 1).expand_as(policy_logps)
                loss = -(policy_logps * adv_expanded * mask).sum() / mask.sum()
                
                # --- Step 6: 优化 ---
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "Loss": f"{loss.item() * self.config.gradient_accumulation_steps:.4f}",
                    "R_Avg": f"{rewards_tensor.mean().item():.2f}",
                    "KL": f"{-kl_penalty.mean().item():.3f}"
                })
                
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
            
            # --- Epoch End ---
            save_path = os.path.join(self.config.output_dir, f"checkpoint-epoch-{epoch+1}")
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"Saved checkpoint to {save_path}")

        # 清理
        try:
            shutil.rmtree(self.temp_lora_path)
        except:
            pass
        progress_bar.close()

# ==========================================
# 3. Main
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_adapter_path", type=str, required=True, help="SFT训练结果路径")
    parser.add_argument("--base_model_path", type=str, default="/root/autodl-tmp/CELPO/model/OREAL/OREAL-7B")
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/CELPO/datasets/exam/irdcl_data.json")
    args = parser.parse_args()

    # 输出目录自动生成
    output_dir = os.path.join(os.path.dirname(args.sft_adapter_path), "grpo_vllm_output")
    
    set_seed(42)
    
    config = GRPOConfig(
        base_model_path=args.base_model_path,
        sft_adapter_path=args.sft_adapter_path,
        data_path=args.data_path,   
        output_dir=output_dir,
        group_size=8,
        gradient_accumulation_steps=4
    )
    
    setup_logging(output_dir)
    logger.info(f"Config: {config}")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. [显存优化] Load Policy Model (Trainable)
    logger.info("Loading Policy Model (BF16)...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, config.sft_adapter_path, is_trainable=True)
    
    # ================= 核心修复部分 =================
    # 必须显式开启输入梯度，否则开启 Gradient Checkpointing 后会断开梯度图
    model.enable_input_require_grads()
    
    # 建议关闭 use_reentrant 以提高兼容性
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # ==============================================
    
    model.config.use_cache = False
    
    # 3. [显存优化] Load Reference Model (4-bit Quantized)
    # Ref Model 不需要梯度，用 4bit 加载极省显存
    logger.info("Loading Reference Model (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_path, 
        quantization_config=bnb_config, # 量化加载
        device_map="auto", 
        trust_remote_code=True
    )
    # 加载 SFT 权重作为 Reference
    ref_model = PeftModel.from_pretrained(ref_model, config.sft_adapter_path)
    ref_model.eval()
    
    # 4. Dataset
    dataset = MathProblemDataset(config.data_path)
    
    # 5. Train
    trainer = VLLMGRPOTrainer(config, model, ref_model, tokenizer, dataset)
    trainer.train()

if __name__ == "__main__":
    main()

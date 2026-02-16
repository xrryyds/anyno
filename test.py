import os
import torch
from transformers import AutoTokenizer
import logging

# ================= 配置 =================
# 替换为你的本地 Qwen2.5 模型路径
MODEL_PATH = "/root/autodl-tmp/CELPO/model/OREAL/OREAL-7B" 
model_path = "/root/autodl-tmp/CELPO/model/OREAL/OREAL-7B"
# 保持与你的训练/推理代码一致的常量
SYSTEM_PROMPT = "Please reason step by step and put your final answer within \\boxed{}."
GEN_HINTS_WIH_ANSWER = "# known:\n{hints}\n{answer}"
HINT_PREFIX_TEMPLATE = "# known:\n{hint}\n"  # 推理时用的模板

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def load_tokenizer():
    try:
        logger.info(f"Loading tokenizer from {MODEL_PATH}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return None

# ================= 验证 1: Mask 定位精准度 =================
def verify_mask_alignment(tokenizer):
    logger.info("=" * 60)
    logger.info("TEST 1: Verifying Mask Alignment (Train Logic)")
    logger.info("=" * 60)

    # 模拟一条数据
    q = "Calculate 12 + 34."
    h = "First add the units digits, then the tens digits." # Hint
    a = "46" # Answer

    # --- 1. 复现 FixedModeCollator 的逻辑 ---
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q}
    ]
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_str, add_special_tokens=False).input_ids
    len_prompt = len(prompt_ids)

    # 构造完整目标文本
    target_text = GEN_HINTS_WIH_ANSWER.format(hints=h, answer=a)
    # 模拟 Answer 结尾加 EOS (训练代码逻辑)
    target_ids = tokenizer(target_text, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    
    full_ids = prompt_ids + target_ids

    # 计算 Hint 长度 (训练代码核心逻辑)
    hint_only_text = f"# known:\n{h}\n"
    hint_ids_only = tokenizer(hint_only_text, add_special_tokens=False).input_ids
    len_hint_part = len(hint_ids_only)

    hint_end_idx = min(len_prompt + len_hint_part, len(full_ids))

    # 生成 Mask
    h_mask = [0] * len(full_ids)
    a_mask = [0] * len(full_ids)

    for i in range(len_prompt, hint_end_idx):
        h_mask[i] = 1
    for i in range(hint_end_idx, len(full_ids)):
        a_mask[i] = 1

    # --- 2. 验证提取内容 ---
    # 将 list 转 tensor 方便操作
    full_tensor = torch.tensor(full_ids)
    h_mask_tensor = torch.tensor(h_mask)
    a_mask_tensor = torch.tensor(a_mask)

    # 提取被 mask 标记为 1 的 token
    extracted_hint_ids = full_tensor[h_mask_tensor == 1].tolist()
    extracted_answer_ids = full_tensor[a_mask_tensor == 1].tolist()

    decoded_hint = tokenizer.decode(extracted_hint_ids)
    decoded_answer = tokenizer.decode(extracted_answer_ids)

    # --- 3. 打印结果与断言 ---
    logger.info(f"[Original Hint]   : {hint_only_text.__repr__()}")
    logger.info(f"[Extracted Hint]  : {decoded_hint.__repr__()}")
    logger.info(f"[Original Answer] : {a.__repr__()} (+ EOS)")
    logger.info(f"[Extracted Answer]: {decoded_answer.__repr__()}")

    # 验证 Hint
    # 注意：decode 可能会产生轻微空格差异，所以这里去头尾空格比较，或者包含比较
    if hint_only_text.strip() == decoded_hint.strip():
        logger.info("✅ Hint Mask Alignment: SUCCESS")
    else:
        logger.error("❌ Hint Mask Alignment: FAILED")
        logger.error(f"Expected: {hint_only_text.__repr__()}")
        logger.error(f"Got:      {decoded_hint.__repr__()}")

    # 验证 Answer (注意 extracted 包含 EOS，decode 不会显示 EOS，所以这很安全)
    if decoded_answer.strip() == a.strip():
        logger.info("✅ Answer Mask Alignment: SUCCESS")
    else:
        logger.error("❌ Answer Mask Alignment: FAILED")


# ================= 验证 2: 训练 vs 推理 序列一致性 =================
def verify_train_inference_consistency(tokenizer):
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Verifying Train vs. Inference Consistency")
    logger.info("=" * 60)

    q = "Solve x + 5 = 10."
    h = "Subtract 5 from both sides."
    a = "5" # 假设这是推理生成出来的答案，我们需要验证序列拼接是否一致

    # --- 方式 A: 训练模式 (List Concatenation) ---
    # 逻辑：token(Prompt) + token(Hint_Block) + token(Answer)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_str, add_special_tokens=False).input_ids
    
    # 训练中 Hint 和 Answer 是在一个字符串里 tokenized 的
    target_text = GEN_HINTS_WIH_ANSWER.format(hints=h, answer=a)
    target_ids = tokenizer(target_text, add_special_tokens=False).input_ids
    
    train_seq_ids = prompt_ids + target_ids

    # --- 方式 B: 推理模式 (String Concatenation) ---
    # 逻辑：vLLM 接收的是 full_prompt 字符串，然后一次性 tokenize
    # Inference logic: full_prompt = base_prompt + prefix_text + (generation)
    # 我们模拟生成完后的完整字符串
    
    # 1. Base Prompt (System + User + Assistant Header)
    base_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 2. Prefix (Hint)
    prefix_text = HINT_PREFIX_TEMPLATE.format(hint=h)
    
    # 3. Generated Answer (模拟)
    generated_text = a
    
    # 4. 拼接完整字符串
    full_inference_str = base_prompt + prefix_text + generated_text
    
    # 5. Tokenize
    inference_seq_ids = tokenizer(full_inference_str, add_special_tokens=False).input_ids

    # --- 比较 ---
    logger.info(f"Train Sequence Length:     {len(train_seq_ids)}")
    logger.info(f"Inference Sequence Length: {len(inference_seq_ids)}")

    if train_seq_ids == inference_seq_ids:
        logger.info("✅ Consistency Check: PASS (Perfect Match)")
    else:
        logger.error("❌ Consistency Check: FAILED (Mismatch Detected!)")
        # 找出哪里不一样
        min_len = min(len(train_seq_ids), len(inference_seq_ids))
        for i in range(min_len):
            if train_seq_ids[i] != inference_seq_ids[i]:
                logger.error(f"Mismatch at index {i}:")
                logger.error(f"  Train Token: {train_seq_ids[i]} ({tokenizer.decode([train_seq_ids[i]])})")
                logger.error(f"  Infer Token: {inference_seq_ids[i]} ({tokenizer.decode([inference_seq_ids[i]])})")
                # 通常这里不一样是因为边界融合
                context_start = max(0, i-5)
                context_end = min(min_len, i+5)
                logger.info(f"  Context around mismatch (Train): {tokenizer.decode(train_seq_ids[context_start:context_end])}")
                break

if __name__ == "__main__":
    tokenizer = load_tokenizer()
    if tokenizer:
        verify_mask_alignment(tokenizer)
        verify_train_inference_consistency(tokenizer)

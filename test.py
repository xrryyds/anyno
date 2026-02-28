import os
import torch
from transformers import AutoTokenizer

# ==========================================
# 配置与模板 (严格复制自你的代码)
# ==========================================
SYSTEM_PROMPT = "Please reason step by step and put your final answer within \\boxed{}."

# Training 模板
GEN_HINTS_WIH_ANSWER = "# known:\n{hints}\n{answer}"

# Inference 模板
HINT_PREFIX_TEMPLATE = "# known:\n{hint}\n"

class ConsistencyDebugger:
    def __init__(self, model_path):
        print(f"Loading tokenizer from: {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
    def debug_consistency(self, question, hints, answer):
        print("\n" + "#" * 60)
        print("🔍 深度一致性检查 (Deep Consistency Check)")
        print("#" * 60)

        # ==============================================================================
        # Part 1: 模拟训练输入 (Mode B Generation)
        # 逻辑来源: FixedModeCollator
        # ==============================================================================
        print("\n" + "="*20 + " [1] TRAINING INPUT (Mode B) " + "="*20)
        
        # 1.1 构建 Prompt ID
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": str(question)}]
        prompt_str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = self.tokenizer(prompt_str, add_special_tokens=False).input_ids
        
        # 1.2 构建 Target ID (Hint + Answer)
        target_text = GEN_HINTS_WIH_ANSWER.format(hints=hints, answer=answer)
        target_ids = self.tokenizer(target_text, add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]
        
        # 1.3 拼接 (List Concatenation) -> 这是模型在训练时实际接收到的序列
        train_full_ids = prompt_ids + target_ids
        
        # 1.4 计算 Mask 边界 (为了截取展示)
        hint_only_text = f"# known:\n{hints}\n"
        hint_part_ids = self.tokenizer(hint_only_text, add_special_tokens=False).input_ids
        
        len_prompt = len(prompt_ids)
        len_hint = len(hint_part_ids)
        
        # 截取各部分
        real_prompt_part = train_full_ids[:len_prompt]
        real_hint_part = train_full_ids[len_prompt : len_prompt + len_hint]
        real_answer_part = train_full_ids[len_prompt + len_hint :]
        
        # 解码回字符串 (模型看到的实际内容)
        decoded_train_full = self.tokenizer.decode(train_full_ids)
        decoded_train_prompt = self.tokenizer.decode(real_prompt_part)
        decoded_train_hint = self.tokenizer.decode(real_hint_part)
        decoded_train_answer = self.tokenizer.decode(real_answer_part)

        print(f"【完整输入 (Decoded String)】:\n{repr(decoded_train_full)}")
        print(f"\n【截取: Prompt部分】:\n{repr(decoded_train_prompt)}")
        print(f"\n【截取: Hints部分 (Train)】:\n{repr(decoded_train_hint)}")
        print(f"\n【截取: Answer部分 (Train)】:\n{repr(decoded_train_answer)}")
        print(f"\n【Token IDs (前10个)】: {train_full_ids[:10]}")
        print(f"【Token IDs (接缝处 ±5个)】: ... {train_full_ids[len_prompt-5 : len_prompt+5]} ...")

        # ==============================================================================
        # Part 2: 模拟推理输入 (Inference / vLLM)
        # 逻辑来源: exam_with_hints
        # ==============================================================================
        print("\n" + "="*20 + " [2] INFERENCE INPUT (Prompt + Hint) " + "="*20)
        
        # 2.1 构建 Base Prompt
        base_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # 2.2 构建 Prefix (Hint)
        prefix_text = HINT_PREFIX_TEMPLATE.format(hint=hints)
        
        # 2.3 字符串拼接 (String Concatenation) -> 这是喂给 vLLM 的原始字符串
        infer_full_prompt_str = base_prompt + prefix_text
        
        # 2.4 vLLM 内部会做的 Tokenize
        infer_full_ids = self.tokenizer(infer_full_prompt_str, add_special_tokens=False).input_ids

        print(f"【完整Prompt (String喂给vLLM)】:\n{repr(infer_full_prompt_str)}")
        print(f"\n【vLLM Tokenize后的 IDs (前10个)】: {infer_full_ids[:10]}")
        
        # 找到接缝处 (Base Prompt 结束的地方)
        # 注意：这里我们大致估算接缝位置，通过长度对比
        boundary_idx = len(prompt_ids) 
        if boundary_idx < len(infer_full_ids):
             print(f"【Token IDs (接缝处 ±5个)】: ... {infer_full_ids[boundary_idx-5 : boundary_idx+5]} ...")

        # ==============================================================================
        # Part 3: 核心对比结果
        # ==============================================================================
        print("\n" + "="*20 + " [3] COMPARISON RESULT " + "="*20)
        
        # 比较点：模型在生成 Answer 之前，看到的输入是否一致？
        # 训练时看到的 = Prompt IDs + Hint IDs
        # 推理时看到的 = Tokenize(Prompt Str + Hint Str)
        
        train_input_prefix_ids = train_full_ids[:len_prompt + len_hint]
        infer_input_prefix_ids = infer_full_ids # 推理时整个就是输入
        
        # 检查长度
        len_train = len(train_input_prefix_ids)
        len_infer = len(infer_input_prefix_ids)
        
        print(f"Length Check: Train={len_train} tokens vs Inference={len_infer} tokens")
        
        if train_input_prefix_ids == infer_input_prefix_ids:
            print("\n✅ [PERFECT MATCH] 完美一致！")
            print("训练代码中的 List拼接 与 推理代码中的 String拼接 产生了完全相同的 Token 序列。")
            print("你可以放心地使用这套代码。")
        else:
            print("\n❌ [MISMATCH] 不一致！")
            print("警告：训练时模型看到的 Token 与推理时喂给模型的 Token 不一样。")
            
            # 寻找第一个不一致的地方
            min_len = min(len_train, len_infer)
            for i in range(min_len):
                if train_input_prefix_ids[i] != infer_input_prefix_ids[i]:
                    t_id = train_input_prefix_ids[i]
                    i_id = infer_input_prefix_ids[i]
                    print(f"\n第一个差异点 (Index {i}):")
                    print(f"  Train Token: {t_id} -> {repr(self.tokenizer.decode([t_id]))}")
                    print(f"  Infer Token: {i_id} -> {repr(self.tokenizer.decode([i_id]))}")
                    
                    # 打印上下文
                    start = max(0, i-10)
                    print(f"  Context (Train): {self.tokenizer.decode(train_input_prefix_ids[start:i+1])}")
                    print(f"  Context (Infer): {self.tokenizer.decode(infer_input_prefix_ids[start:i+1])}")
                    break
            
            if len_train != len_infer:
                print(f"\n长度差异原因分析: 可能是字符串拼接处的空格或换行符被 Tokenizer 合并了。")
                print("Train (List Join): [End_Prompt] + [Start_Hint]")
                print("Infer (Str Join) : '...End_Prompt' + 'Start_Hint...' -> Tokenize")

# ==========================================
# 运行配置
# ==========================================
if __name__ == "__main__":
    # 请修改为你的模型路径
    # model_path = "/root/autodl-tmp/model/OREAL-7B" 
    # 如果你在本地测试，可以用 Qwen/Qwen2.5-7B-Instruct 等
    model_path =  "/root/autodl-tmp/CELPO/model/OREAL/OREAL-7B"

    # 检查路径
    if not os.path.exists(model_path):
        print(f"Error: 路径 {model_path} 不存在，请修改脚本底部的 model_path。")
    else:
        debugger = ConsistencyDebugger(model_path)
        
        # 测试用例 (包含 LaTeX 和 换行，这是最容易出错的地方)
        q = "Calculate $\\int x dx$."
        h = "Recall that $\\int x^n dx = \\frac{x^{n+1}}{n+1}$."
        a = "The answer is $\\frac{x^2}{2} + C$."
        
        debugger.debug_consistency(q, h, a)

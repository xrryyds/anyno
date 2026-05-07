import os
import torch
from transformers import AutoTokenizer

# ==========================================
# ==========================================
SYSTEM_PROMPT = "Please reason step by step and put your final answer within \\boxed{}."

GEN_HINTS_WIH_ANSWER = "{hints}{answer}"

HINT_PREFIX_TEMPLATE = "{hint}"

class ConsistencyDebugger:
    def __init__(self, model_path):
        print(f"Loading tokenizer from: {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
    def debug_consistency(self, question, hints, answer):
        print("\n" + "#" * 60)
        print("🔍  (Deep Consistency Check)")
        print("#" * 60)

        # ==============================================================================
        # ==============================================================================
        print("\n" + "="*20 + " [1] TRAINING INPUT (Mode B) " + "="*20)
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": str(question)}]
        prompt_str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = self.tokenizer(prompt_str, add_special_tokens=False).input_ids
        
        target_text = GEN_HINTS_WIH_ANSWER.format(hints=hints, answer=answer)
        target_ids = self.tokenizer(target_text, add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]
        
        train_full_ids = prompt_ids + target_ids
        
        hint_only_text = f"{hints}"
        hint_part_ids = self.tokenizer(hint_only_text, add_special_tokens=False).input_ids
        
        len_prompt = len(prompt_ids)
        len_hint = len(hint_part_ids)
        
        real_prompt_part = train_full_ids[:len_prompt]
        real_hint_part = train_full_ids[len_prompt : len_prompt + len_hint]
        real_answer_part = train_full_ids[len_prompt + len_hint :]
        
        decoded_train_full = self.tokenizer.decode(train_full_ids)
        decoded_train_prompt = self.tokenizer.decode(real_prompt_part)
        decoded_train_hint = self.tokenizer.decode(real_hint_part)
        decoded_train_answer = self.tokenizer.decode(real_answer_part)

        print(f" (Decoded String):\n{repr(decoded_train_full)}")
        print(f"\n: Prompt:\n{repr(decoded_train_prompt)}")
        print(f"\n: Hints (Train):\n{repr(decoded_train_hint)}")
        print(f"\n: Answer (Train):\n{repr(decoded_train_answer)}")
        print(f"\nToken IDs (10): {train_full_ids[:10]}")
        print(f"Token IDs ( ±5): ... {train_full_ids[len_prompt-5 : len_prompt+5]} ...")

        # ==============================================================================
        # ==============================================================================
        print("\n" + "="*20 + " [2] INFERENCE INPUT (Prompt + Hint) " + "="*20)
        
        base_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        prefix_text = HINT_PREFIX_TEMPLATE.format(hint=hints)
        
        infer_full_prompt_str = base_prompt + prefix_text
        
        infer_full_ids = self.tokenizer(infer_full_prompt_str, add_special_tokens=False).input_ids

        print(f"Prompt (StringvLLM):\n{repr(infer_full_prompt_str)}")
        print(f"\nvLLM Tokenize IDs (10): {infer_full_ids[:10]}")
        
        boundary_idx = len(prompt_ids) 
        if boundary_idx < len(infer_full_ids):
             print(f"Token IDs ( ±5): ... {infer_full_ids[boundary_idx-5 : boundary_idx+5]} ...")

        # ==============================================================================
        # ==============================================================================
        print("\n" + "="*20 + " [3] COMPARISON RESULT " + "="*20)
        
        
        train_input_prefix_ids = train_full_ids[:len_prompt + len_hint]
        infer_input_prefix_ids = infer_full_ids
        
        len_train = len(train_input_prefix_ids)
        len_infer = len(infer_input_prefix_ids)
        
        print(f"Length Check: Train={len_train} tokens vs Inference={len_infer} tokens")
        
        if train_input_prefix_ids == infer_input_prefix_ids:
            print("\n✅ [PERFECT MATCH] ")
            print(" List   String  Token ")
            print("")
        else:
            print("\n❌ [MISMATCH] ")
            print(" Token  Token ")
            
            min_len = min(len_train, len_infer)
            for i in range(min_len):
                if train_input_prefix_ids[i] != infer_input_prefix_ids[i]:
                    t_id = train_input_prefix_ids[i]
                    i_id = infer_input_prefix_ids[i]
                    print(f"\n (Index {i}):")
                    print(f"  Train Token: {t_id} -> {repr(self.tokenizer.decode([t_id]))}")
                    print(f"  Infer Token: {i_id} -> {repr(self.tokenizer.decode([i_id]))}")
                    
                    start = max(0, i-10)
                    print(f"  Context (Train): {self.tokenizer.decode(train_input_prefix_ids[start:i+1])}")
                    print(f"  Context (Infer): {self.tokenizer.decode(infer_input_prefix_ids[start:i+1])}")
                    break
            
            if len_train != len_infer:
                print(f"\n:  Tokenizer ")
                print("Train (List Join): [End_Prompt] + [Start_Hint]")
                print("Infer (Str Join) : '...End_Prompt' + 'Start_Hint...' -> Tokenize")

# ==========================================
# ==========================================
if __name__ == "__main__":
    # model_path = "/root/autodl-tmp/model/OREAL-7B" 
    model_path =  "/root/autodl-tmp/CELPO/model/OREAL/OREAL-7B"

    if not os.path.exists(model_path):
        print(f"Error:  {model_path}  model_path")
    else:
        debugger = ConsistencyDebugger(model_path)
        
        q = "Calculate $\\int x dx$."
        h = "Recall that $\\int x^n dx = \\frac{x^{n+1}}{n+1}$."
        a = "The answer is $\\frac{x^2}{2} + C$."
        
        debugger.debug_consistency(q, h, a)

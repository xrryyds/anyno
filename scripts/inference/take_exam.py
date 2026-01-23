from debugpy._vendored import project_root
from builtins import print
from builtins import print
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from transformers import set_seed 
from utils import FileIOUtils, extract_hints ,extract_boxed_content, normalize_answer
import numpy as np


class TakeExam:
    def __init__(self,  model_path: str = "/root/autodl-tmp/model/Qwen/Qwen/Qwen2.5-Math-7B-Instruct"):
        
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file_path)) 
        project_root = os.path.dirname(project_root) 
        exam_result_json_path = os.path.join(project_root, "datasets", "exam", "exam.json")
        set_seed(42)

        print(exam_result_json_path)
        self.BATCH_SIZE = 64  
        self.MAX_NEW_TOKENS = 4096
        self.MAX_SEQ_LENGTH = 6000
    
        self.LOCAL_MODEL_PATH = model_path
        self.OUTPUT_JSON_PATH = exam_result_json_path
        
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

        print(f"Loading model from {self.LOCAL_MODEL_PATH}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.LOCAL_MODEL_PATH, trust_remote_code=True)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.LOCAL_MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        self.tokenizer.padding_side = "left" 


    def answer_single_question(self, question):
        try:
            q_text = str(question)
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": q_text}],
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.MAX_SEQ_LENGTH
            ).to(self.model.device)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.MAX_NEW_TOKENS,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.9,
                    use_cache=True
                )
            
            input_ids_len = inputs["input_ids"].shape[1]
            generated_text = self.tokenizer.decode(
                outputs[0, input_ids_len:],
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"[Error] Failed to answer question: {e}")
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
            return ""

    def exam_test(self, question, solution, answer, question_idx):
        # 初始化统计变量
        correct_count = 0
        total_count = 0
        
        total_batches = (len(question) + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        
        for i in tqdm(range(0, len(question), self.BATCH_SIZE), total=total_batches, desc="Inferencing"):
            batch_questions = question[i:i+self.BATCH_SIZE]
            batch_ref_answers = answer[i:i+self.BATCH_SIZE]
            # batch_ref_solution = solution[i:i+self.BATCH_SIZE] # 既然不保存文件，solution如果不参与计算可忽略
            # batch_question_idx = question_idx[i:i+self.BATCH_SIZE]

            try:
                batch_prompts = []
                for q in batch_questions:
                    q_text = str(q)
                    prompt = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": q_text}],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    batch_prompts.append(prompt)

                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.MAX_SEQ_LENGTH
                ).to(self.model.device)

                # Inference Mode
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.MAX_NEW_TOKENS,
                        pad_token_id=self.tokenizer.pad_token_id,
                        do_sample=True,
                        temperature=0.1, 
                        top_p=0.9,
                        use_cache=True 
                    )

                input_ids_len = inputs["input_ids"].shape[1]
                decoded_outputs = self.tokenizer.batch_decode(
                    outputs[:, input_ids_len:], 
                    skip_special_tokens=True
                )

                # === 计算准确率部分 ===
                for idx, generated_text in enumerate(decoded_outputs):
                    # 获取模型生成结果和参考答案
                    pred_answer = extract_boxed_content(generated_text)
                    ref_answer = str(batch_ref_answers[idx]).strip()
                    pred_answer = normalize_answer(pred_answer)
                    ref_answer = normalize_answer(ref_answer)
                    
                    is_correct = (pred_answer == ref_answer)
                    if is_correct:
                        correct_count += 1
                    
                    total_count += 1
                
                # 实时打印当前进度准确率 (可选)
                current_acc = correct_count / total_count if total_count > 0 else 0
                print(f"Batch {i//self.BATCH_SIZE} done. Current Acc: {current_acc:.2%}")

            except Exception as e:
                print(f"\n[Error] Batch {i//self.BATCH_SIZE} failed: {e}")
                if "out of memory" in str(e):
                    print("显存不足提示: 请降低 Batch Size。")
                torch.cuda.empty_cache()
                continue

        # 计算最终准确率
        final_accuracy = correct_count / total_count if total_count > 0 else 0.0
        print(f"\n========================================")
        print(f"Inference Done.")
        print(f"Correct: {correct_count}/{total_count}")
        print(f"Final Accuracy: {final_accuracy:.2%}")
        print(f"========================================")
        
        return final_accuracy

  



    def compute_shannon_entropy(self, logits_sequence, normalize=True):
        if len(logits_sequence.shape) == 3:
            entropies = []
            for single_logits in logits_sequence:
                entropy = self.compute_shannon_entropy(single_logits, normalize)
                entropies.append(entropy)
            return np.array(entropies)
        
        L, vocab_size = logits_sequence.shape
        
        probs = torch.softmax(logits_sequence, dim=-1)  
        
        log_probs = torch.log(probs + 1e-10) 
        step_entropies = -torch.sum(probs * log_probs, dim=-1)  
        
        avg_entropy = torch.mean(step_entropies).item()
        
        if normalize:
            max_entropy = np.log(vocab_size)
            normalized_entropy = avg_entropy / max_entropy
            return normalized_entropy
        
        return avg_entropy


    def exam(self, question, solution, answer, question_idx):
        results = []
        total_batches = (len(question) + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        
        for i in tqdm(range(0, len(question), self.BATCH_SIZE), total=total_batches, desc="Inferencing"):
            batch_questions = question[i:i+self.BATCH_SIZE]
            batch_ref_answers = answer[i:i+self.BATCH_SIZE]
            batch_ref_solution = solution[i:i+self.BATCH_SIZE]
            batch_question_idx = question_idx[i:i+self.BATCH_SIZE]

            try:
                batch_prompts = []
                for q in batch_questions:
                    q_text = str(q)
                    prompt = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": q_text}],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    batch_prompts.append(prompt)

                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.MAX_SEQ_LENGTH
                ).to(self.model.device)

                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.MAX_NEW_TOKENS,
                        pad_token_id=self.tokenizer.pad_token_id,
                        do_sample=True,
                        temperature=0.1, 
                        top_p=0.9,
                        use_cache=True
                    )

                input_ids_len = inputs["input_ids"].shape[1]
                
                generated_sequences = outputs[:, input_ids_len:]
                
                decoded_outputs = self.tokenizer.batch_decode(
                    generated_sequences, 
                    skip_special_tokens=True
                )
                

                for idx, generated_text in enumerate(decoded_outputs):
                    results.append({
                        "question": batch_questions[idx],
                        "answer": generated_text.strip(),
                        "ref_answer": batch_ref_answers[idx].strip(),
                        "ref_solution": batch_ref_solution[idx].strip(),
                        "question_idx": batch_question_idx[idx]
                    })
                    
                if (i // self.BATCH_SIZE) % 10 == 0:
                    with open(self.OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"\n[Error] Batch {i//self.BATCH_SIZE} failed: {e}")
                if "out of memory" in str(e):
                    print("显存不足提示: 如果 BS=8 依然 OOM，请改回 4。")
                torch.cuda.empty_cache()
                continue

        with open(self.OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        
        print(f"Done! Results saved to {self.OUTPUT_JSON_PATH}")


        with open(self.OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Done! Results saved to {self.OUTPUT_JSON_PATH}")












    def exam_multi_answer(self, question, solution, answer, question_idx, num_samples=8, temperature=0.7):
        results = []
        
        total_batches = (len(question) + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        
        for i in tqdm(range(0, len(question), self.BATCH_SIZE), total=total_batches, desc="Inferencing"):
            batch_questions = question[i:i+self.BATCH_SIZE]
            batch_ref_answers = answer[i:i+self.BATCH_SIZE]
            batch_ref_solution = solution[i:i+self.BATCH_SIZE]
            batch_question_idx = question_idx[i:i+self.BATCH_SIZE]

            try:
                batch_prompts = []
                for q in batch_questions:
                    q_text = str(q)
                    prompt = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": q_text}],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    batch_prompts.append(prompt)

                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.MAX_SEQ_LENGTH
                ).to(self.model.device)

                input_ids_len = inputs["input_ids"].shape[1]

                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.MAX_NEW_TOKENS,
                        pad_token_id=self.tokenizer.pad_token_id,
                        do_sample=True, # 必须开启采样才能生成多样化结果
                        
                        # === 修改点 1: 传入控制参数 ===
                        temperature=temperature, 
                        num_return_sequences=num_samples, # 关键参数：一次生成多条
                        
                        top_p=0.9,
                        use_cache=True
                    )

                # outputs 的 shape 变成了 [batch_size * num_samples, seq_len]
                generated_sequences = outputs[:, input_ids_len:]
                
                decoded_outputs = self.tokenizer.batch_decode(
                    generated_sequences, 
                    skip_special_tokens=True
                )
                
                # === 修改点 2: 结果匹配逻辑 ===
                # HuggingFace 的 generate 输出顺序是：
                # [Q1_A1, Q1_A2, ..., Q1_An, Q2_A1, Q2_A2, ...]
                # 所以我们需要两层循环来对应回原始数据
                for batch_idx in range(len(batch_questions)):
                    # 当前问题对应的所有答案在 decoded_outputs 中的起始位置
                    start_pos = batch_idx * num_samples
                    
                    for sample_idx in range(num_samples):
                        # 获取具体的某一个生成结果
                        generated_text = decoded_outputs[start_pos + sample_idx]
                        
                        results.append({
                            "question": batch_questions[batch_idx],
                            "answer": generated_text.strip(),
                            "ref_answer": batch_ref_answers[batch_idx].strip(),
                            "ref_solution": batch_ref_solution[batch_idx].strip(),
                            "question_idx": batch_question_idx[batch_idx]
                        })
                    
                # 定期保存
                if (i // self.BATCH_SIZE) % 10 == 0:
                    with open(self.OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"\n[Error] Batch {i//self.BATCH_SIZE} failed: {e}")
                if "out of memory" in str(e):
                    print(f"显存不足提示: 当前 num_samples={num_samples}, 实际负载增加了 {num_samples} 倍。请尝试减小 BATCH_SIZE。")
                torch.cuda.empty_cache()
                continue

        # 最终保存
        with open(self.OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Done! Results saved to {self.OUTPUT_JSON_PATH}")



if __name__ == "__main__":
    from configs import GRPOConfig
    from data_math import Math_500
    
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path)) 
    project_root = os.path.dirname(os.path.dirname(project_root))
    exam_file_path = os.path.join(project_root, "CELPO", "configs", "celpo_train.yaml")
    print(exam_file_path)
    config = GRPOConfig.load_yaml(exam_file_path)
    math_500 = Math_500(config)
    test_dataset = math_500.get_test_data()
    train_dataset= math_500.get_train_data()
    question = test_dataset.problems + train_dataset.problems
    solution = test_dataset.solutions + train_dataset.solutions
    answer = test_dataset.answers + train_dataset.answers
    print(f"dataset_len_check: {len(question)} {len(solution)} {len(answer)}")
    take_exam = TakeExam("/root/project/data/xrr/Qwen/Qwen2.5-Math-7B-Instruct")
    question_idx = []
    for idx in range(len(question)):
        question_idx.append(idx)
    take_exam.exam(question, solution, answer, question_idx)

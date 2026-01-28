import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed
)

from utils import (
    FileIOUtils,
    extract_hints,
    extract_boxed_content,
    normalize_answer
)


class TakeExam:
    def __init__(
        self,
        model_path: str = "/root/autodl-tmp/model/Qwen/Qwen2.5-Math-7B-Instruct"
    ):
        # ================== Path ==================
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file_path))
        project_root = os.path.dirname(project_root)

        self.OUTPUT_JSON_PATH = os.path.join(
            project_root, "datasets", "exam", "exam.json"
        )

        # ================== Config ==================
        set_seed(42)

        self.BATCH_SIZE = 32
        self.MAX_NEW_TOKENS = 2048
        self.MAX_SEQ_LENGTH = 3096

        self.LOCAL_MODEL_PATH = model_path

        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

        # ================== Load tokenizer ==================
        print(f"Loading tokenizer from {self.LOCAL_MODEL_PATH} ...")

        # 🚑 关键修复：强制使用 slow tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.LOCAL_MODEL_PATH,
            trust_remote_code=True,
            use_fast=False,          # ⭐⭐⭐ 必须
        )

        # ================== Load model ==================
        print(f"Loading model from {self.LOCAL_MODEL_PATH} ...")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.LOCAL_MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        # ================== Pad token ==================
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        self.tokenizer.padding_side = "left"

        print("Model & Tokenizer loaded successfully.")

    # =====================================================
    # 单题推理
    # =====================================================
    def answer_single_question(self, question: str) -> str:
        try:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": str(question)}],
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.MAX_SEQ_LENGTH,
            ).to(self.model.device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.MAX_NEW_TOKENS,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.9,
                    use_cache=True,
                )

            gen_text = self.tokenizer.decode(
                outputs[0, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            return gen_text.strip()

        except Exception as e:
            print(f"[Error] single question failed: {e}")
            torch.cuda.empty_cache()
            return ""

    # =====================================================
    # 批量考试（算准确率）
    # =====================================================
    def exam_test(self, question, solution, answer, question_idx):
        correct_count, total_count = 0, 0

        total_batches = (len(question) + self.BATCH_SIZE - 1) // self.BATCH_SIZE

        for i in tqdm(
            range(0, len(question), self.BATCH_SIZE),
            total=total_batches,
            desc="Inferencing",
        ):
            try:
                batch_questions = question[i:i + self.BATCH_SIZE]
                batch_answers = answer[i:i + self.BATCH_SIZE]

                prompts = [
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": str(q)}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for q in batch_questions
                ]

                inputs = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.MAX_SEQ_LENGTH,
                ).to(self.model.device)

                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.MAX_NEW_TOKENS,
                        pad_token_id=self.tokenizer.pad_token_id,
                        do_sample=True,
                        temperature=0.1,
                        top_p=0.9,
                        use_cache=True,
                    )

                input_len = inputs["input_ids"].shape[1]
                decoded = self.tokenizer.batch_decode(
                    outputs[:, input_len:],
                    skip_special_tokens=True,
                )

                for pred, ref in zip(decoded, batch_answers):
                    pred_ans = normalize_answer(
                        extract_boxed_content(pred)
                    )
                    ref_ans = normalize_answer(str(ref))

                    if pred_ans == ref_ans:
                        correct_count += 1
                    total_count += 1

            except Exception as e:
                print(f"[Error] batch {i // self.BATCH_SIZE} failed: {e}")
                torch.cuda.empty_cache()

        acc = correct_count / max(total_count, 1)
        print(f"\nFinal Accuracy: {acc:.2%}")
        return acc

    # =====================================================
    # 批量考试（保存所有结果）
    # =====================================================
    def exam(self, question, solution, answer, question_idx):
        results = []

        total_batches = (len(question) + self.BATCH_SIZE - 1) // self.BATCH_SIZE

        for i in tqdm(
            range(0, len(question), self.BATCH_SIZE),
            total=total_batches,
            desc="Inferencing",
        ):
            try:
                batch_questions = question[i:i + self.BATCH_SIZE]
                batch_solutions = solution[i:i + self.BATCH_SIZE]
                batch_answers = answer[i:i + self.BATCH_SIZE]
                batch_ids = question_idx[i:i + self.BATCH_SIZE]

                prompts = [
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": str(q)}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for q in batch_questions
                ]

                inputs = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.MAX_SEQ_LENGTH,
                ).to(self.model.device)

                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.MAX_NEW_TOKENS,
                        pad_token_id=self.tokenizer.pad_token_id,
                        do_sample=True,
                        temperature=0.1,
                        top_p=0.9,
                        use_cache=True,
                    )

                input_len = inputs["input_ids"].shape[1]
                decoded = self.tokenizer.batch_decode(
                    outputs[:, input_len:],
                    skip_special_tokens=True,
                )

                for q, a, ra, rs, idx in zip(
                    batch_questions,
                    decoded,
                    batch_answers,
                    batch_solutions,
                    batch_ids,
                ):
                    results.append({
                        "question": q,
                        "answer": a.strip(),
                        "ref_answer": ra.strip(),
                        "ref_solution": rs.strip(),
                        "question_idx": idx,
                    })

            except Exception as e:
                print(f"[Error] batch {i // self.BATCH_SIZE} failed: {e}")
                torch.cuda.empty_cache()

        os.makedirs(os.path.dirname(self.OUTPUT_JSON_PATH), exist_ok=True)
        with open(self.OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Done! Results saved to {self.OUTPUT_JSON_PATH}")


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    from configs import GRPOConfig
    from data_math import Math_500

    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path))
    project_root = os.path.dirname(os.path.dirname(project_root))

    exam_file_path = os.path.join(
        project_root, "CELPO", "configs", "celpo_train.yaml"
    )

    config = GRPOConfig.load_yaml(exam_file_path)
    math_500 = Math_500(config)

    test_dataset = math_500.get_test_data()
    train_dataset = math_500.get_train_data()

    question = test_dataset.problems + train_dataset.problems
    solution = test_dataset.solutions + train_dataset.solutions
    answer = test_dataset.answers + train_dataset.answers
    question_idx = list(range(len(question)))

    print(f"Dataset size: {len(question)}")

    take_exam = TakeExam(
        "/root/project/data/xrr/Qwen/Qwen2.5-Math-7B-Instruct"
    )
    take_exam.exam(question, solution, answer, question_idx)

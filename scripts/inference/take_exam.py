import os
import json
import torch
import numpy as np
import logging
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

# =====================================================
# Logger
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
        logger.info(f"Loading tokenizer from {self.LOCAL_MODEL_PATH}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.LOCAL_MODEL_PATH,
            trust_remote_code=True,
            use_fast=False,  # ⭐ 必须使用 slow tokenizer
        )

        # ================== Load model ==================
        logger.info(f"Loading model from {self.LOCAL_MODEL_PATH}")

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

        logger.info("Model & Tokenizer loaded successfully.")

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
            logger.error(f"Single question failed: {e}")
            torch.cuda.empty_cache()
            return ""

    # =====================================================
    # 批量考试（batch 级保存 + entropy）
    # =====================================================
    def exam(self, question, solution, answer, question_idx):
        results = []

        total_batches = (len(question) + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        os.makedirs(os.path.dirname(self.OUTPUT_JSON_PATH), exist_ok=True)

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
                        return_dict_in_generate=True,
                        output_scores=True,
                    )

                input_len = inputs["input_ids"].shape[1]
                generated_ids = outputs.sequences[:, input_len:]

                decoded = self.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )

                entropies = self._compute_sequence_entropy(
                    generated_ids,
                    outputs.scores,
                )

                del outputs
                torch.cuda.empty_cache()

                for q, a, ra, rs, idx, ent in zip(
                    batch_questions,
                    decoded,
                    batch_answers,
                    batch_solutions,
                    batch_ids,
                    entropies,
                ):
                    results.append({
                        "question": q,
                        "answer": a.strip(),
                        "ref_answer": ra.strip(),
                        "ref_solution": rs.strip(),
                        "question_idx": idx,
                        "entropy": float(ent),
                    })

                # ================== ⭐ batch 级保存 ==================
                with open(self.OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

                logger.info(
                    f"Batch {i // self.BATCH_SIZE + 1}/{total_batches} saved "
                    f"({len(results)} samples)"
                )

            except Exception as e:
                logger.error(
                    f"Batch {i // self.BATCH_SIZE + 1} failed: {e}"
                )
                torch.cuda.empty_cache()

        logger.info(f"All done! Final results saved to {self.OUTPUT_JSON_PATH}")

    # =====================================================
    # 显存友好的 entropy 计算
    # =====================================================
    def _compute_sequence_entropy(self, generated_ids, scores):
        """
        H(a) = -1/L * sum_i log p_{i, y_i}
        仅对生成 token 计算 log-prob，避免 vocab 级 softmax
        """
        if len(scores) == 0:
            return [0.0] * generated_ids.shape[0]

        batch_size = generated_ids.shape[0]
        seq_len = min(len(scores), generated_ids.shape[1])

        entropies = []

        for b in range(batch_size):
            total_nll = 0.0
            valid_tokens = 0

            for t in range(seq_len):
                token_id = generated_ids[b, t].item()

                if token_id in (
                    self.tokenizer.pad_token_id,
                    self.tokenizer.eos_token_id,
                ):
                    continue

                log_probs = torch.log_softmax(scores[t][b], dim=-1)
                total_nll += -log_probs[token_id].item()
                valid_tokens += 1

            entropies.append(
                total_nll / valid_tokens if valid_tokens > 0 else 0.0
            )

        return entropies


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

    logger.info(f"Dataset size: {len(question)}")

    take_exam = TakeExam(
        "/root/project/data/xrr/Qwen/Qwen2.5-Math-7B-Instruct"
    )
    take_exam.exam(question, solution, answer, question_idx)

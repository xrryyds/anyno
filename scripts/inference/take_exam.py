# import os
# import json
# import torch
# import numpy as np
# import logging
# from tqdm import tqdm
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     set_seed
# )
# # ⭐ 改动 1: 导入 PeftModel
# from peft import PeftModel

# from utils import (
#     FileIOUtils,
#     extract_hints,
#     extract_boxed_content,
#     normalize_answer
# )

# # =====================================================
# # Logger
# # =====================================================
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
# )
# logger = logging.getLogger(__name__)


# class TakeExam:
#     def __init__(
#         self,
#         model_path: str = "/root/autodl-tmp/model/Qwen/Qwen2.5-Math-7B-Instruct",
#         use_lora: bool = False,      
#         adapter_path: str = None     
#     ):
#         # ================== Path ==================
#         current_file_path = os.path.abspath(__file__)
#         project_root = os.path.dirname(os.path.dirname(current_file_path))
#         project_root = os.path.dirname(project_root)

#         self.OUTPUT_JSON_PATH = os.path.join(
#             project_root, "datasets", "exam", "exam.json"
#         )

#         self.OUTPUT_JSON_PATH_jsonl = os.path.join(
#             project_root, "datasets", "exam", "exam.jsonl"
#         )

#         self.OUTPUT_JSON_PATH_ROLL = os.path.join(
#             project_root, "datasets", "exam", "exam_roll.json"
#         )

#         self.OUTPUT_JSON_PATH_TEST = os.path.join(
#             project_root, "datasets", "exam", "exam_test.json"
#         )

#         self.OUTPUT_JSON_PATH_EHC_TEST = os.path.join(
#             project_root, "datasets", "exam", "exam_ehc_test.json"
#         )

#         # ================== Config ==================
#         set_seed(42)

#         self.BATCH_SIZE = 32
#         self.MAX_NEW_TOKENS = 2048
#         self.MAX_SEQ_LENGTH = 3096

#         self.LOCAL_MODEL_PATH = model_path

#         os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

#         # ================== Load tokenizer ==================
#         logger.info(f"Loading tokenizer from {self.LOCAL_MODEL_PATH}")

#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.LOCAL_MODEL_PATH,
#             trust_remote_code=True,
#             use_fast=False,  # ⭐ 必须使用 slow tokenizer
#         )

#         # ================== Load model ==================
#         logger.info(f"Loading base model from {self.LOCAL_MODEL_PATH}")

#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.LOCAL_MODEL_PATH,
#             device_map="auto",
#             torch_dtype=torch.float16,
#             low_cpu_mem_usage=True,
#             trust_remote_code=True,
#         )

#         # ================== ⭐ 改动 3: 加载 LoRA 逻辑 ==================
#         if use_lora:
#             if adapter_path and os.path.exists(adapter_path):
#                 logger.info(f"Loading LoRA adapter from: {adapter_path}")
#                 try:
#                     self.model = PeftModel.from_pretrained(
#                         self.model,
#                         adapter_path,
#                         torch_dtype=torch.float16
#                     )
#                     logger.info("LoRA adapter loaded successfully!")
#                 except Exception as e:
#                     logger.error(f"Failed to load LoRA adapter: {e}")
#                     raise e
#             else:
#                 logger.warning(f"use_lora=True but adapter_path '{adapter_path}' is invalid. Using base model.")

#         # ================== Pad token ==================
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#             self.model.config.pad_token_id = self.model.config.eos_token_id

#         self.tokenizer.padding_side = "left"

#         logger.info("Model & Tokenizer loaded successfully.")

#     # =====================================================
#     # 单题推理
#     # =====================================================
#     def answer_single_question(self, question: str) -> str:
#         try:
#             prompt = self.tokenizer.apply_chat_template(
#                 [{"role": "user", "content": str(question)}],
#                 tokenize=False,
#                 add_generation_prompt=True,
#             )

#             inputs = self.tokenizer(
#                 prompt,
#                 return_tensors="pt",
#                 truncation=True,
#                 max_length=self.MAX_SEQ_LENGTH,
#             ).to(self.model.device)

#             with torch.inference_mode():
#                 outputs = self.model.generate(
#                     **inputs,
#                     max_new_tokens=self.MAX_NEW_TOKENS,
#                     pad_token_id=self.tokenizer.pad_token_id,
#                     do_sample=True,
#                     temperature=0.1,
#                     top_p=0.9,
#                     use_cache=True,
#                 )

#             gen_text = self.tokenizer.decode(
#                 outputs[0, inputs["input_ids"].shape[1]:],
#                 skip_special_tokens=True,
#             )

#             return gen_text.strip()

#         except Exception as e:
#             logger.error(f"Single question failed: {e}")
#             torch.cuda.empty_cache()
#             return ""

#     # =====================================================
#     # 批量考试（batch 级保存 + entropy）
#     # =====================================================
#     def exam_with_cal_entropy(self, question, solution, answer, question_idx):
#         results = []

#         total_batches = (len(question) + self.BATCH_SIZE - 1) // self.BATCH_SIZE
#         os.makedirs(os.path.dirname(self.OUTPUT_JSON_PATH), exist_ok=True)

#         for i in tqdm(
#             range(0, len(question), self.BATCH_SIZE),
#             total=total_batches,
#             desc="Inferencing",
#         ):
#             try:
#                 batch_questions = question[i:i + self.BATCH_SIZE]
#                 batch_solutions = solution[i:i + self.BATCH_SIZE]
#                 batch_answers = answer[i:i + self.BATCH_SIZE]
#                 batch_ids = question_idx[i:i + self.BATCH_SIZE]

#                 prompts = [
#                     self.tokenizer.apply_chat_template(
#                         [{"role": "user", "content": str(q)}],
#                         tokenize=False,
#                         add_generation_prompt=True,
#                     )
#                     for q in batch_questions
#                 ]

#                 inputs = self.tokenizer(
#                     prompts,
#                     return_tensors="pt",
#                     padding=True,
#                     truncation=True,
#                     max_length=self.MAX_SEQ_LENGTH,
#                 ).to(self.model.device)

#                 with torch.inference_mode():
#                     outputs = self.model.generate(
#                         **inputs,
#                         max_new_tokens=self.MAX_NEW_TOKENS,
#                         pad_token_id=self.tokenizer.pad_token_id,
#                         do_sample=True,
#                         temperature=0.1,
#                         top_p=0.9,
#                         use_cache=True,
#                         return_dict_in_generate=True,
#                         output_scores=True,
#                     )

#                 input_len = inputs["input_ids"].shape[1]
#                 generated_ids = outputs.sequences[:, input_len:]

#                 decoded = self.tokenizer.batch_decode(
#                     generated_ids,
#                     skip_special_tokens=True,
#                 )

#                 entropies = self._compute_sequence_entropy(
#                     generated_ids,
#                     outputs.scores,
#                 )

#                 del outputs
#                 torch.cuda.empty_cache()

#                 for q, a, ra, rs, idx, ent in zip(
#                     batch_questions,
#                     decoded,
#                     batch_answers,
#                     batch_solutions,
#                     batch_ids,
#                     entropies,
#                 ):
#                     results.append({
#                         "question": q,
#                         "answer": a.strip(),
#                         "ref_answer": ra.strip(),
#                         "ref_solution": rs.strip(),
#                         "question_idx": idx,
#                         "entropy": float(ent),
#                     })

#                 # ================== ⭐ batch 级保存 ==================
#                 with open(self.OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
#                     json.dump(results, f, ensure_ascii=False, indent=2)

#                 logger.info(
#                     f"Batch {i // self.BATCH_SIZE + 1}/{total_batches} saved "
#                     f"({len(results)} samples)"
#                 )

#             except Exception as e:
#                 logger.error(
#                     f"Batch {i // self.BATCH_SIZE + 1} failed: {e}"
#                 )
#                 torch.cuda.empty_cache()

#         logger.info(f"All done! Final results saved to {self.OUTPUT_JSON_PATH}")

#     # =====================================================
#     # 批量 Roll K 次考试
#     # =====================================================
#     def exam_roll_k(
#         self,
#         question,
#         solution,
#         answer,
#         question_idx,
#         k: int = 8,
#         temperature: float = 0.7
#     ):
#         """
#         对每个问题采样 k 次。
#         结果不合并为列表，而是展平存储，保持与 exam 相同的字典结构（去掉 entropy）。
#         """
#         results = []

#         # 动态调整 Batch Size 防止显存爆炸
#         # 如果 k=8, 实际生成序列数会变为原来的8倍，因此输入 batch 需除以 k
#         real_batch_size = max(1, self.BATCH_SIZE // k)

#         total_batches = (len(question) + real_batch_size - 1) // real_batch_size
#         os.makedirs(os.path.dirname(self.OUTPUT_JSON_PATH_ROLL), exist_ok=True)

#         logger.info(
#             f"Starting Roll-K Exam: k={k}, temp={temperature}, "
#             f"real_batch_size={real_batch_size}"
#         )

#         for i in tqdm(
#             range(0, len(question), real_batch_size),
#             total=total_batches,
#             desc=f"Rolling (k={k})",
#         ):
#             try:
#                 batch_questions = question[i:i + real_batch_size]
#                 batch_solutions = solution[i:i + real_batch_size]
#                 batch_answers = answer[i:i + real_batch_size]
#                 batch_ids = question_idx[i:i + real_batch_size]

#                 # 构建 Prompts
#                 prompts = [
#                     self.tokenizer.apply_chat_template(
#                         [{"role": "user", "content": str(q)}],
#                         tokenize=False,
#                         add_generation_prompt=True,
#                     )
#                     for q in batch_questions
#                 ]

#                 inputs = self.tokenizer(
#                     prompts,
#                     return_tensors="pt",
#                     padding=True,
#                     truncation=True,
#                     max_length=self.MAX_SEQ_LENGTH,
#                 ).to(self.model.device)

#                 with torch.inference_mode():
#                     # 生成: num_return_sequences=k
#                     outputs = self.model.generate(
#                         **inputs,
#                         max_new_tokens=self.MAX_NEW_TOKENS,
#                         pad_token_id=self.tokenizer.pad_token_id,
#                         do_sample=True,             # 必须开启采样
#                         temperature=temperature,    # 温度
#                         top_p=0.9,
#                         num_return_sequences=k,     # 每个问题返回 k 个回答
#                         use_cache=True,
#                     )

#                 input_len = inputs["input_ids"].shape[1]
#                 generated_ids = outputs[:, input_len:]

#                 decoded_flat = self.tokenizer.batch_decode(
#                     generated_ids,
#                     skip_special_tokens=True,
#                 )

#                 del outputs, inputs, generated_ids
#                 torch.cuda.empty_cache()

#                 # 整理结果
#                 # decoded_flat 的长度是 len(batch_questions) * k
#                 # 顺序是: [q1_roll1, q1_roll2... q1_rollk, q2_roll1...]
#                 for j, (q, ra, rs, idx) in enumerate(zip(
#                     batch_questions,
#                     batch_answers,
#                     batch_solutions,
#                     batch_ids
#                 )):
#                     # 截取属于当前问题的 k 个回答
#                     start_idx = j * k
#                     end_idx = start_idx + k
#                     k_responses = decoded_flat[start_idx:end_idx]

#                     # 将这 k 个回答分别作为独立的条目保存
#                     for resp in k_responses:
#                         results.append({
#                             "question": q,
#                             "answer": resp.strip(),  # 这里是具体的某一次 roll 的结果
#                             "ref_answer": ra.strip(),
#                             "ref_solution": rs.strip(),
#                             "question_idx": idx,
#                             # "entropy" 字段不需要
#                         })

#                 # ================== 实时保存 ==================
#                 with open(self.OUTPUT_JSON_PATH_ROLL, "w", encoding="utf-8") as f:
#                     json.dump(results, f, ensure_ascii=False, indent=2)

#                 logger.info(
#                     f"Batch {i // real_batch_size + 1}/{total_batches} saved. "
#                     f"Total entries so far: {len(results)}"
#                 )

#             except Exception as e:
#                 logger.error(
#                     f"Roll Batch {i // real_batch_size + 1} failed: {e}"
#                 )
#                 torch.cuda.empty_cache()

#         logger.info(f"Roll-K Exam done! Results saved to {self.OUTPUT_JSON_PATH_ROLL}")

#     # =====================================================
#     # 显存友好的 entropy 计算
#     # =====================================================
#     def _compute_sequence_entropy(self, generated_ids, scores):
#         """
#         H(a) = -1/L * sum_i log p_{i, y_i}
#         仅对生成 token 计算 log-prob，避免 vocab 级 softmax
#         """
#         if len(scores) == 0:
#             return [0.0] * generated_ids.shape[0]

#         batch_size = generated_ids.shape[0]
#         seq_len = min(len(scores), generated_ids.shape[1])

#         entropies = []

#         for b in range(batch_size):
#             total_nll = 0.0
#             valid_tokens = 0

#             for t in range(seq_len):
#                 token_id = generated_ids[b, t].item()

#                 if token_id in (
#                     self.tokenizer.pad_token_id,
#                     self.tokenizer.eos_token_id,
#                 ):
#                     continue

#                 log_probs = torch.log_softmax(scores[t][b], dim=-1)
#                 total_nll += -log_probs[token_id].item()
#                 valid_tokens += 1

#             entropies.append(
#                 total_nll / valid_tokens if valid_tokens > 0 else 0.0
#             )

#         return entropies
    
#     # =====================================================
#     # 标准 Exam
#     # =====================================================
#     def exam(self, question, solution, answer, question_idx):

#         results = []

#         total_batches = (len(question) + self.BATCH_SIZE - 1) // self.BATCH_SIZE
#         os.makedirs(os.path.dirname(self.OUTPUT_JSON_PATH), exist_ok=True)

#         for i in tqdm(
#             range(0, len(question), self.BATCH_SIZE),
#             total=total_batches,
#             desc="Inferencing",
#         ):
#             try:
#                 batch_questions = question[i:i + self.BATCH_SIZE]
#                 batch_solutions = solution[i:i + self.BATCH_SIZE]
#                 batch_answers = answer[i:i + self.BATCH_SIZE]
#                 batch_ids = question_idx[i:i + self.BATCH_SIZE]

#                 prompts = [
#                     self.tokenizer.apply_chat_template(
#                         [{"role": "user", "content": str(q)}],
#                         tokenize=False,
#                         add_generation_prompt=True,
#                     )
#                     for q in batch_questions
#                 ]

#                 inputs = self.tokenizer(
#                     prompts,
#                     return_tensors="pt",
#                     padding=True,
#                     truncation=True,
#                     max_length=self.MAX_SEQ_LENGTH,
#                 ).to(self.model.device)

#                 with torch.inference_mode():
#                     outputs = self.model.generate(
#                         **inputs,
#                         max_new_tokens=self.MAX_NEW_TOKENS,
#                         pad_token_id=self.tokenizer.pad_token_id,
#                         do_sample=True,
#                         temperature=0.1,
#                         top_p=0.9,
#                         use_cache=True,
#                         # return_dict_in_generate=True,
#                         # ❌ output_scores=True  ← 删除
#                     )
#                 input_len = inputs["input_ids"].shape[1]
#                 generated_ids = outputs[:, input_len:]

#                 decoded = self.tokenizer.batch_decode(
#                     generated_ids,
#                     skip_special_tokens=True,
#                 )

#                 del outputs
#                 torch.cuda.empty_cache()

#                 for q, a, ra, rs, idx in zip(
#                     batch_questions,
#                     decoded,
#                     batch_answers,
#                     batch_solutions,
#                     batch_ids,
#                 ):
#                     results.append({
#                         "question": q,
#                         "answer": a.strip(),
#                         "ref_answer": ra.strip(),
#                         "ref_solution": rs.strip(),
#                         "question_idx": idx,
#                         # ❌ "entropy": float(ent)  ← 删除
#                     })

#                 # ================== ⭐ batch 级保存 ==================
#                 with open(self.OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
#                     json.dump(results, f, ensure_ascii=False, indent=2)

#                 logger.info(
#                     f"Batch {i // self.BATCH_SIZE + 1}/{total_batches} saved "
#                     f"({len(results)} samples)"
#                 )

#             except Exception as e:
#                 logger.error(
#                     f"Batch {i // self.BATCH_SIZE + 1} failed: {e}"
#                 )
#                 torch.cuda.empty_cache()

#         logger.info(f"All done! Final results saved to {self.OUTPUT_JSON_PATH}")


# # =====================================================
# # Main
# # =====================================================
# if __name__ == "__main__":
#     from configs import GRPOConfig
#     from data_math import Math_500

#     current_file_path = os.path.abspath(__file__)
#     project_root = os.path.dirname(os.path.dirname(current_file_path))
#     project_root = os.path.dirname(os.path.dirname(project_root))

#     exam_file_path = os.path.join(
#         project_root, "CELPO", "configs", "celpo_train.yaml"
#     )

#     # 尝试加载配置和数据
#     try:
#         config = GRPOConfig.load_yaml(exam_file_path)
#         math_500 = Math_500(config)

#         test_dataset = math_500.get_test_data()
#         train_dataset = math_500.get_train_data()

#         question = test_dataset.problems + train_dataset.problems
#         solution = test_dataset.solutions + train_dataset.solutions
#         answer = test_dataset.answers + train_dataset.answers
#         question_idx = list(range(len(question)))

#         logger.info(f"Dataset size: {len(question)}")
        
#         LORA_PATH = os.path.join(project_root, "CELPO", "output", "hint_sft_XXXX_XXXX") 

#         take_exam = TakeExam(
#             model_path="/root/project/data/xrr/Qwen/Qwen2.5-Math-7B-Instruct",
#             use_lora=True,          # 开启 LoRA 加载
#             adapter_path=LORA_PATH  # 指定路径
#         )
        
#         # 使用 Roll-K 模式
#         # k=8, temperature=0.7
#         take_exam.exam_roll_k(
#             question, 
#             solution, 
#             answer, 
#             question_idx, 
#             k=8, 
#             temperature=0.7
#         )

#     except Exception as e:
#         logger.error(f"Initialization or execution failed: {e}")
import os
import json
import logging
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

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
        model_path: str = "/root/autodl-tmp/model/Qwen/Qwen2.5-Math-7B-Instruct",
        use_lora: bool = False,      
        adapter_path: str = None     
    ):
        # ================== Path (保持不变) ==================
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file_path))
        project_root = os.path.dirname(project_root)

        self.OUTPUT_JSON_PATH = os.path.join(
            project_root, "datasets", "exam", "exam.json"
        )
        self.OUTPUT_JSON_PATH_jsonl = os.path.join(
            project_root, "datasets", "exam", "exam.jsonl"
        )
        self.OUTPUT_JSON_PATH_ROLL = os.path.join(
            project_root, "datasets", "exam", "exam_roll.json"
        )
        self.OUTPUT_JSON_PATH_TEST = os.path.join(
            project_root, "datasets", "exam", "exam_test.json"
        )
        self.OUTPUT_JSON_PATH_EHC_TEST = os.path.join(
            project_root, "datasets", "exam", "exam_ehc_test.json"
        )

        # ================== Config ==================
        # 1. 设置全局种子 (影响 numpy/torch)
        set_seed(42)
        # 2. 保存类成员变量供 vLLM 使用 (这是复现的关键)
        self.seed = 42

        self.MAX_NEW_TOKENS = 2048
        self.MAX_MODEL_LEN = 4096 

        self.LOCAL_MODEL_PATH = model_path
        self.use_lora = use_lora
        self.adapter_path = adapter_path

        # ================== Load tokenizer ==================
        logger.info(f"Loading tokenizer from {self.LOCAL_MODEL_PATH}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.LOCAL_MODEL_PATH,
            trust_remote_code=True,
            use_fast=False, 
        )

        # ================== Initialize vLLM ==================
        logger.info(f"Initializing vLLM Engine from {self.LOCAL_MODEL_PATH}...")
        
        num_gpus = torch.cuda.device_count()
        logger.info(f"Detected {num_gpus} GPUs. Setting tensor_parallel_size={num_gpus}")

        # vLLM 初始化时也可以传入 seed，但这主要影响模型权重初始化的随机性
        # 对采样随机性的控制主要在 SamplingParams
        self.llm = LLM(
            model=self.LOCAL_MODEL_PATH,
            trust_remote_code=True,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=0.9,
            max_model_len=self.MAX_MODEL_LEN,
            enable_lora=use_lora,
            max_lora_rank=64,
            enforce_eager=False,
            seed=self.seed, # 建议在这里也加一个，虽然对 generation 影响有限
        )

        self.lora_request = None
        if use_lora and adapter_path and os.path.exists(adapter_path):
            logger.info(f"LoRA enabled. Adapter path: {adapter_path}")
            self.lora_request = LoRARequest("adapter", 1, adapter_path)
        elif use_lora:
            logger.warning(f"use_lora=True but path '{adapter_path}' is invalid.")

        logger.info("vLLM Engine loaded successfully.")

    # =====================================================
    # 单题推理 (适配 vLLM)
    # =====================================================
    def answer_single_question(self, question: str) -> str:
        try:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": str(question)}],
                tokenize=False,
                add_generation_prompt=True,
            )
            
            # ⭐ 修复点 1：传入 self.seed
            sampling_params = SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=self.MAX_NEW_TOKENS,
                stop_token_ids=[self.tokenizer.eos_token_id],
                seed=self.seed 
            )

            outputs = self.llm.generate(
                [prompt], 
                sampling_params, 
                lora_request=self.lora_request,
                use_tqdm=False 
            )
            return outputs[0].outputs[0].text.strip()

        except Exception as e:
            logger.error(f"Single question failed: {e}")
            return ""

    def exam_with_cal_entropy(self, question, solution, answer, question_idx):
        logger.warning("exam_with_cal_entropy is running without entropy calculation in vLLM mode to ensure speed.")
        self.exam(question, solution, answer, question_idx)

    # =====================================================
    # 批量 Roll K 次考试 (vLLM 重构核心加速版)
    # =====================================================
    def exam_roll_k(
        self,
        question,
        solution,
        answer,
        question_idx,
        k: int = 8,
        temperature: float = 0.7
    ):
        """
        使用 vLLM 进行 Roll K 推理。
        """
        logger.info(f"Starting vLLM Roll-K Exam: k={k}, temp={temperature}, total_questions={len(question)}")
        
        logger.info("Preparing prompts...")
        prompts = []
        for q in question:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": str(q)}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(text)

        # ⭐ 修复点 2：传入 self.seed
        # 即使 temperature > 0，固定 seed 也能保证每次 Roll 出来的 K 个结果是固定的
        sampling_params = SamplingParams(
            n=k,
            temperature=temperature,
            top_p=0.9,
            max_tokens=self.MAX_NEW_TOKENS,
            stop_token_ids=[self.tokenizer.eos_token_id],
            seed=self.seed
        )

        outputs = self.llm.generate(
            prompts, 
            sampling_params, 
            lora_request=self.lora_request
        )

        results = []
        logger.info("Processing outputs...")
        
        for i, output in enumerate(outputs):
            q_text = question[i]
            ref_ans = answer[i]
            ref_sol = solution[i]
            q_idx = question_idx[i]

            for sample in output.outputs:
                results.append({
                    "question": q_text,
                    "answer": sample.text.strip(),
                    "ref_answer": ref_ans.strip(),
                    "ref_solution": ref_sol.strip(),
                    "question_idx": q_idx,
                })

        os.makedirs(os.path.dirname(self.OUTPUT_JSON_PATH_ROLL), exist_ok=True)
        with open(self.OUTPUT_JSON_PATH_ROLL, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Roll-K Exam done! {len(results)} entries saved to {self.OUTPUT_JSON_PATH_ROLL}")
    
    # =====================================================
    # 标准 Exam (适配 vLLM)
    # =====================================================
    def exam(self, question, solution, answer, question_idx):
        logger.info(f"Starting vLLM Standard Exam: total_questions={len(question)}")
        
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": str(q)}],
                tokenize=False,
                add_generation_prompt=True,
            ) for q in question
        ]

        # ⭐ 修复点 3：传入 self.seed
        sampling_params = SamplingParams(
            n=1,
            temperature=0,
            top_p=0.9,
            max_tokens=self.MAX_NEW_TOKENS,
            stop_token_ids=[self.tokenizer.eos_token_id],
            seed=self.seed
        )

        outputs = self.llm.generate(prompts, sampling_params, lora_request=self.lora_request)

        results = []
        for i, output in enumerate(outputs):
            results.append({
                "question": question[i],
                "answer": output.outputs[0].text.strip(),
                "ref_answer": answer[i].strip(),
                "ref_solution": solution[i].strip(),
                "question_idx": question_idx[i],
            })

        os.makedirs(os.path.dirname(self.OUTPUT_JSON_PATH), exist_ok=True)
        with open(self.OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Standard Exam done! Saved to {self.OUTPUT_JSON_PATH}")


    def _compute_sequence_entropy(self, generated_ids, scores):
        return []


if __name__ == "__main__":
    from configs import GRPOConfig
    from data_math import Math_500

    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path))
    project_root = os.path.dirname(os.path.dirname(project_root))

    exam_file_path = os.path.join(
        project_root, "CELPO", "configs", "celpo_train.yaml"
    )

    try:
        config = GRPOConfig.load_yaml(exam_file_path)
        math_500 = Math_500(config)

        test_dataset = math_500.get_test_data()
        train_dataset = math_500.get_train_data()

        question = test_dataset.problems + train_dataset.problems
        solution = test_dataset.solutions + train_dataset.solutions
        answer = test_dataset.answers + train_dataset.answers
        question_idx = list(range(len(question)))

        logger.info(f"Dataset size: {len(question)}")
        
        LORA_PATH = os.path.join(project_root, "CELPO", "output", "hint_sft_XXXX_XXXX") 

        take_exam = TakeExam(
            model_path="/root/project/data/xrr/Qwen/Qwen2.5-Math-7B-Instruct",
            use_lora=True,          
            adapter_path=LORA_PATH  
        )
        
        take_exam.exam_roll_k(
            question, 
            solution, 
            answer, 
            question_idx, 
            k=8, 
            temperature=0.7
        )

    except Exception as e:
        logger.error(f"Initialization or execution failed: {e}")
        raise e

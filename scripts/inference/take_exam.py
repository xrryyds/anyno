import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
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

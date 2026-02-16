import os
# ==================== 环境变量设置 ====================
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # CUDA 同步
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 确定性 CUBLAS
os.environ["PYTHONHASHSEED"] = "42"  # Python 哈希种子

import json
import logging
import random
import numpy as np
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

# Qwen-Math 的标准 System Prompt，这对激发数学能力至关重要
SYSTEM_PROMPT = "Please reason step by step and put your final answer within \\boxed{}."

# =====================================================
# 全局种子设置函数
# =====================================================
def set_all_seeds(seed=42):
    """确保所有随机性来源都使用相同种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # PyTorch 确定性设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 使用确定性算法（warn_only=True 避免某些操作不支持时报错）
    torch.use_deterministic_algorithms(True, warn_only=True)

# 初始化全局种子
set_all_seeds(42)
set_seed(42)  # Transformers 的种子

class TakeExam:
    def __init__(
        self,
        model_path: str = "/root/autodl-tmp/model/Qwen/Qwen2.5-Math-7B-Instruct",
        use_lora: bool = False,      
        adapter_path: str = None     
    ):
        # ================== Path ==================
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file_path))
        project_root = os.path.dirname(project_root)

        self.OUTPUT_JSON_PATH = os.path.join(
            project_root, "datasets", "exam", "exam.json"
        )
        self.OUTPUT_JSON_PATH_ROLL = os.path.join(
            project_root, "datasets", "exam", "exam_roll.json"
        )

        # ================== Config ==================
        self.seed = 42
        set_all_seeds(self.seed)  # 再次确保种子设置

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

        # 准备 Qwen 特有的 Stop Tokens ID
        # 151645: <|im_end|>, 151643: <|endoftext|>
        self.stop_token_ids = [self.tokenizer.eos_token_id, 151643, 151645]

        # ================== Initialize vLLM ==================
        logger.info(f"Initializing vLLM Engine from {self.LOCAL_MODEL_PATH}...")
        
        # 单 GPU 设置，确保完全确定性
        logger.info("Using single GPU (tensor_parallel_size=1) for deterministic results")

        self.llm = LLM(
            model=self.LOCAL_MODEL_PATH,
            trust_remote_code=True,
            tensor_parallel_size=1,  # ✅ 单 GPU，确保确定性
            gpu_memory_utilization=0.9,
            max_model_len=self.MAX_MODEL_LEN,
            enable_lora=use_lora,
            max_lora_rank=64,
            enforce_eager=True,  # ✅ 强制 eager 模式，避免编译优化带来的不确定性
            seed=self.seed,
            dtype="bfloat16"  # ✅ 明确指定精度，避免自动选择的不确定性
        )

        self.lora_request = None
        if use_lora and adapter_path and os.path.exists(adapter_path):
            logger.info(f"LoRA enabled. Adapter path: {adapter_path}")
            self.lora_request = LoRARequest("adapter", 1, adapter_path)
        elif use_lora:
            logger.warning(f"use_lora=True but path '{adapter_path}' is invalid.")

        logger.info("vLLM Engine loaded successfully.")
        
        # vLLM 初始化后再次设置种子，确保不被污染
        set_all_seeds(self.seed)

    def _build_prompts(self, questions):
        """
        统一构建带有 System Prompt 的输入
        """
        prompts = []
        for q in questions:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(q)}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(text)
        return prompts

    # =====================================================
    # 单题推理
    # =====================================================
    def answer_single_question(self, question: str) -> str:
        try:
            # 每次推理前重置种子，确保一致性
            set_all_seeds(self.seed)
            
            prompts = self._build_prompts([question])
            
            sampling_params = SamplingParams(
                temperature=0.0,  # ✅ 贪心解码
                top_p=1.0,        # ✅ temperature=0 时必须为 1.0
                max_tokens=self.MAX_NEW_TOKENS,
                stop_token_ids=self.stop_token_ids,
                seed=self.seed 
            )

            outputs = self.llm.generate(
                prompts, 
                sampling_params, 
                lora_request=self.lora_request,
                use_tqdm=False 
            )
            return outputs[0].outputs[0].text.strip()

        except Exception as e:
            logger.error(f"Single question failed: {e}")
            return ""

    def exam_with_cal_entropy(self, question, solution, answer, question_idx):
        logger.warning("exam_with_cal_entropy is running standard exam (entropy skipped for speed).")
        self.exam(question, solution, answer, question_idx)

    # =====================================================
    # 批量 Roll K 次考试
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
        注意：如果用于 Pass@1 测试，请在调用时传入 k=1, temperature=0.0
        """
        logger.info(f"Starting vLLM Roll-K Exam: k={k}, temp={temperature}, total_questions={len(question)}")
        
        # 推理前重置种子
        set_all_seeds(self.seed)
        
        prompts = self._build_prompts(question)

        sampling_params = SamplingParams(
            n=k,
            temperature=temperature,
            top_p=1.0 if temperature == 0 else 0.9,  # temperature=0 时 top_p 必须为 1
            max_tokens=self.MAX_NEW_TOKENS,
            stop_token_ids=self.stop_token_ids,
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
    # 标准 Exam (Pass@1)
    # =====================================================
    def exam(self, question, solution, answer, question_idx):
        logger.info(f"Starting vLLM Standard Exam (Greedy): total_questions={len(question)}")
        
        # 推理前重置种子
        set_all_seeds(self.seed)
        
        prompts = self._build_prompts(question)

        # ✅ 强制 greedy search，确保确定性输出
        sampling_params = SamplingParams(
            n=1,
            temperature=0.0,  # ✅ 必须为 0
            top_p=1.0,        # ✅ 必须为 1.0
            max_tokens=self.MAX_NEW_TOKENS,
            stop_token_ids=self.stop_token_ids,
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


# =====================================================
# 一致性测试函数
# =====================================================
def test_consistency(take_exam, question, n_runs=3):
    """测试多次运行结果是否完全一致"""
    logger.info(f"Testing consistency with {n_runs} runs...")
    results = []
    
    for i in range(n_runs):
        # 每次运行前重置种子
        set_all_seeds(42)
        
        output = take_exam.answer_single_question(question)
        results.append(output)
        logger.info(f"Run {i+1} output (first 100 chars): {output[:100]}...")
    
    # 检查一致性
    unique_results = set(results)
    if len(unique_results) == 1:
        logger.info("✅ 结果完全一致！输出稳定可靠。")
        return True
    else:
        logger.error("❌ 结果不一致！")
        for i, r in enumerate(results):
            logger.error(f"  Run {i+1}: {r[:200]}")
        return False

    # =====================================================
    # 新增：带 Hint 的 Prefix-Forcing 考试 (Q + H -> A)
    # =====================================================
def exam_with_hints(self, question, solution, answer, question_idx, hints):
    """
    使用 Prefix-Forcing 模式进行推理。
    
    原理：
    构造 Prompt = System + User(Question) + Assistant(# known:\n{Hint}\n)
    强制模型认为它已经输出了 Hint，从而接着生成 Answer。
    这保证了推理时的上下文状态与训练时完全一致。
    """
    logger.info(f"Starting vLLM Exam (Prefix-Forcing): total_questions={len(question)}")
    
    # 1. 确保随机种子一致
    set_all_seeds(self.seed)
    
    # 2. 构建带有预填充(Pre-fill)的 Prompts
    prompts = []
    # 必须严格匹配训练代码中的 GEN_HINTS_WIH_ANSWER 格式: "# known:\n{hints}\n{answer}"
    # 因此前缀应该是 "# known:\n{hint}\n"
    HINT_PREFIX_TEMPLATE = "# known:\n{hint}\n"

    for i, q in enumerate(question):
        # A. 构建基础 ChatML (System + User)
        # add_generation_prompt=True 会在最后加上 <|im_start|>assistant\n
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(q)}
        ]
        base_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # B. 拼接 Hint 到 Assistant 的开头
        current_hint = hints[i] if i < len(hints) else ""
        
        if current_hint and current_hint.strip() != "":
            # 拼接逻辑：Assistant标签 + Hint前缀
            prefix_text = HINT_PREFIX_TEMPLATE.format(hint=current_hint)
            full_prompt = base_prompt + prefix_text
        else:
            # 如果没有 Hint，就按普通模式推理
            full_prompt = base_prompt
            
        prompts.append(full_prompt)

    # 3. 设置采样参数 (Greedy Search)
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=self.MAX_NEW_TOKENS,
        stop_token_ids=self.stop_token_ids,
        seed=self.seed
    )

    # 4. 执行推理
    # vLLM 会自动识别 prompt 中的 prefix，并从 prefix 后面继续生成
    outputs = self.llm.generate(prompts, sampling_params, lora_request=self.lora_request)

    # 5. 处理结果
    results = []
    for i, output in enumerate(outputs):
        # generated_text 仅包含模型新生成的 Answer 部分 (不包含 Hint)
        generated_answer = output.outputs[0].text.strip()
        
        current_hint = hints[i] if i < len(hints) else ""
        
        # 为了数据完整性，我们模拟拼接出完整的 Assistant 输出
        if current_hint:
            full_response = HINT_PREFIX_TEMPLATE.format(hint=current_hint) + generated_answer
        else:
            full_response = generated_answer

        results.append({
            "question": question[i],
            "provided_hint": current_hint,      # 记录输入的 Hint
            "model_answer": generated_answer,   # 模型生成的 Answer
            "full_response": full_response,     # 完整回复 (Hint + Answer)，可直接用于后续 SFT 数据构造
            "ref_answer": answer[i].strip(),
            "ref_solution": solution[i].strip(),
            "question_idx": question_idx[i],
        })

    # 6. 保存结果
    os.makedirs(os.path.dirname(self.OUTPUT_JSON_PATH), exist_ok=True)
    with open(self.OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Prefix-Forcing Exam done! Saved to {self.OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    # 假设你也有这些文件
    try:
        from configs import GRPOConfig
        from data_math import Math_500
    except ImportError:
        # Fallback for standalone testing
        class MockDataset:
            def __init__(self):
                self.problems = ["What is 1+1?"]
                self.solutions = ["1+1=2"]
                self.answers = ["2"]
        logger.warning("Could not import configs/data_math. Using mock data.")
        Math_500 = lambda x: MockDataset()

    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path))
    project_root = os.path.dirname(os.path.dirname(project_root))

    exam_file_path = os.path.join(project_root, "CELPO", "configs", "celpo_train.yaml")
    
    # 修改为你的模型实际路径
    MODEL_PATH = "/root/autodl-tmp/CELPO/model/OREAL/OREAL-7B"

    try:
        # --- 测试用 Mock 数据 ---
        question = ["Find the value of x if 2x + 3 = 7.", "Calculate 15 * 15."]
        solution = ["2x=4 -> x=2", "225"]
        answer = ["2", "225"]
        question_idx = list(range(len(question)))

        logger.info(f"Dataset size: {len(question)}")
        
        take_exam = TakeExam(
            model_path=MODEL_PATH,
            use_lora=False,
            adapter_path=None  
        )
        
        # ✅ 先进行一致性测试
        logger.info("=" * 60)
        logger.info("Running consistency test...")
        logger.info("=" * 60)
        test_consistency(take_exam, question[0], n_runs=5)
        
        # ✅ 然后运行正式考试
        logger.info("=" * 60)
        logger.info("Running standard exam...")
        logger.info("=" * 60)
        take_exam.exam(
            question, 
            solution, 
            answer, 
            question_idx
        )

    except Exception as e:
        logger.error(f"Initialization or execution failed: {e}")
        raise e

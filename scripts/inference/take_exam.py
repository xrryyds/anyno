import os
import json
import logging
import random
import numpy as np
import torch
import multiprocessing as mp
from typing import List, Dict, Any, Sequence, Optional
from transformers import AutoTokenizer, set_seed
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "42"

# =====================================================
# Logger
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MAX_SEQ_LENGTH = 2048

SYSTEM_PROMPT = "Please reason step by step and put your final answer within \\boxed{}."

# =====================================================
# =====================================================
def set_all_seeds(seed=42):
    """"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

set_all_seeds(42)
set_seed(42)

class TakeExam:
    def __init__(
        self,
        model_path: str = "/root/autodl-tmp/model/Qwen/Qwen2.5-Math-7B-Instruct",
        use_lora: bool = False,
        adapter_path: str = None,
        max_seq_length: Optional[int] = None,
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
        set_all_seeds(self.seed)

        self.max_seq_length = max_seq_length or MAX_SEQ_LENGTH
        self.MAX_NEW_TOKENS = self.max_seq_length
        self.MAX_MODEL_LEN = self.max_seq_length + 1024

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

        # 151645: <|im_end|>, 151643: <|endoftext|>
        self.stop_token_ids = [self.tokenizer.eos_token_id, 151643, 151645]

        # ================== Initialize vLLM ==================
        logger.info(f"Initializing vLLM Engine from {self.LOCAL_MODEL_PATH}...")
        
        logger.info("Using single GPU (tensor_parallel_size=1) for deterministic results")

        self.llm = LLM(
            model=self.LOCAL_MODEL_PATH,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=self.MAX_MODEL_LEN,
            enable_lora=use_lora,
            max_lora_rank=64,
            enforce_eager=True,
            seed=self.seed,
            dtype="bfloat16"
        )

        self.lora_request = None
        if use_lora and adapter_path and os.path.exists(adapter_path):
            logger.info(f"LoRA enabled. Adapter path: {adapter_path}")
            self.lora_request = LoRARequest("adapter", 1, adapter_path)
        elif use_lora:
            logger.warning(f"use_lora=True but path '{adapter_path}' is invalid.")

        logger.info("vLLM Engine loaded successfully.")
        
        set_all_seeds(self.seed)

    def _build_prompts(self, questions):
        """
         System Prompt 
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
    # =====================================================
    def compute_answer_vocab_loss_vector(self, question, answer):
        """Compute per-vocab average loss using vLLM prompt_logprobs (no second model needed).

        Args:
            question: List[str] of questions.
            answer: List[str] of reference answers, same length as question.

        Returns:
            torch.Tensor of shape [vocab_size] with average loss per token id.
        """
        if len(question) != len(answer):
            raise ValueError(f"question and answer must have same length, got {len(question)} and {len(answer)}")

        vocab_size = self.tokenizer.vocab_size

        # Build full prompts (prefix + answer) and record prefix lengths
        prompts = []
        answer_starts = []
        for q, a in zip(question, answer):
            messages_prefix = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(q)},
            ]
            prefix_text = self.tokenizer.apply_chat_template(
                messages_prefix, tokenize=False, add_generation_prompt=True,
            )
            full_text = prefix_text + str(a)

            prefix_len = len(self.tokenizer(
                prefix_text, add_special_tokens=False, truncation=True,
                max_length=self.MAX_MODEL_LEN,
            )["input_ids"])
            full_len = len(self.tokenizer(
                full_text, add_special_tokens=False, truncation=True,
                max_length=self.MAX_MODEL_LEN,
            )["input_ids"])

            if full_len <= prefix_len:
                continue

            prompts.append(full_text)
            answer_starts.append(prefix_len)

        if not prompts:
            return torch.zeros(vocab_size)

        # Use vLLM with prompt_logprobs=1 and max_tokens=1 to get per-token log-probs
        sampling_params = SamplingParams(
            max_tokens=1,
            prompt_logprobs=1,
            temperature=0.0,
            seed=self.seed,
        )
        outputs = self.llm.generate(prompts, sampling_params=sampling_params)

        loss_sum = torch.zeros(vocab_size, dtype=torch.float32)
        count = torch.zeros(vocab_size, dtype=torch.float32)

        for out, answer_start in zip(outputs, answer_starts):
            prompt_logprobs = out.prompt_logprobs  # list of None | dict, length = num_prompt_tokens
            if prompt_logprobs is None:
                continue
            # prompt_logprobs[0] is None (no logprob for first token)
            # positions [answer_start .. seq_len-1] are answer tokens
            for pos in range(answer_start, len(prompt_logprobs)):
                lp_dict = prompt_logprobs[pos]
                if lp_dict is None:
                    continue
                # token id at this position
                token_id = out.prompt_token_ids[pos]
                if token_id >= vocab_size:
                    continue
                lp = lp_dict.get(token_id)
                if lp is None:
                    # take the first available entry (the token's own logprob)
                    lp = next(iter(lp_dict.values()))
                # logprob -> NLL loss
                log_prob = lp.logprob if hasattr(lp, "logprob") else float(lp)
                loss_sum[token_id] += -log_prob
                count[token_id] += 1

        avg_loss_per_vocab = loss_sum / (count + 1e-8)
        return avg_loss_per_vocab

    # =====================================================
    # =====================================================
    def answer_single_question(self, question: str) -> str:
        try:
            set_all_seeds(self.seed)
            
            prompts = self._build_prompts([question])
            
            sampling_params = SamplingParams(
                temperature=0.0,
                top_p=1.0,
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
         vLLM  Roll K 
         Pass@1  k=1, temperature=0.0
        """
        logger.info(f"Starting vLLM Roll-K Exam: k={k}, temp={temperature}, total_questions={len(question)}")
        
        set_all_seeds(self.seed)
        
        prompts = self._build_prompts(question)

        sampling_params = SamplingParams(
            n=k,
            temperature=temperature,
            top_p=1.0 if temperature == 0 else 0.9,
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
    # =====================================================
    def _exam_core(self, question, solution, answer, question_idx):
        """Core implementation of exam() that returns in-memory results.

        This helper performs pure compute using vLLM and does not touch
        the filesystem. Public APIs like exam() and exam_multi_gpu are
        responsible for any file I/O.
        """
        logger.info(f"Running exam core on {len(question)} questions.")

        prompts = self._build_prompts(question)

        sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.MAX_NEW_TOKENS,
            stop_token_ids=self.stop_token_ids,
            seed=self.seed,
        )

        outputs = self.llm.generate(
            prompts,
            sampling_params,
            lora_request=self.lora_request,
        )

        results = []
        for i, output in enumerate(outputs):
            results.append({
                "question": question[i],
                "answer": output.outputs[0].text.strip(),
                "ref_answer": answer[i].strip(),
                "ref_solution": solution[i].strip(),
                "question_idx": question_idx[i],
            })

        return results

    def exam(self, question, solution, answer, question_idx):
        logger.info(f"Starting vLLM Standard Exam (Greedy): total_questions={len(question)}")

        set_all_seeds(self.seed)

        results = self._exam_core(question, solution, answer, question_idx)

        os.makedirs(os.path.dirname(self.OUTPUT_JSON_PATH), exist_ok=True)
        with open(self.OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Standard Exam done! Saved to {self.OUTPUT_JSON_PATH}")

    def exam_multi_gpu(
        self,
        question,
        solution,
        answer,
        question_idx,
        device_ids=None,
        num_workers=None,
        write_output=True,
    ):
        """Data-parallel multi-GPU variant of exam().

        This method does not change the existing single-GPU behavior; it is an
        opt-in helper that shards the input questions across multiple GPUs,
        runs _exam_core on each shard in a separate process, and then merges
        the results.

        Args:
            question, solution, answer, question_idx: Lists aligned by index.
            device_ids: Optional sequence of CUDA device indices to use.
                If None, uses all available GPUs.
            num_workers: Optional number of worker processes to spawn.
                Defaults to min(len(device_ids), len(question)).
            write_output: If True, write merged results to OUTPUT_JSON_PATH.

        Returns:
            List[dict]: merged results in the same order as the input lists.
        """
        total = len(question)
        if total == 0:
            logger.warning("exam_multi_gpu called with empty question list.")
            if write_output:
                os.makedirs(os.path.dirname(self.OUTPUT_JSON_PATH), exist_ok=True)
                with open(self.OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
            return []

        if device_ids is None:
            if not torch.cuda.is_available():
                logger.warning(
                    "No CUDA devices available; falling back to single-GPU behavior."
                )
                # Single-GPU fallback: behave like exam()
                set_all_seeds(self.seed)
                results = self._exam_core(question, solution, answer, question_idx)
                if write_output:
                    os.makedirs(os.path.dirname(self.OUTPUT_JSON_PATH), exist_ok=True)
                    with open(self.OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                return results
            n_gpus = torch.cuda.device_count()
            device_ids = list(range(n_gpus))

        if isinstance(device_ids, int):
            device_ids = [device_ids]

        if not device_ids:
            logger.warning(
                "Empty device_ids passed to exam_multi_gpu; falling back to single-GPU behavior."
            )
            set_all_seeds(self.seed)
            results = self._exam_core(question, solution, answer, question_idx)
            if write_output:
                os.makedirs(os.path.dirname(self.OUTPUT_JSON_PATH), exist_ok=True)
                with open(self.OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            return results

        if num_workers is None:
            num_workers = min(len(device_ids), total)
        else:
            num_workers = min(num_workers, len(device_ids), total)

        if num_workers <= 1:
            logger.info(
                "exam_multi_gpu called with a single worker; using single-GPU behavior."
            )
            set_all_seeds(self.seed)
            results = self._exam_core(question, solution, answer, question_idx)
            if write_output:
                os.makedirs(os.path.dirname(self.OUTPUT_JSON_PATH), exist_ok=True)
                with open(self.OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            return results

        logger.info(
            f"Running exam_multi_gpu with {num_workers} workers on devices {device_ids[:num_workers]}"
        )

        # Build contiguous index shards to preserve original ordering
        indices = list(range(total))
        shard_size = (total + num_workers - 1) // num_workers
        index_shards = [
            indices[i * shard_size : min((i + 1) * shard_size, total)]
            for i in range(num_workers)
            if i * shard_size < total
        ]

        args_list = []
        for local_rank, (device_id, idx_shard) in enumerate(
            zip(device_ids, index_shards)
        ):
            if not idx_shard:
                continue
            q_shard = [question[i] for i in idx_shard]
            s_shard = [solution[i] for i in idx_shard]
            a_shard = [answer[i] for i in idx_shard]
            qi_shard = [question_idx[i] for i in idx_shard]
            args_list.append(
                (
                    local_rank,
                    device_id,
                    self.LOCAL_MODEL_PATH,
                    self.use_lora,
                    self.adapter_path,
                    self.max_seq_length,
                    self.seed,
                    q_shard,
                    s_shard,
                    a_shard,
                    qi_shard,
                )
            )

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(args_list)) as pool:
            shard_results = pool.map(_run_exam_shard_worker, args_list)

        merged_results = []
        for part in shard_results:
            merged_results.extend(part)

        # Because we use contiguous shards and _exam_core preserves input
        # ordering within each shard, merged_results is already aligned with
        # the original question order. No extra sorting is required.

        if write_output:
            os.makedirs(os.path.dirname(self.OUTPUT_JSON_PATH), exist_ok=True)
            with open(self.OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(merged_results, f, ensure_ascii=False, indent=2)
            logger.info(
                "Multi-GPU Standard Exam done! Saved to %s", self.OUTPUT_JSON_PATH
            )

        return merged_results

    # =====================================================
    # =====================================================
    def exam_with_hints(self, question, solution, answer, question_idx, hints):
        """
         Prefix-Forcing 
        
        
         Prompt = System + User(Question) + Assistant({Hint})
         Hint Answer
        """
        logger.info(f"Starting vLLM Exam (Prefix-Forcing): total_questions={len(question)}")
        
        set_all_seeds(self.seed)
        
        prompts = []
        HINT_PREFIX_TEMPLATE = "{hint}"

        for i, q in enumerate(question):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(q)}
            ]
            base_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            current_hint = hints[i] if i < len(hints) else ""
            
            if current_hint and current_hint.strip() != "":
                prefix_text = HINT_PREFIX_TEMPLATE.format(hint=current_hint)
                full_prompt = base_prompt + prefix_text
            else:
                full_prompt = base_prompt
                
            prompts.append(full_prompt)

        sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.MAX_NEW_TOKENS,
            stop_token_ids=self.stop_token_ids,
            seed=self.seed
        )

        outputs = self.llm.generate(prompts, sampling_params, lora_request=self.lora_request)

        results = []
        for i, output in enumerate(outputs):
            generated_answer = output.outputs[0].text.strip()
            
            current_hint = hints[i] if i < len(hints) else ""
            
            if current_hint:
                full_response = HINT_PREFIX_TEMPLATE.format(hint=current_hint) + generated_answer
            else:
                full_response = generated_answer

            results.append({
                "question": question[i],
                "answer": generated_answer,       
                "provided_hint": current_hint,
                "full_response": full_response,
                "ref_answer": answer[i].strip(),
                "ref_solution": solution[i].strip(),
                "question_idx": question_idx[i],
            })

        os.makedirs(os.path.dirname(self.OUTPUT_JSON_PATH), exist_ok=True)
        with open(self.OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Prefix-Forcing Exam done! Saved to {self.OUTPUT_JSON_PATH}")


    # =====================================================
    # =====================================================
    def exam_roll_k_with_hints(
        self,
        question,
        solution,
        answer,
        question_idx,
        hints,
        k: int = 8,
        temperature: float = 0.7
    ):
        """
         Prefix-Forcing  Roll-K 
        
        
        1.  Hint 
        2.  K  (Sampling)
        """
        logger.info(f"Starting vLLM Roll-K Exam with Hints: k={k}, temp={temperature}, total_questions={len(question)}")
        
        set_all_seeds(self.seed)
        
        prompts = []
        HINT_PREFIX_TEMPLATE = "{hint}"

        for i, q in enumerate(question):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(q)}
            ]
            base_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            current_hint = hints[i] if i < len(hints) else ""
            
            if current_hint and current_hint.strip() != "":
                prefix_text = HINT_PREFIX_TEMPLATE.format(hint=current_hint)
                full_prompt = base_prompt + prefix_text
            else:
                full_prompt = base_prompt
            
            prompts.append(full_prompt)

        sampling_params = SamplingParams(
            n=k,
            temperature=temperature,
            top_p=1.0 if temperature == 0 else 0.9,
            max_tokens=self.MAX_NEW_TOKENS,
            stop_token_ids=self.stop_token_ids,
            seed=self.seed
        )

        outputs = self.llm.generate(prompts, sampling_params, lora_request=self.lora_request)

        results = []
        logger.info("Processing Roll-K outputs...")

        for i, output in enumerate(outputs):
            q_text = question[i]
            ref_ans = answer[i]
            ref_sol = solution[i]
            q_idx = question_idx[i]
            current_hint = hints[i] if i < len(hints) else ""

            for sample in output.outputs:
                generated_answer = sample.text.strip()
                
                if current_hint:
                    full_response = HINT_PREFIX_TEMPLATE.format(hint=current_hint) + generated_answer
                else:
                    full_response = generated_answer

                results.append({
                    "question": q_text,
                    "answer": generated_answer,
                    "provided_hint": current_hint,
                    "full_response": full_response,
                    "ref_answer": ref_ans.strip(),
                    "ref_solution": ref_sol.strip(),
                    "question_idx": q_idx,
                })

        os.makedirs(os.path.dirname(self.OUTPUT_JSON_PATH_ROLL), exist_ok=True)
        with open(self.OUTPUT_JSON_PATH_ROLL, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Roll-K Exam with Hints done! {len(results)} entries saved to {self.OUTPUT_JSON_PATH_ROLL}")


def _run_exam_shard_worker(args):
    """Worker function for exam_multi_gpu.

    Defined at module scope so it can be pickled by multiprocessing.
    It reconstructs a TakeExam instance on a specific GPU and runs
    _exam_core() on the provided shard.
    """
    (
        local_rank,
        device_id,
        model_path,
        use_lora,
        adapter_path,
        max_seq_length,
        seed,
        question_shard,
        solution_shard,
        answer_shard,
        question_idx_shard,
    ) = args

    # Pin this worker process to a single CUDA device.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    logger.info(
        "[exam worker %d] Using CUDA device %s for %d questions.",
        local_rank,
        device_id,
        len(question_shard),
    )

    worker = TakeExam(
        model_path=model_path,
        use_lora=use_lora,
        adapter_path=adapter_path,
        max_seq_length=max_seq_length,
    )
    worker.seed = seed
    set_all_seeds(worker.seed)

    return worker._exam_core(
        question_shard,
        solution_shard,
        answer_shard,
        question_idx_shard,
    )


# =====================================================
# =====================================================
def test_consistency(take_exam, question, n_runs=3):
    """"""
    logger.info(f"Testing consistency with {n_runs} runs...")
    results = []

    for i in range(n_runs):
        set_all_seeds(42)

        output = take_exam.answer_single_question(question)
        results.append(output)
        logger.info(f"Run {i+1} output (first 100 chars): {output[:100]}...")

    unique_results = set(results)
    if len(unique_results) == 1:
        logger.info("✅ ")
        return True
    else:
        logger.error("❌ ")
        for i, r in enumerate(results):
            logger.error(f"  Run {i+1}: {r[:200]}")
        return False


if __name__ == "__main__":
    MODEL_PATH = "/root/autodl-tmp/CELPO/model/OREAL/OREAL-7B"

    try:
        question = ["Find the value of x if 2x + 3 = 7.", "Calculate 15 * 15."]
        solution = ["2x=4 -> x=2", "225"]
        answer = ["2", "225"]
        hints = ["First subtract 3 from both sides.", "Multiply 10 by 15 first then add 5 times 15."] 
        question_idx = list(range(len(question)))

        logger.info(f"Dataset size: {len(question)}")
        
        take_exam = TakeExam(
            model_path=MODEL_PATH,
            use_lora=False,
            adapter_path=None  
        )
        
        logger.info("=" * 60)
        logger.info("Running consistency test...")
        logger.info("=" * 60)
        test_consistency(take_exam, question[0], n_runs=3)
        
        logger.info("=" * 60)
        logger.info("Running Exam with Hints...")
        logger.info("=" * 60)
        take_exam.exam_with_hints(
            question, 
            solution, 
            answer, 
            question_idx,
            hints
        )

    except Exception as e:
        logger.error(f"Initialization or execution failed: {e}")
        raise e

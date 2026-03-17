import os
from utils import FileIOUtils, extract_hints ,extract_boxed_content, normalize_answer
from openai import OpenAI, RateLimitError, APIError
from prompt.prompts import TEACHER_CORRECT_PROMPT, OREAL_CORRECT_PROMPT
import time
import torch
import multiprocessing as mp
from typing import List, Sequence, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional vLLM import; may be unavailable in some environments.
try:
    from vllm import LLM, SamplingParams
    _VLLM_IMPORT_ERROR: Optional[Exception] = None
except Exception as e:  # pragma: no cover - import-time environment specific
    LLM = None  # type: ignore[assignment]
    SamplingParams = None  # type: ignore[assignment]
    _VLLM_IMPORT_ERROR = e

# Backend selection for teacher_hints_self.
# Allowed values:
#   "auto" (default): use vLLM when available, otherwise fall back to transformers.
#   "vllm": require vLLM; raise if unavailable.
#   "transformers": force transformers backend.
TEACHER_HINTS_BACKEND: str = os.environ.get("TEACHER_HINTS_BACKEND", "auto").lower()

# How often to checkpoint partial hints to disk.
# Can be overridden via the TEACHER_HINTS_CHECKPOINT_INTERVAL environment variable.
try:
    TEACHER_HINTS_CHECKPOINT_INTERVAL: int = int(
        os.environ.get("TEACHER_HINTS_CHECKPOINT_INTERVAL", "10")
    )
except ValueError:
    TEACHER_HINTS_CHECKPOINT_INTERVAL = 10


def _is_vllm_available() -> bool:
    """Return True if vLLM is importable and CUDA is available.

    This helper is conservative: it only checks for a successful import and
    presence of a CUDA device. More detailed configuration errors will be
    surfaced when initializing the vLLM engine.
    """
    if LLM is None or SamplingParams is None:
        return False
    if not torch.cuda.is_available():
        return False
    return True

base_url = "https://wanqing-api.corp.kuaishou.com/api/agent/v1/apps"
api_key = "k1y21hll8l0eurf7t3dg4enb56g0hhjjszf4"


class TeacherCorrecter:
    def __init__(self):
        self.file = FileIOUtils()
        self.acc = 0
        self.err_count = 0
        self.toolong_count = 0
        self.acc_count = 0

    def teacher_hints(self) -> bool:
        print("Starting teacher hinting...")
        print("load mistakes...")
        self.file.load_mistakes()
        (
            m_question_idx,
            m_question,
            m_answer,
            m_ref_answer,
            m_ref_solution,
            m_entropy,
        ) = self.file.parse_data(self.file.mistakes)
        print("mistakes size:", len(m_question))

        h_question = []
        h_hints = []
        h_ref_solution = []
        h_ref_answer = []
        h_question_idx = []

        print(f"generating hints({len(m_question)})...")
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        print("----- standard request -----")
        for idx in range(len(m_question)):
            prompt = TEACHER_CORRECT_PROMPT.format(
                problem=m_question[idx],
                student_answer=m_answer[idx],
                ref_solution=m_ref_solution[idx],
            )
            response = None
            while True:
                try:
                    completion = client.chat.completions.create(
                        model="app-xkp5mg-1764855493646070178",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful assistant who good at math",
                            },
                            {"role": "user", "content": prompt},
                        ],
                    )
                    response = completion.choices[0].message.content
                    break

                except RateLimitError:
                    print(
                        f"Rate limit reached at idx {idx}. Sleeping for 20 seconds..."
                    )
                    time.sleep(20)
                except Exception as e:
                    print(f"An unexpected error occurred at idx {idx}: {e}")
                    raise e

            hints = extract_hints(response)

            # 将当前结果加入列表
            h_question_idx.append(m_question_idx[idx])
            h_question.append(m_question[idx])
            h_hints.append(hints)
            h_ref_solution.append(m_ref_solution[idx])
            h_ref_answer.append(m_ref_answer[idx])

            # -----------------------------------------------------------
            # [修改部分] 每隔 10 条保存一次
            # -----------------------------------------------------------
            if (idx + 1) % 10 == 0:
                print(f"Auto-saving checkpoint at count {idx + 1}...")
                # 注意：m_answer 和 m_entropy 是原始完整列表，
                # 这里使用 [:idx+1] 进行切片，确保传入的长度与当前 h_question 一致
                self.file.save_hints(
                    h_question,
                    h_hints,
                    h_ref_solution,
                    h_ref_answer,
                    h_question_idx,
                    m_answer[: idx + 1],
                    m_entropy[: idx + 1],
                )
            # -----------------------------------------------------------

        print("saving final hints...")
        # 循环结束后保存完整数据（防止总数不是10的倍数导致最后几条没存）
        self.file.save_hints(
            h_question,
            h_hints,
            h_ref_solution,
            h_ref_answer,
            h_question_idx,
            m_answer,
            m_entropy,
        )
        return True

    def _generate_hints_for_indices(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        indices: Sequence[int],
        m_question_idx: Sequence[int],
        m_question: Sequence[str],
        m_answer: Sequence[str],
        m_ref_solution: Sequence[str],
        m_ref_answer: Sequence[str],
        worker_tag: Optional[str] = None,
        log_every: int = 5,
    ) -> Tuple[List[int], List[str], List[str], List[str], List[str]]:
        """Core local-model hint generation over a subset of indices.

        This helper performs pure compute and returns results in memory
        without any file I/O, so it can be reused by both the sequential
        and multi-GPU pipelines.
        """
        h_question_idx: List[int] = []
        h_question: List[str] = []
        h_hints: List[str] = []
        h_ref_solution: List[str] = []
        h_ref_answer: List[str] = []

        if worker_tag is None:
            worker_tag = "[teacher_hints_self]"

        total = len(indices)

        for local_idx, idx in enumerate(indices):
            prompt = TEACHER_CORRECT_PROMPT.format(
                problem=m_question[idx],
                student_answer=m_answer[idx],
                ref_solution=m_ref_solution[idx],
            )
            q_idx = m_question_idx[idx]
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant who good at math",
                },
                {"role": "user", "content": prompt},
            ]

            try:
                input_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = tokenizer(
                    input_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096,
                ).to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=True,
                    )

                generated_ids = outputs[0]
                prompt_len = inputs["input_ids"].shape[1]
                gen_only_ids = generated_ids[prompt_len:]
                response = tokenizer.decode(
                    gen_only_ids,
                    skip_special_tokens=True,
                ).strip()

                hints = extract_hints(response)
            except Exception as e:
                print(f"{worker_tag} Error at idx {idx}: {e}")
                hints = ""

            h_question_idx.append(q_idx)
            h_question.append(m_question[idx])
            h_hints.append(hints)
            h_ref_solution.append(m_ref_solution[idx])
            h_ref_answer.append(m_ref_answer[idx])

            if (local_idx + 1) % max(log_every, 1) == 0:
                print(
                    f"{worker_tag} processed {local_idx + 1}/{total} items..."
                )

        return (
            h_question_idx,
            h_question,
            h_hints,
            h_ref_solution,
            h_ref_answer,
        )

    def teacher_hints_self(self, model_path: str) -> bool:
        """Generate teacher hints using a local backend.

        By default this prefers a vLLM backend for batched generation when
        available, falling back to a standard transformers-based implementation.

        Backend selection is controlled via the module-level
        TEACHER_HINTS_BACKEND flag (or the TEACHER_HINTS_BACKEND environment
        variable), with allowed values:

        - "auto" (default): use vLLM when available, otherwise transformers.
        - "vllm": require vLLM; raise a clear error if unavailable.
        - "transformers": always use the transformers backend.

        Args:
            model_path: Local path or model ID for the chat model.

        Returns:
            True if the full pipeline runs to completion.
        """
        print("Starting teacher hinting (local model)...")
        print("load mistakes...")
        self.file.load_mistakes()
        (
            m_question_idx,
            m_question,
            m_answer,
            m_ref_answer,
            m_ref_solution,
            m_entropy,
        ) = self.file.parse_data(self.file.mistakes)
        total = len(m_question)
        print("mistakes size:", total)

        if total == 0:
            print("[teacher_hints_self] No mistakes to process.")
            # Preserve previous behavior by writing an empty hints file so
            # downstream consumers can still load it without errors.
            self.file.save_hints([], [], [], [], [], [], [])
            return True

        backend = TEACHER_HINTS_BACKEND
        if backend not in {"auto", "vllm", "transformers"}:
            print(
                f"[teacher_hints_self] Unknown TEACHER_HINTS_BACKEND='{backend}', "
                "falling back to 'auto'."
            )
            backend = "auto"

        # Explicit vLLM-only mode.
        if backend == "vllm":
            if not _is_vllm_available():
                raise RuntimeError(
                    "TEACHER_HINTS_BACKEND is set to 'vllm', but vLLM is not "
                    "available or CUDA is not enabled. Please install vLLM and "
                    "ensure a CUDA device is visible, or set "
                    "TEACHER_HINTS_BACKEND='transformers' to force the "
                    "transformers backend."
                )
            print("[teacher_hints_self] Using vLLM backend.")
            return self._teacher_hints_vllm_orchestrate(
                model_path,
                m_question_idx,
                m_question,
                m_answer,
                m_ref_solution,
                m_ref_answer,
                m_entropy,
            )

        # Explicit transformers-only mode.
        if backend == "transformers":
            print("[teacher_hints_self] Using transformers backend.")
            return self._teacher_hints_transformers(
                model_path,
                m_question_idx,
                m_question,
                m_answer,
                m_ref_solution,
                m_ref_answer,
                m_entropy,
            )

        # Auto mode: prefer vLLM when available, otherwise fall back.
        if _is_vllm_available():
            print("[teacher_hints_self] vLLM available; using vLLM backend.")
            return self._teacher_hints_vllm_orchestrate(
                model_path,
                m_question_idx,
                m_question,
                m_answer,
                m_ref_solution,
                m_ref_answer,
                m_entropy,
            )

        print(
            "[teacher_hints_self] vLLM not available; falling back to "
            "transformers backend."
        )
        return self._teacher_hints_transformers(
            model_path,
            m_question_idx,
            m_question,
            m_answer,
            m_ref_solution,
            m_ref_answer,
            m_entropy,
        )

    def _teacher_hints_transformers(
        self,
        model_path: str,
        m_question_idx: Sequence[int],
        m_question: Sequence[str],
        m_answer: Sequence[str],
        m_ref_solution: Sequence[str],
        m_ref_answer: Sequence[str],
        m_entropy: Sequence[float],
        checkpoint_interval: int = TEACHER_HINTS_CHECKPOINT_INTERVAL,
    ) -> bool:
        """Transformers-based implementation of `teacher_hints_self`.

        This preserves the original behavior of loading a local
        AutoModelForCausalLM and generating hints sequentially, including
        periodic checkpointing.
        """
        total = len(m_question)
        print(f"[teacher_hints_self] (transformers) generating hints({total})...")

        h_question: List[str] = []
        h_hints: List[str] = []
        h_ref_solution: List[str] = []
        h_ref_answer: List[str] = []
        h_question_idx: List[int] = []

        # Load tokenizer and model once
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        all_indices = list(range(total))
        (
            local_q_idx,
            local_q,
            local_hints,
            local_ref_sol,
            local_ref_ans,
        ) = self._generate_hints_for_indices(
            model,
            tokenizer,
            all_indices,
            m_question_idx,
            m_question,
            m_answer,
            m_ref_solution,
            m_ref_answer,
            worker_tag="[teacher_hints_self]",
            log_every=5,
        )

        # Accumulate and periodically checkpoint to disk.
        for i, q_idx in enumerate(local_q_idx):
            h_question_idx.append(q_idx)
            h_question.append(local_q[i])
            h_hints.append(local_hints[i])
            h_ref_solution.append(local_ref_sol[i])
            h_ref_answer.append(local_ref_ans[i])

            if checkpoint_interval > 0 and (i + 1) % checkpoint_interval == 0:
                print(
                    f"[teacher_hints_self] Auto-saving checkpoint at count {i + 1}..."
                )
                self.file.save_hints(
                    h_question,
                    h_hints,
                    h_ref_solution,
                    h_ref_answer,
                    h_question_idx,
                    m_answer[: i + 1],
                    m_entropy[: i + 1],
                )

        print("[teacher_hints_self] saving final hints (transformers)...")
        self.file.save_hints(
            h_question,
            h_hints,
            h_ref_solution,
            h_ref_answer,
            h_question_idx,
            m_answer,
            m_entropy,
        )
        return True

    def _teacher_hints_vllm_orchestrate(
        self,
        model_path: str,
        m_question_idx: Sequence[int],
        m_question: Sequence[str],
        m_answer: Sequence[str],
        m_ref_solution: Sequence[str],
        m_ref_answer: Sequence[str],
        m_entropy: Sequence[float],
        batch_size: int = 32,
        checkpoint_interval: int = TEACHER_HINTS_CHECKPOINT_INTERVAL,
        device_id: Optional[int] = None,
    ) -> bool:
        """vLLM-based implementation of `teacher_hints_self`.

        This helper runs batched generation via `_teacher_hints_vllm_single_gpu`
        and mirrors the checkpointing semantics of the transformers backend.
        """
        total = len(m_question)
        print(
            f"[teacher_hints_self] (vLLM) generating hints for {total} items "
            f"with batch_size={batch_size}..."
        )

        (
            h_question_idx,
            h_question,
            h_hints,
            h_ref_solution,
            h_ref_answer,
        ) = self._teacher_hints_vllm_single_gpu(
            model_path,
            m_question_idx,
            m_question,
            m_answer,
            m_ref_solution,
            m_ref_answer,
            device_id=device_id,
            batch_size=batch_size,
        )

        if len(h_question) != total:
            raise RuntimeError(
                "[teacher_hints_self] (vLLM) Mismatch between number of "
                f"inputs ({total}) and outputs ({len(h_question)})."
            )

        acc_q_idx: List[int] = []
        acc_q: List[str] = []
        acc_hints: List[str] = []
        acc_ref_sol: List[str] = []
        acc_ref_ans: List[str] = []

        for i in range(total):
            acc_q_idx.append(h_question_idx[i])
            acc_q.append(h_question[i])
            acc_hints.append(h_hints[i])
            acc_ref_sol.append(h_ref_solution[i])
            acc_ref_ans.append(h_ref_answer[i])

            if checkpoint_interval > 0 and (i + 1) % checkpoint_interval == 0:
                print(
                    "[teacher_hints_self] (vLLM) Auto-saving checkpoint at "
                    f"count {i + 1}..."
                )
                self.file.save_hints(
                    acc_q,
                    acc_hints,
                    acc_ref_sol,
                    acc_ref_ans,
                    acc_q_idx,
                    m_answer[: i + 1],
                    m_entropy[: i + 1],
                )

        print("[teacher_hints_self] saving final hints (vLLM)...")
        self.file.save_hints(
            acc_q,
            acc_hints,
            acc_ref_sol,
            acc_ref_ans,
            acc_q_idx,
            m_answer,
            m_entropy,
        )
        return True

    def _teacher_hints_vllm_single_gpu(
        self,
        model_path: str,
        m_question_idx: Sequence[int],
        m_question: Sequence[str],
        m_answer: Sequence[str],
        m_ref_solution: Sequence[str],
        m_ref_answer: Sequence[str],
        device_id: Optional[int] = None,
        batch_size: int = 32,
    ) -> Tuple[List[int], List[str], List[str], List[str], List[str]]:
        """Single-GPU vLLM path for teacher hints.

        This helper performs batched generation on a single GPU using vLLM and
        returns in-memory results without touching the filesystem.
        """
        # Optionally pin to a specific CUDA device.
        if device_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
            print(
                f"[teacher_hints_vllm_single_gpu] Using CUDA device {device_id} for vLLM."
            )

        # Initialize tokenizer.
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Prepare stop token ids (model-agnostic: only EOS).
        stop_token_ids = []
        if tokenizer.eos_token_id is not None:
            stop_token_ids.append(tokenizer.eos_token_id)

        # Initialize vLLM engine on a single GPU.
        try:
            llm = LLM(
                model=model_path,
                trust_remote_code=True,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
                max_model_len=4096,
                enforce_eager=True,
                dtype="bfloat16",
            )
        except Exception as e:
            raise RuntimeError(
                "[teacher_hints_vllm_single_gpu] Failed to initialize vLLM "
                f"engine: {e}"
            ) from e

        total = len(m_question)
        prompts: List[str] = []
        processed = 0

        for idx in range(total):
            prompt = TEACHER_CORRECT_PROMPT.format(
                problem=m_question[idx],
                student_answer=m_answer[idx],
                ref_solution=m_ref_solution[idx],
            )
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant who good at math",
                },
                {"role": "user", "content": prompt},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt_text)

        sampling_params = SamplingParams(
            n=1,
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
            stop_token_ids=stop_token_ids or None,
        )

        h_question_idx: List[int] = []
        h_question: List[str] = []
        h_hints: List[str] = []
        h_ref_solution: List[str] = []
        h_ref_answer: List[str] = []

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_prompts = prompts[start:end]
            batch_count = end - start
            try:
                outputs = llm.generate(batch_prompts, sampling_params)
            except Exception as e:
                print(
                    f"[teacher_hints_vllm_single_gpu] Error generating batch "
                    f"{start}-{end}: {e}"
                )
                # On failure, append empty hints for this batch to keep alignment.
                for idx in range(start, end):
                    h_question_idx.append(m_question_idx[idx])
                    h_question.append(m_question[idx])
                    h_hints.append("")
                    h_ref_solution.append(m_ref_solution[idx])
                    h_ref_answer.append(m_ref_answer[idx])
            else:
                for local_i, out in enumerate(outputs):
                    idx = start + local_i
                    response_text = out.outputs[0].text.strip()
                    try:
                        hints = extract_hints(response_text)
                    except Exception as e:
                        print(
                            f"[teacher_hints_vllm_single_gpu] Error parsing hints at "
                            f"idx {idx}: {e}"
                        )
                        hints = ""

                    h_question_idx.append(m_question_idx[idx])
                    h_question.append(m_question[idx])
                    h_hints.append(hints)
                    h_ref_solution.append(m_ref_solution[idx])
                    h_ref_answer.append(m_ref_answer[idx])

            processed += batch_count
            print(
                f"[teacher_hints_self] (vLLM) processed {processed}/{total} items "
                f"({processed / total:.2%})"
            )

        return (
            h_question_idx,
            h_question,
            h_hints,
            h_ref_solution,
            h_ref_answer,
        )

    def teacher_hints_self_parallel(
        self,
        model_path: str,
        device_ids: Optional[Sequence[int]] = None,
        num_workers: Optional[int] = None,
    ) -> bool:
        """Data-parallel multi-GPU variant of teacher_hints_self.

        This method shards the mistakes dataset over multiple GPUs, runs
        local-model generation in separate worker processes, then merges
        results and writes hints once at the end.

        Existing single-GPU behavior of teacher_hints_self is unchanged:
        this is an opt-in API.
        """
        print("Starting teacher hinting (local model, multi-GPU)...")
        print("load mistakes...")
        self.file.load_mistakes()
        (
            m_question_idx,
            m_question,
            m_answer,
            m_ref_answer,
            m_ref_solution,
            m_entropy,
        ) = self.file.parse_data(self.file.mistakes)
        total = len(m_question)
        print("mistakes size:", total)

        if total == 0:
            print("[teacher_hints_self_parallel] No mistakes to process.")
            return True

        # Resolve available devices
        if device_ids is None:
            if not torch.cuda.is_available():
                print(
                    "[teacher_hints_self_parallel] No CUDA devices; falling back to single-GPU teacher_hints_self."
                )
                return self.teacher_hints_self(model_path)
            n_gpus = torch.cuda.device_count()
            device_ids = list(range(n_gpus))
        elif isinstance(device_ids, int):
            device_ids = [device_ids]

        if not device_ids:
            print(
                "[teacher_hints_self_parallel] Empty device_ids; falling back to single-GPU teacher_hints_self."
            )
            return self.teacher_hints_self(model_path)

        if num_workers is None:
            num_workers = min(len(device_ids), total)
        else:
            num_workers = min(num_workers, len(device_ids), total)

        if num_workers <= 1:
            print(
                "[teacher_hints_self_parallel] num_workers <= 1; preferring single-GPU vLLM path."
            )

            # If vLLM is not available, gracefully fall back to the
            # sequential teacher_hints_self implementation.
            if not _is_vllm_available():
                print(
                    "[teacher_hints_self_parallel] vLLM not available; "
                    "falling back to teacher_hints_self()."
                )
                return self.teacher_hints_self(model_path)

            # Use the shared vLLM orchestration helper so that checkpointing
            # behavior matches teacher_hints_self.
            single_device_id: Optional[int] = None
            if isinstance(device_ids, (list, tuple)) and len(device_ids) == 1:
                single_device_id = device_ids[0]

            return self._teacher_hints_vllm_orchestrate(
                model_path,
                m_question_idx,
                m_question,
                m_answer,
                m_ref_solution,
                m_ref_answer,
                m_entropy,
                batch_size=32,
                device_id=single_device_id,
            )

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
            args_list.append(
                (
                    local_rank,
                    device_id,
                    model_path,
                    idx_shard,
                    m_question_idx,
                    m_question,
                    m_answer,
                    m_ref_solution,
                    m_ref_answer,
                )
            )

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(args_list)) as pool:
            shard_results = pool.map(_teacher_hints_shard_worker, args_list)

        # Merge shard results
        merged_idx: List[int] = []
        merged_q: List[str] = []
        merged_hints: List[str] = []
        merged_ref_sol: List[str] = []
        merged_ref_ans: List[str] = []

        for (
            h_question_idx,
            h_question,
            h_hints,
            h_ref_solution,
            h_ref_answer,
        ) in shard_results:
            merged_idx.extend(h_question_idx)
            merged_q.extend(h_question)
            merged_hints.extend(h_hints)
            merged_ref_sol.extend(h_ref_solution)
            merged_ref_ans.extend(h_ref_answer)

        # Sort by original question index to restore order
        order = sorted(range(len(merged_idx)), key=lambda i: merged_idx[i])
        h_question_idx_sorted: List[int] = [merged_idx[i] for i in order]
        h_question_sorted: List[str] = [merged_q[i] for i in order]
        h_hints_sorted: List[str] = [merged_hints[i] for i in order]
        h_ref_solution_sorted: List[str] = [merged_ref_sol[i] for i in order]
        h_ref_answer_sorted: List[str] = [merged_ref_ans[i] for i in order]

        # Final single save (no per-10-item checkpointing in parallel mode)
        print("saving final hints (parallel)...")
        print(
            f"[teacher_hints_self_parallel] Completed generating hints for {total} items across {len(args_list)} workers."
        )
        self.file.save_hints(
            h_question_sorted,
            h_hints_sorted,
            h_ref_solution_sorted,
            h_ref_answer_sorted,
            h_question_idx_sorted,
            m_answer,
            m_entropy,
        )
        return True

    def teacher_hints_gtp(self) -> bool:
        print("Starting teacher hinting (GPT-4o)...")
        print("load mistakes...")
        self.file.load_mistakes()
        (
            m_question_idx,
            m_question,
            m_answer,
            m_ref_answer,
            m_ref_solution,
            m_entropy,
        ) = self.file.parse_data(self.file.mistakes)
        print("mistakes size:", len(m_question))

        h_question = []
        h_hints = []
        h_ref_solution = []
        h_ref_answer = []
        h_question_idx = []

        print(f"generating hints({len(m_question)})...")

        # 初始化 OpenAI 客户端
        # 建议将 key 放入环境变量 OPENAI_API_KEY 中，或者在这里直接替换字符串
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            # 如果你使用的是国内中转/代理，取消下面这行的注释并填入地址
            # base_url="https://api.openai-proxy.com/v1"
        )

        print("----- standard request (GPT-4o) -----")
        for idx in range(len(m_question)):
            prompt = TEACHER_CORRECT_PROMPT.format(
                problem=m_question[idx],
                student_answer=m_answer[idx],
                ref_solution=m_ref_solution[idx],
            )

            response = None
            max_retries = 5  # 增加最大重试次数防止死循环
            retry_count = 0

            while retry_count < max_retries:
                try:
                    # 调用 GPT-4o
                    completion = client.chat.completions.create(
                        model="app-7c54im-1766977238437488331",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful assistant who is good at math.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.7,  # 适当增加一点随机性，避免过于死板，或设为0保持确定性
                    )
                    response = completion.choices[0].message.content
                    break

                except RateLimitError:
                    print(
                        f"Rate limit reached at idx {idx}. Sleeping for 20 seconds..."
                    )
                    time.sleep(20)
                    retry_count += 1
                except APIError as e:
                    print(f"OpenAI API Error at idx {idx}: {e}. Retrying...")
                    time.sleep(5)
                    retry_count += 1
                except Exception as e:
                    print(f"An unexpected error occurred at idx {idx}: {e}")
                    # 如果是严重错误，可以选择 break 或者 raise
                    raise e

            if response:
                hints = extract_hints(response)
                h_question_idx.append(m_question_idx[idx])
                h_question.append(m_question[idx])
                h_hints.append(hints)
                h_ref_solution.append(m_ref_solution[idx])
                h_ref_answer.append(m_ref_answer[idx])

                # 打印进度，防止在此处看起来像卡死
                if idx % 5 == 0:
                    print(f"Processed {idx + 1}/{len(m_question)}")
            else:
                print(f"Failed to get response for idx {idx}")

        print("saving hints...")
        self.file.save_hints(
            h_question,
            h_hints,
            h_ref_solution,
            h_ref_answer,
            h_question_idx,
            m_answer,
            m_entropy,
        )
        return True

    def teacher_mark_paper_with_save(self) -> bool:
        incorrect_data, correct_data = self.teacher_mark_paper()
        (
            err_question_idx,
            err_questions,
            err_answers,
            err_ref_solutions,
            err_ref_answers,
            err_entropy,
        ) = incorrect_data
        (
            correct_question_idx,
            correct_questions,
            correct_answers,
            correct_ref_solutions,
            correct_ref_answers,
            correct_entropy,
        ) = correct_data
        self.file.save_mistakes(
            err_question_idx,
            err_questions,
            err_answers,
            err_ref_solutions,
            err_ref_answers,
            err_entropy,
        )
        self.file.save_right(
            correct_question_idx,
            correct_questions,
            correct_answers,
            correct_ref_solutions,
            correct_ref_answers,
            correct_entropy,
        )
        return True

    def judge_and_gen_hints(self):
        print("Starting judge and generate hints...")
        self.teacher_mark_paper_with_save()
        self.teacher_hints()

    def teacher_mark_paper(self, roll: bool = False):
        print("Starting teacher marking...")
        self.file.load_exam(roll)
        (
            question_idx,
            question,
            answer,
            ref_answer,
            ref_solution,
            entropy,
        ) = self.file.parse_data(self.file.data)
        size = len(question)

        self.acc_count = 0
        self.err_count = 0
        self.toolong_count = 0

        err_question_idx = []
        err_questions = []
        err_answers = []
        err_ref_solutions = []
        err_ref_answers = []
        err_entropy = []

        correct_question_idx = []
        correct_questions = []
        correct_answers = []
        correct_ref_solutions = []
        correct_ref_answers = []
        correct_entropy = []

        print("----- standard request -----")
        for idx in range(size):
            final_answer = extract_boxed_content(answer[idx])
            final_answer = normalize_answer(final_answer)
            ref_final_answer = normalize_answer(ref_answer[idx])

            if final_answer == ref_final_answer:
                self.acc_count += 1
                correct_question_idx.append(question_idx[idx])
                correct_questions.append(question[idx])
                correct_answers.append(answer[idx])
                correct_ref_solutions.append(ref_solution[idx])
                correct_ref_answers.append(ref_answer[idx])
                correct_entropy.append(entropy[idx])
            else:
                self.err_count += 1
                err_question_idx.append(question_idx[idx])
                err_questions.append(question[idx])
                err_answers.append(answer[idx])
                err_ref_solutions.append(ref_solution[idx])
                err_ref_answers.append(ref_answer[idx])
                err_entropy.append(entropy[idx])

            if idx % 5 == 0:
                left = size - idx
                print(
                    f"finished: {idx}, left: {left}, acc:{self.acc_count}, err:{self.err_count}, toolong:{self.toolong_count}"
                )

        print(f"Accuracy: {self.acc_count}/{size}")
        print(f"Error count: {self.err_count}")

        return (
            (
                err_question_idx,
                err_questions,
                err_answers,
                err_ref_solutions,
                err_ref_answers,
                err_entropy,
            ),
            (
                correct_question_idx,
                correct_questions,
                correct_answers,
                correct_ref_solutions,
                correct_ref_answers,
                correct_entropy,
            ),
        )

    def check_answers_equivalence(self) -> int:
        print("Loading mistakes for evaluation...")
        self.file.load_mistakes()
        total_questions = len(self.file.mistakes)
        equivalent_count = 0

        print(f"Total items to check: {total_questions}")
        print("----- Starting Evaluation (GPT-4o) -----")

        data = self.file.mistakes

        # ================== 1. 初始化错误数据列表 ==================
        err_question_idx = []
        err_questions = []
        err_answers = []
        err_ref_solutions = []
        err_ref_answers = []
        err_entropy = []
        # ========================================================

        # 确保 client 初始化
        if not hasattr(self, "client"):
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://wanqing-api.corp.kuaishou.com/api/agent/v1/apps",
            )

        for idx, item in enumerate(data):
            question_idx_val = item.get("question_idx", idx)
            question_text = item.get("question", "")
            ref_answer = item.get("ref_answer", "")
            ref_solution = item.get("ref_solution", "")  # 获取解析，保存需要用
            entropy = item.get("entropy", 0.0)  # 获取 entropy

            raw_answer = item.get("answer", "")
            student_answer_core = extract_boxed_content(raw_answer)

            # 构建 Prompt
            prompt = OREAL_CORRECT_PROMPT.format(
                question=question_text,
                gold_answer=ref_answer,
                answer=student_answer_core,
            )

            is_equivalent = False
            response_content = ""

            # 调用 API
            max_retries = 5
            retry_count = 0

            while retry_count < max_retries:
                try:
                    completion = self.client.chat.completions.create(
                        model="app-7c54im-1766977238437488331",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful assistant evaluating math answers.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.0,
                        max_completion_tokens=100,
                    )
                    response_content = completion.choices[0].message.content.strip()
                    break

                except RateLimitError:
                    print(
                        f"[Idx {question_idx_val}] Rate limit reached. Sleeping for 20 seconds..."
                    )
                    time.sleep(20)
                    retry_count += 1
                except APIError as e:
                    print(
                        f"[Idx {question_idx_val}] OpenAI API Error: {e}. Retrying..."
                    )
                    time.sleep(5)
                    retry_count += 1
                except Exception as e:
                    print(f"[Idx {question_idx_val}] Unexpected error: {e}")
                    break

            # 解析结果
            if response_content:
                clean_resp = response_content.upper().replace(".", "")

                if "A" == clean_resp or "CORRECT" in clean_resp:
                    is_equivalent = True
                elif "B" == clean_resp or "INCORRECT" in clean_resp:
                    is_equivalent = False
                else:
                    if "CORRECT" in clean_resp and "INCORRECT" not in clean_resp:
                        is_equivalent = True
                    else:
                        print(
                            f"[Idx {question_idx_val}] Ambiguous response: {response_content}"
                        )

            # 统计与记录
            if is_equivalent:
                equivalent_count += 1
                status = "CORRECT"
            else:
                status = "WRONG"
                # ================== 2. 如果错了，收集数据 ==================
                err_question_idx.append(question_idx_val)
                err_questions.append(question_text)
                err_answers.append(raw_answer)
                err_ref_solutions.append(ref_solution)
                err_ref_answers.append(ref_answer)
                err_entropy.append(entropy)
                # ========================================================

            print(f"Idx {question_idx_val}: {status} | GPT Says: {response_content}")

        # 输出最终统计
        print("-" * 30)
        print("Evaluation Finished.")
        print(f"Total Questions: {total_questions}")
        print(f"Equivalent (Correct) Answers: {equivalent_count}")
        print(f"Genuine Mistakes (Saved): {len(err_questions)}")

        if total_questions > 0:
            print(f"Accuracy: {equivalent_count / total_questions * 100:.2f}%")

        # ================== 3. 保存错题 ==================
        if len(err_questions) > 0:
            print("Saving verified mistakes to file...")
            self.file.save_mistakes(
                err_question_idx,
                err_questions,
                err_answers,
                err_ref_solutions,
                err_ref_answers,
                err_entropy,
            )
        else:
            print("No mistakes found to save!")
        # ================================================

        return equivalent_count


def _teacher_hints_shard_worker(args):
    """Worker function for teacher_hints_self_parallel.

    Reconstructs a local model on a specific GPU and runs
    _generate_hints_for_indices() over a shard of indices.
    """
    (
        local_rank,
        device_id,
        model_path,
        indices,
        m_question_idx,
        m_question,
        m_answer,
        m_ref_solution,
        m_ref_answer,
    ) = args

    # Pin worker process to a single CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    print(
        f"[teacher_hints worker {local_rank}] Using CUDA device {device_id} for {len(indices)} items."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Use a temporary TeacherCorrecter instance only for access to the helper.
    tmp = TeacherCorrecter()
    worker_tag = f"[teacher_hints worker {local_rank}]"
    result = tmp._generate_hints_for_indices(
        model,
        tokenizer,
        indices,
        m_question_idx,
        m_question,
        m_answer,
        m_ref_solution,
        m_ref_answer,
        worker_tag=worker_tag,
        log_every=5,
    )

    print(
        f"[teacher_hints worker {local_rank}] completed {len(indices)} items."
    )

    return result


if __name__ == "__main__":
    corrector = TeacherCorrecter()
    corrector.check_answers_equivalence()

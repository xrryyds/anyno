import importlib
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from datasets import Dataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "Please reason step by step and put your final answer within \\boxed{}."
DEFAULT_MAX_TOTAL_LENGTH = 2048
DEFAULT_TEACHER_MAX_PROMPT_LENGTH = 4096
DEFAULT_NUM_PROMPTS_PER_BATCH = 4
SAVE_TOTAL_LIMIT = 2


@dataclass
class SDFTWrapperConfig:
    model_path: str
    data_path: str
    output_base_dir: str
    num_train_epochs: int
    learning_rate: float = 5e-5
    seed: int = 42
    num_prompts_per_batch: int = DEFAULT_NUM_PROMPTS_PER_BATCH
    per_device_train_batch_size: int = 1
    max_total_length: int = DEFAULT_MAX_TOTAL_LENGTH
    max_completion_length: int = DEFAULT_MAX_TOTAL_LENGTH
    teacher_max_prompt_length: int = DEFAULT_TEACHER_MAX_PROMPT_LENGTH
    ref_model_mixup_alpha: float = 0.01
    ref_model_sync_steps: int = 1
    alpha: float = 0.0
    num_loss_tokens_to_skip: int = 3
    output_dir: Optional[str] = None
    lora_path: Optional[str] = None
    use_vllm: bool = True
    vllm_mode: str = "colocate"
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.3
    vllm_enable_sleep_mode: bool = True
    logging_steps: int = 1
    save_steps: int = 100
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    disable_dropout: bool = False
    report_to: str = "none"
    bf16: bool = True
    fp16: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: float = 1.0


def _is_peft_adapter_dir(path: Optional[str]) -> bool:
    if not path or not os.path.isdir(path):
        return False
    adapter_config = os.path.join(path, "adapter_config.json")
    adapter_weights = (
        os.path.join(path, "adapter_model.safetensors"),
        os.path.join(path, "adapter_model.bin"),
    )
    return os.path.isfile(adapter_config) and any(os.path.isfile(candidate) for candidate in adapter_weights)


def _project_root() -> str:
    current_file_path = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))


def _sdft_repo_root() -> str:
    return os.path.join(_project_root(), "sdft", "Self-Distillation")


def _resolve_torch_dtype(config: SDFTWrapperConfig):
    torch, _, _, _, _, _ = _import_training_modules()
    if config.fp16:
        return torch.float16
    if config.bf16:
        return torch.bfloat16
    return torch.float32


def _validate_runtime_paths(config: SDFTWrapperConfig) -> None:
    sdft_root = _sdft_repo_root()
    if not os.path.isdir(sdft_root):
        raise FileNotFoundError(f"Original sdft repo not found: {sdft_root}")
    if not os.path.isdir(config.model_path):
        raise FileNotFoundError(f"Model path not found or not a directory: {config.model_path}")
    if not os.path.isfile(config.data_path):
        raise FileNotFoundError(f"Training data path not found: {config.data_path}")
    if config.lora_path is not None and not _is_peft_adapter_dir(config.lora_path):
        raise FileNotFoundError(
            "Provided lora_path is not a valid PEFT adapter directory "
            f"(missing adapter_config.json / adapter weights): {config.lora_path}"
        )


def _import_sdft_modules():
    sdft_root = _sdft_repo_root()
    if sdft_root not in sys.path:
        sys.path.insert(0, sdft_root)

    try:
        DistilConfig = importlib.import_module("distil_config").DistilConfig
        DistilTrainer = importlib.import_module("distil_trainer").DistilTrainer
        return DistilConfig, DistilTrainer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Failed to import original sdft modules. Please install the dependencies from "
            f"`{os.path.join(sdft_root, 'requirements.txt')}` in your training environment."
        ) from exc


def _import_training_modules():
    try:
        import torch
        from peft import LoraConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        return torch, LoraConfig, PeftModel, AutoModelForCausalLM, AutoTokenizer, set_seed
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing training dependency. Please install transformers/peft/torch and the original sdft requirements "
            "in your training environment."
        ) from exc


def _extract_ref_solution(item: Dict[str, Any]) -> str:
    value = item.get("ref_solution")
    if value is not None and str(value).strip():
        return str(value)
    value = item.get("ref_solutions")
    if isinstance(value, list):
        for candidate in value:
            if candidate is not None and str(candidate).strip():
                return str(candidate)
    elif value is not None and str(value).strip():
        return str(value)
    return ""


def _build_student_prompt(question: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(question)},
    ]


def _build_teacher_prompt(question: str, ref_solution: str):
    teacher_user_content = (
        f"{question}\n\n"
        "This is an example for a response to the question:\n"
        f"{ref_solution}\n\n"
        "Now answer with a response of your own, including the thinking process."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": teacher_user_content},
    ]


def _load_adv_hints_dataset(data_path: str, seed: int) -> Dataset:
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError(f"Expected list dataset in {data_path}, got {type(raw_data).__name__}")

    rows = []
    for idx, item in enumerate(raw_data):
        question = item.get("question")
        ref_solution = _extract_ref_solution(item)
        if question is None or not str(question).strip():
            logger.warning(f"Skipping sample {idx}: empty question")
            continue
        if not ref_solution.strip():
            logger.warning(f"Skipping sample {idx}: empty ref_solution/ref_solutions")
            continue
        rows.append(
            {
                "prompt": _build_student_prompt(str(question)),
                "teacher_prompt": _build_teacher_prompt(str(question), ref_solution),
                "question": str(question),
                "ref_solution": ref_solution,
            }
        )

    if not rows:
        raise ValueError(f"No valid question/ref_solution pairs found in {data_path}")

    dataset = Dataset.from_list(rows)
    dataset = dataset.shuffle(seed=seed)
    logger.info(f"Loaded {len(dataset)} training samples from {data_path}")
    return dataset


def _build_output_dir(config: SDFTWrapperConfig) -> str:
    if config.output_dir is not None:
        return config.output_dir
    return os.path.join(
        config.output_base_dir,
        f"sdft_{config.num_train_epochs}ep_{datetime.now().strftime('%m%d_%H%M')}",
    )


def _build_save_steps(dataset_size: int, config: SDFTWrapperConfig) -> int:
    effective_batch = max(1, config.num_prompts_per_batch)
    steps_per_epoch = max(1, math.ceil(dataset_size / effective_batch))
    total_steps = max(1, steps_per_epoch * max(1, config.num_train_epochs))
    return max(1, total_steps // SAVE_TOTAL_LIMIT)


def _build_models(config: SDFTWrapperConfig):
    torch, LoraConfig, PeftModel, AutoModelForCausalLM, _, _ = _import_training_modules()
    torch_dtype = _resolve_torch_dtype(config)

    student_model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    student_model.config.use_cache = False
    teacher_model.config.use_cache = False

    if config.lora_path:
        logger.info(f"Loading existing LoRA adapter from {config.lora_path}")
        student_model = PeftModel.from_pretrained(student_model, config.lora_path, is_trainable=True)
        teacher_model = PeftModel.from_pretrained(teacher_model, config.lora_path, is_trainable=False)
        peft_config = None
    else:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
            bias="none",
        )

    return student_model, teacher_model, peft_config


def _build_tokenizer(model_path: str):
    _, _, _, _, AutoTokenizer, _ = _import_training_modules()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return tokenizer


def _build_distil_config(
    DistilConfig,
    config: SDFTWrapperConfig,
    dataset_size: int,
    output_dir: str,
):
    save_steps = _build_save_steps(dataset_size, config)
    return DistilConfig(
        seed=config.seed,
        output_dir=output_dir,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        bf16=config.bf16,
        fp16=config.fp16,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.num_prompts_per_batch,
        max_prompt_length=config.max_total_length,
        teacher_max_prompt_length=config.teacher_max_prompt_length,
        max_completion_length=config.max_completion_length,
        max_total_length=config.max_total_length,
        num_train_epochs=config.num_train_epochs,
        num_iterations=1,
        num_generations=1,
        save_steps=save_steps,
        save_total_limit=SAVE_TOTAL_LIMIT,
        max_grad_norm=1.0,
        report_to=config.report_to,
        log_completions=False,
        sync_ref_model=True,
        ref_model_sync_steps=config.ref_model_sync_steps,
        ref_model_mixup_alpha=config.ref_model_mixup_alpha,
        vllm_importance_sampling_correction=True,
        num_loss_tokens_to_skip=config.num_loss_tokens_to_skip,
        alpha=config.alpha,
        use_vllm=config.use_vllm,
        vllm_mode=config.vllm_mode,
        vllm_tensor_parallel_size=config.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=config.vllm_enable_sleep_mode,
        disable_dropout=config.disable_dropout,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        min_p=config.min_p,
        repetition_penalty=config.repetition_penalty,
        remove_unused_columns=False,
        dataloader_drop_last=True,
    )


def run_sdft_training_baseline(
    model_path: str,
    data_path: str,
    batch_size: int,
    real_data_epochs: int,
    output_base_dir: Optional[str] = None,
    device_num: int = 1,
    lora_path: Optional[str] = None,
    learning_rate: float = 5e-5,
    num_prompts_per_batch: Optional[int] = None,
    max_seq_length: int = DEFAULT_MAX_TOTAL_LENGTH,
    teacher_max_seq_length: int = DEFAULT_TEACHER_MAX_PROMPT_LENGTH,
    ref_model_mixup_alpha: float = 0.01,
    ref_model_sync_steps: int = 1,
    alpha: float = 0.0,
    num_loss_tokens_to_skip: int = 3,
    use_vllm: bool = True,
    vllm_gpu_memory_utilization: float = 0.3,
):
    DistilConfig, DistilTrainer = _import_sdft_modules()
    _, _, _, _, _, set_seed = _import_training_modules()

    if output_base_dir is None:
        output_base_dir = os.path.join(_project_root(), "output")
    os.makedirs(output_base_dir, exist_ok=True)

    effective_num_prompts = num_prompts_per_batch or batch_size
    wrapper_config = SDFTWrapperConfig(
        model_path=model_path,
        data_path=data_path,
        output_base_dir=output_base_dir,
        num_train_epochs=real_data_epochs,
        learning_rate=learning_rate,
        num_prompts_per_batch=effective_num_prompts,
        max_total_length=max_seq_length,
        max_completion_length=max_seq_length,
        teacher_max_prompt_length=teacher_max_seq_length,
        ref_model_mixup_alpha=ref_model_mixup_alpha,
        ref_model_sync_steps=ref_model_sync_steps,
        alpha=alpha,
        num_loss_tokens_to_skip=num_loss_tokens_to_skip,
        lora_path=lora_path,
        use_vllm=use_vllm,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=1,
    )
    _validate_runtime_paths(wrapper_config)

    if device_num != 1:
        logger.warning(
            "run_sdft_training_baseline currently keeps colocated vLLM in single-GPU mode for training stability. "
            f"Ignoring device_num={device_num}."
        )

    set_seed(wrapper_config.seed)
    dataset = _load_adv_hints_dataset(wrapper_config.data_path, wrapper_config.seed)
    output_dir = _build_output_dir(wrapper_config)
    os.makedirs(output_dir, exist_ok=True)
    wrapper_config.output_dir = output_dir

    logger.info(f"SDFT model_path: {model_path}")
    logger.info(f"SDFT data_path: {data_path}")
    logger.info(f"SDFT output_dir: {output_dir}")
    logger.info(f"SDFT epochs: {real_data_epochs}")
    logger.info(f"SDFT num_prompts_per_batch: {effective_num_prompts}")
    logger.info(f"SDFT max_seq_length(student total): {max_seq_length}")
    logger.info(f"SDFT teacher_max_seq_length: {teacher_max_seq_length}")
    logger.info(f"SDFT use_vllm: {use_vllm}")

    tokenizer = _build_tokenizer(wrapper_config.model_path)
    student_model, teacher_model, peft_config = _build_models(wrapper_config)
    distil_config = _build_distil_config(
        DistilConfig=DistilConfig,
        config=wrapper_config,
        dataset_size=len(dataset),
        output_dir=output_dir,
    )

    trainer = DistilTrainer(
        model=student_model,
        ref_model=teacher_model,
        args=distil_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    if not _is_peft_adapter_dir(output_dir):
        raise RuntimeError(
            "Training finished, but the saved output directory does not look like a valid PEFT adapter: "
            f"{output_dir}"
        )

    meta = {
        "train_type": "sdft_original_wrapper",
        "base_model_path": model_path,
        "data_path": data_path,
        "system_prompt": SYSTEM_PROMPT,
        "student_prompt_template": "take_exam-aligned system+user chat prompt",
        "teacher_prompt_template": "original sdft demonstration-conditioned prompt",
        "num_train_epochs": real_data_epochs,
        "num_prompts_per_batch": effective_num_prompts,
        "max_seq_length": max_seq_length,
        "teacher_max_seq_length": teacher_max_seq_length,
        "ref_model_mixup_alpha": ref_model_mixup_alpha,
        "ref_model_sync_steps": ref_model_sync_steps,
        "alpha": alpha,
        "num_loss_tokens_to_skip": num_loss_tokens_to_skip,
        "use_vllm": use_vllm,
        "continued_from_lora": bool(lora_path),
    }
    with open(os.path.join(output_dir, "sdft_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info(f"SDFT training finished. Adapter saved to {output_dir}")
    logger.info(
        f"TakeExam usage: TakeExam(model_path='{model_path}', use_lora=True, adapter_path='{output_dir}')"
    )
    return output_dir


if __name__ == "__main__":
    root = _project_root()
    default_model_path = os.path.join(root, "model", "OREAL", "OREAL-7B")
    default_data_path = os.path.join(root, "datasets", "exam", "adv_hints.json")
    default_output_base_dir = os.path.join(root, "output")

    run_sdft_training_baseline(
        model_path=default_model_path,
        data_path=default_data_path,
        batch_size=4,
        real_data_epochs=2,
        output_base_dir=default_output_base_dir,
    )

import importlib
import json
import logging
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "Please reason step by step and put your final answer within \\boxed{}."
DEFAULT_MAX_PROMPT_LENGTH = 2048
DEFAULT_MAX_RESPONSE_LENGTH = 2048
DEFAULT_NUM_PROMPTS_PER_BATCH = 4


@dataclass
class SDPOWrapperConfig:
    model_path: str
    data_path: str
    output_base_dir: str
    num_train_epochs: int
    learning_rate: float = 1e-5
    seed: int = 42
    train_batch_size: int = 4
    rollout_batch_size: int = DEFAULT_NUM_PROMPTS_PER_BATCH
    max_prompt_length: int = DEFAULT_MAX_PROMPT_LENGTH
    max_response_length: int = DEFAULT_MAX_RESPONSE_LENGTH
    alpha: float = 0.5
    dont_reprompt_on_self_success: bool = True
    include_environment_feedback: bool = False
    output_dir: Optional[str] = None
    lora_path: Optional[str] = None
    lora_rank: int = 16
    lora_alpha: int = 32
    n_gpus_per_node: int = 1
    val_n: int = 16
    ppo_mini_batch_size: Optional[int] = None
    exp_name: Optional[str] = None


def _project_root() -> str:
    current_file_path = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))


def _sdpo_repo_root() -> str:
    return os.path.join(_project_root(), "sdpo", "SDPO")


def _validate_runtime_paths(config: SDPOWrapperConfig) -> None:
    sdpo_root = _sdpo_repo_root()
    if not os.path.isdir(sdpo_root):
        raise FileNotFoundError(f"Original SDPO repo not found: {sdpo_root}")
    if not config.model_path or not str(config.model_path).strip():
        raise FileNotFoundError("Model path is empty.")
    if not os.path.isfile(config.data_path):
        raise FileNotFoundError(f"Training data path not found: {config.data_path}")
    if config.lora_path is not None and not os.path.isdir(config.lora_path):
        raise FileNotFoundError(f"LoRA adapter path not found or not a directory: {config.lora_path}")


def _is_peft_adapter_dir(path: Optional[str]) -> bool:
    if not path or not os.path.isdir(path):
        return False
    adapter_config = os.path.join(path, "adapter_config.json")
    adapter_weights = (
        os.path.join(path, "adapter_model.safetensors"),
        os.path.join(path, "adapter_model.bin"),
    )
    return os.path.isfile(adapter_config) and any(os.path.isfile(candidate) for candidate in adapter_weights)


def _import_sdpo_preprocess_module():
    sdpo_root = _sdpo_repo_root()
    if sdpo_root not in sys.path:
        sys.path.insert(0, sdpo_root)
    return importlib.import_module("data.preprocess")


def _extract_ground_truth(item: Dict[str, Any]) -> str:
    for key in ("ref_answer", "answer"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _load_raw_dataset(data_path: str) -> List[Dict[str, Any]]:
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError(f"Expected list dataset in {data_path}, got {type(raw_data).__name__}")
    return raw_data


def _build_sdpo_record(item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    question = item.get("question")
    answer = _extract_ground_truth(item)

    if question is None or not str(question).strip():
        logger.warning(f"Skipping sample {idx}: empty question")
        return None
    if not answer.strip():
        logger.warning(f"Skipping sample {idx}: empty ref_answer/answer")
        return None

    prompt = f"{str(question).strip()}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    return {
        "idx": int(item.get("question_idx", idx)),
        "kind": "math",
        "dataset": "math500",
        "answer": answer,
        "elo": 1500,
        "prompt": prompt,
        "description": str(question).strip(),
        "tests": "-",
        "embedding": [],
        "system": None,
    }


def _split_train_val(records: List[Dict[str, Any]], seed: int) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if len(records) < 2:
        return records, records

    import random

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, math.ceil(len(shuffled) * 0.1))
    if val_size >= len(shuffled):
        val_size = 1
    val_records = shuffled[:val_size]
    train_records = shuffled[val_size:]
    if not train_records:
        train_records = val_records
    return train_records, val_records


def _write_json_array(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def _prepare_sdpo_dataset_dir(config: SDPOWrapperConfig, output_dir: str) -> str:
    raw_rows = _load_raw_dataset(config.data_path)
    records = []
    for idx, item in enumerate(raw_rows):
        record = _build_sdpo_record(item, idx)
        if record is not None:
            records.append(record)

    if not records:
        raise ValueError(f"No valid SDPO records could be built from {config.data_path}")

    dataset_dir = os.path.join(output_dir, "sdpo_dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    train_rows, val_rows = _split_train_val(records, config.seed)
    train_json = os.path.join(dataset_dir, "train.json")
    test_json = os.path.join(dataset_dir, "test.json")
    _write_json_array(train_json, train_rows)
    _write_json_array(test_json, val_rows)

    preprocess = _import_sdpo_preprocess_module()
    preprocess.run_proprocessing(dataset_dir)

    train_parquet = os.path.join(dataset_dir, "train.parquet")
    test_parquet = os.path.join(dataset_dir, "test.parquet")
    if not os.path.isfile(train_parquet) or not os.path.isfile(test_parquet):
        raise RuntimeError("SDPO preprocessing did not produce train.parquet/test.parquet")

    logger.info(f"Prepared SDPO dataset: train={len(train_rows)}, val={len(val_rows)}, dir={dataset_dir}")
    return dataset_dir


def _build_output_dir(config: SDPOWrapperConfig) -> str:
    if config.output_dir is not None:
        return config.output_dir
    return os.path.join(
        config.output_base_dir,
        f"sdpo_{config.num_train_epochs}ep_{datetime.now().strftime('%m%d_%H%M')}",
    )


def _find_latest_sdpo_adapter_dir(output_dir: str) -> str:
    tracker_path = os.path.join(output_dir, "latest_checkpointed_iteration.txt")
    candidate_steps: List[int] = []

    if os.path.isfile(tracker_path):
        with open(tracker_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content.isdigit():
            candidate_steps.append(int(content))

    for name in os.listdir(output_dir):
        if name.startswith("global_step_"):
            suffix = name.split("global_step_", 1)[1]
            if suffix.isdigit():
                candidate_steps.append(int(suffix))

    if not candidate_steps:
        raise FileNotFoundError(
            f"No SDPO checkpoint folders were found under {output_dir}. "
            "Training may have exited before the actor checkpoint was saved."
        )

    for step in sorted(set(candidate_steps), reverse=True):
        adapter_dir = os.path.join(output_dir, f"global_step_{step}", "actor", "lora_adapter")
        if _is_peft_adapter_dir(adapter_dir):
            return adapter_dir

    raise FileNotFoundError(
        f"SDPO checkpoints were found under {output_dir}, but none contained a valid LoRA adapter directory."
    )


def _build_sdpo_command(config: SDPOWrapperConfig, output_dir: str, dataset_dir: str) -> List[str]:
    sdpo_root = _sdpo_repo_root()
    reward_fn_path = os.path.join(sdpo_root, "verl", "utils", "reward_score", "feedback", "__init__.py")
    train_parquet = os.path.join(dataset_dir, "train.parquet")
    test_parquet = os.path.join(dataset_dir, "test.parquet")
    exp_name = config.exp_name or os.path.basename(output_dir)
    ppo_mini_batch_size = config.ppo_mini_batch_size or config.train_batch_size

    cmd = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        "--config-name",
        "_generated_ppo_trainer",
        f"data.train_files={train_parquet}",
        f"data.val_files={test_parquet}",
        f"data.train_batch_size={config.train_batch_size}",
        f"data.max_prompt_length={config.max_prompt_length}",
        f"data.max_response_length={config.max_response_length}",
        "data.filter_overlong_prompts=False",
        "data.truncation=right",
        f"trainer.total_epochs={config.num_train_epochs}",
        "trainer.val_before_train=False",
        "trainer.logger=[console]",
        f"trainer.n_gpus_per_node={config.n_gpus_per_node}",
        "trainer.nnodes=1",
        f"trainer.project_name=CELPO-SDPO",
        f"trainer.group_name={exp_name}",
        f"trainer.experiment_name={exp_name}",
        f"trainer.default_local_dir={output_dir}",
        "trainer.save_freq=5",
        "trainer.max_actor_ckpt_to_keep=1",
        f"actor_rollout_ref.model.path={config.model_path}",
        "actor_rollout_ref.model.trust_remote_code=True",
        f"actor_rollout_ref.model.lora_rank={config.lora_rank}",
        f"actor_rollout_ref.model.lora_alpha={config.lora_alpha}",
        "actor_rollout_ref.model.target_modules=all-linear",
        f"actor_rollout_ref.actor.optim.lr={config.learning_rate}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={ppo_mini_batch_size}",
        "actor_rollout_ref.actor.policy_loss.loss_mode=sdpo",
        f"actor_rollout_ref.actor.self_distillation.alpha={config.alpha}",
        "actor_rollout_ref.actor.self_distillation.distillation_topk=100",
        "actor_rollout_ref.actor.self_distillation.max_reprompt_len=10240",
        "actor_rollout_ref.actor.self_distillation.is_clip=2.0",
        f"actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success={str(config.dont_reprompt_on_self_success)}",
        f"actor_rollout_ref.actor.self_distillation.include_environment_feedback={str(config.include_environment_feedback)}",
        "actor_rollout_ref.actor.optim.lr_warmup_steps=10",
        f"actor_rollout_ref.rollout.n={config.rollout_batch_size}",
        "actor_rollout_ref.rollout.calculate_log_probs=True",
        f"actor_rollout_ref.rollout.val_kwargs.n={config.val_n}",
        f"actor_rollout_ref.rollout.max_model_len={config.max_prompt_length + config.max_response_length}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={config.max_prompt_length + config.max_response_length}",
        "algorithm.adv_estimator=grpo",
        "algorithm.norm_adv_by_std_in_grpo=False",
        "algorithm.rollout_correction.rollout_is=token",
        "algorithm.rollout_correction.rollout_is_threshold=2.0",
        f"custom_reward_function.path={reward_fn_path}",
    ]

    if config.lora_path:
        cmd.append(f"actor_rollout_ref.model.lora_adapter_path={config.lora_path}")

    return cmd


def run_sdpo_training_baseline(
    model_path: str,
    data_path: str,
    batch_size: int,
    real_data_epochs: int,
    output_base_dir: Optional[str] = None,
    device_num: int = 1,
    lora_path: Optional[str] = None,
    learning_rate: float = 1e-5,
    num_prompts_per_batch: Optional[int] = None,
    max_seq_length: int = DEFAULT_MAX_PROMPT_LENGTH,
    alpha: float = 0.5,
):
    if output_base_dir is None:
        output_base_dir = os.path.join(_project_root(), "output")
    os.makedirs(output_base_dir, exist_ok=True)

    rollout_batch_size = num_prompts_per_batch or batch_size
    config = SDPOWrapperConfig(
        model_path=model_path,
        data_path=data_path,
        output_base_dir=output_base_dir,
        num_train_epochs=real_data_epochs,
        learning_rate=learning_rate,
        train_batch_size=batch_size,
        rollout_batch_size=rollout_batch_size,
        max_prompt_length=max_seq_length,
        max_response_length=max_seq_length,
        alpha=alpha,
        lora_path=lora_path,
        n_gpus_per_node=max(1, device_num),
    )
    _validate_runtime_paths(config)

    output_dir = _build_output_dir(config)
    os.makedirs(output_dir, exist_ok=True)
    config.output_dir = output_dir

    logger.info(f"SDPO model_path: {model_path}")
    logger.info(f"SDPO data_path: {data_path}")
    logger.info(f"SDPO output_dir: {output_dir}")
    logger.info(f"SDPO epochs: {real_data_epochs}")
    logger.info(f"SDPO train_batch_size: {batch_size}")
    logger.info(f"SDPO rollout_batch_size: {rollout_batch_size}")
    logger.info(f"SDPO max_seq_length: {max_seq_length}")

    dataset_dir = _prepare_sdpo_dataset_dir(config, output_dir)
    command = _build_sdpo_command(config, output_dir, dataset_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = _sdpo_repo_root() + os.pathsep + env.get("PYTHONPATH", "")
    env["VLLM_USE_V1"] = env.get("VLLM_USE_V1", "1")

    with open(os.path.join(output_dir, "sdpo_launch_command.sh"), "w", encoding="utf-8") as f:
        f.write(subprocess.list2cmdline(command) + "\n")

    subprocess.run(command, cwd=_sdpo_repo_root(), env=env, check=True)
    adapter_dir = _find_latest_sdpo_adapter_dir(output_dir)

    meta = {
        "train_type": "sdpo_original_wrapper",
        "base_model_path": model_path,
        "data_path": data_path,
        "prepared_dataset_dir": dataset_dir,
        "system_prompt": SYSTEM_PROMPT,
        "num_train_epochs": real_data_epochs,
        "train_batch_size": batch_size,
        "rollout_batch_size": rollout_batch_size,
        "max_seq_length": max_seq_length,
        "alpha": alpha,
        "continued_from_lora": bool(lora_path),
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "take_exam_adapter_path": adapter_dir,
    }
    with open(os.path.join(output_dir, "sdpo_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info(f"SDPO training finished. Output saved under {output_dir}")
    logger.info(
        f"TakeExam usage: TakeExam(model_path='{model_path}', use_lora=True, adapter_path='{adapter_dir}')"
    )
    return adapter_dir


if __name__ == "__main__":
    root = _project_root()
    default_model_path = os.path.join(root, "model", "DS", "DeepSeek-R1-Distill-Qwen-7B")
    default_data_path = os.path.join(root, "datasets", "exam", "adv_hints.json")
    default_output_base_dir = os.path.join(root, "output")

    run_sdpo_training_baseline(
        model_path=default_model_path,
        data_path=default_data_path,
        batch_size=4,
        real_data_epochs=1,
        output_base_dir=default_output_base_dir,
    )

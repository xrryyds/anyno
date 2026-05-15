import json
import logging
import os
import shutil
from datasets import Dataset
from .load_dataset import LoadDataset

logger = logging.getLogger(__name__)

_LIVEMATHBENCH_LOCAL_PATH  = "./datasets/data/LiveMathBench-en"
_LIVEMATHBENCH_JSONL_PATH  = "./datasets/data/LiveMathBench-en/livemathbench_test.jsonl"
_LIVEMATHBENCH_HF_NAME     = "llm-2025-sahara/LiveMathBench-en"


class LiveMathBench:
    def __init__(self, split: str = "test", max_size: int = None):
        """Wrapper for llm-2025-sahara/LiveMathBench-en.

        加载优先级（从高到低）：
        1. 本地 JSONL 文件（livemathbench_test.jsonl）—— 完全不依赖 datasets 版本
        2. load_from_disk（本地 arrow 缓存）
        3. 从 HuggingFace / 镜像重新下载

        Args:
            split: Dataset split to load (default: "test").
            max_size: If provided, limit the dataset to the first `max_size` rows
                for faster debugging or smaller runs.
        """
        self.problems, self.solutions, self.answers, self.data_len = \
            self._load_problems_answers(max_size)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _load_problems_answers(cls, max_size):
        """直接返回 (problems, solutions, answers, data_len)，不经过 Dataset 对象。"""

        # ── 优先：从 JSONL 文件读取（版本无关，最稳定）──────────────────
        if os.path.isfile(_LIVEMATHBENCH_JSONL_PATH):
            problems, solutions, answers = [], [], []
            try:
                with open(_LIVEMATHBENCH_JSONL_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        q = row.get("question", "")
                        a = row.get("answer", "")
                        if q and a:
                            problems.append(q)
                            solutions.append(a)
                            answers.append(a)
                            if max_size is not None and len(problems) >= max_size:
                                break
                logger.info(
                    f"[LiveMathBench] Loaded {len(problems)} rows from JSONL: "
                    f"{_LIVEMATHBENCH_JSONL_PATH}"
                )
                return problems, solutions, answers, len(problems)
            except Exception as e:
                logger.warning(f"[LiveMathBench] Failed to read JSONL ({e}), falling back...")

        # ── 备用：通过 datasets 加载（可能因版本问题失败）───────────────
        dataset = cls._load_dataset_obj("test")
        if max_size is not None and max_size < len(dataset):
            dataset = dataset.select(range(max_size))

        problems, solutions, answers = [], [], []
        for row in dataset:
            q = row.get("question", "")
            a = row.get("answer", "")
            if q and a:
                problems.append(q)
                solutions.append(a)
                answers.append(a)
        logger.info(
            f"Loaded LiveMathBench-en: {len(problems)} usable rows from raw size {len(dataset)}"
        )
        return problems, solutions, answers, len(problems)

    @staticmethod
    def _load_dataset_obj(split: str) -> Dataset:
        """Load LiveMathBench-en Dataset object with automatic fallback."""
        local_path = _LIVEMATHBENCH_LOCAL_PATH

        # 第一次尝试：从本地 arrow 缓存加载
        if os.path.isdir(local_path):
            try:
                from datasets import load_from_disk as _load_from_disk
                dataset = _load_from_disk(local_path)
                logger.info(
                    f"[LiveMathBench] Loaded from local cache: {local_path} "
                    f"({len(dataset)} rows)"
                )
                return dataset
            except Exception as e:
                logger.warning(
                    f"[LiveMathBench] load_from_disk failed ({e}). "
                    "Deleting broken cache and re-downloading from HuggingFace..."
                )
                try:
                    shutil.rmtree(local_path)
                except Exception as rm_e:
                    logger.warning(f"[LiveMathBench] Failed to remove cache: {rm_e}")

        # 第二次尝试：从 HuggingFace / 镜像下载
        if not os.environ.get("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            logger.info("[LiveMathBench] HF_ENDPOINT not set, using https://hf-mirror.com")

        logger.info(
            f"[LiveMathBench] Downloading {_LIVEMATHBENCH_HF_NAME} (split={split}) "
            f"from {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}..."
        )
        from datasets import load_dataset as _load_dataset
        dataset = _load_dataset(
            path=_LIVEMATHBENCH_HF_NAME,
            split=split,
            cache_dir="./datasets/cache",
        )
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            dataset.save_to_disk(local_path)
            logger.info(f"[LiveMathBench] Saved fresh cache to {local_path}")
        except Exception as save_e:
            logger.warning(f"[LiveMathBench] Failed to save cache: {save_e}")

        return dataset

    def extract_data(self, dataset: Dataset):
        """Extract parallel lists of problems/solutions/answers from the HF dataset.

        LiveMathBench-en rows are expected to have at least `question` and `answer`
        fields. We duplicate `answer` into `solutions` as a placeholder so that
        downstream code, which expects a full solution string, continues to work
        without modification.
        """
        problems = []
        solutions = []
        answers = []

        for data in dataset:
            question = data.get("question", None)
            answer = data.get("answer", None)

            if not question or not answer:
                continue

            problems.append(question)
            # Placeholder: use final answer as both solution and answer
            solutions.append(answer)
            answers.append(answer)

        logger.info(
            f"Loaded LiveMathBench-en: {len(problems)} usable rows from raw size {len(dataset)}"
        )
        return problems, solutions, answers, len(problems)

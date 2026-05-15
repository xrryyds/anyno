import logging
import os
import shutil
from datasets import Dataset
from .load_dataset import LoadDataset

logger = logging.getLogger(__name__)

_LIVEMATHBENCH_LOCAL_PATH = "./datasets/data/LiveMathBench-en"
_LIVEMATHBENCH_HF_NAME    = "llm-2025-sahara/LiveMathBench-en"


class LiveMathBench:
    def __init__(self, split: str = "test", max_size: int = None):
        """Wrapper for llm-2025-sahara/LiveMathBench-en.

        Args:
            split: Dataset split to load (default: "test").
            max_size: If provided, limit the dataset to the first `max_size` rows
                for faster debugging or smaller runs.
        """
        dataset = self._load_dataset(split)

        if max_size is not None and max_size < len(dataset):
            dataset = dataset.select(range(max_size))

        self.problems, self.solutions, self.answers, self.data_len = self.extract_data(
            dataset
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_dataset(split: str) -> Dataset:
        """Load LiveMathBench-en with automatic fallback.

        Strategy:
        1. Try load_from_disk (fast, uses local arrow cache).
        2. If that fails (e.g. old datasets version can't parse 'List' type),
           delete the broken cache and re-download from HuggingFace as parquet,
           then save fresh arrow cache for next time.
        """
        local_path = _LIVEMATHBENCH_LOCAL_PATH

        # ── 第一次尝试：从本地 arrow 缓存加载 ──────────────────────────
        if os.path.isdir(local_path):
            try:
                from datasets import load_from_disk as _load_from_disk
                dataset = _load_from_disk(local_path)
                if split != "test":
                    # load_from_disk 返回的是单个 split Dataset，不需要再 split
                    pass
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

        # ── 第二次尝试：从 HuggingFace 下载（parquet → arrow 缓存）──────
        # 优先使用 HF_ENDPOINT 镜像（如 https://hf-mirror.com），若未设置则尝试设置默认镜像
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
        # 保存到本地 arrow 缓存，下次直接读
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

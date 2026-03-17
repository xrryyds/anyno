import logging
from datasets import Dataset
from .load_dataset import LoadDataset

logger = logging.getLogger(__name__)


class LiveMathBench:
    def __init__(self, split: str = "test", max_size: int = None):
        """Wrapper for llm-2025-sahara/LiveMathBench-en.

        Args:
            split: Dataset split to load (default: "test").
            max_size: If provided, limit the dataset to the first `max_size` rows
                for faster debugging or smaller runs.
        """
        dataset_loader = LoadDataset(
            dataset_name="llm-2025-sahara/LiveMathBench-en",
            split=split,
            local_path="./datasets/data/LiveMathBench-en",
            config=None,
        )

        if max_size is not None:
            try:
                dataset_loader.set_dataset_size(max_size)
            except Exception as e:
                logger.warning(f"Failed to set dataset size to {max_size}: {e}")

        self.problems, self.solutions, self.answers, self.data_len = self.extract_data(
            dataset_loader.get_dataset()
        )

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

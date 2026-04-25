import logging
from datasets import Dataset
from .load_dataset import LoadDataset
from prompt import QUESTION_PROMPT, ANSWER_PROMPT
from sklearn.model_selection import train_test_split 
from .math_dataset import Math_DataSet
from .math_data_util import Math_data


logger = logging.getLogger(__name__)


class AIME_1983_2024():
    def __init__(self):
        dataset_name = 'gneubig/aime-1983-2024'

        dataset_loader = LoadDataset(
            dataset_name=dataset_name,
            split='train',
            local_path=f'./datasets/data/aime_1983_2024',
        )

        self.problems, self.solutions, self.answers, self.data_len = self.extract_data(
            dataset_loader.get_dataset())
        print("######################")
        print(len(self.problems))

    def extract_data(self, dataset: Dataset) -> tuple[list, list, list, int]:
        problems = []
        solutions = []
        answers = []

        for data in dataset:
            problem = data.get("Question", None)
            solution = data.get("Answer", None)
            answer = data.get("Answer", None)
            if problem and solution and answer:
                problems.append(problem)
                solutions.append(solution)
                answers.append(answer)
        return problems, solutions, answers, len(problems)

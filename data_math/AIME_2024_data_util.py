import logging
from datasets import Dataset
from .load_dataset import LoadDataset
from prompt import QUESTION_PROMPT, ANSWER_PROMPT
from sklearn.model_selection import train_test_split 
from .math_dataset import Math_DataSet
from .math_data_util import Math_data


logger = logging.getLogger(__name__)


class AIME2024():
    def __init__(self, train:bool=True):
        dataset_loader = LoadDataset(
            dataset_name='HuggingFaceH4/aime_2024',
            split='train',
            local_path='./datasets/data/aime_2024',
        )

        self.problems, self.solutions, self.answers, self.data_len = self.extract_data(
            dataset_loader.get_dataset())

    def extract_data(self, dataset: Dataset) -> tuple[list, list, list, int]:
        problems = []
        solutions = []
        answers = []

        for data in dataset:
            problem = data.get("problem", None)
            solution = data.get("solution", None)
            answer = data.get("answer", None)

            if problem and solution and answer:
                problems.append(problem)
                solutions.append(solution)
                answers.append(answer)
        return problems, solutions, answers, len(problems)
            
def main():
   data = AIME2024(False)
   split ="#" * 20
   print("problems:" + data.problems[0])
   print(split)
   print("train_answer:" + data.answers[0])
   print(split)
   print("solution:"+ data.solutions[0])
         


if __name__ == "__main__":
    main()

import logging
from datasets import Dataset
from .load_dataset import LoadDataset
from prompt import QUESTION_PROMPT, ANSWER_PROMPT
from configs import GRPOConfig
from sklearn.model_selection import train_test_split 
from .math_dataset import Math_DataSet
from .math_data_util import Math_data
from utils import extract_boxed_content


logger = logging.getLogger(__name__)


class Math_500():
    def __init__(self):
        dataset_loader = LoadDataset(
            dataset_name='HuggingFaceH4/MATH-500',
            split='test',
            local_path='./datasets/data/MATH-500'
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

    def gen_prompt(self, data: list, max_token: int = 512):
        for i in range(len(data)):
            data[i] = QUESTION_PROMPT.format(
                max_token=max_token,
                problem_text=data[i]
            )
            
    
def main():
    math_500 = Math_500()

    spilt = "=============================="


    print("problems:" + math_500.problems[0])
    print(spilt)
    print("train_solution:" + math_500.solutions[0])
    print(spilt)
    print("train_answer:" + math_500.answers[0])
    print(spilt)

         


if __name__ == "__main__":
    main()


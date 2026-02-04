import logging
import os
from datasets import Dataset
from .load_dataset import LoadDataset
from prompt import QUESTION_PROMPT
from utils import extract_boxed_content

logger = logging.getLogger(__name__)

class Math_Subset():
    def __init__(self, subset: str, train: bool = True):
        """
            subset options: "algebra", "counting_and_probability", "geometry", 
                       "intermediate_algebra", "number_theory", "prealgebra", "precalculus"
        """
        
        if train:
            target_split = 'train'
            local_path = os.path.join('./datasets/data/MATH/train')
        else:
            target_split = 'test'
            local_path = os.path.join('./datasets/data/MATH/test')

        logger.info(f"Loading MATH subset")

        dataset_loader = LoadDataset(
            dataset_name='qwedsacf/competition_math',
            split=target_split,
            local_path=local_path,
            # config=subset
        )

        self.problems, self.solutions, self.answers, self.data_len = self.extract_data(
            dataset_loader.get_dataset())

    def extract_data(self, dataset: Dataset) -> tuple[list, list, list, int]:
        problems = []
        solutions = []
        answers = []

        if dataset is None:
            return [], [], [], 0

        for data in dataset:
            problem = data.get("problem", "").strip()
            solution = data.get("solution", "").strip()
            answer = data.get("answer", "")

            if not answer and solution:
                answer = extract_boxed_content(solution)

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
    math_subset = Math_Subset(subset=subset_name, train=False)

    if math_subset.data_len == 0:
        print("Error: No data loaded.")
        return

    spilt = "=============================="

    print(f"Total Data Loaded: {math_subset.data_len}")
    print(spilt)
    print("problems:" + math_subset.problems[0])
    print(spilt)
    print("solution:" + math_subset.solutions[0])
    print(spilt)
    print("answer:" + math_subset.answers[0])
    print(spilt)


if __name__ == "__main__":
    main()

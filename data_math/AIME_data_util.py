import logging
from datasets import Dataset
from .load_dataset import LoadDataset
from prompt import QUESTION_PROMPT, ANSWER_PROMPT
from sklearn.model_selection import train_test_split 
from .math_dataset import Math_DataSet
from .math_data_util import Math_data


logger = logging.getLogger(__name__)


class AIME():
    def __init__(self, year: int = 2024):
        dataset_mapping = {
            2024: 'math-ai/aime24',
            2025: 'math-ai/aime25',
            2026: 'math-ai/aime26',
        }

        if year not in dataset_mapping:
            raise ValueError("year must be 2024, 2025 or 2026")

        dataset_loader = LoadDataset(
            dataset_name=dataset_mapping[year],
            split='test',
            local_path=f'./datasets/data/aime_{year}',
        )

        self.year = year
        self.problems, self.solutions, self.answers, self.data_len = self.extract_data(
            dataset_loader.get_dataset(), year)
        print("######################")
        print(len(self.problems))

    def extract_data(self, dataset: Dataset, year) -> tuple[list, list, list, int]:
        problems = []
        solutions = []
        answers = []

        if year == 2024:
            for data in dataset:
                problem = data.get("problem", None)
                solution = data.get("solution", None)
                answer = data.get("solution", None)
                if problem and solution and answer:
                    problems.append(problem)
                    solutions.append(solution)
                    answers.append(answer)
        else:
            for data in dataset:
                problem = data.get("problem", None)
                solution = data.get("answer", None)
                answer = data.get("answer", None)
                if problem and solution and answer:
                    problems.append(problem)
                    solutions.append(solution)
                    answers.append(answer)
        return problems, solutions, answers, len(problems)
            
def main():
   data = AIME(False, year=2024)
   split ="#" * 20
   print("problems:" + data.problems[0])
   print(split)
   print("train_answer:" + data.answers[0])
   print(split)
   print("solution:"+ data.solutions[0])
         


if __name__ == "__main__":
    main()

import logging
from datasets import Dataset
from .load_dataset import LoadDataset
from prompt import QUESTION_PROMPT, ANSWER_PROMPT
from sklearn.model_selection import train_test_split 
from .math_dataset import Math_DataSet

class Math_data():
    def __init__(self):
        dataset_loader = LoadDataset(
            dataset_name='',
            split='',
            local_path=''
        )

        self.problems, self.solutions, self.answers, self.data_len = self.extract_data(
            dataset_loader.get_dataset())


        self.gen_prompt(self.problems, max_token=2048)
        self.gen_answer(self.answers)
        
        (self.train_problems, self.test_problems,
         self.train_solutions, self.test_solutions,
         self.train_answers, self.test_answers) = train_test_split(
            self.problems, self.solutions, self.answers,
            test_size=0.2, 
            random_state=42,  
            shuffle=True  
        )

        self.train_data =  Math_DataSet(self.train_problems, self.train_solutions,  self.train_answers)
        self.test_data = Math_DataSet(self.test_problems, self.test_solutions, self.test_answers)

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
            
    def gen_answer(self):
        for i in range(len(self.answers)):
            self.answers[i] = ANSWER_PROMPT.format(
                answer = self.answers[i],
                thinking = self.solutions[i]
            )
            
    def get_data(self):
        return self.train_problems, self.train_solutions, self.train_answers


    def get_train_data(self):
        return self.train_data
    
    def get_test_data(self):
        return self.test_data
    
    def get_dataset(self):
        return self.train_data, self.test_data
    
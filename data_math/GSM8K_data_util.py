import logging
from datasets import Dataset
from .load_dataset import LoadDataset
from prompt import QUESTION_PROMPT, ANSWER_PROMPT
from configs import GRPOConfig
from sklearn.model_selection import train_test_split 
from .math_dataset import Math_DataSet
from .math_data_util import Math_data


logger = logging.getLogger(__name__)


class GSM8K():
    def __init__(self, train:bool=True):
        if train:
            dataset_loader = LoadDataset(
                dataset_name='gsm8k',
                split='train',
                local_path='./datasets/data/gsm8k',
                config='main'
            )
        else:
            dataset_loader = LoadDataset(
                dataset_name='gsm8k',
                split='test',
                local_path='./datasets/data/gsm8k_testpythp',
                config='main'
            ) 

        self.problems, self.solutions, self.answers, self.data_len = self.extract_data(
            dataset_loader.get_dataset())

    def extract_data(self, dataset: Dataset) -> tuple[list, list, list, int]:
        problems = []
        solutions = []
        answers = []

        for data in dataset:
            question = data.get("question", "").strip()
            answer_text = data.get("answer", "").strip()

            if not question or not answer_text:
                continue
            
            # 初始化为空，防止读取上一轮的数据
            solution_text = None
            final_answer = None

            if "####" in answer_text:
                parts = answer_text.split("####")
                if len(parts) >= 2: # 增加安全性检查，防止 split 后长度不足
                    final_answer = parts[1].strip()  
            else:
                print("error:" + answer_text)

            # 只有当本次循环成功提取到内容时才追加
            if final_answer:
                problems.append(question)
                solutions.append(answer_text)
                answers.append(final_answer)

        return problems, solutions, answers, len(problems)
            
def main():
   gms8k = GSM8K(False)
   split ="#" * 20
   print("problems:" + gms8k.problems[0])
   print(split)
   print("train_answer:" + gms8k.answers[0])
   print(split)
   print("solution:"+ gms8k.solutions[0])
         


if __name__ == "__main__":
    main()

import logging
import os
from datasets import Dataset, concatenate_datasets
from .load_dataset import LoadDataset
from prompt import QUESTION_PROMPT, ANSWER_PROMPT
from utils import extract_boxed_content

logger = logging.getLogger(__name__)

class Math_All():
    def __init__(self, train: bool = True):
        # MATH 数据集的子集列表
        target_subsets = [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus"
        ]

        loaded_datasets = []
        
        # 根据 train 参数设定分片和基础路径
        if train:
            target_split = 'train'
            local_path_base = './datasets/data/MATH/train' # 为了更清晰，建议加上 /train
        else:
            target_split = 'test'
            local_path_base = './datasets/data/MATH/test'

        logger.info(f"Loading MATH dataset (Split: {target_split})...")
        print(f"Start loading MATH dataset (Split: {target_split})...")

        for subset in target_subsets:
            try:
                # 【关键修改】: 
                # 每个子集必须有独立的文件夹，否则不同子集的元数据会互相覆盖导致报错
                current_local_path = os.path.join(local_path_base, subset)
                
                dataset_loader = LoadDataset(
                    dataset_name='HuggingFaceH4/MATH',
                    split=target_split,
                    local_path=current_local_path, # 传入独立的路径
                    config=subset 
                )
                
                ds = dataset_loader.get_dataset()
                # 只有当成功获取到 dataset 对象时才加入列表
                if ds is not None:
                    loaded_datasets.append(ds)
                    logger.info(f"Successfully loaded subset: {subset}")
                    
            except Exception as e:
                logger.warning(f"Failed to load subset '{subset}' split '{target_split}': {e}")
                print(f"Warning: Failed to load {subset}. Error: {e}")

        if not loaded_datasets:
            raise ValueError(f"No datasets loaded for split {target_split}. Please check network or clean cache.")

        # 合并所有子集
        full_dataset = concatenate_datasets(loaded_datasets)
        
        # 提取并处理数据
        self.problems, self.solutions, self.answers, self.data_len = self.extract_data(full_dataset)

    def extract_data(self, dataset: Dataset) -> tuple[list, list, list, int]:
        problems = []
        solutions = []
        answers = []

        for data in dataset:
            problem = data.get("problem", "").strip()
            solution = data.get("solution", "").strip()
            answer = data.get("answer", "") # 可能为 None 或 ""

            # 基础校验
            if not problem or not solution:
                continue
            
            # 如果没有直接的 answer 字段，尝试从 solution 中提取
            if not answer:
                answer = extract_boxed_content(solution)
            
            # 只有提取到有效答案才加入
            if answer:
                problems.append(problem)
                solutions.append(solution)
                answers.append(answer)

        return problems, solutions, answers, len(problems)
            
def main():
    # ---------------------------------------------------------
    # 示例 1: 下载并查看 Train 集
    # ---------------------------------------------------------
    print("="*20 + " Loading TRAIN Set " + "="*20)
    try:
        math_train = Math_All(train=True)
        print(f"Total Train Data: {math_train.data_len}")
        if math_train.data_len > 0:
            print(f"Sample Problem (Train): {math_train.problems[0][:100]}...")
            print(f"Sample Answer (Train): {math_train.answers[0]}")
    except Exception as e:
        print(f"Train Load Error: {e}")

    print("\n")

    # ---------------------------------------------------------
    # 示例 2: 下载并查看 Test 集
    # ---------------------------------------------------------
    print("="*20 + " Loading TEST Set " + "="*20)
    try:
        math_test = Math_All(train=True)
        split_line = "#" * 20
        print(f"Total Test Data: {math_test.data_len}")
        
        if math_test.data_len > 0:
            print(split_line)
            print("Sample Problem (Test):" + math_test.problems[0])
            print(split_line)
            print("Sample Answer (Test):" + math_test.answers[0])
            print(split_line)
            print("Sample solution (Test):" + math_test.solutions[0])
    except Exception as e:
        print(f"Test Load Error: {e}")

if __name__ == "__main__":
    main()

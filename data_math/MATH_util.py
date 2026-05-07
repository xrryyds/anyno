import logging
import os
import json
from datasets import Dataset, concatenate_datasets
from .load_dataset import LoadDataset
from utils import extract_boxed_content

logger = logging.getLogger(__name__)

class Math_All():
    def __init__(self, train: bool = True, subset_name: str = "all", shuffle: bool = True):
        """
        Args:
            train (bool): True , False 
            subset_name (str):  "algebra"
                                "all"
            shuffle (bool): 
        """
        all_possible_subsets = [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus"
        ]

        if subset_name.lower() == "all":
            target_subsets = all_possible_subsets
            logger.info(f"Mode: Load ALL subsets and MIX.")
        elif subset_name in all_possible_subsets:
            target_subsets = [subset_name]
            logger.info(f"Mode: Load single subset '{subset_name}'.")
        else:
            raise ValueError(f"Invalid subset_name: '{subset_name}'. Must be 'all' or one of {all_possible_subsets}")

        if train:
            target_split = 'train'
            local_path_base = './datasets/data/MATH/train'
        else:
            target_split = 'test'
            local_path_base = './datasets/data/MATH/test'

        loaded_datasets = []
        print(f"Start loading MATH dataset (Split: {target_split}, Target: {subset_name})...")

        for subset in target_subsets:
            try:
                current_local_path = os.path.join(local_path_base, subset)
                
                dataset_loader = LoadDataset(
                    dataset_name='HuggingFaceH4/MATH',
                    split=target_split,
                    local_path=current_local_path,
                    config=subset 
                )
                
                ds = dataset_loader.get_dataset()
                
                if ds is not None:
                    loaded_datasets.append(ds)
                    print(f" - Loaded: {subset} ({len(ds)} rows)")
            except Exception as e:
                logger.warning(f"Failed to load {subset}: {e}")

        if not loaded_datasets:
            raise ValueError(f"No datasets loaded. Please check paths or network.")

        full_dataset = concatenate_datasets(loaded_datasets)
        
        if shuffle:
            print("Shuffling (mixing) data...")
            full_dataset = full_dataset.shuffle(seed=42)

        self.problems, self.solutions, self.answers, self.data_len = self.extract_data(full_dataset)
        print(f"Done. Total valid samples: {self.data_len}")

    def extract_data(self, dataset: Dataset) -> tuple[list, list, list, int]:
        problems = []
        solutions = []
        answers = []

        for data in dataset:
            problem = data.get("problem", "").strip()
            solution = data.get("solution", "").strip()
            answer = data.get("answer", "") 

            if not problem or not solution:
                continue
            
            if not answer:
                answer = extract_boxed_content(solution)
            
            if answer:
                problems.append(problem)
                solutions.append(solution)
                answers.append(answer)

        return problems, solutions, answers, len(problems)

def save_to_jsonl(math_obj, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for p, s, a in zip(math_obj.problems, math_obj.solutions, math_obj.answers):
            entry = {"problem": p, "solution": s, "answer": a}
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def main():
    # ==========================================
    # ==========================================
    try:
        print("\n--- 1. Loading ALL Train Data (Mixed) ---")
        math_all_train = Math_All(train=True, subset_name="all")
        
        save_to_jsonl(math_all_train, "./processed_data/math_train_all_mixed.jsonl")
        
        print("Preview (Problem Start):")
        for i in range(3):
            print(f" {i+1}. {math_all_train.problems[i][:50]}...")

    except Exception as e:
        print(f"Error: {e}")

    # ==========================================
    # ==========================================
    try:
        print("\n--- 2. Loading ONLY Geometry Train Data ---")
        math_geo_train = Math_All(train=True, subset_name="geometry")
        
        save_to_jsonl(math_geo_train, "./processed_data/math_train_geometry.jsonl")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

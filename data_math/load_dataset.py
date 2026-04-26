import logging
import os
from .hf_datasets import load_dataset, load_from_disk

logger = logging.getLogger(__name__)


class LoadDataset:
    def __init__(self, dataset_name, split, local_path, config=None):
        logger.info(f"Loading Dataset: {dataset_name} ({split})...")
        self.local_path = local_path
        dataset = None
        try:
            if os.path.exists(local_path):
                dataset = load_from_disk(local_path)
            else:
                dataset = load_dataset(
                    path=dataset_name,
                    name = config,
                    split=split,
                    cache_dir="./datasets/cache" 
                )
                dataset.save_to_disk(local_path)
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise e
        self.dataset = dataset



    def __len__(self): return len(self.dataset)
    
    def get_dataset(self): return self.dataset
    
    def set_dataset_size(self, size: int):
        if size > len(self.dataset):
            logger.warning(f"Requested size {size} is larger than dataset length {len(self.dataset)}. Keeping original dataset.")
            return
        self.dataset = self.dataset.select(range(size))

def main():
    dataset_loader = LoadDataset(
        dataset_name='HuggingFaceH4/MATH-500',
        split='test',
        local_path='./datasets/data/MATH-500'
    )
    print(f"Dataset length: {len(dataset_loader.get_dataset())}")
    
    print(dataset_loader.get_dataset()[0])

if __name__ == "__main__":
    main()

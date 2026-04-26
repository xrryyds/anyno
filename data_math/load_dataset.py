import logging
import os
import shutil
from .hf_datasets import load_dataset, load_from_disk

logger = logging.getLogger(__name__)


class LoadDataset:
    def __init__(self, dataset_name, split, local_path, config=None):
        logger.info(f"Loading Dataset: {dataset_name} ({split})...")
        self.local_path = local_path
        dataset = None
        try:
            if self._is_valid_saved_dataset(local_path):
                logger.info(f"Loading dataset from local cache: {local_path}")
                dataset = load_from_disk(local_path)
            elif os.path.exists(local_path):
                logger.warning(
                    f"Local path exists but is not a valid saved dataset: {local_path}. "
                    "Removing it and re-downloading."
                )
                self._remove_local_path(local_path)

            if dataset is None:
                logger.info(f"Downloading dataset from hub: {dataset_name}")
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                dataset = load_dataset(
                    path=dataset_name,
                    name=config,
                    split=split,
                    cache_dir="./datasets/cache"
                )
                dataset.save_to_disk(local_path)
                logger.info(f"Saved dataset to local cache: {local_path}")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise e
        self.dataset = dataset

    @staticmethod
    def _is_valid_saved_dataset(local_path: str) -> bool:
        if not local_path or not os.path.isdir(local_path):
            return False

        state_path = os.path.join(local_path, "state.json")
        dataset_info_path = os.path.join(local_path, "dataset_info.json")
        return os.path.isfile(state_path) or os.path.isfile(dataset_info_path)

    @staticmethod
    def _remove_local_path(local_path: str) -> None:
        if not os.path.exists(local_path):
            return
        if os.path.isdir(local_path):
            shutil.rmtree(local_path)
        else:
            os.remove(local_path)



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

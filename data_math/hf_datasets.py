import importlib
import os
import sys


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _import_hf_datasets():
    project_root = os.path.realpath(_project_root())
    original_sys_path = list(sys.path)
    try:
        filtered_sys_path = []
        for entry in sys.path:
            normalized = os.path.realpath(entry or os.getcwd())
            if normalized == project_root:
                continue
            filtered_sys_path.append(entry)
        sys.path = filtered_sys_path
        try:
            return importlib.import_module("datasets")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Hugging Face 'datasets' package is not installed in the current Python environment. "
                "Please install it with `pip install datasets` or use the project's expected conda environment."
            ) from exc
    finally:
        sys.path = original_sys_path


_hf_datasets = _import_hf_datasets()

Dataset = _hf_datasets.Dataset
load_dataset = _hf_datasets.load_dataset
load_from_disk = _hf_datasets.load_from_disk

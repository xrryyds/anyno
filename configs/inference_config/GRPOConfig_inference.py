import logging
import yaml
from dataclasses import dataclass
import numpy as np
import datetime 
import torch

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfigInference:
    model_path: str = ".../outputs/grpo_qwen/final_model"
    base_model: str = "Qwen/Qwen2-1.5B-Instruct"
    max_length: int = 1024
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    num_return_sequences: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    @classmethod
    def load_yaml(cls, yaml_path: str):
        logger.info(f"Loading config from {yaml_path}")
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        return cls(
            # Model
            model_path=cfg['model']['model_path'],
            base_model=cfg['model']['base_model'],
            max_length=cfg['inference']['max_length'],
            max_new_tokens=cfg['inference']['max_new_tokens'],
            temperature=cfg['inference']['temperature'],
            top_p=cfg['inference']['top_p'],
            top_k=cfg['inference']['top_k'],
            num_return_sequences=cfg['inference']['num_return_sequences'],
            device=cfg['inference']['device'],
            use_lora=cfg['model']['use_lora'],
            lora_r=cfg['lora']['r'],
            lora_alpha=cfg['lora']['alpha'],
            lora_dropout=cfg['lora']['dropout'],
        )
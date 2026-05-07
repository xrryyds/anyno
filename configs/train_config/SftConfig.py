import logging
import yaml
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SftConfig:
    model_name: str = "Qwen/Qwen2-1.5B-Instruct"
    output_dir: str = "/home/xrrfolder/CELPO/outputs/sft"
    bf16: bool = True
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    save_steps: int = 50
    logging_steps: int = 10
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    num_train_epochs: int = 1
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    report_to: str = "none"
    
    lora_r: int = 8
    lora_alpha: int = 16
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    max_seq_length: int = 1024

    @classmethod
    def load_yaml(cls, yaml_path: str):
        """YAMLConfig"""
        logger.info(f"Loading config from {yaml_path}")
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        return cls(
            # Model
            model_name=cfg['model']['name'],
            max_seq_length=cfg['model']['max_seq_length'],
            
            # LoRA
            lora_r=cfg['lora']['r'],
            lora_alpha=cfg['lora']['alpha'],
            bias=cfg['lora']['bias'],
            task_type=cfg['lora']['task_type'],
            
            # Training
            learning_rate=float(cfg['training']['learning_rate']),
            per_device_train_batch_size=cfg['training']['batch_size'],
            gradient_accumulation_steps=cfg['training']['gradient_accumulation_steps'],
            num_train_epochs=cfg['training']['num_epochs'],
            max_grad_norm=float(cfg['training']['max_grad_norm']),
            warmup_ratio=float(cfg['training']['warmup_ratio']),
            lr_scheduler_type=cfg['training']['lr_scheduler_type'],
            bf16=cfg['training']['bf16'],
            
            # System
            output_dir=cfg['system']['output_dir'],
            save_steps=cfg['system']['save_steps'],
            logging_steps=cfg['system']['logging_steps'],
            report_to=cfg['system']['report_to']
        )

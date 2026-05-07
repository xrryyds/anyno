import os
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_to_base_model(
    base_model_path: str,
    lora_path: str,
    output_model_path: str,
    trust_remote_code: bool = True,
) -> None:
    """ LoRA 

    
        base_model_path:  HuggingFace Hub 
        lora_path:  LoRA 
        output_model_path: 
        trust_remote_code:  True

    
        -  peft.PeftModel.from_pretrained  LoRA 
        -  merge_and_unload()  LoRA “”
        -  model.save_pretrained(output_model_path) 
        -  tokenizer  output_model_path 
    """

    os.makedirs(output_model_path, exist_ok=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=trust_remote_code,
    )

    lora_model = PeftModel.from_pretrained(base_model, lora_path)

    merged_model = lora_model.merge_and_unload()

    merged_model.save_pretrained(output_model_path)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )
        tokenizer.save_pretrained(output_model_path)
    except Exception:
        pass

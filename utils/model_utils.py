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
    """将 LoRA 权重与基座模型合并，并将合并后的完整模型保存到指定目录。

    参数：
        base_model_path: 基座模型目录或 HuggingFace Hub 名称。
        lora_path: 已训练好的 LoRA 权重目录。
        output_model_path: 合并后模型的输出目录（如不存在会自动创建）。
        trust_remote_code: 是否信任自定义模型代码，默认 True，与项目中其它加载方式保持一致。

    说明：
        - 使用 peft.PeftModel.from_pretrained 加载 LoRA 到基座模型上。
        - 调用 merge_and_unload() 将 LoRA 权重真正合并进模型权重中，得到一个“纯”模型。
        - 使用 model.save_pretrained(output_model_path) 导出模型权重。
        - 同时尝试将基座模型对应的 tokenizer 一并保存到 output_model_path 方便后续直接加载。
    """

    os.makedirs(output_model_path, exist_ok=True)

    # 1. 加载基座模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=trust_remote_code,
    )

    # 2. 加载 LoRA 并合并
    lora_model = PeftModel.from_pretrained(base_model, lora_path)

    # merge_and_unload 会把 LoRA 权重真正写回到 base_model 的权重里，并移除 PEFT 结构
    merged_model = lora_model.merge_and_unload()

    # 3. 保存合并后的完整模型
    merged_model.save_pretrained(output_model_path)

    # 4. 同步保存 tokenizer（如果存在的话）
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )
        tokenizer.save_pretrained(output_model_path)
    except Exception:
        # tokenizer 保存失败并不会影响模型本身的导出，按需在调用方处理
        pass

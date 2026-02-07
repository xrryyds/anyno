#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" 
import logging
from huggingface_hub import snapshot_download
# =====================================================
# Logger
# # =====================================================
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
# )
# logger = logging.getLogger(__name__)

# MY_TOKEN = ""

# current_file_path = os.path.abspath(__file__)
# project_root = os.path.dirname(os.path.dirname(current_file_path)) 
# model_dir = os.path.join(project_root, "CELPO", "model", "OREAL")
# logger.info("downloading...")
# snapshot_download(
#     repo_id="internlm/OREAL-32B",
#     local_dir= os.path.join(model_dir, "OREAL-32B"),
#     token=MY_TOKEN,
#     max_workers=1,                 # 🔴 关键
#     # resume_download=True,          # 🔴 关键
#     # local_dir_use_symlinks=False
# )
# logger.info("finished！")



import os
from modelscope.hub.snapshot_download import snapshot_download

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path)) 
# ModelScope 下载不需要 token (通常)
save_dir = os.path.join(project_root, "CELPO", "model", "DS", "DeepSeek-R1-Distill-Qwen-7B")

print("正在从魔搭社区下载...")
snapshot_download(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", # ModelScope 上的对应 ID
    cache_dir=None,       # 设为 None 以便直接下载到 local_dir
    local_dir=save_dir,
    revision='master'
)
print("下载完成！")


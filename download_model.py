#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" 
import logging
from huggingface_hub import snapshot_download
# =====================================================
# Logger
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MY_TOKEN = ""

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path)) 
model_dir = os.path.join(project_root, "CELPO", "model", "OREAL")
logger.info("downloading...")
snapshot_download(
    repo_id="internlm/OREAL-32B",
    local_dir= os.path.join(model_dir, "OREAL-32B"),
    token=MY_TOKEN,
    max_workers=1,                 # 🔴 关键
    resume_download=True,          # 🔴 关键
    local_dir_use_symlinks=False
)
logger.info("finished！")
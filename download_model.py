import os
from huggingface_hub import snapshot_download
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" 

from huggingface_hub import snapshot_download

# 替换成你的 HF Token，以 hf_ 开头
MY_TOKEN = ""



current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path)) 
model_dir = os.path.join(project_root, "CELPO", "model", "OREAL")
print("开始下载模型...")
snapshot_download(
    repo_id="internlm/OREAL-7B",
    local_dir= os.path.join(model_dir, "OREAL-7B"),
    token=MY_TOKEN,
    max_workers=1,                 # 🔴 关键
    resume_download=True,          # 🔴 关键
    local_dir_use_symlinks=False
)
print("下载完成！")

# current_file_path = os.path.abspath(__file__)
# project_root = os.path.dirname(os.path.dirname(current_file_path)) 
# model_dir = os.path.join(project_root, "CELPO", "model", "Qwen")


# print("开始下载模型...")
# snapshot_download(
#     repo_id="Qwen/Qwen2.5-7B-Instruct",
#     local_dir= os.path.join(model_dir, "Qwen2.5-Math-7B-Instruct"),
#     max_workers=8,
#     token=MY_TOKEN 
# )
# print("下载完成！")
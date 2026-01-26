import os
from huggingface_hub import snapshot_download

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 替换成你的 HF Token，以 hf_ 开头
MY_TOKEN = ""

# print("开始下载模型...")
# snapshot_download(
#     repo_id="internlm/OREAL-7B",
#     local_dir="/root/project/data/xrr/OREAL-7B",
#     max_workers=8,
#     token=MY_TOKEN  # <--- 加上这一行
# )
# print("下载完成！")

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path)) 
model_dir = os.path.join(project_root, "CELPO", "model", "Qwen")


print("开始下载模型...")
snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir= os.path.join(model_dir, "Qwen2.5-Math-7B-Instruct"),
    max_workers=8,
    token=MY_TOKEN 
)
print("下载完成！")
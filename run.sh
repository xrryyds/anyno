#!/bin/bash

# --- Slurm 资源配置 ---
# 如果你直接运行此脚本，这些参数会生效 (sbatch 模式)
#SBATCH --job-name=OREAL_exam
#SBATCH --partition=ai_science
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# --- 变量配置 ---
CONTAINER_PATH="ubuntu22.04-pytorch2.3.0-py3.10-gpu-cuda12.4.1-deepspeed0.14.2-cudnn9.1.0.70-allreduce-llamafactory0.8.1_v1.0.0.sif"
CONDA_PATH="/mnt/petrelfs/wanhaiyuan/miniforge3/etc/profile.d/conda.sh"
PROJECT_DIR="/mnt/petrelfs/wanhaiyuan/xrr/CELPO" # 请确保这是 main.py 的绝对路径

# --- 核心命令 ---
# 使用 apptainer exec 直接在容器内执行一系列命令
apptainer exec --nv --cleanenv --writable-tmpfs --bind /mnt:/mnt $CONTAINER_PATH /bin/bash << EOF
    # 进入项目目录
    cd $PROJECT_DIR
    
    # 激活 Conda 环境
    source $CONDA_PATH
    conda activate xiong
    
    # 执行程序
    echo "Starting Python main.py..."
    python main.py
EOF

echo "Task finished."
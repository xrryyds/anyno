#!/bin/bash

#SBATCH --job-name=OREAL_exam
#SBATCH --partition=ai_science
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

CONTAINER_PATH="ubuntu22.04-pytorch2.3.0-py3.10-gpu-cuda12.4.1-deepspeed0.14.2-cudnn9.1.0.70-allreduce-llamafactory0.8.1_v1.0.0.sif"
CONDA_PATH="/mnt/petrelfs/wanhaiyuan/miniforge3/etc/profile.d/conda.sh"
PROJECT_DIR="/mnt/petrelfs/wanhaiyuan/xrr/CELPO"

apptainer exec --nv --cleanenv --writable-tmpfs --bind /mnt:/mnt $CONTAINER_PATH /bin/bash << EOF
    cd $PROJECT_DIR
    
    source $CONDA_PATH
    conda activate xiong
    
    echo "Starting Python main.py..."
    python main.py
EOF

echo "Task finished."
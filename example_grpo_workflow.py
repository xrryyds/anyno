"""
GRPO on MATH500 完整工作流示例

这个脚本展示了如何：
1. 使用 run_sira_training_v2 进行 SIRA 训练
2. 使用训练结果在 MATH500 上进行 GRPO
3. 测试 GRPO 后的模型性能

使用方法:
    # 单卡训练
    python example_grpo_workflow.py --step 1  # SIRA 训练
    python example_grpo_workflow.py --step 2  # GRPO 训练
    python example_grpo_workflow.py --step 3  # 测试模型
    
    # 多卡 GRPO 训练
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 example_grpo_workflow.py --step 2
"""

import os
import sys
import argparse
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    grpo_on_MATH500,
    test_grpo_on_MATH500,
    model_path
)
from scripts import run_sira_training_v2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def step1_sira_training():
    """
    步骤 1: SIRA 训练
    
    这一步会：
    - 加载 irdcl_data.json 数据集
    - 使用 SIRA 方法训练模型
    - 当 mode_b_loss_raw 达到目标值时自动停止
    - 保存 checkpoint 到 output/sira_sft_50ep_XXXX/
    """
    logger.info("="*80)
    logger.info("STEP 1: SIRA Training")
    logger.info("="*80)
    
    # 配置参数
    batch_size = 8
    real_data_epochs = 50
    target_mode_b = 0.023  # 目标 Mode B Loss，达到后自动停止
    
    logger.info(f"Configuration:")
    logger.info(f"  - Model Path: {model_path}")
    logger.info(f"  - Batch Size: {batch_size}")
    logger.info(f"  - Max Epochs: {real_data_epochs}")
    logger.info(f"  - Target Mode B Loss: {target_mode_b}")
    logger.info("")
    
    # 运行 SIRA 训练
    run_sira_training_v2(
        model_path=model_path,
        batch_size=batch_size,
        real_data_epochs=real_data_epochs,
        target_mode_b=target_mode_b
    )
    
    logger.info("")
    logger.info("✅ SIRA Training completed!")
    logger.info("📁 Check output directory for checkpoint path")
    logger.info("   Example: output/sira_sft_50ep_0302_1046/checkpoint-target-reached-epoch-10")
    logger.info("")


def step2_grpo_training(sira_checkpoint_path: str):
    """
    步骤 2: GRPO 训练
    
    这一步会：
    - 加载 SIRA 训练的 checkpoint
    - 在 MATH500 数据集上进行 GRPO 训练
    - 保存最终模型到 output/grpo_stable/
    
    Args:
        sira_checkpoint_path: SIRA 训练的 checkpoint 路径
    """
    logger.info("="*80)
    logger.info("STEP 2: GRPO Training on MATH500")
    logger.info("="*80)
    
    if not os.path.exists(sira_checkpoint_path):
        logger.error(f"❌ SIRA checkpoint not found: {sira_checkpoint_path}")
        logger.error("Please run Step 1 first or provide correct checkpoint path")
        return
    
    logger.info(f"Using SIRA checkpoint: {sira_checkpoint_path}")
    logger.info("")
    
    # 运行 GRPO 训练
    grpo_on_MATH500(
        lora_path=sira_checkpoint_path,
        num_generations=8  # 每个问题生成 8 个样本
    )
    
    logger.info("")
    logger.info("✅ GRPO Training completed!")
    logger.info("📁 Model saved to: output/grpo_stable/")
    logger.info("")


def step3_test_model(grpo_checkpoint_path: str):
    """
    步骤 3: 测试 GRPO 模型
    
    这一步会：
    - 加载 GRPO 训练的模型
    - 在 MATH500 测试集上评估性能
    - 输出准确率等统计信息
    
    Args:
        grpo_checkpoint_path: GRPO 训练的 checkpoint 路径
    """
    logger.info("="*80)
    logger.info("STEP 3: Testing GRPO Model on MATH500")
    logger.info("="*80)
    
    if not os.path.exists(grpo_checkpoint_path):
        logger.error(f"❌ GRPO checkpoint not found: {grpo_checkpoint_path}")
        logger.error("Please run Step 2 first or provide correct checkpoint path")
        return
    
    logger.info(f"Using GRPO checkpoint: {grpo_checkpoint_path}")
    logger.info("")
    
    # 测试模型
    results = test_grpo_on_MATH500(grpo_lora_path=grpo_checkpoint_path)
    
    logger.info("")
    logger.info("✅ Testing completed!")
    logger.info(f"📊 Final Accuracy: {results['accuracy']:.2f}%")
    logger.info("")


def main():
    parser = argparse.ArgumentParser(description="GRPO on MATH500 完整工作流")
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="选择执行步骤: 1=SIRA训练, 2=GRPO训练, 3=测试模型"
    )
    parser.add_argument(
        "--sira-checkpoint",
        type=str,
        default="/root/autodl-tmp/CELPO/output/sira_all_Math500/checkpoint-target-reached-epoch-10",
        help="SIRA checkpoint 路径 (用于步骤 2)"
    )
    parser.add_argument(
        "--grpo-checkpoint",
        type=str,
        default="/root/autodl-tmp/CELPO/output/grpo_stable",
        help="GRPO checkpoint 路径 (用于步骤 3)"
    )
    
    args = parser.parse_args()
    
    if args.step == 1:
        step1_sira_training()
        
    elif args.step == 2:
        step2_grpo_training(args.sira_checkpoint)
        
    elif args.step == 3:
        step3_test_model(args.grpo_checkpoint)


if __name__ == "__main__":
    main()

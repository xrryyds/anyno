import json
import matplotlib.pyplot as plt
import os
import glob

# =================配置区域=================
# 这里填你代码里 HintSFTConfig 中的 output_base_dir
BASE_OUTPUT_DIR = "/root/autodl-tmp/output" 
# =========================================

def find_latest_log_file(base_dir):
    """自动查找目录下最新的 hint_sft_* 文件夹中的 epoch_metrics.jsonl"""
    # 找所有以 hint_sft_ 开头的文件夹
    search_pattern = os.path.join(base_dir, "hint_sft_*")
    dirs = glob.glob(search_pattern)
    
    if not dirs:
        return None
    
    # 按修改时间排序，找最新的
    latest_dir = max(dirs, key=os.path.getmtime)
    log_file = os.path.join(latest_dir, "epoch_metrics.jsonl")
    
    if os.path.exists(log_file):
        print(f"✅ 自动定位到最新日志: {log_file}")
        return log_file
    else:
        print(f"❌ 在最新的文件夹 {latest_dir} 中没找到 metrics 文件")
        return None

def plot_training_metrics(log_file_path, output_image_path="training_visualization.png"):
    if not log_file_path or not os.path.exists(log_file_path):
        print(f"Error: 文件无效")
        return

    epochs, total_losses, anchor_losses, mode_b_losses, gate_values = [], [], [], [], []

    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
                epochs.append(data['epoch'])
                total_losses.append(data['avg_train_loss'])
                anchor_losses.append(data['avg_anchor_loss'])
                mode_b_losses.append(data['avg_mode_b_loss'])
                gate_values.append(data['avg_gate_value'])
            except:
                pass

    if not epochs:
        print("数据为空，无法绘图")
        return

    # 绘图逻辑
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 图1：Loss
    ax1.plot(epochs, total_losses, 'o-', label='Total Loss', color='#333333')
    ax1.plot(epochs, mode_b_losses, 's--', label='Mode B (Generation)', color='#d62728', alpha=0.7)
    ax1.plot(epochs, anchor_losses, '^--', label='Anchor (Stability)', color='#1f77b4', alpha=0.7)
    ax1.set_title('Loss Dynamics', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 图2：Gate
    ax2.plot(epochs, gate_values, 'o-', color='#9467bd', linewidth=2, label='Avg Gate')
    ax2.set_title('Adaptive Gate Evolution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Gate Value')
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    print(f"📊 图表已生成: {output_image_path}")

if __name__ == "__main__":
    # 自动寻找
    target_file = find_latest_log_file(BASE_OUTPUT_DIR)
    
    if target_file:
        plot_training_metrics(target_file)
    else:
        print("未找到任何日志文件，请检查 BASE_OUTPUT_DIR 是否正确")

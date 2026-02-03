import json
import matplotlib.pyplot as plt
import os
import sys

# ==========================================
# 配置：请修改这里的路径
# ==========================================
# 指向你的 epoch_metrics.jsonl 文件路径
LOG_FILE_PATH = "/root/autodl-tmp/CELPO/output/hint_sft_0203_1131/epoch_metrics.jsonl" 

def plot_training_metrics(log_path):
    if not os.path.exists(log_path):
        print(f"错误: 找不到文件 {log_path}")
        return

    epochs = []
    total_losses = []
    anchor_losses = []
    mode_b_losses = []
    
    # 额外读取 Alpha 和 Gate 值，用于辅助分析（可选画图）
    alphas = []
    gates = []

    # 1. 读取数据
    print(f"正在读取日志: {log_path} ...")
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                epochs.append(data['epoch'])
                total_losses.append(data['avg_train_loss'])
                anchor_losses.append(data['avg_anchor_loss_weighted'])
                mode_b_losses.append(data['avg_mode_b_loss'])
                
                alphas.append(data.get('final_alpha', 0))
                gates.append(data.get('avg_gate_value', 0))
            except json.JSONDecodeError:
                continue

    if not epochs:
        print("日志文件为空或格式不正确。")
        return

    # 2. 设置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid') # 如果报错，可改为 'ggplot' 或删掉这行
    plt.figure(figsize=(12, 8))

    # 3. 绘制三条 Loss 曲线
    # Line 1: Mode B (Generation) Loss
    plt.plot(epochs, mode_b_losses, marker='o', linestyle='-', linewidth=2, 
             color='#1f77b4', label='Mode B Loss (Generation)')
    
    # Line 2: Anchor Loss (Weighted by Alpha)
    plt.plot(epochs, anchor_losses, marker='s', linestyle='--', linewidth=2, 
             color='#ff7f0e', label='Anchor Loss (Weighted)')
    
    # Line 3: Total Average Loss
    plt.plot(epochs, total_losses, marker='^', linestyle='-', linewidth=3, 
             color='#2ca02c', label='Total Train Loss')

    # 4. 图表细节装饰
    plt.title('SIRA Training Loss Dynamics', fontsize=16, pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss Value', fontsize=14)
    plt.legend(fontsize=12, loc='best', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 防止 X 轴刻度太密或不是整数
    if len(epochs) < 20:
        plt.xticks(epochs)

    # 5. 保存与显示
    output_img_path = log_path.replace('.jsonl', '_loss_curve.png')
    plt.savefig(output_img_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {output_img_path}")
    
    # 如果是在本地运行且支持弹窗，取消下面注释可以显示
    # plt.show()

    # ==========================================
    # (可选) 额外绘制 Alpha 和 Gate 变化的图
    # ==========================================
    plt.figure(figsize=(12, 6))
    
    ax1 = plt.gca()
    ax1.plot(epochs, alphas, 'r-o', label='Alpha (Balance Term)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Alpha', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    
    ax2 = ax1.twinx()
    ax2.plot(epochs, gates, 'b-s', label='Avg Gate Value')
    ax2.set_ylabel('Gate Value', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    
    plt.title('SIRA Dynamics: Alpha & Gate Evolution')
    output_param_path = log_path.replace('.jsonl', '_params_curve.png')
    plt.savefig(output_param_path, dpi=300, bbox_inches='tight')
    print(f"参数变化图已保存至: {output_param_path}")

if __name__ == "__main__":
    # 如果通过命令行传参: python plot.py path/to/log.jsonl
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = LOG_FILE_PATH
        
    plot_training_metrics(log_file)

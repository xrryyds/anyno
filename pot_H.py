import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import rcParams
from matplotlib import font_manager # 引入字体管理器

# ================= Configuration =================
ADV_HINTS_PATH = "/root/autodl-tmp/CELPO/datasets/exam/adv_hints.json"
DISADV_HINTS_PATH = "/root/autodl-tmp/CELPO/datasets/exam/disadv_hints.json"
OUTPUT_DIR = "/root/autodl-tmp/CELPO"
FONT_PATH = "/root/autodl-tmp/CELPO/times.ttf" # 字体文件路径

# ================= Font Setup (Critical Fix) =================
def configure_fonts():
    """
    强制加载本地 Times New Roman 字体，解决 Linux 服务器缺字体问题
    """
    # 1. 检查字体文件是否存在
    if not os.path.exists(FONT_PATH):
        # 如果没有 times.ttf，尝试下载或者使用系统自带的 DejaVu Serif 作为保底
        print(f"Warning: Font file not found at {FONT_PATH}.")
        print("Attempting to use built-in 'DejaVu Serif' as a fallback.")
        rcParams['font.family'] = 'serif'
        rcParams['font.serif'] = ['DejaVu Serif'] 
        return

    # 2. 动态加载字体文件
    try:
        font_manager.fontManager.addfont(FONT_PATH)
        prop = font_manager.FontProperties(fname=FONT_PATH)
        # 获取注册后的字体名称（通常是 'Times New Roman'）
        font_name = prop.get_name()
        
        # 3. 设置全局字体
        rcParams['font.family'] = font_name
        rcParams['font.sans-serif'] = [font_name] # 强制覆盖
        
        print(f"Successfully loaded font: {font_name}")
    except Exception as e:
        print(f"Font loading failed: {e}. Using default.")

configure_fonts()

# ================= Academic Style Settings =================
rcParams['font.size'] = 14
rcParams['axes.labelsize'] = 16
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['legend.fontsize'] = 12
rcParams['mathtext.fontset'] = 'stix' # 数学公式使用类似 Times 的字体
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

# 配色
COLOR_CORRECT = "#4c72b0"
COLOR_INCORRECT = "#c44e52"

def load_data(file_path, group_name):
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found.")
        return pd.DataFrame()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    extracted = []
    for item in data:
        if "entropy_original" in item and "entropy_with_hints" in item:
            extracted.append({
                "Original": item["entropy_original"],
                "With_Hints": item["entropy_with_hints"],
                "Group": group_name
            })
    return pd.DataFrame(extracted)

def plot_scatter_comparison_academic(df_correct, df_incorrect, save_path):
    if df_correct.empty and df_incorrect.empty: return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    all_vals = pd.concat([df_correct, df_incorrect])
    max_val = max(all_vals["Original"].max(), all_vals["With_Hints"].max()) * 1.1
    
    datasets = [
        (df_correct, "Correct Group", axes[0], COLOR_CORRECT), 
        (df_incorrect, "Incorrect Group", axes[1], COLOR_INCORRECT)
    ]

    for df, title, ax, color in datasets:
        if df.empty: continue
        
        # KDE 可能会因为数据点重叠完全一致而报错，加个 try
        try:
            sns.kdeplot(
                data=df, x="Original", y="With_Hints", 
                ax=ax, color=color, alpha=0.3, levels=5, fill=True, warn_singular=False
            )
        except:
            pass 
        
        ax.scatter(
            df["Original"], df["With_Hints"], 
            c=color, alpha=0.5, s=25, edgecolor='white', linewidth=0.5
        )
        
        ax.plot([0, max_val], [0, max_val], ls="--", c=".15", linewidth=1.2, label=r'Baseline ($y=x$)')
        
        # Wilcoxon Test
        try:
            stat, p_val = stats.wilcoxon(df["Original"], df["With_Hints"])
            p_text = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
        except:
            p_text = "N/A" # 数据太少无法计算

        stats_text = f"N = {len(df)}\nWilcoxon Test:\n{p_text}"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='#dddddd'))

        ax.set_title(title, fontsize=16, fontweight='bold', pad=12)
        ax.set_xlabel(r"Original Entropy ($\mathcal{H}_{orig}$)")
        if ax == axes[0]:
            ax.set_ylabel(r"Entropy with Hints ($\mathcal{H}_{hints}$)")
        
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "entropy_scatter_academic.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(save_path, "entropy_scatter_academic.png"), dpi=300, bbox_inches='tight')
    print("Scatter plot saved.")
    plt.close()

def plot_violin_academic(df_correct, df_incorrect, save_path):
    if df_correct.empty and df_incorrect.empty: return

    def melt_df(df):
        return df.melt(id_vars=["Group"], value_vars=["Original", "With_Hints"], 
                       var_name="State", value_name="Entropy")

    df_long = pd.concat([melt_df(df_correct), melt_df(df_incorrect)])
    df_long["State"] = df_long["State"].replace({"Original": "Original", "With_Hints": "w/ Hints"})

    plt.figure(figsize=(10, 6))
    
    ax = sns.violinplot(
        data=df_long, x="Group", y="Entropy", hue="State",
        split=True, inner=None,
        palette={"Original": "#e0e0e0", "w/ Hints": COLOR_CORRECT},
        linewidth=1, cut=0
    )
    
    sns.boxplot(
        data=df_long, x="Group", y="Entropy", hue="State",
        ax=ax, fliersize=0, boxprops={'facecolor':'none', "zorder": 2},
        width=0.3, linewidth=1.2, zorder=2, dodge=True
    )

    handles, labels = ax.get_legend_handles_labels()
    # 只需要前两个图例
    if len(handles) >= 2:
        ax.legend(handles[:2], [r'$\mathcal{H}_{orig}$', r'$\mathcal{H}_{hints}$'], 
                  loc='upper center', frameon=False, ncol=2, fontsize=13)

    ax.set_xlabel("")
    ax.set_ylabel(r"Entropy Value ($\mathcal{H}$)")
    ax.set_title("Distribution Shift of Token Entropy", fontsize=16, fontweight='bold', pad=15)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "entropy_violin_academic.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(save_path, "entropy_violin_academic.png"), dpi=300, bbox_inches='tight')
    print("Violin plot saved.")
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_correct = load_data(ADV_HINTS_PATH, "Correct")
    df_incorrect = load_data(DISADV_HINTS_PATH, "Incorrect")
    
    print(f"Loaded: Correct ({len(df_correct)}), Incorrect ({len(df_incorrect)})")
    plot_scatter_comparison_academic(df_correct, df_incorrect, OUTPUT_DIR)
    plot_violin_academic(df_correct, df_incorrect, OUTPUT_DIR)

if __name__ == "__main__":
    main()
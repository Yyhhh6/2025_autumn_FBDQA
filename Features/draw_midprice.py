import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_features(df, features_to_plot, save_path):
    """
    绘制指定特征随时间变化的图
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12,6))
    for feat in features_to_plot:
        plt.plot(df['idx'], df[feat], label=feat)
    plt.xlabel("Index")
    plt.ylabel("Feature Value")
    plt.title("Selected Features Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Figure saved to: {save_path}")


# ---------- 批量绘图 ----------
file_paths = [
    "data/data_raw/snapshot_sym0_date0_am.csv",
    "data/data_raw/snapshot_sym0_date0_pm.csv",
    "data/data_raw/snapshot_sym0_date20_am.csv",
    "data/data_raw/snapshot_sym0_date20_pm.csv",
    "data/data_raw/snapshot_sym0_date40_am.csv",
    "data/data_raw/snapshot_sym0_date40_pm.csv",
    "data/data_raw/snapshot_sym0_date57_am.csv",
    "data/data_raw/snapshot_sym0_date57_pm.csv",
    "data/data_raw/snapshot_sym0_date70_am.csv",
    "data/data_raw/snapshot_sym0_date70_pm.csv",
]

save_dir = "Features/figures/sym0"  # 保存目录
features_to_plot = ['n_midprice']  # 指定要绘制的特征

for file_path in file_paths:
    # 读取 CSV
    df = pd.read_csv(file_path)

    # 给 df 添加 idx 列
    df['idx'] = range(len(df))

    # 构造保存文件名
    file_name = os.path.basename(file_path).replace('.csv', '_features.png')
    save_path = os.path.join(save_dir, file_name)

    # 绘图
    plot_features(df, features_to_plot, save_path)

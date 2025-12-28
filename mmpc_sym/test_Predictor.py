import os
import numpy as np
import pandas as pd
from .Predictor import Predictor  # 确保 Predictor 在包路径中可用

# ================= 配置 =================
data_dir = "data/data_sym4_test"
tick_size = 100  # 每 100 个 tick 分一段
N = 20           # 用于计算 PNL 的步长
# =======================================

pred = Predictor()

# 汇总所有 csv 的预测结果和真实标签
all_test_labels = []
all_y_pred = []
pnl = []

# 遍历文件夹中的 CSV
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
for file_name in csv_files:
    path = os.path.join(data_dir, file_name)
    df = pd.read_csv(path)
    print(f"\nProcessing {file_name}, shape={df.shape}")

    # 分割成连续 100 tick 的小 DataFrame
    segments = []
    test_labels = []
    for start_idx in range(0, len(df) - tick_size + 1):
        seg = df.iloc[start_idx:start_idx + tick_size].copy()
        segments.append(seg)
        test_labels.append(seg[f"label_{N}"].iloc[-1])  # 最后一个 tick 的 label_N

    # 预测
    # print("len(segments): ", len(segments))
    y_pred_list = pred.predict(segments)
    y_pred = np.array([s for s in y_pred_list]).flatten()
    test_labels = np.array(test_labels)
    midprice = df["n_midprice"][99:].values

    print("shape of test_labels: ", test_labels.shape)
    print("shape of y_pred: ", y_pred.shape)
    print("shape of midprice: ", midprice.shape)

    # 累积
    all_test_labels.append(test_labels)
    all_y_pred.append(y_pred)

    for i, s in enumerate(y_pred):
        if i + N >= len(midprice):
            continue
        if s == 2:       # Long
            pnl.append(midprice[i + N] - midprice[i])
        elif s == 0:     # Short
            pnl.append(midprice[i] - midprice[i + N])

# 将所有结果拼接成一维
all_test_labels = np.concatenate(all_test_labels)
all_y_pred = np.concatenate(all_y_pred)

# ----------------- 计算指标 -----------------
index_recall = all_test_labels != 1
recall = np.sum(all_y_pred[index_recall] == all_test_labels[index_recall]) / np.sum(index_recall)

index_precision = all_y_pred != 1
precision = np.sum(all_y_pred[index_precision] == all_test_labels[index_precision]) / np.sum(index_precision)

beta = 0.5
f05 = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F0.5:      {f05:.4f}")

# ----------------- PNL 计算 -----------------
pnl = np.array(pnl)
total_pnl = pnl.sum()
avg_pnl = pnl.mean()
trade_pnl = pnl[pnl != 0]

win_rate = (trade_pnl > 0).mean() if len(trade_pnl) > 0 else 0
num_trades = len(trade_pnl)

final_score = f05 * (avg_pnl - 0.0006) ** 2 * 10000 * 10000
if avg_pnl - 0.0006 < 0:
    final_score = -final_score

print(f"Total PNL:   {total_pnl:.4f}")
print(f"Avg PNL:     {avg_pnl:.6f}")
print(f"Trades:      {num_trades}")
print(f"Win Rate:    {win_rate:.3f}")
print(f"Final Score: {final_score:.3f}")

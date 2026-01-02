from .data_process import *
from .model import XGBModel
# from .test import extract_feature
import os
from .Predictor import preprocess_local as preprocess
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def extract_feature(files_dir, N):
    csv_files = files_dir
    def process_file(file, N):
        if os.path.exists(file):
            df = pd.read_csv(file)#[:-N]
            n_midprice = df['n_midprice'].values
            amount_delta = df['amount_delta'].values
            if df.empty:
                raise ValueError(f"File {file} is empty.")
            df = df.reset_index(drop=True)
            df, labels, _ = preprocess(df, N)
            df = df.squeeze(axis=0)
            # labels = labels[99:]
            # df_list = df[99:]
            n_midprice = n_midprice[99:]
            amount_delta = amount_delta[99:]
        else:
            print("file: ", file)
            raise FileNotFoundError(f"File {file} not found.")
        return df, labels, n_midprice, amount_delta

    data = []
    labels_list = []
    midprice_list = []
    amount_delta_list = []

    for file in tqdm(csv_files, total=len(csv_files), desc="Extracting features"):
        df, labels, n_midprice, amount_delta = process_file(file, N)
        data.append(df)
        labels_list.append(labels)
        midprice_list.append(n_midprice)
        amount_delta_list.append(amount_delta)
    data = np.concatenate(data, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    # midprice_list = np.concatenate(midprice_list, axis=0)
    # amount_delta_list = np.concatenate(amount_delta_list, axis=0)

    return data, labels_list, midprice_list, amount_delta_list

test_dir = "./data/data_sym7_test"
test_files = [
    os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".csv")
]
model = XGBModel("/hdd/yyh/src/quant/models_ZZZ_0/model_20_all_20260101_235606.json")

test_data, test_labels, n_midprice, amount_delta = extract_feature(files_dir=test_files, N=20)

y_pred = model.predict(test_data)   # (N, 3)
confidence = np.max(y_pred, axis=1)
signal = np.argmax(y_pred, axis=1)
signal[confidence < 0.65] = 1

import matplotlib.pyplot as plt
import numpy as np

def plot_prediction_results(mid_prices, amount_deltas, signals, labels, start_idx=0, length=500):
    """
    mid_prices: 原始中间价序列
    amount_deltas: 成交量变化序列
    """
    # 合并列表为 numpy 数组
    mid_prices = np.concatenate(mid_prices, axis=0)
    amount_deltas = np.concatenate(amount_deltas, axis=0)

    end_idx = min(start_idx + length, len(mid_prices))
    
    prices = mid_prices[start_idx:end_idx]
    deltas = amount_deltas[start_idx:end_idx]
    preds = signals[start_idx:end_idx]
    actuals = labels[start_idx:end_idx]
    x = np.arange(len(prices)) + start_idx

    fig, ax1 = plt.subplots(figsize=(15, 7))

    # --- 绘制 Price (左轴) ---
    ax1.plot(x, prices, color='black', alpha=0.3, label='Mid Price', linewidth=1)
    ax1.set_xlabel("Ticks")
    ax1.set_ylabel("Price", color='black')
    ax1.grid(True, alpha=0.2)

    # --- 绘制 Amount Delta (右轴) ---
    ax2 = ax1.twinx()  
    # 使用填充区域或柱状图更能区分成交量，这里使用 alpha 较低的浅蓝色
    ax2.fill_between(x, deltas, color='blue', alpha=0.1, label='Amount Delta')
    ax2.set_ylabel("Amount Delta", color='blue')
    # 限制右轴范围，避免遮挡主要信号点（可选）
    # ax2.set_ylim(min(deltas)*1.5, max(deltas)*1.5) 

    # --- 绘制预测信号 (在 ax1 价格曲线上) ---
    for i in range(len(preds)):
        if preds[i] == 1: continue
        color = 'green' if preds[i] == actuals[i] else 'red'
        marker = '^' if preds[i] == 2 else 'v'
        ax1.scatter(x[i], prices[i], color=color, marker=marker, s=50, alpha=0.8, zorder=3)

    # 合并图例
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='black', lw=1, alpha=0.3),
        Line2D([0], [0], color='blue', lw=4, alpha=0.2), # 代表 amount_delta
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='red', markersize=10)
    ]
    ax1.legend(custom_lines, ['Mid Price', 'Amount Delta', 'Correct Prediction', 'Wrong Prediction'], loc='upper left')
    
    plt.title(f"Price & Amount Delta with Predictions (Ticks {start_idx} to {end_idx})")
    plt.tight_layout()
    plt.show()
    plt.savefig("analysis_plot.png", dpi=300)
# 调用函数进行绘图（假设 n_midprice, signal, test_labels 已经准备好）
# 注意：确保 mid_price 的长度与 signal 一致
plot_prediction_results(n_midprice, amount_delta, signal, test_labels, start_idx=0, length=5000)

index_recall = test_labels != 1
recall = sum(signal[index_recall] == test_labels[index_recall]) / sum(index_recall)
# Precision：预测上涨/下跌中，预测正确的比例
index_precision = signal != 1
precision = sum(signal[index_precision] == test_labels[index_precision]) / sum(index_precision)
beta = 0.5
f05 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F0.5:      {f05:.4f}")
pnl = []

start = 0
j = 0
for i in range(len(n_midprice)):
    length = len(n_midprice[i])
    for j in range(length):
        if j + 20 >= length:
            break
        if signal[j+start] == 2:      # Long
            pnl.append(n_midprice[i][j+20] - n_midprice[i][j])
        elif signal[j+start] == 0:    # Short
            pnl.append(n_midprice[i][j] - n_midprice[i][j+20])
    start += length
    
pnl = np.array(pnl)
total_pnl = pnl.sum()
avg_pnl = pnl.mean()
trade_pnl = pnl[pnl != 0]

win_rate = (trade_pnl > 0).mean()
num_trades = len(trade_pnl)

final_score = f05 * (avg_pnl - 0.0006) * (avg_pnl - 0.0006) * 10000 * 10000
if avg_pnl - 0.0006 < 0:
    final_score = -final_score

print(f"Total PNL:   {total_pnl:.4f}")
print(f"Avg PNL:     {avg_pnl:.6f}")
print(f"Trades:      {num_trades}")
print(f"Win Rate:    {win_rate:.3f}")
print(f"Final Score:    {final_score:.3f}")

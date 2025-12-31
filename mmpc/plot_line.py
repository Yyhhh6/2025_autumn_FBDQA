from .data_process import *
from .model import XGBModel
from .test import extract_feature
import os

test_dir = "./data/data_sym7_test"
test_files = [
    os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".csv")
]
model = XGBModel("/hdd/yyh/src/quant/mmpc/model_20_sym7_20251231_072506.json")

test_data, test_labels, n_midprice = extract_feature(files_dir=test_files, N=20)

y_pred = model.predict(test_data)   # (N, 3)
confidence = np.max(y_pred, axis=1)
signal = np.argmax(y_pred, axis=1)
signal[confidence < 0.65] = 1

import matplotlib.pyplot as plt
import numpy as np

def plot_prediction_results(mid_prices, signals, labels, start_idx=0, length=500):
    """
    mid_prices: 原始中间价序列 (n_midprice)
    signals: 模型生成的预测信号 (0:下, 1:平, 2:上)
    labels: 真实标签 (0:下, 1:平, 2:上)
    """
    # 截取特定窗口的数据
    mid_prices = np.concatenate(mid_prices, axis=0)

    # end_idx = len(mid_prices)
    end_idx = 14500
    start_idx = 13000
    prices = mid_prices[start_idx:end_idx]
    preds = signals[start_idx:end_idx]
    actuals = labels[start_idx:end_idx]
    x = np.arange(len(prices))+start_idx

    plt.figure(figsize=(15, 7))
    plt.plot(x, prices, color='black', alpha=0.3, label='Mid Price', linewidth=1)

    # 定义标记形状：上涨用上三角，下跌用下三角
    # 定义颜色逻辑：预测正确为绿，预测错误为红
    
    for i in range(len(preds)):
        # 只观察预测了上涨(2)或下跌(0)的点，忽略预测为“不变(1)”的点
        if preds[i] == 1:
            continue
            
        color = 'green' if preds[i] == actuals[i] else 'red'
        marker = '^' if preds[i] == 2 else 'v'
        # if i + start_idx < 5000 and i + start_idx > 1800:  
            # 仅打印部分点的信息，避免信息过载
        print(f"Tick {start_idx + i}: Price={prices[i]:.4f}, Predicted={preds[i]}, Actual={actuals[i]}, Color={color}, Marker={marker}, x={x[i]}, y={prices[i]}")
        print(f"1 ticks price change: {mid_prices[start_idx + i + 1] - mid_prices[start_idx + i]:.4f}")
        print(f"20 ticks price change: {mid_prices[start_idx + i + 20] - mid_prices[start_idx + i]:.4f}")
        print("-----")
        
        plt.scatter(x[i], prices[i], color=color, marker=marker, s=50, alpha=0.8)

    # 制作图例
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='black', lw=1, alpha=0.3),
                    Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10),
                    Line2D([0], [0], marker='v', color='w', markerfacecolor='red', markersize=10)]
    
    plt.legend(custom_lines, ['Mid Price', 'Correct Prediction', 'Wrong Prediction'])
    plt.title(f"Model Predictions vs Actual Movements (Ticks {start_idx} to {end_idx})")
    plt.xlabel("Ticks")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.2)
    plt.show()
    plt.savefig("mmpc/prediction_results_part1.png", dpi=300)

# 调用函数进行绘图（假设 n_midprice, signal, test_labels 已经准备好）
# 注意：确保 mid_price 的长度与 signal 一致
plot_prediction_results(n_midprice, signal, test_labels, start_idx=1000, length=500)

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

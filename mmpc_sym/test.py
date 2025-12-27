from .model import XGBModel
from .Predictor import preprocess
from .data_process import *
import os
import numpy as np
import pandas as pd
import random
import re
from collections import defaultdict
from tqdm import tqdm

# 去掉涨跌停的文件
EXCLUDE_FILES = [
    './data/data_raw/snapshot_sym0_date64_pm.csv',
    './data/data_raw/snapshot_sym0_date71_pm.csv',
    './data/data_raw/snapshot_sym0_date63_am.csv',
]

TRAIN_RATIO = 0.75
VAL_RATIO = 0.0
SEED = 42
N_list = [20]
file_dir = "./data/data_raw"
model_dir = "./models_sym"

alpha_map = {5: 0.0005, 10: 0.0005, 20: 0.001, 40: 0.001, 60: 0.001}

# ------------------ CSV 文件处理 ------------------
def split_csv_files(data_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=SEED):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    csv_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".csv") and os.path.join(data_dir, f) not in EXCLUDE_FILES
    ]

    csv_files.sort()
    random.seed(seed)
    random.shuffle(csv_files)

    n = len(csv_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = csv_files[:n_train]
    val_files = csv_files[n_train:n_train + n_val]
    test_files = csv_files[n_train + n_val:]
    print(f"Total CSV files: {n}, Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    return train_files, val_files, test_files

def extract_feature(files_list, N):
    def process_file(file, N):
        if not os.path.exists(file):
            raise FileNotFoundError(f"File {file} not found.")
        df = pd.read_csv(file)
        if df.empty:
            raise ValueError(f"File {file} is empty.")
        df = df.reset_index(drop=True)
        df, labels = preprocess(df, N)
        df = df.squeeze(axis=0)
        labels = labels[99:]
        df_list = df[99:]
        n_midprice = df['n_midprice'].values[99:]
        return df_list, labels, n_midprice

    data, labels_list, midprice_list = [], [], []
    for file in tqdm(files_list, total=len(files_list), desc="Extracting features"):
        df, labels, n_midprice = process_file(file, N)
        data.append(df)
        labels_list.append(labels)
        midprice_list.append(n_midprice)

    data = np.concatenate(data, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    midprice_list = np.concatenate(midprice_list, axis=0)
    return data, labels_list, midprice_list

# ------------------ 按 sym 分组 ------------------
def group_files_by_sym(file_list):
    sym_dict = defaultdict(list)
    for file in file_list:
        match = re.search(r'sym(\d+)', file)
        if match:
            sym = int(match.group(1))
            sym_dict[sym].append(file)
    return sym_dict

# ------------------ 模型按 sym 分类 ------------------
def get_model_files_by_sym(model_dir):
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".json")]
    sym_model_dict = defaultdict(list)
    for f in model_files:
        match = re.search(r'sym(\d+)', f)
        if match:
            sym = int(match.group(1))
            sym_model_dict[sym].append(os.path.join(model_dir, f))
    # 取每个 sym 的最新模型（按文件名排序）
    for sym in sym_model_dict:
        sym_model_dict[sym].sort()
        sym_model_dict[sym] = sym_model_dict[sym][-1]
    return sym_model_dict

# ------------------ 主评测流程 ------------------
overall_metrics = []
sym_model_dict = get_model_files_by_sym(model_dir)

for N in N_list:
    train_files, val_files, test_files = split_csv_files(
        data_dir=file_dir,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=1-TRAIN_RATIO-VAL_RATIO,
        seed=SEED
    )

    sym_files_dict = group_files_by_sym(test_files)

    for sym, sym_files in sym_files_dict.items():
        model_path = sym_model_dict.get(sym)
        if not model_path:
            print(f"No model found for sym{sym}, skipping.")
            continue
        print(f"Evaluating sym{sym} with {len(sym_files)} files using model {model_path}")
        test_data, test_labels, n_midprice = extract_feature(sym_files, N)
        model = XGBModel(model_path)
        y_pred = model.predict(test_data)

        target_confidences = [0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9]

        for target_confidence in target_confidences:
            confidence = np.max(y_pred, axis=1)
            signal = np.argmax(y_pred, axis=1)
            signal[confidence < target_confidence] = 1  # Hold

            y = signal
            index_recall = test_labels != 1
            recall = sum(y[index_recall] == test_labels[index_recall]) / sum(index_recall)
            index_precision = y != 1
            precision = sum(y[index_precision] == test_labels[index_precision]) / sum(index_precision)
            beta = 0.5
            f05 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

            pnl = []
            for i, s in enumerate(signal):
                if i + N >= len(n_midprice):
                    continue
                if s == 2:
                    pnl.append(n_midprice[i+N] - n_midprice[i])
                elif s == 0:
                    pnl.append(n_midprice[i] - n_midprice[i+N])

            pnl = np.array(pnl)
            total_pnl = pnl.sum()
            avg_pnl = pnl.mean()
            trade_pnl = pnl[pnl != 0]
            win_rate = (trade_pnl > 0).mean() if len(trade_pnl) > 0 else 0
            num_trades = len(trade_pnl)

            overall_metrics.append({
                "sym": sym,
                "target_confidence": target_confidence,
                "Precision": precision,
                "Recall": recall,
                "F0.5": f05,
                "Total_PNL": total_pnl,
                "Avg_PNL": avg_pnl,
                "Win_Rate": win_rate,
                "Num_Trades": num_trades
            })

# ------------------ 输出指标 ------------------
df_metrics = pd.DataFrame(overall_metrics)
print("Per-sym metrics:\n", df_metrics)

overall_summary = df_metrics.groupby('target_confidence').mean()
print("\nOverall summary by target_confidence:\n", overall_summary)

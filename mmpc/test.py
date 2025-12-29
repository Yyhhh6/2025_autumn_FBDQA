from .model import XGBModel
from .Predictor import preprocess
from .data_process import *
import os
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, mean_squared_log_error
from tqdm import tqdm

# 去掉涨跌停的文件
# EXCLUDE_FILES = ['./data/data_raw/snapshot_sym1_date33_pm.csv', './data/data_raw/snapshot_sym7_date42_am.csv', './data/data_raw/snapshot_sym1_date25_am.csv', './data/data_raw/snapshot_sym6_date32_pm.csv', './data/data_raw/snapshot_sym4_date33_pm.csv', './data/data_raw/snapshot_sym2_date59_pm.csv', './data/data_raw/snapshot_sym1_date26_pm.csv', './data/data_raw/snapshot_sym2_date59_am.csv', './data/data_raw/snapshot_sym2_date57_pm.csv', './data/data_raw/snapshot_sym4_date34_am.csv', './data/data_raw/snapshot_sym5_date38_am.csv', './data/data_raw/snapshot_sym0_date64_pm.csv', './data/data_raw/snapshot_sym1_date33_am.csv', './data/data_raw/snapshot_sym1_date34_pm.csv', './data/data_raw/snapshot_sym0_date63_pm.csv', './data/data_raw/snapshot_sym0_date71_pm.csv', './data/data_raw/snapshot_sym7_date10_pm.csv', './data/data_raw/snapshot_sym4_date32_pm.csv', './data/data_raw/snapshot_sym6_date42_pm.csv', './data/data_raw/snapshot_sym4_date33_am.csv', './data/data_raw/snapshot_sym7_date42_pm.csv', './data/data_raw/snapshot_sym0_date63_am.csv', './data/data_raw/snapshot_sym2_date42_pm.csv', './data/data_raw/snapshot_sym4_date34_pm.csv', './data/data_raw/snapshot_sym1_date25_pm.csv', './data/data_raw/snapshot_sym5_date23_pm.csv', './data/data_raw/snapshot_sym6_date33_pm.csv', './data/data_raw/snapshot_sym4_date31_pm.csv', './data/data_raw/snapshot_sym7_date10_am.csv']
# EXCLUDE_FILES = ['./data/data_raw/snapshot_sym0_date64_pm.csv', './data/data_raw/snapshot_sym0_date71_pm.csv', './data/data_raw/snapshot_sym0_date63_am.csv',]
EXCLUDE_FILES = []

TRAIN_RATIO = 0.0
VAL_RATIO = 0.0
SEED = 42

# N_list = [5, 10, 20, 40, 60]
N_list = [20]
alpha_map = {5: 0.0005, 10: 0.0005, 20: 0.001, 40: 0.001, 60: 0.001}
file_dir="data/data_sym3_test"

def split_csv_files(
    data_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=SEED
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    csv_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".csv") and 
        os.path.join(data_dir, f) not in EXCLUDE_FILES
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
    print(f"test_files: {test_files}")
    return train_files, val_files, test_files

def extract_feature(files_dir, N):
    csv_files = files_dir
    def process_file(file, N):
        if os.path.exists(file):
            df = pd.read_csv(file)#[:-N]
            n_midprice = df['n_midprice'].values
            if df.empty:
                raise ValueError(f"File {file} is empty.")
            df = df.reset_index(drop=True)
            df, labels = preprocess(df, N)
            df = df.squeeze(axis=0)
            labels = labels[99:]
            df_list = df[99:]
            n_midprice = n_midprice[99:]
        else:
            print("file: ", file)
            raise FileNotFoundError(f"File {file} not found.")
        return df_list, labels, n_midprice

    data = []
    labels_list = []
    midprice_list = []

    for file in tqdm(csv_files, total=len(csv_files), desc="Extracting features"):
        df, labels, n_midprice = process_file(file, N)
        data.append(df)
        labels_list.append(labels)
        midprice_list.append(n_midprice)
    data = np.concatenate(data, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    # midprice_list = np.concatenate(midprice_list, axis=0)

    return data, labels_list, midprice_list

for N in N_list:
    train_files, val_files, test_files = split_csv_files(data_dir=file_dir, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=1-TRAIN_RATIO-VAL_RATIO, seed=SEED)
    test_files = ['data/data_sym5_test/snapshot_sym5_date56_pm.csv', 'data/data_sym5_test/snapshot_sym5_date9_am.csv', 'data/data_sym5_test/snapshot_sym5_date68_pm.csv', 'data/data_sym5_test/snapshot_sym5_date47_pm.csv', 'data/data_sym5_test/snapshot_sym5_date58_pm.csv', 'data/data_sym5_test/snapshot_sym5_date39_pm.csv', 'data/data_sym5_test/snapshot_sym5_date77_pm.csv', 'data/data_sym5_test/snapshot_sym5_date55_pm.csv', 'data/data_sym5_test/snapshot_sym5_date53_am.csv', 'data/data_sym5_test/snapshot_sym5_date49_am.csv', 'data/data_sym5_test/snapshot_sym5_date10_am.csv', 'data/data_sym5_test/snapshot_sym5_date52_am.csv', 'data/data_sym5_test/snapshot_sym5_date12_pm.csv', 'data/data_sym5_test/snapshot_sym5_date27_pm.csv']
    test_data, test_labels, n_midprice = extract_feature(files_dir=test_files, N=N)
    # print(f"test_data shape: {test_data.shape}, test_labels shape: {test_labels.shape}, n_midprice shape: {n_midprice.shape}")
    print(f"test_data shape: {test_data.shape}, test_labels shape: {test_labels.shape}")
    model = XGBModel("models_sym3/model_20_all_20251228_084958.json")
    # print(f"the 1st test sample ground truth: {test_labels[0]}, {test_data[0].shape}")
    y_pred = model.predict(test_data)   # (N, 3)

    target_confidences = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    # target_confidences = [0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9]

    for target_confidence in target_confidences:
        print(f"************target_confidence={target_confidence}************")
        confidence = np.max(y_pred, axis=1)
        signal = np.argmax(y_pred, axis=1)
        signal[confidence < target_confidence] = 1 # 信心不足时，预测为不变
        y = signal
        # signal = y_pred
        # y = y_pred
        print(f"y shape: {y.shape}, test_labels shape: {test_labels.shape}")
        from sklearn.metrics import classification_report, precision_score, recall_score, fbeta_score
        print(f"Results for N={N}:")
        # print(classification_report(test_labels, y, digits=4))
        # Recall：真实上涨/下跌中，被预测正确的比例
        index_recall = test_labels != 1
        print("index_recall: ", index_recall)
        recall = sum(y[index_recall] == test_labels[index_recall]) / sum(index_recall)
        # Precision：预测上涨/下跌中，预测正确的比例
        index_precision = y != 1
        precision = sum(y[index_precision] == test_labels[index_precision]) / sum(index_precision)
        beta = 0.5
        f05 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F0.5:      {f05:.4f}")
        pnl = []

        print("len(n_midprice): ", len(n_midprice))

        start = 0
        j = 0
        for i in range(len(n_midprice)):
            length = len(n_midprice[i])
            for j in range(length):
                if j + N >= length:
                    break
                if signal[j+start] == 2:      # Long
                    pnl.append(n_midprice[i][j+N] - n_midprice[i][j])
                elif signal[j+start] == 0:    # Short
                    pnl.append(n_midprice[i][j] - n_midprice[i][j+N])
            start += length
            
        print("****************start*******************: ", start)
        # exit(0)

        # for i, s in enumerate(signal):
        #     if i + N >= len(n_midprice):
        #         # pnl.append(0)
        #         continue
        #     if s == 2:      # Long
        #         pnl.append(n_midprice[i+N] - n_midprice[i])
        #     elif s == 0:    # Short
        #         pnl.append(n_midprice[i] - n_midprice[i+N])
        #     # else:           # Hold
        #     #     pnl.append(0)

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
    
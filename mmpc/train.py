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


TRAIN_RATIO = 0.7
VAL_RATIO = 0.2  # 测试集自动是 0.1
SEED = 42

# N_list = [5, 10, 20, 40, 60]
N_list = [5]
alpha_map = {5: 0.0005, 10: 0.0005, 20: 0.001, 40: 0.001, 60: 0.001}
file_dir="./data/data_raw"

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
        if f.endswith(".csv")
    ]

    csv_files.sort()  # 保证稳定
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

def extract_feature(files_dir, N):
    csv_files = files_dir
    def process_file(file, N):
        if os.path.exists(file):
            df = pd.read_csv(file)[:-N]
            
            # TODO：数据预处理。还要去掉涨跌停的！！
            # 这里的scalar是提前对数据集计算得到的
            scaler = np.load(os.path.join(os.path.dirname(__file__), 'scaler.npz')) 
            df = data_scale_Z_Score_np(df, mean=scaler['mean'], std=scaler['std'])

            if df.empty:
                raise ValueError(f"File {file} is empty.")
            df = df.reset_index(drop=True)
            df, labels = preprocess(df, N)
            df = df.squeeze(axis=0)
            labels = labels[99:]
            df_list = df[99:]
        else:
            print("file: ", file)
            raise FileNotFoundError(f"File {file} not found.")
        return df_list, labels

    data = []
    labels_list = []

    for file in tqdm(csv_files, total=len(csv_files), desc="Extracting features"):
        df, labels = process_file(file, N)
        data.append(df)
        labels_list.append(labels)

    data = np.concatenate(data, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)

    return data, labels_list

# raw_data → replace inf → replace nan → 去极值 (MAD 或百分位, train-based) → 标准化 (train-based)
for N in N_list:
    train_files, val_files, test_files = split_csv_files(data_dir=file_dir, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=1-TRAIN_RATIO-VAL_RATIO, seed=SEED)
    train_data, train_labels = extract_feature(files_dir=train_files, N=N)
    val_data, val_labels = extract_feature(files_dir=val_files, N=N)
    test_data, test_labels = extract_feature(files_dir=test_files, N=N)
    
    # print(f"train data")
    # nan_idx = np.argwhere(np.isnan(train_data))
    # print(f"NaN 数量: {len(nan_idx)}")
    # print("NaN 位置 (行, 列):", nan_idx)
    # # Inf（包含 +inf 和 -inf）
    # inf_idx = np.argwhere(np.isinf(train_data))
    # print(f"Inf 数量: {len(inf_idx)}")
    # print("Inf 位置 (行, 列):", inf_idx)
    # print(f"val data")
    # nan_idx = np.argwhere(np.isnan(val_data))
    # print(f"NaN 数量: {len(nan_idx)}")
    # print("NaN 位置 (行, 列):", nan_idx)
    # # Inf（包含 +inf 和 -inf）
    # inf_idx = np.argwhere(np.isinf(val_data))
    # print(f"Inf 数量: {len(inf_idx)}")
    # print("Inf 位置 (行, 列):", inf_idx)
    # print(f"test data")
    # nan_idx = np.argwhere(np.isnan(test_data))
    # print(f"NaN 数量: {len(nan_idx)}")
    # print("NaN 位置 (行, 列):", nan_idx)
    # # Inf（包含 +inf 和 -inf）
    # inf_idx = np.argwhere(np.isinf(test_data))
    # print(f"Inf 数量: {len(inf_idx)}")
    # print("NaN 位置 (行, 列):", nan_idx)

    # 替换inf
    # train_data[~np.isfinite(train_data)] = np.nan
    # val_data[~np.isfinite(val_data)]     = np.nan
    # test_data[~np.isfinite(test_data)]   = np.nan

    # 去NaN
    # TODO：tree_method="hist" 时，XGBoost 能处理 NaN，不需要额外处理
    # train_data, val_data, test_data, medians = factors_null_process_np(train=train_data, val=val_data, test=test_data)

    # 去极值（基于训练集统计量）
    # train_data, val_data, test_data, lower, upper = extreme_process_MAD_np(train=train_data, val=val_data, test=test_data, num=3)
    
    # 归一化。TODO：数据处理对每一个特征分别做
    # train_data, val_data, test_data, mean, std = data_scale_Z_Score_np(train=train_data, val=val_data, test=test_data)

    # np.savez(
    #     "mmpc/scaler.npz",
    #     # median=medians,
    #     # mad_lower=lower,
    #     # mad_upper=upper,
    #     mean=mean,
    #     std=std
    # )

    model = XGBModel()
    model.train(
        train_data,
        train_labels,
        val_data,
        val_labels,
        num_boost_round=800,
        early_stopping_rounds=100,
        N=N,
    )
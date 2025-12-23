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


TRAIN_RATIO = 0.9
VAL_RATIO = 0.1 
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

for N in N_list:
    train_files, val_files, test_files = split_csv_files(data_dir=file_dir, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=1-TRAIN_RATIO-VAL_RATIO, seed=SEED)
    train_data, train_labels = extract_feature(files_dir=train_files, N=N)
    val_data, val_labels = extract_feature(files_dir=val_files, N=N)
    test_data, test_labels = extract_feature(files_dir=test_files, N=N)
    
    model = XGBModel()
    model.train(
        train_data,
        train_labels,
        val_data,
        val_labels,
        num_boost_round=1500,
        early_stopping_rounds=150,
        N=N,
    )
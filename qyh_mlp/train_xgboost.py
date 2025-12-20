import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from dataset import LOBWindowDataset
from model import LOBMLP, Predictor
import os
import random

import xgboost as xgb

def build_xgb_dataset(csv_paths, window_size, predictor, label_name="label_5"):
    X_list = []
    y_list = []

    for csv_path in tqdm(csv_paths, desc="Building XGB dataset"):
        df = pd.read_csv(csv_path)

        for i in range(len(df) - window_size + 1):
            window_df = df.iloc[i:i + window_size].copy()

            # === 用你原来的特征工程 ===
            x_hat = predictor.preprocess([window_df])  # [1, 43]
            X_list.append(x_hat.cpu().numpy()[0])

            # === label ===
            y_list.append(int(window_df.iloc[-1][label_name]))

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)

    return X, y

def split_csv_files(
    data_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # 读取所有 csv
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

    return train_files, val_files, test_files

def main():
    data_dir = "data/data_raw"
    train_csvs, val_csvs, test_csvs = split_csv_files(data_dir)

    print(f"Train CSVs: {len(train_csvs)}")
    print(f"Val CSVs:   {len(val_csvs)}")
    print(f"Test CSVs:  {len(test_csvs)}")

    predictor = Predictor()
    predictor.device = "cuda"  # XGBoost 用 numpy
    print(predictor.device)

    # ===== 构建数据 =====
    X_train, y_train = build_xgb_dataset(train_csvs, 100, predictor)
    X_val, y_val     = build_xgb_dataset(val_csvs, 100, predictor)

    print("Train shape:", X_train.shape)
    print("Val shape:", X_val.shape)

    # ===== XGBoost =====
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=True
    )

    model.save_model("qyh_mlp/checkpoints/xgb_model.json")

main()
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab
import random
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, mean_squared_log_error
from utils import *
# from catboost import CatBoostClassifier
from data_analyze import analyze_quantitative_data
from concurrent.futures import ThreadPoolExecutor
import glob
import time
from data_process import *
from classifier_pipeline import *
import itertools
from feature_engineering import *
from tqdm import tqdm
import pandas as pd

def read_csv_parallel(file_list, N):
    def read_single_file(file):
        data = pd.read_csv(file)[:-N]
        data = data.reset_index(drop=True)     # 重置行索引，从 0 开始
        return data
    
    with ThreadPoolExecutor() as executor:
        dfs = list(executor.map(read_single_file, file_list))
    
    return pd.concat(dfs, axis=0, ignore_index=True)

def extract_feature(file_dir):
    raw_data_dir = os.path.join(file_dir, "data_raw")
    save_dir = os.path.join(file_dir, "data_extract_feature")
    os.makedirs(save_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(raw_data_dir, "*.csv"))

    def process_file(file):
        base_name = os.path.basename(file)
        save_name = os.path.splitext(base_name)[0] + "_extract_feature.csv"
        save_path = os.path.join(save_dir, save_name)

        if os.path.exists(save_path):
            try:
                df = pd.read_csv(save_path)
                df = df.reset_index(drop=True)
                df = feature_extractor(df)
                df.to_csv(save_path, index=False)       # 保存到目标文件（新建或覆盖）
            except pd.errors.EmptyDataError:
                df = pd.read_csv(file)
                df = df.reset_index(drop=True)
                df = feature_extractor(df)
                df.to_csv(save_path, index=False)       # 保存到目标文件（新建或覆盖）
        else:
            df = pd.read_csv(file)
            df = df.reset_index(drop=True)
            df = feature_extractor(df)
            df.to_csv(save_path, index=False)       # 保存到目标文件（新建或覆盖）

        return save_path

    results = []
    with ThreadPoolExecutor() as executor:
        for save_path in tqdm(executor.map(process_file, csv_files),
                              total=len(csv_files),
                              desc="Extracting features"):
            results.append(save_path)

    print(f"Feature extraction done, saved {len(results)} files to {save_dir}")
    return results

def split_data(file_dir, train_ratio, val_ratio, seed, N_list):
    random.seed(seed)
    
    raw_data_dir = os.path.join(file_dir, "data_raw")
    save_dir = os.path.join(file_dir, "data_extract_feature_split")
    os.makedirs(save_dir, exist_ok=True)
    
    train_files = []
    val_files = []
    test_files = []
    csv_files = glob.glob(os.path.join(raw_data_dir, "*.csv"))

    for file_path in csv_files:
        if random.random() < train_ratio:
            train_files.append(file_path)
        elif random.random() > 1 - train_ratio - val_ratio:
            test_files.append(file_path)
        else:
            val_files.append(file_path)
    
    for N in N_list:
        print(f"start split_data N={N}")
        train_data = read_csv_parallel(train_files, N)
        val_data = read_csv_parallel(val_files, N)
        test_data = read_csv_parallel(test_files, N)
        
        train_save_path = os.path.join(save_dir, f"train_data_{N}.csv")
        val_save_path = os.path.join(save_dir, f"val_data_{N}.csv")
        test_save_path = os.path.join(save_dir, f"test_data_{N}.csv")
        
        train_data.to_csv(train_save_path, index=False)
        val_data.to_csv(val_save_path, index=False)
        test_data.to_csv(test_save_path, index=False)
        
        print(f"Saved train_data_{N} to {train_save_path}, shape: {train_data.shape}")
        print(f"Saved val_data_{N} to {val_save_path}, shape: {val_data.shape}")
        print(f"Saved test_data_{N} to {test_save_path}, shape: {test_data.shape}")

def load_data(file_dir, N):
    split_data_dir = os.path.join(file_dir, "data_extract_feature_split")
    train_path = os.path.join(split_data_dir, f"train_data_{N}.csv")
    val_path = os.path.join(split_data_dir, f"val_data_{N}.csv")
    test_path = os.path.join(split_data_dir, f"test_data_{N}.csv")
    
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)
    
    return train_data, val_data, test_data
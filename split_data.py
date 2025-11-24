import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab
import random
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, mean_squared_log_error
from utils import *
from catboost import CatBoostClassifier
from data_analyze import analyze_quantitative_data
from concurrent.futures import ThreadPoolExecutor
import glob
import time
from data_process import *
from classifier_pipeline import *
import itertools
from feature_engineering import *

def read_csv_parallel(file_list, N):
    def read_single_file(file, N):
        data = pd.read_csv(file)[:-N]
        data = feature_extractor(data)
        data = data.reset_index(drop=True)
        return data
    
    with ThreadPoolExecutor() as executor:
        dfs = list(executor.map(read_single_file, file_list, itertools.repeat(N)))
    
    return pd.concat(dfs, axis=0, ignore_index=True)


def split_and_save_data(file_dir, N_list, train_val_ratio=0.8, seed=42):
    random.seed(seed)
    save_dir = os.path.join(file_dir, "split_data")
    os.makedirs(save_dir, exist_ok=True)
    train_files = []
    val_files = []
    
    csv_files = glob.glob(os.path.join(file_dir, "*.csv"))
    
    for file_path in csv_files:
        if random.random() < train_val_ratio:
            train_files.append(file_path)
        else:
            val_files.append(file_path)
    
    for N in N_list:
        train_data = read_csv_parallel(train_files, N)
        val_data = read_csv_parallel(val_files, N)

        
        train_save_path = os.path.join(save_dir, f"train_data_N{N}.csv")
        val_save_path = os.path.join(save_dir, f"val_data_N{N}.csv")
        
        train_data.to_csv(train_save_path, index=False)
        val_data.to_csv(val_save_path, index=False)
        
        print(f"Saved train data for N={N} to {train_save_path}, shape: {train_data.shape}")
        print(f"Saved val data for N={N} to {val_save_path}, shape: {val_data.shape}")

def load_data(file_dir, N):
    split_data_dir = os.path.join(file_dir, "split_data")
    train_path = os.path.join(split_data_dir, f"train_data_N{N}.csv")
    val_path = os.path.join(split_data_dir, f"val_data_N{N}.csv")
    
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    
    return train_data, val_data
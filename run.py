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

pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', 1000)        # 设置显示宽度
pd.set_option('display.max_colwidth', None) # 显示完整的列内容

TRAIN_VAL_RATIO = 0.8
seed = 42
random.seed(seed)

file_dir = "./data"

train_files = []
val_files = []
train_data = pd.DataFrame()
val_data = pd.DataFrame()

def read_csv_parallel(file_list):
    def read_single_file(file):
        return pd.read_csv(file)
    
    with ThreadPoolExecutor() as executor:
        dfs = list(executor.map(read_single_file, file_list))
    
    return pd.concat(dfs, axis=0, ignore_index=True)

csv_files = glob.glob(os.path.join(file_dir, "*.csv"))

for file_path in csv_files:
    if random.random() < TRAIN_VAL_RATIO:
        train_files.append(file_path)
    else:
        val_files.append(file_path)

train_data = read_csv_parallel(train_files)
val_data = read_csv_parallel(val_files)


# 自己划分训练集和验证集
# start_time = time.time()
# for file_name in os.listdir(file_dir):
#     if not file_name.endswith(".csv"):
#         continue
#     if random.random() < TRAIN_VAL_RATIO:
#         train_files.append(os.path.join(file_dir,file_name))
#     else:
#         val_files.append(os.path.join(file_dir,file_name))

# print(f"val_files is {val_files}")

# train_dfs = [pd.read_csv(f, low_memory=False) for f in train_files]
# train_data = pd.concat(train_dfs, axis=0, ignore_index=True)
# val_dfs = [pd.read_csv(f, low_memory=False) for f in val_files]
# val_data = pd.concat(val_dfs, axis=0, ignore_index=True)

# end_time = time.time()
# print(f"数据读取耗时: {end_time - start_time:.2f} 秒")
# exit()


analyze_quantitative_data(train_data, val_data, output_dir="./analysis_results")

# 先用原始数据作为特征试试
feature_col_names = [f for f in train_data.columns if f not in ['date','time','sym','label_5','label_10','label_20','label_40','label_60']] # TODO: 时间特征后续可以考虑加入
label_col_name = ['label_5','label_10','label_20','label_40','label_60']

for label in label_col_name:
    print(f'=================== {label} ===================')
    run_tree_model(CatBoostClassifier, train_data[feature_col_names], train_data[label], val_data[feature_col_names], val_data[label], seed)
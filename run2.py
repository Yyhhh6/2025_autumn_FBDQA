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

# pd.set_option('display.max_columns', None)  # 显示所有列
# pd.set_option('display.width', 1000)        # 设置显示宽度
# pd.set_option('display.max_colwidth', None) # 显示完整的列内容

TRAIN_VAL_RATIO = 0.8
seed = 42
random.seed(seed)

N_list = [5, 10, 20, 40, 60]
alpha_map = {5: 0.0005, 10: 0.0005, 20: 0.001, 40: 0.001, 60: 0.001}
file_dir = "./data"

train_files = []
val_files = []
train_data = pd.DataFrame()
val_data = pd.DataFrame()

# 尝试手动打标签
# train_data = pd.read_csv("/mnt/data/yyh/Ovi/tmp/tmp/data/snapshot_sym9_date78_am.csv")
# train_data1 = pd.read_csv("/mnt/data/yyh/Ovi/tmp/tmp/data/snapshot_sym9_date78_pm.csv")
# # train_data2 = pd.read_csv("/mnt/data/yyh/Ovi/tmp/tmp/data/snapshot_sym2_date9_am.csv")
# train_data = pd.concat([train_data, train_data1], axis=0, ignore_index=True)
# # print(train_data.shape)
# # print(train_data["n_midprice"].describe())
# # print(train_data["n_midprice"])
# # print(train_data["n_midprice"].head(20))
# # print("\n")

# labels = ["label_5", "label_10", "label_20", "label_40", "label_60"]
# for label in labels:
#     index = int(label[6:])
#     tmp = train_data["n_midprice"].shift(-index) - train_data["n_midprice"]
#     tmp_label = pd.Series(1, dtype=int, index=tmp.index)
#     # print(f"init tmp label is {tmp_label}")
#     if index == 5 or index == 10:
#         threshold = 0.0005
#     else:
#         threshold = 0.001
#     # print(f"tmp is {tmp}")
#     tmp.fillna(0, inplace=True)
#     tmp_label[tmp > threshold] = 2
#     tmp_label[tmp < -threshold] = 0
#     print(f"tmp_label value counts:\n{tmp_label.value_counts()}")
#     print(f"{index} label: {train_data[label][train_data[label] != tmp_label]}")    
# exit()

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

print(train_data.iloc[:10000].describe())
print(train_data.iloc[:10000].head())
print(train_data.iloc[:10000].tail())
ax = train_data.iloc[:10000][['n_bid1','n_ask1','n_midprice']].plot(figsize=(12, 6))
ax.figure.savefig('price_plot.png', dpi=300, bbox_inches='tight')
exit()
# print(train_data.columns)
# ['date', 'time', 'sym', 'n_close', 'amount_delta', 'n_midprice', 'n_bid1', 'n_bsize1', 'n_bid2', 'n_bsize2', 'n_bid3', 'n_bsize3', 'n_bid4', 'n_bsize4', 'n_bid5', 'n_bsize5', 'n_ask1', 'n_asize1', 'n_ask2', 'n_asize2', 'n_ask3', 'n_asize3', 'n_ask4', 'n_asize4', 'n_ask5', 'n_asize5', 'label_5', 'label_10', 'label_20', 'label_40', 'label_60']

# 随机打乱训练数据并重置索引
# train_data = train_data.sample(frac=1, random_state=seed).reset_index(drop=True)

# 将时间转化为标签
# TODO：time和date合成？
# tick_dates = pd.to_datetime(train_data['date'].astype(str) + train_data['time'].astype(str).str.zfill(6), format='%Y%m%d%H%M%S')
# 上午: 09:40:03-11:19:57
# 下午: 13:10:03-14:49:57
train_data['time_label'] = assign_tick_time_labels(train_data['time'])
val_data['time_label'] = assign_tick_time_labels(val_data['time'])
# tmp = train_data[['time_label', 'time']].value_counts().sort_index() # 标签没问题

# 去极值
feature_names = ['n_close', 'amount_delta', 'n_midprice', 'n_bid1', 'n_bsize1', 'n_bid2', 'n_bsize2', 'n_bid3', 'n_bsize3', 'n_bid4', 'n_bsize4', 'n_bid5', 'n_bsize5', 'n_ask1', 'n_asize1', 'n_ask2', 'n_asize2', 'n_ask3', 'n_asize3', 'n_ask4', 'n_asize4', 'n_ask5', 'n_asize5', 'time_label']
train_data = extreme_process_MAD(train_data, feature_names=feature_names, num=3)
val_data = extreme_process_MAD(val_data, feature_names=feature_names, num=3)

# 归一化
train_data[feature_names] = data_scale_Z_Score(train_data, feature_names=feature_names)
val_data[feature_names] = data_scale_Z_Score(val_data, feature_names=feature_names)
# calc_factor_correlation(train_data, feature_names)
# TODO：sys作为因子？

# 对因子进行对称正交化处理
train_data = lowdin_orthogonal(train_data, col=feature_names)
val_data = lowdin_orthogonal(val_data, col=feature_names)
# calc_factor_correlation(train_data, feature_names, save_path="orthogonal_factor_corr_heatmap.png")
# print(train_data.describe())
# print(train_data.head(20))


results = run_pipeline(train_data, val_data, feature_names, N_list, alpha_map, out_dir="./results", device="cuda")
print(results)

# print(f"总数据量: {sorted(set(tick_dates))}")
# print(f"Training data shape: {train_data.describe()}")
# print(f"head {train_data.head()}")

# 先用原始数据作为特征试试
# feature_col_names = [f for f in train_data.columns if f not in ['date','time','sym','label_5','label_10','label_20','label_40','label_60']] # TODO: 时间特征后续可以考虑加入
# label_col_name = ['label_5','label_10','label_20','label_40','label_60']

# for label in label_col_name:
#     print(f'=================== {label} ===================')
#     run_tree_model(CatBoostClassifier, train_data[feature_col_names], train_data[label], val_data[feature_col_names], val_data[label], seed)
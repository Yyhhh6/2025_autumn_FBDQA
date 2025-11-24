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

TRAIN_VAL_RATIO = 0.8
seed = 42
random.seed(seed)

file_dir = "./data"

train_files = []
val_files = []
train_data = pd.DataFrame()
val_data = pd.DataFrame()

# 自己划分训练集和验证集
for file_name in os.listdir(file_dir):
    if not file_name.endswith(".csv"):
        continue
    if random.random() < TRAIN_VAL_RATIO:
        train_files.append(os.path.join(file_dir,file_name))
    else:
        val_files.append(os.path.join(file_dir,file_name))

print(f"val_files is {val_files}")
    
for file in train_files:
    df = pd.read_csv(file)
    train_data = pd.concat([train_data,df],axis=0,ignore_index=True)
for file in val_files:
    df = pd.read_csv(file)
    val_data = pd.concat([val_data,df],axis=0,ignore_index=True)

print(f"train shape: {train_data.shape}, val shape: {val_data.shape}")

analyze_quantitative_data(train_data, val_data, output_dir="./analysis_results")

# 多折交叉验证
# 这里后面可以自己实现，因为需要按照文件来划分数据
# for i, file_name in enumerate(os.listdir(file_dir)):
#     print(f"Processing file {i+1}: {file_name}")
#     if not file_name.endswith(".csv"):
#         continue
#     train_file = os.path.join(file_dir,file_name)
#     df = pd.read_csv(train_file)
#     train_data = pd.concat([train_data,df],axis=0,ignore_index=True)
# print(f"total data shape: {train_data.shape}")
# 先用原始数据作为特征试试
feature_col_names = [f for f in train_data.columns if f not in ['date','time','sym','label_5','label_10','label_20','label_40','label_60']] # TODO: 时间特征后续可以考虑加入
label_col_name = ['label_5','label_10','label_20','label_40','label_60']

for label in label_col_name:
    print(f'=================== {label} ===================')
    run_tree_model(CatBoostClassifier, train_data[feature_col_names], train_data[label], val_data[feature_col_names], val_data[label], seed)
    # cat_oof, precision_list, recall_list, f1_score_list = cv_model(CatBoostClassifier, train_data[feature_col_names], train_data[label], val_data[feature_col_names], 'cat', seed)
    # train_data[label] = np.argmax(cat_oof, axis=1)
    # # val_data[label] = np.argmax(cat_test, axis=1)
    # print(f"Precision_list: {precision_list}, avg Precision: {np.mean(precision_list)}")
    # print(f"Recall_list: {recall_list}, avg Recall: {np.mean(recall_list)}")
    # print(f"F1_score_list: {f1_score_list}, avg F1_score: {np.mean(f1_score_list)}")
# feature_col_names = ['n_bid1','n_bid2','n_bid3','n_bid4','n_bid5',\
#                      'n_ask1','n_ask2','n_ask3','n_ask4','n_ask5']
# label_col_name = ['label_5']
# train_sample_nums = 20000
# # 别忘了数据形状和存储连续性
# train_data = np.ascontiguousarray(train_data[feature_col_names][:train_sample_nums].values)
# train_label = train_data[label_col_name][:train_sample_nums].values.reshape(-1)

# test_data = np.ascontiguousarray(train_data[feature_col_names][train_sample_nums:].values)
# test_label = train_data[label_col_name][train_sample_nums:].values.reshape(-1)
# # 确定有无na值，若有要进行处理（是否一定能用0填充）
# print("Checking for null values in training and validation data: (True indicates presence of nulls)")
# print(train_data.isnull().values.any())
# print(val_data.isnull().values.any())

# folds = 5
# kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
# oof = np.zeros([train_data.shape[0], 3])
# test_predict = np.zeros([test_data.shape[0], 3])
# cv_scores = []

# for i, (train_index, valid_index) in enumerate(kf.split(train_data, train_label)):
#     print('************************************ {} ************************************'.format(str(i+1)))
#     trn_x, trn_y, val_x, val_y = train_data.iloc[train_index], train_label[train_index], train_data.iloc[valid_index], train_label[valid_index]







# file_name = f"snapshot_sym1_date66_pm.csv"

# df = pd.read_csv(os.path.join(file_dir,file_name))
# print(df.columns)
# Index(['date', 'time', 'sym', 'n_close', 'amount_delta', 'n_midprice',
#        'n_bid1', 'n_bsize1', 'n_bid2', 'n_bsize2', 'n_bid3', 'n_bsize3',
#        'n_bid4', 'n_bsize4', 'n_bid5', 'n_bsize5', 'n_ask1', 'n_asize1',
#        'n_ask2', 'n_asize2', 'n_ask3', 'n_asize3', 'n_ask4', 'n_asize4',
#        'n_ask5', 'n_asize5', 'label_5', 'label_10', 'label_20', 'label_40',
#        'label_60'],
#       dtype='object')
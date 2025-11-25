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
from split_data import split_and_save_data, load_data
from feature_engineering import *
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', 1000)        # 设置显示宽度
pd.set_option('display.max_colwidth', None) # 显示完整的列内容
# train_data = pd.read_csv("./data/snapshot_sym9_date78_am.csv")
# train_data = feature_extractor(train_data[:-5])
# # TODO：针对前几个数据的行为是什么
# train_data = train_data.reset_index(drop=True)
# print("train_data shape:", train_data.shape)
# print("train_data columns:", train_data.columns)
# print("train_data head:\n", train_data.head())
# print("train_data describe:\n", train_data.describe())
# exit()
# print(f"train_data shape: {train_data.shape}")
# print(f"train_data columns: {train_data.columns}")
# print(f"train_data head:\n{train_data.head()}")
# print(f"train_data describe:\n{train_data.describe()}")
# features = extract_features_for_day(train_data, window=100)
# print("--"*30)

# # 拼回标签
# train_data = features.join(train_data)[:-5].reset_index()
# print(f"train_data shape: {train_data.shape}")
# print(f"train_data columns: {train_data.columns}")
# print(f"train_data head:\n{train_data.head()}")
# print(f"train_data describe:\n{train_data.describe()}")
# print(train_data['time'])
# exit()
# pd.set_option('display.max_columns', None)  # 显示所有列
# pd.set_option('display.width', 1000)        # 设置显示宽度
# pd.set_option('display.max_colwidth', None) # 显示完整的列内容

TRAIN_VAL_RATIO = 0.8
seed = 42
random.seed(seed)

N_list = [5, 10, 20, 40, 60]
N_list = [5]
alpha_map = {5: 0.0005, 10: 0.0005, 20: 0.001, 40: 0.001, 60: 0.001}
file_dir = "./data"


# split_and_save_data(file_dir, N_list, train_val_ratio=TRAIN_VAL_RATIO, seed=seed)

feature_names = ['time_label', 'n_close', 'amount_delta', 'n_midprice',
                'n_bid1', 'n_bsize1', 'n_bid2', 'n_bsize2', 'n_bid3', 'n_bsize3',
                'n_bid4', 'n_bsize4', 'n_bid5', 'n_bsize5', 'n_ask1', 'n_asize1', 
                'n_ask2', 'n_asize2', 'n_ask3', 'n_asize3', 'n_ask4', 'n_asize4', 
                'n_ask5', 'n_asize5', 'bid1', 'bid2', 'bid3', 'bid4', 'bid5', 
                'ask1', 'ask2', 'ask3', 'ask4', 'ask5', 'ask1_ma5', 'ask1_ma10', 
                'ask1_ma20', 'ask1_ma40', 'ask1_ma60', 'ask1_ma80', 'ask1_ma100', 
                'bid1_ma5', 'bid1_ma10', 'bid1_ma20', 'bid1_ma40', 'bid1_ma60', 
                'bid1_ma80', 'bid1_ma100', 'spread1', 'spread2', 'spread3', 
                'mid_price1', 'mid_price2', 'mid_price3', 'weighted_ab1', 
                'weighted_ab2', 'weighted_ab3', 'relative_spread1', 'relative_spread2', 
                'relative_spread3', 'bsize1', 'bsize2', 'bsize3', 'bsize4', 'bsize5', 
                'asize1', 'asize2', 'asize3', 'asize4', 'asize5', 'amount']


for N in N_list:
    train_data, val_data = load_data(file_dir, N=N) # TODO：此时是针对不同N训练不同的模型，需要训练一个统一的模型吗？

    # 特征挖掘
    # print(train_data.columns) # 除了date, time, symbol, label外其余均为特征
    # print(train_data.shape) # N=5时，(2300530, 78)
    # exit()

    # 增加时间标签特征
    train_data['time_label'] = assign_tick_time_labels(train_data['time'])
    val_data['time_label'] = assign_tick_time_labels(val_data['time'])

    # 去NaN
    assert not train_data[feature_names].isnull().sum().any(), "train_data中存在NaN值，请检查！"
    assert not val_data[feature_names].isnull().sum().any(), "val_data中存在NaN值，请检查！"

    # 去极值
    train_data = extreme_process_MAD(train_data, feature_names=feature_names, num=3)
    val_data = extreme_process_MAD(val_data, feature_names=feature_names, num=3)

    assert not train_data.isnull().any().any(), "train_data中存在NaN值，请检查！"
    assert not val_data.isnull().any().any(), "val_data中存在NaN值，请检查！"
    def check_finite_pandas(df):
        # 检查无限值和NaN值
        has_inf = np.isinf(df.select_dtypes(include=[np.number])).any().any()
        has_na = df.isna().any().any()
        return not (has_inf or has_na)

    # 修改断言
    assert check_finite_pandas(train_data), "train_data中存在inf或NaN值，请检查！"
    assert check_finite_pandas(val_data), "val_data中存在inf或NaN值，请检查！"
    # assert np.all(np.isfinite(np.array(train_data))), "train_data中存在inf值，请检查！"
    # assert np.all(np.isfinite(np.array(val_data))), "val_data中存在inf值，请检查！"

    # print(train_data['momentum_5'].describe())
    # print(train_data['momentum_5'])
    # 归一化
    train_data[feature_names] = data_scale_Z_Score(train_data, feature_names=feature_names)
    val_data[feature_names] = data_scale_Z_Score(val_data, feature_names=feature_names)
    # print("-"*50)
    # print(train_data['momentum_5'].describe())
    # print(train_data['momentum_5'])
    # # 查看哪些地方有null或inf
    # print("找出train_data中的NaN和inf值所在的行和列...")
    # # 找出train_data中的NaN和inf值所在的行和列...
    # feat_df = train_data[feature_names]

    # # 每列 NaN 和 inf 的统计
    # nan_counts = feat_df.isnull().sum()
    # inf_mask = np.isinf(feat_df.values)
    # inf_counts = pd.Series(inf_mask.sum(axis=0), index=feat_df.columns)

    # print("Columns with NaN in train_data:\n", nan_counts[nan_counts > 0])
    # print("Columns with inf in train_data:\n", inf_counts[inf_counts > 0])

    # # 每行是否含 NaN / inf
    # nan_row_mask = feat_df.isnull().any(axis=1)
    # inf_row_mask = pd.DataFrame(inf_mask, index=feat_df.index, columns=feat_df.columns).any(axis=1)

    # print(f"Number of rows with any NaN: {nan_row_mask.sum()}")
    # print(f"Number of rows with any inf: {inf_row_mask.sum()}")

    # # 列出出现 NaN 的前 20 个行及其列名
    # nan_idxs = train_data.index[nan_row_mask].tolist()
    # if nan_idxs:
    #     print("First 20 row indices with NaN:", nan_idxs[:20])
    #     for idx in nan_idxs[:20]:
    #         cols = feat_df.columns[feat_df.loc[idx].isnull()].tolist()
    #         print(f" Row {idx} NaN columns: {cols}")

    # # 列出出现 inf 的前 20 个行及其列名
    # inf_idxs = train_data.index[inf_row_mask].tolist()
    # if inf_idxs:
    #     print("First 20 row indices with inf:", inf_idxs[:20])
    #     for idx in inf_idxs[:20]:
    #         pos = feat_df.index.get_loc(idx)
    #         cols = feat_df.columns[inf_mask[pos]].tolist()
    #         print(f" Row {idx} inf columns: {cols}")

    # # 合并 NaN 或 inf 的行（若需要查看完整行）
    # any_bad_mask = nan_row_mask | inf_row_mask
    # if any_bad_mask.any():
    #     print("Sample rows containing NaN or inf (first 5):")
    #     print(train_data.loc[any_bad_mask].head())

    # print(train_data[feature_names].isnull().sum())
    # # 检查train_data[feature_names].isnull().sum()中是否有index的值大于0
    # mask = train_data[feature_names].isnull().sum()
    # print(mask[mask > 0])
    # print(np.isinf(train_data[feature_names]).sum())
    # print((np.isinf(train_data[feature_names]).sum() > 0).any())
    # print("检查val_data中的NaN和inf值...")
    # print(val_data[feature_names].isnull().sum())
    # print((val_data[feature_names].isnull().sum() > 0).any())
    # print(np.isinf(val_data[feature_names]).sum())
    # print((np.isinf(val_data[feature_names]).sum() > 0).any())
    # train_data = remove_outliers(train_data, feature_names=feature_names)
    # val_data = remove_outliers(val_data, feature_names=feature_names)

    # TODO：sys作为因子？

    # TODO：市值、行业中性化处理

    # TODO；对因子进行对称正交化处理，是否需要？
    # 增加其他因子后这里在求特征值和特征向量时会出现inf或nan
    # train_data = lowdin_orthogonal(train_data, col=feature_names)
    # val_data = lowdin_orthogonal(val_data, col=feature_names)


    results = run_pipeline(train_data, val_data, feature_names, N_list, alpha_map, out_dir="./results", device="cuda")
    print(results)
    # exit()

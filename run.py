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
from split_data import load_data, extract_feature, split_data
from feature_engineering import *

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2  # 测试集自动是 0.1
SEED = 42

# N_list = [5, 10, 20, 40, 60]
N_list = [5]
alpha_map = {5: 0.0005, 10: 0.0005, 20: 0.001, 40: 0.001, 60: 0.001}
file_dir="./data"

# 特征提取
# extract_feature(file_dir=file_dir)
# split_data(file_dir=file_dir, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=SEED, N_list=N_list)

# feature_names = ['time_label', 'n_close', 'amount_delta', 'n_midprice',
#                 'n_bid1', 'n_bsize1', 'n_bid2', 'n_bsize2', 'n_bid3', 'n_bsize3',
#                 'n_bid4', 'n_bsize4', 'n_bid5', 'n_bsize5', 'n_ask1', 'n_asize1', 
#                 'n_ask2', 'n_asize2', 'n_ask3', 'n_asize3', 'n_ask4', 'n_asize4', 
#                 'n_ask5', 'n_asize5', 'bid1', 'bid2', 'bid3', 'bid4', 'bid5', 
#                 'ask1', 'ask2', 'ask3', 'ask4', 'ask5', 'ask1_ma5', 'ask1_ma10', 
#                 'ask1_ma20', 'ask1_ma40', 'ask1_ma60', 'ask1_ma80', 'ask1_ma100', 
#                 'bid1_ma5', 'bid1_ma10', 'bid1_ma20', 'bid1_ma40', 'bid1_ma60', 
#                 'bid1_ma80', 'bid1_ma100', 'spread1', 'spread2', 'spread3', 
#                 'mid_price1', 'mid_price2', 'mid_price3', 'weighted_ab1', 
#                 'weighted_ab2', 'weighted_ab3', 'relative_spread1', 'relative_spread2', 
#                 'relative_spread3', 'bsize1', 'bsize2', 'bsize3', 'bsize4', 'bsize5', 
#                 'asize1', 'asize2', 'asize3', 'asize4', 'asize5', 'amount']
feature_names = ['time_label', 'n_close', 'amount_delta', 'n_midprice', 'n_bid1', 
                 'n_bsize1', 'n_bid2', 'n_bsize2', 'n_bid3', 'n_bsize3', 'n_bid4', 
                 'n_bsize4', 'n_bid5', 'n_bsize5', 'n_ask1', 'n_asize1', 'n_ask2', 
                 'n_asize2', 'n_ask3', 'n_asize3', 'n_ask4', 'n_asize4', 'n_ask5', 
                 'n_asize5', 'label_5', 'label_10', 'label_20', 'label_40', 'label_60', 
                 'bid1', 'bid2', 'bid3', 'bid4', 'bid5', 'ask1', 'ask2', 'ask3', 'ask4', 
                 'ask5', 'ask1_ma5', 'ask1_ma10', 'ask1_ma20', 'ask1_ma40', 'ask1_ma60', 
                 'ask1_ma80', 'ask1_ma100', 'bid1_ma5', 'bid1_ma10', 'bid1_ma20', 
                 'bid1_ma40', 'bid1_ma60', 'bid1_ma80', 'bid1_ma100', 'spread1', 'spread2', 
                 'spread3', 'mid_price1', 'mid_price2', 'mid_price3', 'weighted_ab1', 
                 'weighted_ab2', 'weighted_ab3', 'relative_spread1', 'relative_spread2', 
                 'relative_spread3', 'bsize1', 'bsize2', 'bsize3', 'bsize4', 'bsize5', 
                 'asize1', 'asize2', 'asize3', 'asize4', 'asize5', 'amount', 'ofi', 
                 'ofi_ma5', 'ofi_ma20', 'ofi_ma50', 'ofi_trend_10', 'obi1', 'obi3', 
                 'obi1_ma10', 'obi3_ma10', 'obi1_ma30', 'microprice', 'microprice_change', 
                 'microprice_change_ma5', 'microprice_deviation', 'microprice_deviation_ma10',
                 'kyle_lambda', 'kyle_lambda_ma10', 'kyle_lambda_ma30', 'realized_vol_5', 
                 'realized_vol_10', 'realized_vol_20', 'realized_vol_50', 'vol_change', 
                 'vol_ratio', 'momentum_5', 'momentum_10', 'momentum_20', 'reversal_3', 
                 'reversal_5', 'price_acceleration', 'effective_spread', 
                 'effective_spread_ma10', 'effective_spread_ma30', 'spread_change', 
                 'spread_volatility', 'volume_ratio_10', 'volume_ratio_30', 'volume_surge', 
                 'volume_surge_ma10', 'short_rsi', 'price_momentum_3', 'price_momentum_5', 
                 'bid_depth_3', 'ask_depth_3', 'depth_imbalance_3', 'bid_depth_change', 
                 'ask_depth_change', 'bid_pressure', 'ask_pressure', 'bid_skew', 'ask_skew', 
                 'price_mean_100', 'price_std_100', 'price_max_100', 'price_min_100', 
                 'price_range_100', 'price_position', 'volume_mean_100', 'volume_std_100', 
                 'spread_mean_100', 'spread_std_100']



for N in N_list:
    train_data, val_data, test_data = load_data(file_dir=file_dir, N=N) # TODO：此时是针对不同N训练不同的模型，需要训练一个统一的模型吗？

    # 增加时间标签特征
    train_data['time_label'] = assign_tick_time_labels(train_data['time'])
    val_data['time_label'] = assign_tick_time_labels(val_data['time'])
    test_data['time_label'] = assign_tick_time_labels(test_data['time'])
    test_data['n_close_origin'] = test_data['n_close']  # 保存原始收盘价用于后续计算PnL

    print("train_data的列名：", train_data.columns.tolist())

    # 去NaN（包括时间标签处理）
    ## 处理时间标签的NaN值，使用0填充或删除
    assert 'time_label' in train_data.columns, "train_data中缺少'time_label'列！"
    assert 'time_label' in val_data.columns, "val_data中缺少'time_label'列！"
    assert 'time_label' in test_data.columns, "test_data中缺少'time_label'列！"
    # train_data['time_label'] = train_data['time_label'].fillna(0)
    # val_data['time_label'] = val_data['time_label'].fillna(0)
    # test_data['time_label'] = test_data['time_label'].fillna(0)
    
    # assert check_finite_pandas(train_data), "train_data中存在inf值，请检查！"
    # assert check_finite_pandas(val_data), "val_data中存在inf值，请检查！"
    # assert check_finite_pandas(test_data), "test_data中存在inf值，请检查！"
    # assert check_nan_pandas(train_data), "train_data中存在NaN值，请检查！"
    # assert check_nan_pandas(val_data), "val_data中存在NaN值，请检查！"
    # assert check_nan_pandas(test_data), "test_data中存在NaN值，请检查！"

    ## 检查特征列中的NaN
    train_data = factors_null_process(train_data, feature_names=feature_names, class_name="训练集")
    val_data = factors_null_process(val_data, feature_names=feature_names, class_name="验证集")
    test_data = factors_null_process(test_data, feature_names=feature_names, class_name="测试集")

    # 去极值（基于训练集统计量）
    train_data = extreme_process_MAD(train_data, feature_names=feature_names, num=3)
    val_data = extreme_process_MAD(val_data, feature_names=feature_names, num=3)
    test_data = extreme_process_MAD(test_data, feature_names=feature_names, num=3)

    # 修改断言
    assert check_finite_pandas(train_data), "train_data中存在inf值，请检查！"
    assert check_finite_pandas(val_data), "val_data中存在inf值，请检查！"
    assert check_finite_pandas(test_data), "test_data中存在inf值，请检查！"
    assert check_nan_pandas(train_data), "train_data中存在NaN值，请检查！"
    assert check_nan_pandas(val_data), "val_data中存在NaN值，请检查！"
    assert check_nan_pandas(test_data), "test_data中存在NaN值，请检查！"

    # 归一化
    train_data[feature_names] = data_scale_Z_Score(train_data, feature_names=feature_names)
    val_data[feature_names] = data_scale_Z_Score(val_data, feature_names=feature_names)
    test_data[feature_names] = data_scale_Z_Score(test_data, feature_names=feature_names)

    results = run_pipeline(train_data, val_data, test_data, feature_names, N_list, alpha_map, out_dir="./results", device="cuda")
    print(results)
    # exit()
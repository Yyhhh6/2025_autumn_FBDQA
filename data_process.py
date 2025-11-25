import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def check_finite_pandas(df): 
    return not np.isinf(df.select_dtypes(include=[np.number])).any().any()
def check_nan_pandas(df):
    return not df.isna().any().any()

def factors_null_process(feature_names: list, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    medians = train[feature_names].median()

    # 使用训练集的中位数填充 train/val/test
    train[feature_names] = train[feature_names].fillna(medians)
    val[feature_names] = val[feature_names].fillna(medians)
    test[feature_names] = test[feature_names].fillna(medians)
    
    return train, val, test

       
def remove_outliers(data: pd.DataFrame, feature_names: list, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> pd.DataFrame:
    ''' 去除异常值，使用分位数法 '''
    data_ = data.copy()
    for feature in feature_names:
        lower_bound = data_[feature].quantile(lower_quantile)
        upper_bound = data_[feature].quantile(upper_quantile)
        data_ = data_[(data_[feature] >= lower_bound) & (data_[feature] <= upper_bound)]
    return data_

def calc_MAD_params(train, feature_names, num=3):
    median = train[feature_names].median(axis=0)
    mad = (train[feature_names].sub(median).abs()).median(axis=0)
    lower = median - num * 1.4826 * mad
    upper = median + num * 1.4826 * mad
    return lower, upper

def extreme_process_MAD(feature_names, train, val, test, num=3) -> pd.DataFrame:
    lower, upper = calc_MAD_params(train, feature_names, num)
    train[feature_names] = train[feature_names].clip(lower=lower, upper=upper, axis=1)
    val[feature_names] = val[feature_names].clip(lower=lower, upper=upper, axis=1)
    test[feature_names] = test[feature_names].clip(lower=lower, upper=upper, axis=1)
    return train, val, test

def data_scale_Z_Score(data, feature_names=None):
    if feature_names is not None:
        data_ = data[feature_names].copy()
        data_.loc[:, feature_names] = (
            data_.loc[:, feature_names] - data_.loc[:, feature_names].mean()) / (data_.loc[:, feature_names].std() + 1e-10)
    else:
        data_ = data.copy()
        data_ = (data_ - data_.mean()) / (data_.std() + 1e-10)
    return data_

def assign_tick_time_labels(tick_series: pd.Series) -> pd.Series:
    """
    为tick数据分配时间标签
    
    Parameters:
    tick_series: pd.Series, 格式为 'HH:MM:SS' 的时间字符串
    
    Returns:
    pd.Series: 时间标签，不在范围内的返回NaN
    """
    # 创建映射字典
    tick_series_ = tick_series.copy()
    time_label_map = {}
    label_counter = 0
    
    # 生成上午时间段 (09:40:03 - 11:19:57)
    am_start = datetime.strptime('09:40:03', '%H:%M:%S')
    am_end = datetime.strptime('11:19:57', '%H:%M:%S')
    
    current_time = am_start
    while current_time <= am_end:
        time_str = current_time.strftime('%H:%M:%S')
        time_label_map[time_str] = label_counter
        label_counter += 1
        current_time += timedelta(seconds=3)
    
    # 生成下午时间段 (13:10:03 - 14:49:57)
    pm_start = datetime.strptime('13:10:03', '%H:%M:%S')
    pm_end = datetime.strptime('14:49:57', '%H:%M:%S')
    
    current_time = pm_start
    while current_time <= pm_end:
        time_str = current_time.strftime('%H:%M:%S')
        time_label_map[time_str] = label_counter
        label_counter += 1
        current_time += timedelta(seconds=3)
    
    # 使用map函数进行映射
    labels = tick_series_.map(time_label_map)
    
    return labels
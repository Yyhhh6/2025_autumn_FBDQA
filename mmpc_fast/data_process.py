import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def factors_null_process_np(train, val=None, test=None, medians=None):
    if medians is None:
        medians = np.nanmedian(train, axis=0)

    if val is not None and test is not None:
        for data in (train, val, test):
            mask = np.isnan(data)
            data[mask] = medians[np.where(mask)[1]]
        return train, val, test, medians
    else:
        mask = np.isnan(train)
        train[mask] = medians[np.where(mask)[1]]
        return train

def calc_MAD_params_np(input, num=3):
    """
    返回 lower, upper: shape (len(feature_idx),)
    """
    median = np.nanmedian(input, axis=0)
    mad = np.nanmedian(np.abs(input - median), axis=0)

    lower = median - num * 1.4826 * mad
    upper = median + num * 1.4826 * mad

    return lower, upper


def extreme_process_MAD_np(train, val=None, test=None, lower=None, upper=None, num=3):
    if lower is None or upper is None:
        lower, upper = calc_MAD_params_np(train, num)

    if val is not None and test is not None:
        for data in (train, val, test):
            np.clip(data, lower, upper, out=data)
        return train, val, test, lower, upper
    else:
        np.clip(train, lower, upper, out=train)
        return train

def calc_mean_std_np(data, mean=None, std=None):
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    return (data - mean) / (std + 1e-10), mean, std

def data_scale_Z_Score_np(train, val=None, test=None, mean=None, std=None):
    if mean is not None and std is not None:
        train = (train - mean) / (std + 1e-10)
        return train
    else:
        train, mean, std = calc_mean_std_np(train)
        val = (val - mean) / (std + 1e-10)
        test = (test - mean) / (std + 1e-10)
        return train, val, test, mean, std


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

def data_scale_Z_Score(data, feature_names=None, mean=None, std=None) -> pd.DataFrame:
    if feature_names is not None:
        data_ = data[feature_names].copy()
        data_.loc[:, feature_names] = (
            data_.loc[:, feature_names] - data_.loc[:, feature_names].mean()) / (data_.loc[:, feature_names].std() + 1e-10)
    else:
        data_ = data.copy()
        if mean is not None and std is not None:
            data_ = (data_ - mean) / (std + 1e-10)
        else:
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


def assign_tick_time_label(tick_str: str) -> int | None:
    """
    为单个tick时间字符串分配标签
    
    Parameters:
    tick_str: str, 格式为 'HH:MM:SS'
    
    Returns:
    int 或 None: 对应标签，不在范围内返回 None
    """
    # 上午时间段
    am_start = datetime.strptime('09:40:03', '%H:%M:%S')
    am_end = datetime.strptime('11:19:57', '%H:%M:%S')
    
    # 下午时间段
    pm_start = datetime.strptime('13:10:03', '%H:%M:%S')
    pm_end = datetime.strptime('14:49:57', '%H:%M:%S')
    
    # 转成 datetime
    try:
        t = datetime.strptime(tick_str, '%H:%M:%S')
    except:
        return None
    
    # 判断上午
    if am_start <= t <= am_end:
        delta_sec = int((t - am_start).total_seconds())
        label = delta_sec // 3
        return label
    
    # 判断下午
    if pm_start <= t <= pm_end:
        delta_sec = int((t - pm_start).total_seconds())
        label = (int((am_end - am_start).total_seconds()) // 3 + 1) + (delta_sec // 3)
        return label
    
    # 不在范围
    return None


def process_file_worker(file_path, out_dir):
    # worker must be top-level for multiprocessing pickling
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import pandas as pd
        from pathlib import Path

        file = Path(file_path)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(file)
        plt.figure(figsize=(10, 6))
        plt.plot(df['time'], df['n_midprice'], label='Midprice')
        plt.xlabel('Time')
        plt.ylabel('Midprice')
        plt.title(f'Midprice vs Time for {file.name}')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        out_file = out / f'{file.name}_midprice_vs_time.png'
        plt.savefig(out_file)
        plt.close()
        return str(out_file)
    except Exception as e:
        return f'ERROR {file_path}: {e}'


def plot_midprice_vs_time(dir_path: str = '/hdd/yyh/src/quant/data/data_raw/',
                          out_dir: str = '/hdd/yyh/src/quant/data/midprice_plot/'):
    """
    多核并行绘图，使用可用核数的80%
    """
    import os
    import math
    import concurrent.futures
    from pathlib import Path
    from itertools import repeat

    p = Path(dir_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = [str(f) for f in p.iterdir() if f.suffix == '.csv']
    if not files:
        print("No CSV files found in", dir_path)
        return

    cpu_count = os.cpu_count() or 1
    num_workers = max(1, int(math.floor(cpu_count * 0.8)))
    print(f'Found {cpu_count} CPU cores, using {num_workers} worker(s) (80%).')

    # 使用 ProcessPoolExecutor 并行处理文件
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in executor.map(process_file_worker, files, repeat(str(out))):
            print(result)


if __name__ == "__main__":
    plot_midprice_vs_time()
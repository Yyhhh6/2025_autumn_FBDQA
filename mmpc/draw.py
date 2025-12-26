import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

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

# 假设你有 rolling_lr_k_r2 函数和 time_fixed_sample 函数
# from your_module import rolling_lr_k_r2, time_fixed_sample, assign_tick_time_labels
def rolling_lr_k_r2(y: np.ndarray):
    if len(y) < 2:
        return 0.0, 0.0

    x = np.arange(len(y), dtype=np.float32)
    y_std = np.std(y)

    if y_std < 1e-8:
        return 0.0, 0.0

    k, _ = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    r2 = r * r if np.isfinite(r) else 0.0
    return k, r2

def time_fixed_sample(df, cols, lags=[1, 2, 3, 5, 10, 20, 30, 50, 80]):
    """
    基于 position（iloc）的固定 lag 采样
    适用于训练 + 推理，index 任意
    """
    n = len(df)
    lag_features = {}

    for lag in lags:
        valid_len = n - lag
        if valid_len <= 0:
            continue

        for c in cols:
            col_name = f'{c}_lag{lag}'
            arr = np.full(n, np.nan, dtype=np.float32)
            arr[lag:] = df[c].iloc[:valid_len].values
            lag_features[col_name] = arr

    lag_df = pd.DataFrame(lag_features, index=df.index)
    df = pd.concat([df, lag_df], axis=1)

    return df

# 1. 读取 CSV
file_path = "data/data_raw/snapshot_sym7_date0_am.csv"
df = pd.read_csv(file_path)

# 2. 计算基本特征
# ---------- 价格+1 ----------
# 价格+1（从涨跌幅还原到前收盘价的比例）
df['bid1'] = df['n_bid1']+1
df['bid2'] = df['n_bid2']+1
df['bid3'] = df['n_bid3']+1
df['bid4'] = df['n_bid4']+1
df['bid5'] = df['n_bid5']+1
df['ask1'] = df['n_ask1']+1
df['ask2'] = df['n_ask2']+1
df['ask3'] = df['n_ask3']+1
df['ask4'] = df['n_ask4']+1
df['ask5'] = df['n_ask5']+1

# ---------- 量价组合 ----------
# 量价组合
df['spread'] = df['ask1'] - df['bid1']
df['spread2'] = df['ask2'] - df['bid2']
df['spread3'] = df['ask3'] - df['bid3']
df['mid_price'] = (df['ask1'] + df['bid1']) / 2
df['mid_price2'] = (df['ask2'] + df['bid2']) / 2
df['mid_price3'] = (df['ask3'] + df['bid3']) / 2
# 加权买卖价，如果立即成交，价格更偏向于哪一方
df['weighted_ab1'] = (df['ask1'] * df['n_bsize1'] + df['bid1'] * df['n_asize1']) / (df['n_asize1'] + df['n_bsize1'])
df['weighted_ab2'] = (df['ask2'] * df['n_bsize2'] + df['bid2'] * df['n_asize2']) / (df['n_asize2'] + df['n_bsize2'])
df['weighted_ab3'] = (df['ask3'] * df['n_bsize3'] + df['bid3'] * df['n_asize3']) / (df['n_asize3'] + df['n_bsize3'])
# 相对价差，无量纲化流动性
df['relative_spread'] = df['spread'] / df['mid_price']
df['relative_spread2'] = df['spread2'] / df['mid_price2']
df['relative_spread3'] = df['spread3'] / df['mid_price3']

# ---------- 对量取对数 ----------
# 对量取对数（量的尺度压缩）
df['bsize1'] = df['n_bsize1'].map(np.log1p)
df['bsize2'] = df['n_bsize2'].map(np.log1p)
df['bsize3'] = df['n_bsize3'].map(np.log1p)
df['bsize4'] = df['n_bsize4'].map(np.log1p)
df['bsize5'] = df['n_bsize5'].map(np.log1p)
df['asize1'] = df['n_asize1'].map(np.log1p)
df['asize2'] = df['n_asize2'].map(np.log1p)
df['asize3'] = df['n_asize3'].map(np.log1p)
df['asize4'] = df['n_asize4'].map(np.log1p)
df['asize5'] = df['n_asize5'].map(np.log1p)
# 从上个tick到当前tick发生的成交金额，单位元
df['amount'] = df['amount_delta'].map(np.log1p)

# ---------- 均线特征 ----------
# 均线特征（TODO: 为什么没有中间价均线）
df['mid_price_ma5'] = df['mid_price'].rolling(window=5, min_periods=1).mean()
df['mid_price_ma10'] = df['mid_price'].rolling(window=10, min_periods=1).mean()
df['mid_price_ma20'] = df['mid_price'].rolling(window=20, min_periods=1).mean()
df['mid_price_ma40'] = df['mid_price'].rolling(window=40, min_periods=1).mean()
df['mid_price_ma60'] = df['mid_price'].rolling(window=60, min_periods=1).mean()
df['ask1_ma5'] = df['ask1'].rolling(window=5, min_periods=1).mean()
df['ask1_ma10'] = df['ask1'].rolling(window=10, min_periods=1).mean()
df['ask1_ma20'] = df['ask1'].rolling(window=20, min_periods=1).mean()
df['ask1_ma40'] = df['ask1'].rolling(window=40, min_periods=1).mean()
df['ask1_ma60'] = df['ask1'].rolling(window=60, min_periods=1).mean()
df['bid1_ma5'] = df['bid1'].rolling(window=5, min_periods=1).mean()
df['bid1_ma10'] = df['bid1'].rolling(window=10, min_periods=1).mean()
df['bid1_ma20'] = df['bid1'].rolling(window=20, min_periods=1).mean()
df['bid1_ma40'] = df['bid1'].rolling(window=40, min_periods=1).mean()
df['bid1_ma60'] = df['bid1'].rolling(window=60, min_periods=1).mean()

# ---------- 时间标签 ----------
df['time_label'] = assign_tick_time_labels(df['time'])  # 如果有这个函数

# ---------- 过去 N tick 的最高价/最低价 ----------
# 过去20、50、100个数据中的最高价和最低价（TODO: 是否有用）
df['high_20'] = df['mid_price'].rolling(window=20, min_periods=1).max()
df['low_20'] = df['mid_price'].rolling(window=20, min_periods=1).min()
df['high_50'] = df['mid_price'].rolling(window=50, min_periods=1).max()
df['low_50'] = df['mid_price'].rolling(window=50, min_periods=1).min()
df['high_100'] = df['mid_price'].rolling(window=100, min_periods=1).max()
df['low_100'] = df['mid_price'].rolling(window=100, min_periods=1).min()

# ---------- 滚动线性回归 ----------
# 中间价线性回归（TODO: 可能是最有用的特征）
k, r2 = rolling_lr_k_r2(df['mid_price'].to_numpy()[-30:])
df['mid_lr_k'] = k   # 斜率
df['mid_lr_r2'] = r2   # 拟合优度

# ---------- 时间衰减盘口特征 ----------
# 时间衰减盘口特征
decay = np.exp(-np.arange(100)[::-1] / 20)  # 越近权重越大
decay = decay / decay.sum()
def decay_mean(x):
    w = decay[-len(x):]
    return np.sum(x * w)
df['bid1_decay'] = df['bid1'].rolling(100, min_periods=1).apply(decay_mean, raw=True)
df['ask1_decay'] = df['ask1'].rolling(100, min_periods=1).apply(decay_mean, raw=True)
df['spread_decay'] = df['spread'].rolling(100, min_periods=1).apply(decay_mean, raw=True)
df['bsize1_decay'] = df['bsize1'].rolling(100, min_periods=1).apply(decay_mean, raw=True)
df['asize1_decay'] = df['asize1'].rolling(100, min_periods=1).apply(decay_mean, raw=True)

# ---------- 盘口不平衡 ----------
# 盘口不平衡
df['obi_1'] = (df['bsize1'] - df['asize1']) / (df['bsize1'] + df['asize1'] + 1e-6)
df['obi_3'] = (
    df['bsize1'] + df['bsize2'] + df['bsize3']
    - df['asize1'] - df['asize2'] - df['asize3']
) / (
    df['bsize1'] + df['bsize2'] + df['bsize3']
    + df['asize1'] + df['asize2'] + df['asize3'] + 1e-6
)

# ---------- mid_price 动量 ----------
# mid_price 动量
df['mid_diff1'] = df['mid_price'].diff().fillna(0)   # 一阶差分：速度
df['mid_diff2'] = df['mid_diff1'].diff().fillna(0)   # 二阶差分：加速度

# ---------- 时间衰减采样历史数据 ----------
# df = time_fixed_sample(df, raw_cols, lags=[1,2,3,5,10,20,30,50,80])

# 3. 绘图功能
def plot_features(df, features_to_plot, save_path="mmpc/figures/features_plot.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12,6))
    for feat in features_to_plot:
        plt.plot(df['idx'], df[feat], label=feat)
    plt.xlabel("Index")
    plt.ylabel("Feature Value")
    plt.title("Selected Features Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Figure saved to: {save_path}")

# 4. 设置 idx
df['idx'] = range(len(df))

# 5. 指定要画的特征
# 均线：
# features_to_plot = ['mid_price', 'mid_price_ma5', 'mid_price_ma10', 'mid_price_ma20', 'mid_price_ma40', 'mid_price_ma60']
# 盘口不均衡
# features_to_plot = ['mid_price', 'obi_1', 'obi_3']
# 一阶动量与二阶动量
features_to_plot = ['mid_price', 'mid_diff1', 'mid_diff2']
# 斜率与拟合优度
# features_to_plot = ['mid_price', 'mid_lr_k', 'mid_lr_r2']

# 6. 调用绘图
plot_features(df, features_to_plot)

import os
from typing import List, Union
import pandas as pd
import numpy as np
from .model import XGBModel
from .data_process import assign_tick_time_labels, data_scale_Z_Score

class Predictor():
    def __init__(self):
        # 指定模型路径，不使用相对路径
        # pth_path = os.path.join(os.path.dirname(__file__), 'model.pth')
        pth_path = os.path.join(os.path.dirname(__file__), 'model_20.json')
        # 加载模型并移动到对应设备，假设模型是整个模型保存，如果是参数字典需要初始化结构
        self.model = self.load_model(pth_path)
        print(f"model loaded from {pth_path}")
        
    def predict(self, x: List[pd.DataFrame]) -> List[List[int]]:
        # 对输入数据进行预处理
        print(f"Received {len(x)} dataframes for prediction.")
        x_hat = self.preprocess(x)
        y = []
        y_pred = self.model.predict(x_hat)   # (N, 3)
        confidence = np.max(y_pred, axis=1)
        signal = np.argmax(y_pred, axis=1)
        signal[confidence < 0.6] = 1 # 信心不足时，预测为不变
        y.append(signal.tolist())
        y = np.array(y).T.tolist()
        # 确保返回格式为 List[List[int]]
        if isinstance(y[0], list):
            return y
        else:
            return [y]
        
    def load_model(self, model_path: str):
        return XGBModel(model_path)

    def preprocess(self, x: Union[List[pd.DataFrame], pd.DataFrame]):
        return preprocess(x)

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

def rolling_lr_features(y, window=30):
    ks = np.zeros(len(y))
    r2s = np.zeros(len(y))

    for i in range(len(y)):
        sub = y[max(0, i-window+1):i+1]
        k, r2 = rolling_lr_k_r2(sub)
        ks[i] = k
        r2s[i] = r2

    return ks, r2s

def time_fixed_sample(df, cols, lags=[1, 2, 3, 4, 5, 10, 20, 30, 40, 60]):
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

def preprocess(x: Union[List[pd.DataFrame], pd.DataFrame], N=None):
    """
    预处理步骤：
    1. 将每个 DataFrame 转换为 numpy 数组，并确保数据内存连续（使用 np.ascontiguousarray）
    2. 转换为 torch.tensor，数据类型转换为 float32
    3. 堆叠所有 tensor 形成一个 batch，并移动到指定设备上
    """
    arrays = []
    raw_cols = ["n_close",
        # "amount_delta", "n_midprice",
        # "n_bid1", "n_bsize1", "n_bid2", "n_bsize2", "n_bid3", "n_bsize3",
        # "n_bid4", "n_bsize4", "n_bid5", "n_bsize5", "n_ask1", "n_asize1",
        # "n_ask2", "n_asize2", "n_ask3", "n_asize3", "n_ask4", "n_asize4",
        # "n_ask5", "n_asize5",
        'bid1', 'bid2', 'bid3', 'bid4', 'bid5', 'ask1', 'ask2', 'ask3', 'ask4','ask5',
        'spread', 'spread2', 'spread3',
        'mid_price', 'mid_price2', 'mid_price3', 'mid_price4', 'mid_price5', 
        'weighted_ab1', 'weighted_ab2', 'weighted_ab3', 
        'relative_spread', 'relative_spread2', 'relative_spread3', 
        'spread_diff1', "spread_diff2", 'spread2_diff1', "spread2_diff2", 'spread3_diff1', "spread3_diff2", 
        'relative_spread_diff1', "relative_spread_diff2", 'relative_spread2_diff1', "relative_spread2_diff2", 'relative_spread3_diff1', "relative_spread3_diff2", 
        'bsize1', 'bsize2', 'bsize3', 'bsize4', 'bsize5', 'asize1', 'asize2', 'asize3', 'asize4', 'asize5', 'amount',  
        'mid_price_ma5', 'mid_price_ma10', 'mid_price_ma20', 'mid_price_ma40', 'mid_price_ma60', 
        "time_label", 'bid1_decay', 'ask1_decay', 'spread_decay', 'bsize1_decay', 'asize1_decay',
        'obi_1', 'obi_3', 'mid_diff1', 'mid_diff2', 
        'trade_impact', 'signed_amount', 'price_up_amount_down', 'amount_price_div', 
        'bid_depth_slope', 'ask_depth_slope', 'obi_sq', 
        'ask1_ma5', 'ask1_ma10', 'ask1_ma20', 
        'bid1_ma5', 'bid1_ma10', 'bid1_ma20', 
        'high_20', 'low_20', 'pos_20', 
        'mid_lr_k', 'mid_lr_r2', "mid_trend_strength",
    ]

    lags=[1, 2, 3, 4, 5, 10, 20, 30, 50, 80]
    new_columns = [
        'ask1_ma40', 'ask1_ma60',
        'bid1_ma40', 'bid1_ma60',
        'high_50', 'low_50', 'high_100', 'low_100', 
        'pos_50', 'pos_100',
    ]
    
    if isinstance(x, pd.DataFrame):
        x = [x]

    if N: # 训练时需要返回标签
        labels = []
        for df in x:
            labels.append(df['label_20'])
            # print("df['label_20'] shape: ", df['label_20'].shape)
        label = pd.concat(labels, axis=0).reset_index(drop=True)
        # print("label shape:", label.shape)

    for i, df in enumerate(x):
        # if N:
        #     df = df.iloc[:, :-5]
        # # 这里的scalar是提前对数据集计算得到的
        # scaler = np.load(os.path.join(os.path.dirname(__file__), 'scaler.npz'))
        # df = data_scale_Z_Score(df, mean=scaler['mean'], std=scaler['std'])

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

        # 量价组合
        df['spread'] = df['ask1'] - df['bid1']
        df['spread2'] = df['ask2'] - df['bid2']
        df['spread3'] = df['ask3'] - df['bid3']
        df['mid_price'] = (df['ask1'] + df['bid1']) / 2
        df['mid_price2'] = (df['ask2'] + df['bid2']) / 2
        df['mid_price3'] = (df['ask3'] + df['bid3']) / 2
        df['mid_price4'] = (df['ask4'] + df['bid4']) / 2
        df['mid_price5'] = (df['ask5'] + df['bid5']) / 2
        # 加权买卖价，如果立即成交，价格更偏向于哪一方
        df['weighted_ab1'] = (df['ask1'] * df['n_bsize1'] + df['bid1'] * df['n_asize1']) / (df['n_asize1'] + df['n_bsize1'])
        df['weighted_ab2'] = (df['ask2'] * df['n_bsize2'] + df['bid2'] * df['n_asize2']) / (df['n_asize2'] + df['n_bsize2'])
        df['weighted_ab3'] = (df['ask3'] * df['n_bsize3'] + df['bid3'] * df['n_asize3']) / (df['n_asize3'] + df['n_bsize3'])
        # 相对价差，无量纲化流动性
        df['relative_spread'] = df['spread'] / df['mid_price']
        df['relative_spread2'] = df['spread2'] / df['mid_price2']
        df['relative_spread3'] = df['spread3'] / df['mid_price3']
        # 价差变化
        df['spread_diff1'] = df['spread'].diff().fillna(0)
        df['spread_diff2'] = df['spread_diff1'].diff().fillna(0)
        df['spread2_diff1'] = df['spread2'].diff().fillna(0)
        df['spread2_diff2'] = df['spread2_diff1'].diff().fillna(0)
        df['spread3_diff1'] = df['spread3'].diff().fillna(0)
        df['spread3_diff2'] = df['spread3_diff1'].diff().fillna(0)
        df['relative_spread_diff1'] = df['relative_spread'].diff().fillna(0)
        df['relative_spread_diff2'] = df['relative_spread_diff1'].diff().fillna(0)
        df['relative_spread2_diff1'] = df['relative_spread2'].diff().fillna(0)
        df['relative_spread2_diff2'] = df['relative_spread2_diff1'].diff().fillna(0)
        df['relative_spread3_diff1'] = df['relative_spread3'].diff().fillna(0)
        df['relative_spread3_diff2'] = df['relative_spread3_diff1'].diff().fillna(0)

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

        # 均线特征
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
        df['mid_price_ma5'] = df['mid_price'].rolling(window=5, min_periods=1).mean()
        df['mid_price_ma10'] = df['mid_price'].rolling(window=10, min_periods=1).mean()
        df['mid_price_ma20'] = df['mid_price'].rolling(window=20, min_periods=1).mean()
        df['mid_price_ma40'] = df['mid_price'].rolling(window=40, min_periods=1).mean()
        df['mid_price_ma60'] = df['mid_price'].rolling(window=60, min_periods=1).mean()

        # 时间标签（TODO: check准确性）
        df['time_label'] = assign_tick_time_labels(df['time'])
        
        # 过去20、50、100个数据中的最高价和最低价（TODO: 是否有用）
        df['high_20'] = df['mid_price'].rolling(window=20, min_periods=1).max()
        df['low_20'] = df['mid_price'].rolling(window=20, min_periods=1).min()
        df['high_50'] = df['mid_price'].rolling(window=50, min_periods=1).max()
        df['low_50'] = df['mid_price'].rolling(window=50, min_periods=1).min()
        df['high_100'] = df['mid_price'].rolling(window=100, min_periods=1).max()
        df['low_100'] = df['mid_price'].rolling(window=100, min_periods=1).min()

        # 相对位置
        df['pos_20'] = (df['mid_price'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-6)
        df['pos_50'] = (df['mid_price'] - df['low_50']) / (df['high_50'] - df['low_50'] + 1e-6)
        df['pos_100'] = (df['mid_price'] - df['low_100']) / (df['high_100'] - df['low_100'] + 1e-6)

        # 中间价线性回归（TODO: 可能是最有用的特征）
        k, r2 = rolling_lr_features(df['mid_price'].to_numpy())
        df['mid_lr_k'] = k   # 斜率
        df['mid_lr_r2'] = r2   # 拟合优度
        df['mid_trend_strength'] = np.sign(k) * r2

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
        
        # 盘口不平衡
        df['obi_1'] = (df['bsize1'] - df['asize1']) / (df['bsize1'] + df['asize1'] + 1e-6)
        df['obi_3'] = (
            df['bsize1'] + df['bsize2'] + df['bsize3']
            - df['asize1'] - df['asize2'] - df['asize3']
        ) / (
            df['bsize1'] + df['bsize2'] + df['bsize3']
            + df['asize1'] + df['asize2'] + df['asize3'] + 1e-6
        )

        # 盘口不对称强度（引入非线性信号给树模型试试）
        df['obi_sq'] = df['obi_3'] * np.abs(df['obi_3'])

        # mid_price 动量
        df['mid_diff1'] = df['mid_price'].diff().fillna(0)   # 一阶差分：速度
        df['mid_diff2'] = df['mid_diff1'].diff().fillna(0)   # 二阶差分：加速度

        # 价格冲击方向
        df['trade_impact'] = df['mid_diff1'] * df['amount']
        df['signed_amount'] = np.sign(df['mid_diff1']) * df['amount']

        # 量价背离
        df['price_up_amount_down'] = (df['mid_diff1'] > 0).astype(int) * (df['amount'].diff() < 0).astype(int)
        df['amount_price_div'] = df['mid_diff1'] / (df['amount'] + 1e-6)

        # 盘口斜率
        df['bid_depth_slope'] = (df['bsize5'] - df['bsize1']) / 4
        df['ask_depth_slope'] = (df['asize5'] - df['asize1']) / 4

        # 时间衰减采样历史数据（TODO: 这个数据太多了，真的有用吗？）
        df = time_fixed_sample(df, raw_cols, lags=lags)
        # sampled_cols = [c for c in df.columns if '_lag' in c]
        # new_columns = new_columns + sampled_cols
        # print(f"new_columns is {new_columns}")
        # print()

        # print(f"df shape after stacking is {df.shape}") # (1994, D')
        df = df.copy()
        # print("len(raw_cols): ", len(raw_cols))
        # print("len(lags): ", len(lags))
        # print("len(new_columns): ", len(new_columns))
        for c in raw_cols:
            for lag in lags:
                col_name = f'{c}_lag{lag}'
                new_columns.append(col_name)
        # print("len(new_columns): ", len(new_columns))
        x[i] = df[new_columns]#.iloc[-1] # 只取最后一行作为特征。TODO：可以对上面的某些单点特征做 rolling 统计或者线性回归
        # print(f"x shape after selecting new_columns is {x[i].shape}") # (1994, D)
    for df in x:
        arr = np.ascontiguousarray(df.values.astype(np.float32))
        arrays.append(arr)
    
    x_hat = np.stack(arrays, axis=0) # (N, 1994, D)
    # print(f"x_hat shape after stacking is {x_hat.shape}")
    if N: # 训练时需要返回标签
        label = np.ascontiguousarray(label.values.astype(np.int8))
        return x_hat, label

    x_hat = x_hat[:, -1, :]
    return x_hat

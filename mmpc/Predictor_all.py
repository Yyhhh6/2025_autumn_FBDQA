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
        pth_path = os.path.join(os.path.dirname(__file__), 'model_20_20251226_205657.json')
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

# def rolling_lr_features(y, window=30):
#     ks = np.zeros(len(y))
#     r2s = np.zeros(len(y))

#     for i in range(len(y)):
#         sub = y[max(0, i-window+1):i+1]
#         k, r2 = rolling_lr_k_r2(sub)
#         ks[i] = k
#         r2s[i] = r2

#     return ks, r2s

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

    lags1=[1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60]
    raw_cols1 = ["n_close", "sym",
        'bid1', 'bid2', 'bid3', 'bid4', 'bid5', 'ask1', 'ask2', 'ask3', 'ask4','ask5',
        'spread', 'spread2', 'spread3',
        'mid_price', 'mid_price2', 'mid_price3', 'mid_price4', 'mid_price5', 
        'weighted_ab1', 'weighted_ab2', 'weighted_ab3', 
        'relative_spread', 'relative_spread2', 'relative_spread3', 
        'spread_diff1', "spread_diff2", 'spread2_diff1', "spread2_diff2", 'spread3_diff1', "spread3_diff2", 
        'relative_spread_diff1', "relative_spread_diff2", 'relative_spread2_diff1', "relative_spread2_diff2", 'relative_spread3_diff1', "relative_spread3_diff2", 
        'bsize1', 'bsize2', 'bsize3', 'bsize4', 'bsize5', 'asize1', 'asize2', 'asize3', 'asize4', 'asize5', 'amount',  
        'mid_price_ma5', 'mid_price_ma10', 'mid_price_ma20', 
        "time_label", 'bid1_decay', 'ask1_decay', 'spread_decay', 'bsize1_decay', 'asize1_decay',
        'obi_1', 'obi_3', 'mid_diff1', 'mid_diff2', 
        'trade_impact', 'signed_amount', 'price_up_amount_down', 'amount_price_div', 
        'bid_depth_slope', 'ask_depth_slope', 'obi_sq', 
        'ask1_ma5', 'ask1_ma10', 'ask1_ma20', 
        'bid1_ma5', 'bid1_ma10', 'bid1_ma20', 
        'high_20', 'low_20', 'pos_20', 
        # 'mid_lr_k', 'mid_lr_r2', "mid_trend_strength",
        # 'mid_diff1_lr_k', 'mid_diff1_lr_r2', 'mid_diff1_trend_strength',
        # 'mid_diff2_lr_k', 'mid_diff2_lr_r2', 'mid_diff2_trend_strength',
        # 'trend_regime', 'trend_strength_gated', 
        # 'price_move_capacity', 'trend_liquidity_ratio'
    ]

    lags2=[1, 2, 3, 4, 5, 10, 15, 20]
    raw_cols2 = [
        # 'trend_persistence', 'trend_flip', 'trend_age',
        'mid_price_ma40', 'mid_price_ma60', 
        'ask1_ma40', 'ask1_ma60',
        'bid1_ma40', 'bid1_ma60',
        'high_50', 'low_50', 'pos_50',
        # 'trend_align_10_30', 'trend_align_30_60', 'trend_confidence',
        # 'mid_lr_k_10', 'mid_lr_r2_10', 'mid_trend_strength_10',
        # 'mid_lr_k_60', 'mid_lr_r2_60', 'mid_trend_strength_60',
    ]

    new_columns = [
        'high_100', 'low_100', 'pos_100',
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

        extra_feats = {}

        # ======================
        # 价格还原（+1）
        # ======================
        extra_feats.update({
            'bid1': df['n_bid1'] + 1,
            'bid2': df['n_bid2'] + 1,
            'bid3': df['n_bid3'] + 1,
            'bid4': df['n_bid4'] + 1,
            'bid5': df['n_bid5'] + 1,
            'ask1': df['n_ask1'] + 1,
            'ask2': df['n_ask2'] + 1,
            'ask3': df['n_ask3'] + 1,
            'ask4': df['n_ask4'] + 1,
            'ask5': df['n_ask5'] + 1,
        })

        # ======================
        # 量价组合
        # ======================
        extra_feats.update({
            'spread': extra_feats['ask1'] - extra_feats['bid1'],
            'spread2': extra_feats['ask2'] - extra_feats['bid2'],
            'spread3': extra_feats['ask3'] - extra_feats['bid3'],

            'mid_price': (extra_feats['ask1'] + extra_feats['bid1']) / 2,
            'mid_price2': (extra_feats['ask2'] + extra_feats['bid2']) / 2,
            'mid_price3': (extra_feats['ask3'] + extra_feats['bid3']) / 2,
            'mid_price4': (extra_feats['ask4'] + extra_feats['bid4']) / 2,
            'mid_price5': (extra_feats['ask5'] + extra_feats['bid5']) / 2,
        })

        extra_feats.update({
            'weighted_ab1': (
                extra_feats['ask1'] * df['n_bsize1']
                + extra_feats['bid1'] * df['n_asize1']
            ) / (df['n_asize1'] + df['n_bsize1']),
            'weighted_ab2': (
                extra_feats['ask2'] * df['n_bsize2']
                + extra_feats['bid2'] * df['n_asize2']
            ) / (df['n_asize2'] + df['n_bsize2']),
            'weighted_ab3': (
                extra_feats['ask3'] * df['n_bsize3']
                + extra_feats['bid3'] * df['n_asize3']
            ) / (df['n_asize3'] + df['n_bsize3']),
        })

        # ======================
        # 相对价差
        # ======================
        extra_feats.update({
            'relative_spread': extra_feats['spread'] / extra_feats['mid_price'],
            'relative_spread2': extra_feats['spread2'] / extra_feats['mid_price2'],
            'relative_spread3': extra_feats['spread3'] / extra_feats['mid_price3'],
        })

        # ======================
        # 价差变化
        # ======================
        extra_feats.update({
            'spread_diff1': extra_feats['spread'].diff().fillna(0),
            'spread_diff2': extra_feats['spread'].diff().diff().fillna(0),

            'spread2_diff1': extra_feats['spread2'].diff().fillna(0),
            'spread2_diff2': extra_feats['spread2'].diff().diff().fillna(0),

            'spread3_diff1': extra_feats['spread3'].diff().fillna(0),
            'spread3_diff2': extra_feats['spread3'].diff().diff().fillna(0),

            'relative_spread_diff1': extra_feats['relative_spread'].diff().fillna(0),
            'relative_spread_diff2': extra_feats['relative_spread'].diff().diff().fillna(0),

            'relative_spread2_diff1': extra_feats['relative_spread2'].diff().fillna(0),
            'relative_spread2_diff2': extra_feats['relative_spread2'].diff().diff().fillna(0),

            'relative_spread3_diff1': extra_feats['relative_spread3'].diff().fillna(0),
            'relative_spread3_diff2': extra_feats['relative_spread3'].diff().diff().fillna(0),
        })

        # ======================
        # 对数盘口量 & 成交量
        # ======================
        extra_feats.update({
            'bsize1': np.log1p(df['n_bsize1']),
            'bsize2': np.log1p(df['n_bsize2']),
            'bsize3': np.log1p(df['n_bsize3']),
            'bsize4': np.log1p(df['n_bsize4']),
            'bsize5': np.log1p(df['n_bsize5']),
            'asize1': np.log1p(df['n_asize1']),
            'asize2': np.log1p(df['n_asize2']),
            'asize3': np.log1p(df['n_asize3']),
            'asize4': np.log1p(df['n_asize4']),
            'asize5': np.log1p(df['n_asize5']),
            'amount': np.log1p(df['amount_delta']),
        })

        # ======================
        # 均线特征
        # ======================
        for w in [5, 10, 20, 40, 60]:
            extra_feats[f'ask1_ma{w}'] = extra_feats['ask1'].rolling(w, min_periods=1).mean()
            extra_feats[f'bid1_ma{w}'] = extra_feats['bid1'].rolling(w, min_periods=1).mean()
            extra_feats[f'mid_price_ma{w}'] = extra_feats['mid_price'].rolling(w, min_periods=1).mean()

        # ======================
        # 一次性合并
        # ======================
        df = pd.concat([df, pd.DataFrame(extra_feats, index=df.index)], axis=1)

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

        # # 中间价线性回归（TODO: 可能是最有用的特征）
        # k, r2 = rolling_lr_features(df['mid_price'].to_numpy())
        # df['mid_lr_k'] = k   # 斜率
        # df['mid_lr_r2'] = r2   # 拟合优度
        # df['mid_trend_strength'] = np.sign(k) * r2

        # # 信号的持续性
        # df['trend_persistence'] = (
        #     df['mid_trend_strength']
        #     .rolling(20)
        #     .apply(lambda x: np.sum(np.sign(x) == np.sign(x.iloc[-1])))
        # )
        # df['trend_flip'] = (np.sign(df['mid_lr_k']).diff() != 0).astype(int)
        # df['trend_age'] = df['trend_flip'].rolling(50).sum()

        # # 门控
        # df['trend_regime'] = (
        #     (df['mid_lr_r2'] > 0.25) &
        #     (np.abs(df['mid_lr_k']) > np.percentile(np.abs(df['mid_lr_k']), 60))
        # ).astype(int)
        # df['trend_strength_gated'] = df['mid_trend_strength'] * df['trend_regime']

        # k_10, r2_10 = rolling_lr_features(df['mid_price'].to_numpy(), window=10)
        # k_30, r2_30 = k, r2
        # k_60, r2_60 = rolling_lr_features(df['mid_price'].to_numpy(), window=60)
        # df['trend_align_10_30'] = (np.sign(k_10) == np.sign(k_30)).astype(int)
        # df['trend_align_30_60'] = (np.sign(k_30) == np.sign(k_60)).astype(int)
        # df['trend_confidence'] = (
        #     np.sign(k_10) * r2_10 +
        #     np.sign(k_30) * r2_30 +
        #     np.sign(k_60) * r2_60
        # )

        # df['mid_lr_k_10'] = k_10   # 斜率
        # df['mid_lr_r2_10'] = r2_10   # 拟合优度
        # df['mid_trend_strength_10'] = np.sign(k_10) * r2_10
        
        # df['mid_lr_k_60'] = k_60   # 斜率
        # df['mid_lr_r2_60'] = r2_60   # 拟合优度
        # df['mid_trend_strength_60'] = np.sign(k) * r2

        # # 盘口是否允许价格往某个方向动？
        # df['price_move_capacity'] = df['mid_lr_k'] / (df['relative_spread'] + 1e-6)
        # df['trend_liquidity_ratio'] = df['mid_trend_strength'] / (df['relative_spread'] + 1e-6)

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
        
        # # mid_diff1线性回归
        # k, r2 = rolling_lr_features(df['mid_diff1'].to_numpy())
        # df['mid_diff1_lr_k'] = k   # 斜率
        # df['mid_diff1_lr_r2'] = r2   # 拟合优度
        # df['mid_diff1_trend_strength'] = np.sign(k) * r2

        # # mid_diff2线性回归
        # k, r2 = rolling_lr_features(df['mid_diff2'].to_numpy())
        # df['mid_diff2_lr_k'] = k   # 斜率
        # df['mid_diff2_lr_r2'] = r2   # 拟合优度
        # df['mid_diff2_trend_strength'] = np.sign(k) * r2

        extra_feats = {}
        # 价格冲击方向
        extra_feats['trade_impact'] = df['mid_diff1'] * df['amount']
        extra_feats['signed_amount'] = np.sign(df['mid_diff1']) * df['amount']
        # 量价背离
        extra_feats['price_up_amount_down'] = (
            (df['mid_diff1'] > 0).astype(int) * (df['amount'].diff() < 0).astype(int)
        )
        extra_feats['amount_price_div'] = df['mid_diff1'] / (df['amount'] + 1e-6)
        # 盘口斜率
        extra_feats['bid_depth_slope'] = (df['bsize5'] - df['bsize1']) / 4
        extra_feats['ask_depth_slope'] = (df['asize5'] - df['asize1']) / 4
        df = pd.concat([df, pd.DataFrame(extra_feats, index=df.index)], axis=1)

        # 时间衰减采样历史数据（TODO: 这个数据太多了，真的有用吗？）
        df = time_fixed_sample(df, raw_cols1, lags=lags1)
        df = df.copy()

        df = time_fixed_sample(df, raw_cols2, lags=lags2)
        df = df.copy()

        lag_cols1 = [f'{c}_lag{lag}' for c in raw_cols1 for lag in lags1]
        lag_cols2 = [f'{c}_lag{lag}' for c in raw_cols2 for lag in lags2]
        final_columns = (
            new_columns
            + lag_cols1
            + lag_cols2
        )
        x[i] = df[final_columns]

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

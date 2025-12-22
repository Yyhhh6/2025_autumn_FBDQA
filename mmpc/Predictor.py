import os
from typing import List, Union
import pandas as pd
import numpy as np
from .model import XGBModel
from .data_process import assign_tick_time_labels, factors_null_process_np, extreme_process_MAD_np, data_scale_Z_Score_np

class Predictor():
    def __init__(self):
        # 指定模型路径，不使用相对路径
        # pth_path = os.path.join(os.path.dirname(__file__), 'model.pth')
        pth_path = os.path.join(os.path.dirname(__file__), 'model.json')
        # 加载模型并移动到对应设备，假设模型是整个模型保存，如果是参数字典需要初始化结构
        self.model = self.load_model(pth_path)
        print(f"model loaded from {pth_path}")
        
    def predict(self, x: List[pd.DataFrame]) -> List[List[int]]:
        # 对输入数据进行预处理
        print(f"Received {len(x)} dataframes for prediction.")
        x_hat = self.preprocess(x)
        y = []
        # for _ in range(1): # TODO
        y_pred = self.model.predict(x_hat)   # (N, 3)
        confidence = np.max(y_pred, axis=1)
        signal = np.argmax(y_pred, axis=1)
        signal[confidence < 0.55] = 1 # 信心不足时，预测为不变
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


def preprocess(x: Union[List[pd.DataFrame], pd.DataFrame], N=None):
    """
    预处理步骤：
    1. 将每个 DataFrame 转换为 numpy 数组，并确保数据内存连续（使用 np.ascontiguousarray）
    2. 转换为 torch.tensor，数据类型转换为 float32
    3. 堆叠所有 tensor 形成一个 batch，并移动到指定设备上
    """
    arrays = []
    new_columns = [
        'bid1', 'bid2', 'bid3', 'bid4', 'bid5', 'ask1', 'ask2', 'ask3', 'ask4','ask5', 
        'spread', 'spread2', 'spread3', 'mid_price', 'mid_price2', 'mid_price3',
        'weighted_ab1', 'weighted_ab2', 'weighted_ab3', 'relative_spread',
        'relative_spread2', 'relative_spread3', 'bsize1', 'bsize2', 'bsize3',
        'bsize4', 'bsize5', 'asize1', 'asize2', 'asize3', 'asize4', 'asize5',
        'amount', 'ask1_ma5', 'ask1_ma10', 'ask1_ma20', 'ask1_ma40', 'ask1_ma60',
        'bid1_ma5', 'bid1_ma10', 'bid1_ma20', 'bid1_ma40', 'bid1_ma60', "time_label",
        'bid1_decay', 'ask1_decay', 'spread_decay', 'bsize1_decay', 'asize1_decay',
        'obi_1', 'obi_3', 'mid_diff1', 'mid_diff2', 'mid_lr_k', 'mid_lr_r2',
    ]
    
    if isinstance(x, pd.DataFrame):
        x = [x]

    if N: # 训练时需要返回标签
        labels = []
        for df in x:
            labels.append(df['label_5'])
        label = pd.concat(labels, axis=0).reset_index(drop=True)

    for i, df in enumerate(x):
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
        df['weighted_ab1'] = (df['ask1'] * df['n_bsize1'] + df['bid1'] * df['n_asize1']) / (df['n_asize1'] + df['n_bsize1'])
        df['weighted_ab2'] = (df['ask2'] * df['n_bsize2'] + df['bid2'] * df['n_asize2']) / (df['n_asize2'] + df['n_bsize2'])
        df['weighted_ab3'] = (df['ask3'] * df['n_bsize3'] + df['bid3'] * df['n_asize3']) / (df['n_asize3'] + df['n_bsize3'])
        df['relative_spread'] = df['spread'] / df['mid_price']
        df['relative_spread2'] = df['spread2'] / df['mid_price2']
        df['relative_spread3'] = df['spread3'] / df['mid_price3']

        # 对量取对数
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

        # 时间标签
        df['time_label'] = assign_tick_time_labels(df['time'])
        
        # 中间价线性回归
        k, r2 = rolling_lr_k_r2(df['mid_price'].to_numpy()[-30:])
        df['mid_lr_k'] = k
        df['mid_lr_r2'] = r2

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

        # mid_price 动量
        df['mid_diff1'] = df['mid_price'].diff().fillna(0)
        df['mid_diff2'] = df['mid_diff1'].diff().fillna(0)

        # print(f"df shape after stacking is {df.shape}") # (1994, D')
        x[i] = df[new_columns]#.iloc[-1] # 只取最后一行作为特征。TODO：可以对上面的某些单点特征做 rolling 统计或者线性回归
        # print(f"x shape after selecting new_columns is {x[i].shape}") # (1994, D)
    for df in x:
        # 使用 np.ascontiguousarray 确保数组内存连续，利于转换和性能
        arr = np.ascontiguousarray(df.values.astype(np.float32))
        arrays.append(arr)
    
    x_hat = np.stack(arrays, axis=0) # (N, 1994, D)
    # print(f"x_hat shape after stacking is {x_hat.shape}")
    if N: # 训练时需要返回标签
        label = np.ascontiguousarray(label.values.astype(np.int8))
        return x_hat, label

    x_hat = x_hat[:, -1, :]
    # scaler = np.load(os.path.join(os.path.dirname(__file__), 'scaler.npz'))
    # x_hat[~np.isfinite(x_hat)] = np.nan
    # x_hat = factors_null_process_np(x_hat, medians=scaler['median'])
    # x_hat = extreme_process_MAD_np(x_hat, lower=scaler['mad_lower'], upper=scaler['mad_upper'], num=3)
    # x_hat = data_scale_Z_Score_np(x_hat, mean=scaler['mean'], std=scaler['std'])
    return x_hat

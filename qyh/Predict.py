import os
from typing import List
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .mlp import LOBMLP

class Predictor():
    def __init__(self):
        pth_path = os.path.join(os.path.dirname(__file__), 'model.pth')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(pth_path)
        self.model.to(self.device)
        self.model.eval() 
        
    def predict(self, x: List[pd.DataFrame]) -> List[List[int]]:
        x_hat = self.preprocess(x)
        with torch.no_grad(): 
            y = []
            for i in range(5):
                y_pred = self.model(x_hat).cpu().numpy()
                y.append(np.argmax(y_pred,axis=1).tolist())
            y = np.array(y).T.tolist()
        if isinstance(y[0], list):
            return y
        else:
            return [y]
        
    def load_model(self, model_path: str):
        hidden_sizes=[200]
        model = LOBMLP(input_size=43, hidden_sizes=hidden_sizes, output_size=3)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model

    def preprocess(self, x: List[pd.DataFrame]):
        """
        预处理步骤：
        1. 将每个 DataFrame 转换为 numpy 数组，并确保数据内存连续（使用 np.ascontiguousarray）
        2. 转换为 torch.tensor，数据类型转换为 float32
        3. 堆叠所有 tensor 形成一个 batch，并移动到指定设备上
        """
        tensors = []
        new_columns = [
            'bid1', 'bid2', 'bid3', 'bid4', 'bid5', 'ask1', 'ask2', 'ask3', 'ask4','ask5', 
            'spread', 'spread2', 'spread3', 'mid_price', 'mid_price2', 'mid_price3',
            'weighted_ab1', 'weighted_ab2', 'weighted_ab3', 'relative_spread',
            'relative_spread2', 'relative_spread3', 'bsize1', 'bsize2', 'bsize3',
            'bsize4', 'bsize5', 'asize1', 'asize2', 'asize3', 'asize4', 'asize5',
            'amount', 'ask1_ma5', 'ask1_ma10', 'ask1_ma20', 'ask1_ma40', 'ask1_ma60',
            'bid1_ma5', 'bid1_ma10', 'bid1_ma20', 'bid1_ma40', 'bid1_ma60'
        ]
        
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
            df['bsize1'] = df['n_bsize1'].map(np.log)
            df['bsize2'] = df['n_bsize2'].map(np.log)
            df['bsize3'] = df['n_bsize3'].map(np.log)
            df['bsize4'] = df['n_bsize4'].map(np.log)
            df['bsize5'] = df['n_bsize5'].map(np.log)
            df['asize1'] = df['n_asize1'].map(np.log)
            df['asize2'] = df['n_asize2'].map(np.log)
            df['asize3'] = df['n_asize3'].map(np.log)
            df['asize4'] = df['n_asize4'].map(np.log)
            df['asize5'] = df['n_asize5'].map(np.log)
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

            x[i] = df[new_columns]
        
        for df in x:
            # 使用 np.ascontiguousarray 确保数组内存连续，利于转换和性能
            arr = np.ascontiguousarray(df.iloc[-1,:].values.astype(np.float32))
            tensor = torch.from_numpy(arr)
            tensors.append(tensor)
        
        x_hat = torch.stack(tensors, dim=0)
        x_hat = x_hat.to(self.device)
        return x_hat

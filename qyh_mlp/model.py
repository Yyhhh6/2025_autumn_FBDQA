import os
from typing import List
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LOBMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes:list=[100], output_size=3):
        super().__init__()
        self.fc_list = nn.ModuleList()
        self.fc_list.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(len(hidden_sizes)-1):
            self.fc_list.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        self.fc_list.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for fc in self.fc_list[0:-1]:
            x = F.leaky_relu(fc(x))
        x = self.fc_list[-1](x)
        return x

class Predictor():
    def __init__(self):
        # # 指定模型路径，不使用相对路径
        # pth_path = os.path.join(os.path.dirname(__file__), 'model.pth')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加载模型并移动到对应设备，假设模型是整个模型保存，如果是参数字典需要初始化结构
        # self.model = self.load_model(pth_path)
        # self.model.to(self.device)
        # self.model.eval()  # 切换到评估模式以关闭 dropout、batchnorm 等层
        # pass
        
    # def predict(self, x: List[pd.DataFrame]) -> List[List[int]]:
    #     # 输入x: List[pd.DataFrame]，List大小为config.json中的batch; 
    #         # pd.DataFrame为过去100个tick的数据; 
    #         # pd.DataFrame的列名为配置文件中的feature名
    #     # 输出List[List[int]]格式的预测结果：
    #         # 外层List大小为batch，即配置文件中的batch
    #         # 内层List大小为label个数，即配置文件中的label的个数（好像可以用不同模型来预测不同的label？？）
    #         # int为预测标签(0/1/2)
    #     # 对输入数据进行预处理
    #     x_hat = self.preprocess(x)
    #     with torch.no_grad():  # 禁用梯度计算，加速推理
    #         y = []
    #         for i in range(5):
    #             y_pred = self.model(x_hat).cpu().numpy()
    #             y.append(np.argmax(y_pred,axis=1).tolist())
    #         y = np.array(y).T.tolist()
    #     # 确保返回格式为 List[List[int]]
    #     if isinstance(y[0], list):
    #         return y
    #     else:
    #         return [y]
        
    # def load_model(self, model_path: str):
    #     hidden_sizes=[200]
    #     model = LOBMLP(input_size=43, hidden_sizes=hidden_sizes, output_size=3)
    #     model.load_state_dict(torch.load(model_path, map_location=self.device))
    #     return model

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

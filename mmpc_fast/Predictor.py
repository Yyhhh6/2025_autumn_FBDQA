import os
from typing import List, Union
import pandas as pd
import numpy as np
from .model import XGBModel
from .data_process import assign_tick_time_labels, assign_tick_time_label, data_scale_Z_Score
from tqdm import tqdm
from contextlib import redirect_stdout, redirect_stderr

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
        target_confidence = 0.75
        y_pred = self.model.predict(x_hat)   # (N, 3)
        confidence = np.max(y_pred, axis=1)
        signal = np.argmax(y_pred, axis=1)
        signal[confidence < target_confidence] = 1 # 信心不足时，预测为不变
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
        if n - lag <= 0:
            continue

        for c in cols:
            col_name = f'{c}_lag{lag}'
            arr = np.full(n, np.nan, dtype=np.float32)
            arr[lag - 1:] = df[c].iloc[:n - lag + 1].values
            lag_features[col_name] = arr

    lag_df = pd.DataFrame(lag_features, index=df.index)
    df = pd.concat([df, lag_df], axis=1)

    return df

def split_df_sliding(df: pd.DataFrame, window=100):
    """
    滑动窗口切分：
    df[0:100], df[1:101], df[2:102], ...
    """
    xs = []
    for start in range(0, len(df) - window + 1):
        xs.append(df.iloc[start:start + window].reset_index(drop=True))
    return xs


def compare_dfs(df1, df2, tol=1e-5):
    # --- 统一转成 DataFrame ---
    if isinstance(df1, pd.Series):
        df1 = df1.to_frame()
    if isinstance(df2, pd.Series):
        df2 = df2.to_frame()

    print("Shape df1:", df1.shape, "df2:", df2.shape)
    cols1, cols2 = set(df1.columns), set(df2.columns)
    print("Columns only in df1:", cols1 - cols2)
    print("Columns only in df2:", cols2 - cols1)
    
    idx1, idx2 = set(df1.index), set(df2.index)
    print("Rows only in df1:", idx1 - idx2)
    print("Rows only in df2:", idx2 - idx1)
    
    common_cols = df1.columns.intersection(df2.columns)
    common_idx = df1.index.intersection(df2.index)
    
    # 对数值型列使用绝对误差判断
    df1_common = df1.loc[common_idx, common_cols]
    df2_common = df2.loc[common_idx, common_cols]
    
    diff_mask = (df1_common - df2_common).abs() > tol
    num_diff = diff_mask.sum().sum()
    print("Number of different values (tol={}):".format(tol), num_diff)
    
    if num_diff > 0:
        diff_positions = diff_mask.stack()
        diff_positions = diff_positions[diff_positions]  # 只保留 True
        print("Positions with differences (row, column):")
        print(diff_positions)

        # 创建差异 DataFrame
        diff_df = pd.DataFrame(columns=['row', 'column', 'df1_value', 'df2_value'])
        for (row, col) in diff_positions.index:
            diff_df = diff_df._append({
                'row': row,
                'column': col,
                'df1_value': df1_common.loc[row, col],
                'df2_value': df2_common.loc[row, col]
            }, ignore_index=True)
        
        print("\nSample of differences:")
        print(diff_df.head(20))  # 输出前 20 个差异
        return diff_df
    return None


def preprocess_slice(x: list[pd.DataFrame], lags=[1, 2, 3, 4, 5, 10, 20, 30, 40, 50]):
    # 只处理 100 个tick
    x_extract = []
    # print("len(x): ", len(x))
    lags=[1, ]

    for df in tqdm(
        x,
        total=len(x),
        desc="Preprocess",
        ncols=100
    ):
        df = df.reset_index(drop=True)
        assert len(df) == 100

        # 基础价格还原
        for i in range(1, 6):
            df[f'bid{i}'] = df[f'n_bid{i}'] + 1
            df[f'ask{i}'] = df[f'n_ask{i}'] + 1

        # 基础量价组合
        for i in range(1, 4):
            df[f'spread{i}'] = df[f'ask{i}'] - df[f'bid{i}']
            df[f'mid_price{i}'] = (df[f'ask{i}'] + df[f'bid{i}']) / 2

        # 中间价一阶差分
        df['mid_diff1'] = df['mid_price1'].diff().fillna(0)   # 一阶差分：速度

        # 对数量 & 成交量
        for i in range(1, 6):
            df[f'bsize{i}'] = np.log1p(df[f'n_bsize{i}'])
            df[f'asize{i}'] = np.log1p(df[f'n_asize{i}'])
        df['amount'] = np.log1p(df['amount_delta'])

        # 先计算加权中间价特征
        weighted_feats = {}
        for i in range(1, 4):
            weighted_feats[f'weighted_ab{i}'] = (
                df[f'ask{i}'] * df[f'n_bsize{i}'] + df[f'bid{i}'] * df[f'n_asize{i}']
            ) / (df[f'n_asize{i}'] + df[f'n_bsize{i}'])

        # 再计算价差变化特征
        spread_diff_feats = {}
        for i in range(1, 4):
            spread_diff_feats[f'spread{i}_diff1'] = df[f'spread{i}'].diff().fillna(0)
            spread_diff_feats[f'spread{i}_diff2'] = df[f'spread{i}'].diff().diff().fillna(0)

            spread_diff_feats[f'relative_spread{i}_diff1'] = (df[f'spread{i}']/df[f'mid_price{i}']).diff().fillna(0)
            spread_diff_feats[f'relative_spread{i}_diff2'] = (df[f'spread{i}']/df[f'mid_price{i}']).diff().diff().fillna(0)

        # 用字典存每个lag的特征
        lag_feats = {}

        for lag in lags:
            # 用iloc直接取lag对应行
            row = df.iloc[-lag]
            feat_dict = {}

            # 基础价格还原
            for i in range(1, 6):
                feat_dict[f'bid{i}_lag{lag}'] = row[f'n_bid{i}'] + 1
                feat_dict[f'ask{i}_lag{lag}'] = row[f'n_ask{i}'] + 1

            # 价格 & 价差 & 相对价差
            for i in range(1, 4):
                feat_dict[f'spread{i}_lag{lag}'] = row[f'spread{i}']
                feat_dict[f'mid_price{i}_lag{lag}'] = row[f'mid_price{i}']
                feat_dict[f'relative_spread{i}_lag{lag}'] = row[f'spread{i}'] / row[f'mid_price{i}']

            # log量 & 成交量
            for i in range(1, 6):
                feat_dict[f'bsize{i}_lag{lag}'] = row[f'bsize{i}']
                feat_dict[f'asize{i}_lag{lag}'] = row[f'asize{i}']
            feat_dict[f'amount_lag{lag}'] = row['amount']

            # 均线特征（只对ask1, bid1, mid_price计算）
            for w in [5, 10, 20, 40]:
                feat_dict[f'ask1_ma{w}_lag{lag}'] = df['ask1'].iloc[-lag-w+1:None if lag == 1 else -lag+1].mean()
                feat_dict[f'bid1_ma{w}_lag{lag}'] = df['bid1'].iloc[-lag-w+1:None if lag == 1 else -lag+1].mean()
                feat_dict[f'mid_price1_ma{w}_lag{lag}'] = df['mid_price1'].iloc[-lag-w+1:None if lag == 1 else -lag+1].mean()

            # 把加权和价差变化加入 lag 特征
            for k, v in weighted_feats.items():
                feat_dict[f'{k}_lag{lag}'] = v.iloc[-lag]  # 对应当前 lag 的值
            for k, v in spread_diff_feats.items():
                feat_dict[f'{k}_lag{lag}'] = v.iloc[-lag]

            # 时间特征
            feat_dict[f'time_label_lag{lag}'] = assign_tick_time_label(row['time'])

            # ----- 中间价线性回归 -----
            window_size = 30  # 可以根据需要调整
            window_prices = df['mid_price1'].iloc[-lag+1-window_size:None if lag == 1 else -lag+1]

            # 滑动窗口内线性回归
            k_30, r2_30 = rolling_lr_k_r2(window_prices)

            feat_dict[f'mid_lr_k_lag{lag}'] = k_30
            feat_dict[f'mid_lr_r2_lag{lag}'] = r2_30
            feat_dict[f'mid_trend_strength_lag{lag}'] = np.sign(k_30) * r2_30

            feat_dict[f'price_move_capacity_lag{lag}'] = k_30 / (feat_dict[f'relative_spread1_lag{lag}'] + 1e-6)
            feat_dict[f'trend_liquidity_ratio_lag{lag}'] = np.sign(k_30) * r2_30 / (feat_dict[f'relative_spread1_lag{lag}'] + 1e-6)

            # ===== 盘口不平衡（仅采样点） =====
            b1, a1 = row['bsize1'], row['asize1']
            b3 = row['bsize1'] + row['bsize2'] + row['bsize3']
            a3 = row['asize1'] + row['asize2'] + row['asize3']

            obi_1 = (b1 - a1) / (b1 + a1 + 1e-6)
            obi_3 = (b3 - a3) / (b3 + a3 + 1e-6)

            feat_dict[f'obi_1_lag{lag}'] = obi_1
            feat_dict[f'obi_3_lag{lag}'] = obi_3
            feat_dict[f'obi_sq_lag{lag}'] = obi_3 * abs(obi_3)

            # ===== 深度斜率（盘口形状） =====
            feat_dict[f'bid_depth_slope_lag{lag}'] = (row['bsize5'] - row['bsize1']) / 4
            feat_dict[f'ask_depth_slope_lag{lag}'] = (row['asize5'] - row['asize1']) / 4

            # 一阶 / 二阶差分（注意边界）
            feat_dict[f'mid_diff1_lag{lag}'] = df['mid_price1'].iloc[-lag] - df['mid_price1'].iloc[-lag-1]

            # ===== mid_diff1 的局部线性趋势（小窗口） =====
            window_size = 30
            mid_diff1_window = df['mid_diff1'].iloc[-lag+1-window_size:None if lag == 1 else -lag+1]

            k_d1, r2_d1 = rolling_lr_k_r2(mid_diff1_window)

            feat_dict[f'mid_diff1_lr_k_lag{lag}'] = k_d1
            feat_dict[f'mid_diff1_lr_r2_lag{lag}'] = r2_d1
            feat_dict[f'mid_diff1_trend_strength_lag{lag}'] = np.sign(k_d1) * r2_d1

            # ===== 量价关系（仅采样点） =====
            # 价格冲击
            feat_dict[f'trade_impact_lag{lag}'] = row['amount'] * row['mid_diff1']
            feat_dict[f'signed_amount_lag{lag}'] = np.sign(row['mid_diff1']) * row['amount']
            # 量价背离
            feat_dict[f'price_up_amount_down_lag{lag}'] = (
                int(row['mid_diff1'] > 0) * int(row['amount'] < df['amount'].iloc[-lag-1])
            )
            feat_dict[f'amount_price_div_lag{lag}'] = row['mid_diff1'] / (row['amount'] + 1e-6)

            # ===== 区间位置（range position）=====
            for w in [20, 50, 100]:
                prices = df['mid_price1'].iloc[-lag+1-w:None if lag == 1 else -lag+1]

                high = prices.max()
                low = prices.min()
                pos = (row['mid_price1'] - low) / (high - low + 1e-6)

                feat_dict[f'high_{w}_lag{lag}'] = high
                feat_dict[f'low_{w}_lag{lag}'] = low
                feat_dict[f'pos_{w}_lag{lag}'] = pos

            # ===== 多尺度中间价线性趋势 =====
            lr_windows = [10, 60]  # window_size=30前面已经算过
            k_dict = {}
            r2_dict = {}
            k_dict[30] = k_30
            r2_dict[30] = r2_30

            for w in lr_windows:
                # ----- 中间价线性回归 -----
                window_prices = df['mid_price1'].iloc[-lag+1-w:None if lag == 1 else -lag+1]

                # 滑动窗口内线性回归
                k, r2 = rolling_lr_k_r2(window_prices)

                k_dict[w] = k
                r2_dict[w] = r2

                feat_dict[f'mid_lr_k_{w}_lag{lag}'] = k
                feat_dict[f'mid_lr_r2_{w}_lag{lag}'] = r2
                feat_dict[f'mid_trend_strength_{w}_lag{lag}'] = np.sign(k) * r2

            feat_dict[f'trend_align_10_30_lag{lag}'] = int(
                np.sign(k_dict[10]) == np.sign(k_dict[30])
            )
            feat_dict[f'trend_align_30_60_lag{lag}'] = int(
                np.sign(k_dict[30]) == np.sign(k_dict[60])
            )
            feat_dict[f'trend_align_10_60_lag{lag}'] = (
                np.sign(k_dict[10]) != np.sign(k_dict[60])
            ) * abs(k_dict[10] - k_dict[60])

            feat_dict[f'trend_confidence_lag{lag}'] = (
                np.sign(k_dict[10]) * r2_dict[10] +
                np.sign(k_dict[30]) * r2_dict[30] +
                np.sign(k_dict[60]) * r2_dict[60]
            )

            lag_feats.update(feat_dict)

        x_extract.append(lag_feats)

    return pd.DataFrame(x_extract)


def preprocess_platform(x: Union[List[pd.DataFrame], pd.DataFrame], is_local=True):
    """
    更改 preprocess 逻辑，适用于公榜评测的更快推理
    """
    if True:    
        # 本地评测，pd.DataFrame 先切分为 100 个 tick 个片段
        if isinstance(x, pd.DataFrame):
            x = [x]
        assert len(x) == 1, "训练中列表 x 的长度必须为 1 "

        labels = []
        x_slice = []
        for df in x:
            labels.append(df['label_20'][99:])    
            df_slice = split_df_sliding(df, window=100)
            x_slice.extend(df_slice)
            assert len(df_slice) == len(df['label_20'][99:]), f"len(df_slice): {len(df_slice)}    len(df['label_20'][99:]): {len(df['label_20'][99:])}"
        
        label = pd.concat(labels, axis=0).reset_index(drop=True)
        assert len(x_slice) == len(label)

        x_extract1 = preprocess_slice(x_slice)
        # print("x_extract1.shape: ", x_extract1.shape)
        x_extract2 = preprocess_local(x_slice, is_slice=True)
        # print("x_extract2.shape: ", x_extract2.shape)

        # # 对列排序
        # x_extract1 = x_extract1.reindex(sorted(x_extract1.columns), axis=1)
        # x_extract2 = x_extract2.reindex(sorted(x_extract2.columns), axis=1)
        # x_extract1.to_csv("x_extract1.csv", index=True)
        # x_extract2.to_csv("x_extract2.csv", index=True)

        compare_dfs(x_extract1, x_extract2)
        exit(0)

        if is_local==False:
            # 方法一
            x_extract = preprocess_slice(x_slice)
        else:
            # 方法二
            x_extract, _label = preprocess_local(x_slice, is_train=True, is_slice=True)
            compare_dfs(label, _label)  # 二者相同！说明没问题

        assert len(x_extract) == len(label), f"len(x_extract): {len(x_extract)}    len(label): {len(label)}"
        return x_extract, label

    else:    
        # 公榜评测，pd.DataFrame 中都是 100 个 tick
        return preprocess_slice(x_slice)

def preprocess_local(x: Union[List[pd.DataFrame], pd.DataFrame], is_train=False, is_slice=False):
    """
    预处理步骤：
    1. 将每个 DataFrame 转换为 numpy 数组，并确保数据内存连续（使用 np.ascontiguousarray）
    2. 转换为 torch.tensor，数据类型转换为 float32
    3. 堆叠所有 tensor 形成一个 batch，并移动到指定设备上
    """
    arrays = []
    arrays2 = []

    # 低阶特征
    lags1 = [1,]
    # lags1 = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]
    raw_cols1 = [
        # "n_close", "sym",
        'bid1', 'bid2', 'bid3', 'bid4', 'bid5', 'ask1', 'ask2', 'ask3', 'ask4','ask5',
        'spread1', 'spread2', 'spread3',
        'mid_price1', 'mid_price2', 'mid_price3',
        'weighted_ab1', 'weighted_ab2', 'weighted_ab3', 
        'relative_spread1', 'relative_spread2', 'relative_spread3', 
        'spread1_diff1', "spread1_diff2", 'spread2_diff1', "spread2_diff2", 'spread3_diff1', "spread3_diff2", 
        'relative_spread1_diff1', "relative_spread1_diff2", 'relative_spread2_diff1', "relative_spread2_diff2", 'relative_spread3_diff1', "relative_spread3_diff2", 
        'bsize1', 'bsize2', 'bsize3', 'bsize4', 'bsize5', 'asize1', 'asize2', 'asize3', 'asize4', 'asize5', 'amount',  
        'mid_price1_ma5', 'mid_price1_ma10', 'mid_price1_ma20', 'mid_price1_ma40', 
        "time_label", 
        # 'bid1_decay', 'ask1_decay', 'spread_decay', 'bsize1_decay', 'asize1_decay',
        'obi_1', 'obi_3', 
        'mid_diff1', 
        # 'mid_diff2', 
        'trade_impact', 'signed_amount', 'price_up_amount_down', 'amount_price_div', 
        'bid_depth_slope', 'ask_depth_slope', 'obi_sq', 
        'ask1_ma5', 'ask1_ma10', 'ask1_ma20', 'ask1_ma40', 
        'bid1_ma5', 'bid1_ma10', 'bid1_ma20', 'bid1_ma40', 
        'high_20', 'low_20', 'pos_20', 
        'high_50', 'low_50', 'pos_50', 
        'high_100', 'low_100', 'pos_100', 
        'mid_lr_k', 'mid_lr_r2', "mid_trend_strength",
        'mid_diff1_lr_k', 'mid_diff1_lr_r2', 'mid_diff1_trend_strength',
        # 'trend_persistence', 'trend_flip', 'trend_age',
        # 'trend_regime', 'trend_strength_gated', 
        'price_move_capacity', 'trend_liquidity_ratio',
        'trend_align_10_30', 'trend_align_30_60', 'trend_align_10_60', 'trend_confidence',
        'mid_lr_k_10', 'mid_lr_r2_10', 'mid_trend_strength_10', 
        'mid_lr_k_60', 'mid_lr_r2_60', 'mid_trend_strength_60',
    ]

    # 高阶特征
    lags2 = []
    raw_cols2 = []
    # lags2 = [1, 10, 30]
    # raw_cols2 = [
    #     # "n_close", "sym",
    #     'bid1', 'bid2', 'bid3', 'bid4', 'bid5', 'ask1', 'ask2', 'ask3', 'ask4','ask5',
    #     'spread1', 'spread2', 'spread3',
    #     'mid_price1', 'mid_price2', 'mid_price3',
    #     'weighted_ab1', 'weighted_ab2', 'weighted_ab3', 
    #     'relative_spread1', 'relative_spread2', 'relative_spread3', 
    #     'spread1_diff1', "spread1_diff2", 'spread2_diff1', "spread2_diff2", 'spread3_diff1', "spread3_diff2", 
    #     'relative_spread1_diff1', "relative_spread1_diff2", 'relative_spread2_diff1', "relative_spread2_diff2", 'relative_spread3_diff1', "relative_spread3_diff2", 
    #     'bsize1', 'bsize2', 'bsize3', 'bsize4', 'bsize5', 'asize1', 'asize2', 'asize3', 'asize4', 'asize5', 'amount',  
    #     'mid_price1_ma5', 'mid_price1_ma10', 'mid_price1_ma20', 'mid_price1_ma40', 
    #     "time_label", 
    #     # 'bid1_decay', 'ask1_decay', 'spread_decay', 'bsize1_decay', 'asize1_decay',
    #     # 'obi_1', 'obi_3', 'mid_diff1', 'mid_diff2', 
    #     # 'trade_impact', 'signed_amount', 'price_up_amount_down', 'amount_price_div', 
    #     # 'bid_depth_slope', 'ask_depth_slope', 'obi_sq', 
    #     'ask1_ma5', 'ask1_ma10', 'ask1_ma20', 'ask1_ma40', 
    #     'bid1_ma5', 'bid1_ma10', 'bid1_ma20', 'bid1_ma40', 
    #     # 'high_20', 'low_20', 'pos_20', 
    #     'mid_lr_k', 'mid_lr_r2', "mid_trend_strength",
    #     # 'trend_persistence', 'trend_flip', 'trend_age'
    #     # 'mid_diff1_lr_k', 'mid_diff1_lr_r2', 'mid_diff1_trend_strength',
    #     # 'mid_diff2_lr_k', 'mid_diff2_lr_r2', 'mid_diff2_trend_strength',
    #     # 'trend_regime', 'trend_strength_gated', 
    #     # 'price_move_capacity', 'trend_liquidity_ratio'
    # ]
    
    if isinstance(x, pd.DataFrame):
        x = [x]

    if is_train: # 训练时需要返回标签
        labels = []
        for df in x:
            labels.append(df['label_20'][99:])
        label = pd.concat(labels, axis=0).reset_index(drop=True)

    # 带进度条
    for i, df in tqdm(
        enumerate(x),
        total=len(x),
        desc="Preprocess",
        ncols=100
    ):
    # for i, df in enumerate(x):
        extra_feats = {}

        # 基础价格还原
        for i in range(1, 6):
            extra_feats[f'bid{i}'] = df[f'n_bid{i}'] + 1
            extra_feats[f'ask{i}'] = df[f'n_ask{i}'] + 1

        # 基础量价组合
        for i in range(1, 4):
            extra_feats[f'spread{i}'] = extra_feats[f'ask{i}'] - extra_feats[f'bid{i}']
            extra_feats[f'mid_price{i}'] = (extra_feats[f'ask{i}'] + extra_feats[f'bid{i}']) / 2

        # 对数盘口量 & 成交量
        for i in range(1, 6):
            extra_feats[f'bsize{i}'] = np.log1p(df[f'n_bsize{i}'])
            extra_feats[f'asize{i}'] = np.log1p(df[f'n_asize{i}'])
        extra_feats['amount'] = np.log1p(df['amount_delta'])

        # 相对价差
        extra_feats.update({
            'relative_spread1': extra_feats['spread1'] / extra_feats['mid_price1'],
            'relative_spread2': extra_feats['spread2'] / extra_feats['mid_price2'],
            'relative_spread3': extra_feats['spread3'] / extra_feats['mid_price3'],
        })

        # 均线特征
        for w in [5, 10, 20, 40]:
            extra_feats[f'ask1_ma{w}'] = extra_feats['ask1'].rolling(w, min_periods=1).mean()
            extra_feats[f'bid1_ma{w}'] = extra_feats['bid1'].rolling(w, min_periods=1).mean()
            extra_feats[f'mid_price1_ma{w}'] = extra_feats['mid_price1'].rolling(w, min_periods=1).mean()

        # 加权中间价
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

        # 价差变化
        extra_feats.update({
            'spread1_diff1': extra_feats['spread1'].diff().fillna(0),
            'spread1_diff2': extra_feats['spread1'].diff().diff().fillna(0),

            'spread2_diff1': extra_feats['spread2'].diff().fillna(0),
            'spread2_diff2': extra_feats['spread2'].diff().diff().fillna(0),

            'spread3_diff1': extra_feats['spread3'].diff().fillna(0),
            'spread3_diff2': extra_feats['spread3'].diff().diff().fillna(0),

            'relative_spread1_diff1': extra_feats['relative_spread1'].diff().fillna(0),
            'relative_spread1_diff2': extra_feats['relative_spread1'].diff().diff().fillna(0),

            'relative_spread2_diff1': extra_feats['relative_spread2'].diff().fillna(0),
            'relative_spread2_diff2': extra_feats['relative_spread2'].diff().diff().fillna(0),

            'relative_spread3_diff1': extra_feats['relative_spread3'].diff().fillna(0),
            'relative_spread3_diff2': extra_feats['relative_spread3'].diff().diff().fillna(0),
        })

        # 一次性合并
        df = pd.concat([df, pd.DataFrame(extra_feats, index=df.index)], axis=1)

        # 时间标签
        df['time_label'] = assign_tick_time_labels(df['time'])

        # 中间价线性回归
        k, r2 = rolling_lr_features(df['mid_price1'].to_numpy())
        df['mid_lr_k'] = k   # 斜率
        df['mid_lr_r2'] = r2   # 拟合优度
        df['mid_trend_strength'] = np.sign(k) * r2

        # 信号的持续性
        df['trend_persistence'] = (
            df['mid_trend_strength']
            .rolling(20)
            .apply(lambda x: np.sum(np.sign(x) == np.sign(x.iloc[-1])))
        )
        df['trend_flip'] = (np.sign(df['mid_lr_k']).diff() != 0).astype(int)
        df['trend_age'] = df['trend_flip'].rolling(50).sum()

        # 门控
        df['trend_regime'] = (
            (df['mid_lr_r2'] > 0.25) &
            (np.abs(df['mid_lr_k']) > np.percentile(np.abs(df['mid_lr_k']), 60))
        ).astype(int)
        df['trend_strength_gated'] = df['mid_trend_strength'] * df['trend_regime']

        k_10, r2_10 = rolling_lr_features(df['mid_price1'].to_numpy(), window=10)
        k_30, r2_30 = k, r2
        k_60, r2_60 = rolling_lr_features(df['mid_price1'].to_numpy(), window=60)
        df['trend_align_10_30'] = (np.sign(k_10) == np.sign(k_30)).astype(int)
        df['trend_align_30_60'] = (np.sign(k_30) == np.sign(k_60)).astype(int)
        df['trend_align_10_60'] = (np.sign(k_10) != np.sign(k_60)).astype(int) * abs(k_10 - k_60)
        df['trend_confidence'] = (
            np.sign(k_10) * r2_10 +
            np.sign(k_30) * r2_30 +
            np.sign(k_60) * r2_60
        )

        df['mid_lr_k_10'] = k_10   # 斜率
        df['mid_lr_r2_10'] = r2_10   # 拟合优度
        df['mid_trend_strength_10'] = np.sign(k_10) * r2_10
        
        df['mid_lr_k_60'] = k_60   # 斜率
        df['mid_lr_r2_60'] = r2_60   # 拟合优度
        df['mid_trend_strength_60'] = np.sign(k_60) * r2_60

        # 盘口是否允许价格往某个方向动？
        df['price_move_capacity'] = df['mid_lr_k'] / (df['relative_spread1'] + 1e-6)
        df['trend_liquidity_ratio'] = df['mid_trend_strength'] / (df['relative_spread1'] + 1e-6)

        # 过去20、50、100个数据中的最高价和最低价
        df['high_20'] = df['mid_price1'].rolling(window=20, min_periods=1).max()
        df['low_20'] = df['mid_price1'].rolling(window=20, min_periods=1).min()
        df['high_50'] = df['mid_price1'].rolling(window=50, min_periods=1).max()
        df['low_50'] = df['mid_price1'].rolling(window=50, min_periods=1).min()
        df['high_100'] = df['mid_price1'].rolling(window=100, min_periods=1).max()
        df['low_100'] = df['mid_price1'].rolling(window=100, min_periods=1).min()

        # 相对位置
        df['pos_20'] = (df['mid_price1'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-6)
        df['pos_50'] = (df['mid_price1'] - df['low_50']) / (df['high_50'] - df['low_50'] + 1e-6)
        df['pos_100'] = (df['mid_price1'] - df['low_100']) / (df['high_100'] - df['low_100'] + 1e-6)

        # # 时间衰减盘口特征
        # decay = np.exp(-np.arange(100)[::-1] / 20)  # 越近权重越大
        # decay = decay / decay.sum()
        # def decay_mean(x):
        #     w = decay[-len(x):]
        #     return np.sum(x * w)
        # df['bid1_decay'] = df['bid1'].rolling(100, min_periods=1).apply(decay_mean, raw=True)
        # df['ask1_decay'] = df['ask1'].rolling(100, min_periods=1).apply(decay_mean, raw=True)
        # df['spread_decay'] = df['spread'].rolling(100, min_periods=1).apply(decay_mean, raw=True)
        # df['bsize1_decay'] = df['bsize1'].rolling(100, min_periods=1).apply(decay_mean, raw=True)
        # df['asize1_decay'] = df['asize1'].rolling(100, min_periods=1).apply(decay_mean, raw=True)
        
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

        # 盘口斜率
        df['bid_depth_slope'] = (df['bsize5'] - df['bsize1']) / 4
        df['ask_depth_slope'] = (df['asize5'] - df['asize1']) / 4

        # mid_price 动量
        df['mid_diff1'] = df['mid_price1'].diff().fillna(0)   # 一阶差分：速度
        df['mid_diff2'] = df['mid_diff1'].diff().fillna(0)   # 二阶差分：加速度
        
        # mid_diff1线性回归
        k, r2 = rolling_lr_features(df['mid_diff1'].to_numpy())
        df['mid_diff1_lr_k'] = k   # 斜率
        df['mid_diff1_lr_r2'] = r2   # 拟合优度
        df['mid_diff1_trend_strength'] = np.sign(k) * r2

        extra_feats = {}
        # 价格冲击方向
        extra_feats['trade_impact'] = df['mid_diff1'] * df['amount']
        extra_feats['signed_amount'] = np.sign(df['mid_diff1']) * df['amount']
        # 量价背离
        extra_feats['price_up_amount_down'] = (
            (df['mid_diff1'] > 0).astype(int) * (df['amount'].diff() < 0).astype(int)
        )
        extra_feats['amount_price_div'] = df['mid_diff1'] / (df['amount'] + 1e-6)

        df = pd.concat([df, pd.DataFrame(extra_feats, index=df.index)], axis=1)

        # 时间衰减采样历史数据
        df = time_fixed_sample(df, raw_cols1, lags=lags1)
        df = df.copy()

        df = time_fixed_sample(df, raw_cols2, lags=lags2)
        df = df.copy()

        lag_cols = [f'{c}_lag{lag}' for c in raw_cols1 for lag in lags1] + [f'{c}_lag{lag}' for c in raw_cols2 for lag in lags2]
        arr = np.ascontiguousarray(df[lag_cols].values.astype(np.float32))
        arrays.append(arr)

        arrays2.append(df[lag_cols][99:].astype('float32'))    # 去掉前99个tick

    concat_df = pd.concat(arrays2, axis=0, ignore_index=True)
    x_hat = np.stack(arrays, axis=0)   
    if is_slice:
        x_hat = x_hat[:, 99:, :].squeeze(axis=1)
    else:
        x_hat = x_hat[:, 99:, :].squeeze(axis=0)
    
    # # 都是dataframe
    # print("concat_df.shape: ", concat_df.shape)   # (1900, 950)
    # print("label.shape: ", label.shape)   # (1900,)
    # print("x_hat.shape: ", x_hat.shape)   # (1900, 950)

    if is_train:
        return concat_df, label
    else:
        return concat_df
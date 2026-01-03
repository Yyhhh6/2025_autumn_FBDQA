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
        pth_path = os.path.join(os.path.dirname(__file__), 'model_20_all_20260102_235444.json')
        # 加载模型并移动到对应设备，假设模型是整个模型保存，如果是参数字典需要初始化结构
        self.model = self.load_model(pth_path)
        print(f"model loaded from {pth_path}")
        
    def predict(self, x: List[pd.DataFrame]) -> List[List[int]]:
        # 对输入数据进行预处理
        print(f"Received {len(x)} dataframes for prediction.")
        x_hat = self.preprocess(x)

        y = []
        target_confidence = 0.8
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
        return preprocess_platform(x, is_local=False)

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


def compare_dfs(df1, df2, tol=1e-8):
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
    
    # diff_mask = (df1_common - df2_common).abs() > tol
    diff_mask = ~(df1_common.eq(df2_common) | (df1_common.isna() & df2_common.isna()))
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


def preprocess_slice(x: list[pd.DataFrame]):
    # 只处理 100 个tick
    x_extract = []
    # print("len(x): ", len(x))
    lags1=[1, 2, 5, 10, 20, 50]
    # lags1=[1, 2, 5,]
    # lags2=[1, 2, 5]
    lags2=[1]
    lags3=[1]

    for df in tqdm(
        x,
        total=len(x),
        desc="Preprocess",
        ncols=100
    ):
        df = df.reset_index(drop=True)
        assert len(df) == 100

        # 中间价一阶差分
        df['mid_price'] = 1 + df['n_midprice']
        # df['mid_diff1'] = df['mid_price'].diff().fillna(0)   # 一阶差分：速度
        df['amount'] = np.log1p(df['amount_delta'])

        # 真实成交量
        df['real_volume'] = np.log1p(df['amount_delta'] / (df['mid_price'] + 1e-10))

        # ********************下面用循环的方法计算差分好像更快********************
        mid_price = df['mid_price'].to_numpy()
        mid_diff1 = np.empty_like(mid_price)
        mid_diff1[0] = 0.0
        prev = mid_price[0]
        mid_diff1_sum = 0
        for i in range(1, 100):
            diff = mid_price[i] - prev
            mid_diff1[i] = diff
            prev = mid_price[i] 
            mid_diff1_sum += diff if diff > 0 else -diff
        mid_diff1_open_close = mid_price[-1] - mid_price[0]
        df['mid_diff1'] = mid_diff1
        # print("mid_diff1: ", mid_diff1)

        df['trade_sign'] = np.where(
            df['n_close'] == df['n_ask1'],  1,
            np.where(df['n_close'] == df['n_bid1'], -1, 0)
        )
        # print("trade_sign: ", trade_sign)

        df['aggression_eff'] = df['trade_sign'] * np.sign(df['mid_diff1'])
        df['price_control'] = df['trade_sign'] * df['mid_diff1']

        df['close_pos'] = (
            (df['n_close'] - df['n_midprice'])
            / ((df['n_ask1'] - df['n_bid1']) / 2 + 1e-10)
        )

        # 用字典存每个lag的特征
        lag_feats = {}
        feat_dict = {}

        for lag in lags1:
            # 用iloc直接取lag对应行
            row = df.iloc[-lag]

            # 收盘价还原
            feat_dict[f'close_lag{lag}'] = row['n_close'] + 1

            # 基础价格还原
            for i in range(1, 6):
                feat_dict[f'bid{i}_lag{lag}'] = row[f'n_bid{i}'] + 1
                feat_dict[f'ask{i}_lag{lag}'] = row[f'n_ask{i}'] + 1

            # 价格 & 价差 & 相对价差
            feat_dict[f'mid_price_lag{lag}'] = row["mid_price"]
            for i in range(1, 4):
                feat_dict[f'spread{i}_lag{lag}'] = feat_dict[f'ask{i}_lag{lag}'] - feat_dict[f'bid{i}_lag{lag}']
                feat_dict[f'mid_price{i}_lag{lag}'] = (feat_dict[f'bid{i}_lag{lag}'] + feat_dict[f'ask{i}_lag{lag}']) / 2 
                feat_dict[f'relative_spread{i}_lag{lag}'] = feat_dict[f'spread{i}_lag{lag}'] / feat_dict[f'mid_price{i}_lag{lag}']

                # 买卖单相对密度
                feat_dict[f'relative_bid_density{i}_lag{lag}'] = row[f'n_bsize{i}'] / (row[f'n_bsize{i}'] + row[f'n_asize{i}'])
                feat_dict[f'relative_ask_density{i}_lag{lag}'] = row[f'n_asize{i}'] / (row[f'n_bsize{i}'] + row[f'n_asize{i}'])

            # 买卖量差
            feat_dict[f'vol1_rel_diff_lag{lag}'] = (row[f'n_bsize1'] - row[f'n_asize1']) / (row[f'n_bsize1'] + row[f'n_asize1'])
            feat_dict[f'vol3_rel_diff_lag{lag}'] = (row[f'n_bsize1'] + row[f'n_bsize2'] + row[f'n_bsize3'] - row[f'n_asize1'] - row[f'n_asize2'] - row[f'n_asize3']) / (row[f'n_bsize1'] + row[f'n_asize1'] + row[f'n_bsize2'] + row[f'n_asize2'] + row[f'n_bsize3'] + row[f'n_asize3'])
            feat_dict[f'vol5_rel_diff_lag{lag}'] = (row[f'n_bsize1'] + row[f'n_bsize2'] + row[f'n_bsize3'] + row[f'n_bsize4'] + row[f'n_bsize5'] - row[f'n_asize1'] - row[f'n_asize2'] - row[f'n_asize3'] - row[f'n_asize4'] - row[f'n_asize5']) / (row[f'n_bsize1'] + row[f'n_asize1'] + row[f'n_bsize2'] + row[f'n_asize2'] + row[f'n_bsize3'] + row[f'n_asize3'] + row[f'n_bsize4'] + row[f'n_asize4'] + row[f'n_bsize5'] + row[f'n_asize5'])

            # log量 & 成交量
            for i in range(1, 6):
                feat_dict[f'bsize{i}_lag{lag}'] = np.log1p(row[f'n_bsize{i}'])
                feat_dict[f'asize{i}_lag{lag}'] = np.log1p(row[f'n_asize{i}'])
            feat_dict[f'amount_lag{lag}'] = row['amount']
            feat_dict[f'amount_midprice_lag{lag}'] = row['real_volume']

            # 把加权和价差变化加入 lag 特征
            for i in range(1, 4):
                feat_dict[f'weighted_ab{i}_lag{lag}'] = (
                        feat_dict[f'ask{i}_lag{lag}'] * feat_dict[f'bsize{i}_lag{lag}'] \
                        + feat_dict[f'bid{i}_lag{lag}'] * feat_dict[f'asize{i}_lag{lag}']
                    ) / (feat_dict[f'asize{i}_lag{lag}'] + feat_dict[f'bsize{i}_lag{lag}'])

            # # ===== 盘口不平衡（仅采样点） =====
            b1, a1 = feat_dict[f'bsize1_lag{lag}'], feat_dict[f'asize1_lag{lag}']
            b3 = feat_dict[f'bsize1_lag{lag}'] + feat_dict[f'bsize2_lag{lag}'] + feat_dict[f'bsize3_lag{lag}']
            a3 = feat_dict[f'asize1_lag{lag}'] + feat_dict[f'asize2_lag{lag}'] + feat_dict[f'asize3_lag{lag}']

            obi_1 = (b1 - a1) / (b1 + a1 + 1e-10)
            obi_3 = (b3 - a3) / (b3 + a3 + 1e-10)

            feat_dict[f'obi_1_lag{lag}'] = obi_1
            feat_dict[f'obi_3_lag{lag}'] = obi_3
            feat_dict[f'obi_sq_lag{lag}'] = obi_3 * abs(obi_3)

            # ===== 深度斜率（盘口形状） =====
            feat_dict[f'bid_depth_slope_lag{lag}'] = (feat_dict[f'bsize5_lag{lag}'] - feat_dict[f'bsize1_lag{lag}']) / 4
            feat_dict[f'ask_depth_slope_lag{lag}'] = (feat_dict[f'asize5_lag{lag}'] - feat_dict[f'asize1_lag{lag}']) / 4

            # 真实成交量特征
            feat_dict[f'real_volume_lag{lag}'] = row['real_volume']
            # 成交量与价格的比值（反映每单位价格变化的成交量变化）
            feat_dict[f'price_volume_ratio_lag{lag}'] = row['mid_price'] / (feat_dict[f'real_volume_lag{lag}'] + 1e-10)

            feat_dict[f'trade_sign_lag{lag}'] = row['trade_sign']
            feat_dict[f'aggression_eff_lag{lag}'] = row['aggression_eff']
            feat_dict[f'price_control_lag{lag}'] = row['price_control']
            feat_dict[f'close_pos_lag{lag}'] = row['close_pos']

            lag_feats.update(feat_dict)

        # for lag in lags2:
        #     # 用iloc直接取lag对应行
        #     row = df.iloc[-lag]

            # 一阶 / 二阶差分（注意边界）
            feat_dict[f'mid_diff1_lag{lag}'] = row["mid_diff1"]

            # ===== 量价关系（仅采样点） =====
            # 价格冲击
            feat_dict[f'trade_impact_lag{lag}'] = row['amount'] * row['mid_diff1']
            feat_dict[f'signed_amount_lag{lag}'] = np.sign(row['mid_diff1']) * row['amount']
            # 量价背离
            feat_dict[f'price_up_amount_down_lag{lag}'] = (
                int(row['mid_diff1'] > 0) * int(row['amount'] < df['amount'].iloc[-lag-1])
            )
            feat_dict[f'amount_price_div_lag{lag}'] = row['mid_diff1'] / (row['amount'] + 1e-10)

            lag_feats.update(feat_dict)

        for lag in lags3:
            # 用iloc直接取lag对应行
            row = df.iloc[-lag]

            # 区分标的
            feat_dict[f'sym_lag{lag}'] = row['sym']

            # 时间特征
            feat_dict[f'time_label_lag{lag}'] = assign_tick_time_label(row['time'])

            # 还原原始股价
            sym_to_price = {
                0: 1300,
                1: 88.4,
                2: 1004.6,
                3: 61,
                4: 444.4,
                5: 302.4,
                6: 535.2,
                7: 323.2,
                8: 395.4,
                9: 1740,
            }
            feat_dict[f'original_price_lag{lag}'] = sym_to_price[row['sym']]

            # 股价分类
            feat_dict[f'sym_class_lag{lag}'] = (
                0 if feat_dict[f'original_price_lag{lag}'] < 100 else 
                1 if feat_dict[f'original_price_lag{lag}'] < 699 else 
                2 if feat_dict[f'original_price_lag{lag}'] < 1500 else 
                3             
            )

            # 真趋势” vs “来回波动
            feat_dict[f'mid_diff1_open_close_lag{lag}'] = mid_diff1_open_close
            feat_dict[f'mid_diff1_sum_lag{lag}'] = mid_diff1_sum
            feat_dict[f'SignedTrendEff_lag{lag}'] = mid_diff1_open_close / (mid_diff1_sum + 1e-5)

            # time: 处理 HH:MM:SS 转换为以秒为单位的数值
            h, m, s = row['time'].split(':')
            time_sec = int(h) * 3600 + int(m) * 60 + int(s)
            feat_dict[f'time_lag{lag}'] = time_sec

            # 时间分段（每 30 分钟一档）
            feat_dict[f'time_interval_lag{lag}'] = (
                0 if time_sec < 36000 else   # 09:30 - 10:00
                1 if time_sec < 37800 else   # 10:00 - 10:30
                2 if time_sec < 39600 else   # 10:30 - 11:00
                3 if time_sec < 41400 else   # 11:00 - 11:30
                4 if time_sec < 46800 else   # 13:00 - 13:30
                5 if time_sec < 48600 else   # 13:30 - 14:00
                6 if time_sec < 50400 else   # 14:00 - 14:30
                7                            # 14:30 - 15:00
            )

            # 前向差分特征
            for w in [1, 10, 40]:
                feat_dict[f'midprice_delta_rate{w}_lag{lag}'] = row['mid_price'] / df['mid_price'].iloc[-lag-w] - 1
                feat_dict[f'midprice_delta{w}_lag{lag}'] = row['mid_price'] - df['mid_price'].iloc[-lag-w]
                feat_dict[f'close_delta{w}_lag{lag}'] = row['n_close'] - df['n_close'].iloc[-lag-w]
                feat_dict[f'amount_delta{w}_lag{lag}'] = row['amount'] - df['amount'].iloc[-lag-w]
                feat_dict[f'real_volume_delta{w}_lag{lag}'] = row['real_volume'] - df["real_volume"].iloc[-lag-w]  
                
                for i in [1, 3, 5]:
                    feat_dict[f'ask{i}_delta{w}_lag{lag}'] = row[f'n_ask{i}'] - df[f'n_ask{i}'].iloc[-lag-w]
                    feat_dict[f'bid{i}_delta{w}_lag{lag}'] = row[f'n_bid{i}'] - df[f'n_bid{i}'].iloc[-lag-w]
                    
                    feat_dict[f'asize{i}_delta{w}_lag{lag}'] = feat_dict[f'asize{i}_lag{lag}'] - np.log1p(df[f'n_asize{i}'].iloc[-lag-w])
                    feat_dict[f'bsize{i}_delta{w}_lag{lag}'] = feat_dict[f'bsize{i}_lag{lag}'] - np.log1p(df[f'n_bsize{i}'].iloc[-lag-w])

            # 基于n_close成交方向的窗口特征
            for w in [5, 20, 50, 100]:
            # for w in [5, 20, 50]:

                feat_dict[f'trade_sign_sum{w}_lag{lag}'] = df['trade_sign'].iloc[-lag-w+1:None if lag == 1 else -lag+1].sum()
                feat_dict[f'trade_sign_mean{w}_lag{lag}'] = df['trade_sign'].iloc[-lag-w+1:None if lag == 1 else -lag+1].mean()
                feat_dict[f'trade_sign_aggression_eff{w}_lag{lag}'] = df['aggression_eff'].iloc[-lag-w+1:None if lag == 1 else -lag+1].mean()

                # 成交持续性
                feat_dict[f'trade_sign_buy_run{w}_lag{lag}'] = (df['trade_sign'].iloc[-lag-w+1:None if lag == 1 else -lag+1] == 1).sum()
                feat_dict[f'trade_sign_sell_run{w}_lag{lag}'] = (df['trade_sign'].iloc[-lag-w+1:None if lag == 1 else -lag+1] == -1).sum()

                # 势能不对称
                feat_dict[f'trade_sign_aggression_skew{w}_lag{lag}'] = (
                    (feat_dict[f'trade_sign_buy_run{w}_lag{lag}'] - feat_dict[f'trade_sign_sell_run{w}_lag{lag}'])
                    / (w + 1e-10)
                )

                # 主动成交推动率
                feat_dict[f'trade_sign_price_control{w}_lag{lag}'] = (df['price_control'].iloc[-lag-w+1:None if lag == 1 else -lag+1] == -1).mean()

                # rolling 成交压强
                feat_dict[f'trade_sign_close_pressure{w}_lag{lag}'] = (df['close_pos'].iloc[-lag-w+1:None if lag == 1 else -lag+1] == -1).mean()

                # 势能组合指数
                feat_dict[f'aggression_score{w}_lag{lag}'] = (
                    0.4 * feat_dict[f'trade_sign_aggression_skew{w}_lag{lag}']
                    + 0.3 * feat_dict[f'trade_sign_close_pressure{w}_lag{lag}']
                    + 0.3 * feat_dict[f'trade_sign_price_control{w}_lag{lag}']
                )

            # mid_price 窗口特征
            for w in [5, 20, 50, 100]:
            # for w in [5, 20, 50]:
                feat_dict[f'mid_price_ma{w}_lag{lag}'] = df['mid_price'].iloc[-lag-w+1:None if lag == 1 else -lag+1].mean()
                feat_dict[f'mid_price_std{w}_lag{lag}'] = df['mid_price'].iloc[-lag-w+1:None if lag == 1 else -lag+1].std()
                feat_dict[f'mid_price_vol_ratio{w}_lag{lag}'] = feat_dict[f'mid_price_std{w}_lag{lag}'] / (np.abs(feat_dict[f'mid_price_ma{w}_lag{lag}']) + 1e-10)
       
                feat_dict[f'mid_price_max{w}_lag{lag}'] = df['mid_price'].iloc[-lag-w+1:None if lag == 1 else -lag+1].max()
                feat_dict[f'mid_price_min{w}_lag{lag}'] = df['mid_price'].iloc[-lag-w+1:None if lag == 1 else -lag+1].min()
                feat_dict[f'mid_price_max_min{w}_lag{lag}'] = feat_dict[f'mid_price_max{w}_lag{lag}'] - feat_dict[f'mid_price_min{w}_lag{lag}']
                feat_dict[f'pos_{w}_lag{lag}'] = (row['mid_price'] -  feat_dict[f'mid_price_min{w}_lag{lag}']) / (feat_dict[f'mid_price_max{w}_lag{lag}'] -  feat_dict[f'mid_price_min{w}_lag{lag}'] + 1e-10)

            mid_diff1_spike_th = 0.0008
            mid_diff1_threshold_1 = 10
            mid_diff1_threshold_2 = 50

            # mid_diff1 窗口特征
            for w in [5, 20, 50, 100]:
            # for w in [5, 20, 50]:

                feat_dict[f'mid_diff1_ma{w}_lag{lag}'] = df['mid_diff1'].iloc[-lag-w+1:None if lag == 1 else -lag+1].mean()
                feat_dict[f'mid_diff1_std{w}_lag{lag}'] = df['mid_diff1'].iloc[-lag-w+1:None if lag == 1 else -lag+1].std()
                feat_dict[f'mid_diff1_vol_ratio{w}_lag{lag}'] = feat_dict[f'mid_diff1_std{w}_lag{lag}'] / (np.abs(feat_dict[f'mid_diff1_ma{w}_lag{lag}']) + 1e-10)

                feat_dict[f'mid_diff1_max{w}_lag{lag}'] = df['mid_diff1'].iloc[-lag-w+1:None if lag == 1 else -lag+1].max()
                feat_dict[f'mid_diff1_min{w}_lag{lag}'] = df['mid_diff1'].iloc[-lag-w+1:None if lag == 1 else -lag+1].min()
                feat_dict[f'mid_diff1_max_min{w}_lag{lag}'] = feat_dict[f'mid_diff1_max{w}_lag{lag}'] - feat_dict[f'mid_diff1_min{w}_lag{lag}']

                # 窗口内是否曾出现强冲击
                feat_dict[f'mid_diff1_has_spike{w}_lag{lag}'] = (
                    (feat_dict[f'mid_diff1_max{w}_lag{lag}'] > mid_diff1_spike_th) or (-feat_dict[f'mid_diff1_min{w}_lag{lag}'] > mid_diff1_spike_th)
                ).astype(int)

                # 窗口内是否出现 高震荡or平台
                feat_dict[f'mid_diff1_is_flat{w}_lag{lag}'] = (
                    (feat_dict[f'mid_diff1_vol_ratio{w}_lag{lag}'] < mid_diff1_threshold_1)
                ).astype(int)
                feat_dict[f'mid_diff1_is_high_vol{w}_lag{lag}'] = (
                    (feat_dict[f'mid_diff1_vol_ratio{w}_lag{lag}'] > mid_diff1_threshold_2)
                ).astype(int)

                window_diff = df['mid_diff1'].iloc[
                    -lag-w+1 : None if lag == 1 else -lag+1
                ]

                # sign_consistency_20（方向一致性）
                feat_dict[f'sign_consistency{w}_lag{lag}'] = np.abs(
                    np.sign(window_diff).mean()
                )

                # trend_persist_20（趋势持续性）
                signs = np.sign(window_diff)
                last_sign = signs.iloc[-1]
                feat_dict[f'trend_persist{w}_lag{lag}'] = np.sum(
                    signs == last_sign
                )

            # real_volume 窗口特征
            for w in [5, 20, 50, 100]:
            # for w in [5, 20, 50]:

                feat_dict[f'real_volume_ma{w}_lag{lag}'] = df['real_volume'].iloc[-lag-w+1:None if lag == 1 else -lag+1].mean()
                feat_dict[f'real_volume_std{w}_lag{lag}'] = df['real_volume'].iloc[-lag-w+1:None if lag == 1 else -lag+1].std()
                feat_dict[f'real_volume_vol_ratio{w}_lag{lag}'] = feat_dict[f'real_volume_std{w}_lag{lag}'] / (np.abs(feat_dict[f'real_volume_ma{w}_lag{lag}']) + 1e-10)
       
                feat_dict[f'real_volume_max{w}_lag{lag}'] = df['real_volume'].iloc[-lag-w+1:None if lag == 1 else -lag+1].max()
                feat_dict[f'real_volume_min{w}_lag{lag}'] = df['real_volume'].iloc[-lag-w+1:None if lag == 1 else -lag+1].min()
                feat_dict[f'real_volume_max_min{w}_lag{lag}'] = feat_dict[f'real_volume_max{w}_lag{lag}'] - feat_dict[f'real_volume_min{w}_lag{lag}']
                feat_dict[f'real_volume_pos_{w}_lag{lag}'] = (row['real_volume'] -  feat_dict[f'real_volume_min{w}_lag{lag}']) / (feat_dict[f'real_volume_max{w}_lag{lag}'] -  feat_dict[f'real_volume_min{w}_lag{lag}'] + 1e-10)

            # 长短窗口特征对比
            feat_dict[f'mid_diff1_vol_ratio_sl_5_50_lag{lag}'] = feat_dict[f'mid_diff1_vol_ratio5_lag{lag}'] / (feat_dict[f'mid_diff1_vol_ratio50_lag{lag}'] + 1e-6)
            feat_dict[f'real_volume_vol_ratio_sl_5_50_lag{lag}'] = feat_dict[f'real_volume_vol_ratio5_lag{lag}'] / (feat_dict[f'real_volume_vol_ratio50_lag{lag}'] + 1e-6)

            feat_dict[f'mid_diff1_vol_ratio_sl_20_100_lag{lag}'] = feat_dict[f'mid_diff1_vol_ratio20_lag{lag}'] / (feat_dict[f'mid_diff1_vol_ratio100_lag{lag}'] + 1e-6)
            feat_dict[f'real_volume_vol_ratio_sl_20_100_lag{lag}'] = feat_dict[f'real_volume_vol_ratio20_lag{lag}'] / (feat_dict[f'real_volume_vol_ratio100_lag{lag}'] + 1e-6)

            # ===== mid_diff1 的局部线性趋势（小窗口） =====
            window_size = 5
            mid_diff1_window = df['mid_diff1'].iloc[-lag+1-window_size:None if lag == 1 else -lag+1]
            k_d1, r2_d1 = rolling_lr_k_r2(mid_diff1_window)
            feat_dict[f'mid_diff1_lr_k_lag{lag}'] = k_d1
            feat_dict[f'mid_diff1_lr_r2_lag{lag}'] = r2_d1
            feat_dict[f'mid_diff1_trend_strength_lag{lag}'] = np.sign(k_d1) * r2_d1

            feat_dict[f'mid_diff1_price_move_capacity_lag{lag}'] = k_d1 / (feat_dict[f'relative_spread1_lag{lag}'] + 1e-10)
            feat_dict[f'mid_diff1_trend_liquidity_ratio_lag{lag}'] = np.sign(k_d1) * r2_d1 / (feat_dict[f'relative_spread1_lag{lag}'] + 1e-10)

            # 门控趋势特征：趋势筛选 + 趋势强度
            k_threshold = 0.000030
            r2_threshold = 0.4

            feat_dict[f'mid_diff1_trend_regime_lag{lag}'] = (
                (r2_d1 > r2_threshold) &
                (np.abs(k_d1) > k_threshold)
            ).astype(int)
            feat_dict[f'mid_diff1_trend_strength_gated_lag{lag}'] = feat_dict[f'mid_diff1_trend_strength_lag{lag}'] * feat_dict[f'mid_diff1_trend_regime_lag{lag}']

            # ===== 真实量线性回归 =====
            window_size = 5
            real_volume_window = df['real_volume'].iloc[-lag+1-window_size:None if lag == 1 else -lag+1]
            k_d1, r2_d1 = rolling_lr_k_r2(real_volume_window)
            feat_dict[f'real_volume_lr_k_lag{lag}'] = k_d1
            feat_dict[f'real_volume_lr_r2_lag{lag}'] = r2_d1
            feat_dict[f'real_volume_trend_strength_lag{lag}'] = np.sign(k_d1) * r2_d1

            feat_dict[f'real_volume_price_move_capacity_lag{lag}'] = k_d1 / (feat_dict[f'relative_spread1_lag{lag}'] + 1e-10)
            feat_dict[f'real_volume_trend_liquidity_ratio_lag{lag}'] = np.sign(k_d1) * r2_d1 / (feat_dict[f'relative_spread1_lag{lag}'] + 1e-10)

            feat_dict[f'real_volume_trend_regime_lag{lag}'] = (
                (r2_d1 > r2_threshold) &
                (np.abs(k_d1) > k_threshold)
            ).astype(int)
            feat_dict[f'real_volume_trend_strength_gated_lag{lag}'] = feat_dict[f'real_volume_trend_strength_lag{lag}'] * feat_dict[f'real_volume_trend_regime_lag{lag}']

            # ===== 多尺度中间价线性趋势 =====
            lr_windows = [10, 30, 60]  # window_size=30前面已经算过
            k_dict = {}
            r2_dict = {}

            for w in lr_windows:
                # ----- 中间价线性回归 -----
                window_prices = df['mid_price'].iloc[-lag+1-w:None if lag == 1 else -lag+1]

                # 滑动窗口内线性回归
                k, r2 = rolling_lr_k_r2(window_prices)

                k_dict[w] = k
                r2_dict[w] = r2

                feat_dict[f'mid_lr_k_{w}_lag{lag}'] = k
                feat_dict[f'mid_lr_r2_{w}_lag{lag}'] = r2
                feat_dict[f'mid_trend_strength_{w}_lag{lag}'] = np.sign(k) * r2

                feat_dict[f'price_move_capacity_{w}_lag{lag}'] = k / (feat_dict[f'relative_spread1_lag{lag}'] + 1e-10)
                feat_dict[f'trend_liquidity_ratio_{w}_lag{lag}'] = np.sign(k) * r2 / (feat_dict[f'relative_spread1_lag{lag}'] + 1e-10)

                feat_dict[f'trend_regime_{w}_lag{lag}'] = (
                    (r2 > r2_threshold) &
                    (np.abs(k) > k_threshold)
                ).astype(int)
                feat_dict[f'trend_strength_gated_{w}_lag{lag}'] = feat_dict[f'mid_trend_strength_{w}_lag{lag}'] * feat_dict[f'trend_regime_{w}_lag{lag}']

            feat_dict[f'trend_align_10_30_lag{lag}'] = int(
                np.sign(k_dict[10]) == np.sign(k_dict[30])
            )
            feat_dict[f'trend_align_30_60_lag{lag}'] = int(
                np.sign(k_dict[30]) == np.sign(k_dict[60])
            )
            feat_dict[f'trend_align_10_60_lag{lag}'] = int(
                np.sign(k_dict[10]) == np.sign(k_dict[60])
            )

            feat_dict[f'trend_align_amount_10_30_lag{lag}'] = (
                np.sign(k_dict[10]) != np.sign(k_dict[30])
            ) * abs(k_dict[10] - k_dict[30])
            feat_dict[f'trend_align_amount_30_60_lag{lag}'] = (
                np.sign(k_dict[30]) != np.sign(k_dict[60])
            ) * abs(k_dict[30] - k_dict[60])
            feat_dict[f'trend_align_amount_10_60_lag{lag}'] = (
                np.sign(k_dict[10]) != np.sign(k_dict[60])
            ) * abs(k_dict[10] - k_dict[60])

            feat_dict[f'trend_confidence_lag{lag}'] = (
                np.sign(k_dict[10]) * r2_dict[10] +
                np.sign(k_dict[30]) * r2_dict[30] +
                np.sign(k_dict[60]) * r2_dict[60]
            )

            lag_feats.update(feat_dict)

        x_extract.append(lag_feats)

    # concat_df = pd.DataFrame(x_extract)
    concat_df = pd.DataFrame(x_extract).astype('float32')

    # 对列的标签进行排序
    concat_df = concat_df[sorted(concat_df.columns)]

    return concat_df


def preprocess_platform(x: Union[List[pd.DataFrame], pd.DataFrame], is_local=True, is_upload=True):
    """
    更改 preprocess 逻辑，适用于公榜评测的更快推理
    """
    if is_upload==False:    
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

        # TODO:
        x_extract1 = preprocess_slice(x_slice)
        print("x_extract1.shape: ", x_extract1.shape)
        # print("x_extract1['mid_diff1']: ", x_extract1["mid_diff1"])
        # print("x_extract1['mid_lr_k_lag1']: ", x_extract1["mid_lr_k_lag1"])
        # print("x_extract1['mid_lr_r2_lag1']: ", x_extract1["mid_lr_r2_lag1"])
        x_extract2 = preprocess_local(x_slice, is_slice=True)
        print("x_extract2.shape: ", x_extract2.shape)
        # 对列排序
        x_extract1 = x_extract1.reindex(sorted(x_extract1.columns), axis=1)
        x_extract2 = x_extract2.reindex(sorted(x_extract2.columns), axis=1)
        x_extract1.to_csv("x_extract1.csv", index=True)
        x_extract2.to_csv("x_extract2.csv", index=True)
        compare_dfs(x_extract1, x_extract2)
        exit(0)

        if is_local==False:
            # 方法一
            x_extract = preprocess_slice(x_slice)
        else:
            # 方法二
            x_extract, _label, _ = preprocess_local(x_slice, is_train=True, is_slice=True)
            compare_dfs(label, _label)  # 二者相同！说明没问题

        assert len(x_extract) == len(label), f"len(x_extract): {len(x_extract)}    len(label): {len(label)}"
        return x_extract, label

    else:    
        # 公榜评测，pd.DataFrame 中都是 100 个 tick
        if isinstance(x, pd.DataFrame):
            x = [x]

        if is_local==False:
            return preprocess_slice(x)
        else:
            return preprocess_local(x, is_train=False, is_slice=True)

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
    lags1 = [1, 2, 5, 10, 20, 50]
    raw_cols1 = [
        "close", "mid_price",
        'bid1', 'bid2', 'bid3', 'bid4', 'bid5', 'ask1', 'ask2', 'ask3', 'ask4','ask5',
        'spread1', 'spread2', 'spread3',
        'mid_price1', 'mid_price2', 'mid_price3',
        'relative_spread1', 'relative_spread2', 'relative_spread3', 
        'bsize1', 'bsize2', 'bsize3', 'bsize4', 'bsize5', 'asize1', 'asize2', 'asize3', 'asize4', 'asize5', 
        'amount', "amount_midprice",
        'weighted_ab1', 'weighted_ab2', 'weighted_ab3', 
        # 'spread1_diff1', 'spread2_diff1', 'spread3_diff1', 
        # 'relative_spread1_diff1', 'relative_spread2_diff1', 'relative_spread3_diff1', 
        'obi_1', 'obi_3', 'obi_sq',
        'bid_depth_slope', 'ask_depth_slope',  
        # "relative_spread1_diff2", "relative_spread2_diff2", "relative_spread3_diff2",
        # "spread1_diff2", "spread2_diff2", "spread3_diff2",
        "vol1_rel_diff", "vol3_rel_diff", "vol5_rel_diff", 
        'relative_bid_density1', 'relative_bid_density2', 'relative_bid_density3', 
        'relative_ask_density1', 'relative_ask_density2', 'relative_ask_density3',
        'real_volume', 'price_volume_ratio',
        
        'mid_diff1', 
        'trade_impact', 'signed_amount', 'price_up_amount_down', 'amount_price_div', 
        'trade_sign', 'aggression_eff', 'price_control', 'close_pos',
    ]

    # 高阶特征
    # lags2 = [1, 2, 5]
    lags2 = [1]
    raw_cols2 = [
        # 'mid_diff1', 
        # 'trade_impact', 'signed_amount', 'price_up_amount_down', 'amount_price_div', 
    ]

    # 高阶特征
    lags3 = [1]
    raw_cols3 = [
        "sym", "time_label", "time", "time_interval",
        'mid_price_ma5', 'mid_price_ma20', 'mid_price_ma50', 'mid_price_ma100',
        'mid_price_std5', 'mid_price_std20', 'mid_price_std50', 'mid_price_std100',
        "mid_price_vol_ratio5", "mid_price_vol_ratio20", "mid_price_vol_ratio50", "mid_price_vol_ratio100", 
        'mid_diff1_ma5', 'mid_diff1_ma20', 'mid_diff1_ma50', 'mid_diff1_ma100',
        'mid_diff1_std5', 'mid_diff1_std20', 'mid_diff1_std50', 'mid_diff1_std100',
        "mid_diff1_vol_ratio5", "mid_diff1_vol_ratio20", "mid_diff1_vol_ratio50", "mid_diff1_vol_ratio100", 
        'mid_diff1_max5', 'mid_diff1_max20', 'mid_diff1_max50', 'mid_diff1_max100',
        'mid_diff1_min5', 'mid_diff1_min20', 'mid_diff1_min50', 'mid_diff1_min100',
        'mid_diff1_max_min5', 'mid_diff1_max_min20', 'mid_diff1_max_min50', 'mid_diff1_max_min100',
        'mid_price_max5', 'mid_price_max20', 'mid_price_max50', 'mid_price_max100',
        'mid_price_min5', 'mid_price_min20', 'mid_price_min50', 'mid_price_min100',
        'mid_price_max_min5', 'mid_price_max_min20', 'mid_price_max_min50', 'mid_price_max_min100',
        'pos_5', 'pos_20', 'pos_50', 'pos_100', 
        'real_volume_ma5', 'real_volume_ma20', 'real_volume_ma50', 'real_volume_ma100', 
        'real_volume_std5', 'real_volume_std20', 'real_volume_std50', 'real_volume_std100', 
        'real_volume_vol_ratio5', 'real_volume_vol_ratio20', 'real_volume_vol_ratio50', 'real_volume_vol_ratio100', 
        'real_volume_max5', 'real_volume_max20', 'real_volume_max50', 'real_volume_max100', 
        'real_volume_min5', 'real_volume_min20', 'real_volume_min50', 'real_volume_min100', 
        'real_volume_max_min5', 'real_volume_max_min20', 'real_volume_max_min50', 'real_volume_max_min100', 
        'real_volume_pos_5', 'real_volume_pos_20', 'real_volume_pos_50', 'real_volume_pos_100', 
        # 'ask1_ma5', 'ask1_ma10', 'ask1_ma20', 'ask1_ma40', 
        # 'bid1_ma5', 'bid1_ma10', 'bid1_ma20', 'bid1_ma40', 
        'mid_diff1_lr_k', 'mid_diff1_lr_r2', 'mid_diff1_trend_strength',
        "mid_diff1_trend_regime", "mid_diff1_trend_strength_gated",
        "mid_diff1_price_move_capacity", "mid_diff1_trend_liquidity_ratio", 
        'trend_regime_10', 'trend_regime_30', 'trend_regime_60', 
        'trend_strength_gated_10', 'trend_strength_gated_30', 'trend_strength_gated_60', 
        'price_move_capacity_10', 'price_move_capacity_30', 'price_move_capacity_60', 
        'trend_liquidity_ratio_10', 'trend_liquidity_ratio_30', 'trend_liquidity_ratio_60',
        'trend_align_10_30', 'trend_align_30_60', 'trend_align_10_60', 
        'trend_align_amount_10_30', 'trend_align_amount_30_60', 'trend_align_amount_10_60', 
        'trend_confidence',
        'mid_lr_k_10', 'mid_lr_r2_10', 'mid_trend_strength_10', 
        'mid_lr_k_30', 'mid_lr_r2_30', 'mid_trend_strength_30', 
        'mid_lr_k_60', 'mid_lr_r2_60', 'mid_trend_strength_60',
        # # 'bid1_decay', 'ask1_decay', 'spread_decay', 'bsize1_decay', 'asize1_decay',
        "close_delta1", "amount_delta1", "close_delta10", "amount_delta10", "close_delta40", "amount_delta40", 
        "midprice_delta1", "midprice_delta10", "midprice_delta40",
        "midprice_delta_rate1", "midprice_delta_rate10", "midprice_delta_rate40", 
        'ask1_delta1', 'ask3_delta1', 'ask5_delta1',
        'bid1_delta1', 'bid3_delta1', 'bid5_delta1',
        'ask1_delta10', 'ask3_delta10', 'ask5_delta10',
        'bid1_delta10', 'bid3_delta10', 'bid5_delta10',
        'ask1_delta40', 'ask3_delta40', 'ask5_delta40',
        'bid1_delta40', 'bid3_delta40', 'bid5_delta40',
        'asize1_delta1', 'asize3_delta1', 'asize5_delta1',
        'bsize1_delta1', 'bsize3_delta1', 'bsize5_delta1',
        'asize1_delta10', 'asize3_delta10', 'asize5_delta10',
        'bsize1_delta10', 'bsize3_delta10', 'bsize5_delta10',
        'asize1_delta40', 'asize3_delta40', 'asize5_delta40',
        'bsize1_delta40', 'bsize3_delta40', 'bsize5_delta40',
        'real_volume_delta1', 'real_volume_delta10', 'real_volume_delta40',
        'original_price', 'sym_class',
        "mid_diff1_sum", "mid_diff1_open_close", "SignedTrendEff", 
        "real_volume_lr_k", "real_volume_lr_r2", "real_volume_trend_strength", 
        "real_volume_price_move_capacity", "real_volume_trend_liquidity_ratio", 
        "real_volume_trend_regime", "real_volume_trend_strength_gated", 
        "mid_diff1_has_spike5", "mid_diff1_has_spike20", "mid_diff1_has_spike50", "mid_diff1_has_spike100", 
        "mid_diff1_is_flat5", "mid_diff1_is_flat20", "mid_diff1_is_flat50", "mid_diff1_is_flat100", 
        "mid_diff1_is_high_vol5", "mid_diff1_is_high_vol20", "mid_diff1_is_high_vol50", "mid_diff1_is_high_vol100", 
        "sign_consistency5", "sign_consistency20", "sign_consistency50", "sign_consistency100", 
        "trend_persist5", "trend_persist20", "trend_persist50", "trend_persist100", 
        "mid_diff1_vol_ratio_sl_5_50", "real_volume_vol_ratio_sl_5_50", 
        "mid_diff1_vol_ratio_sl_20_100", "real_volume_vol_ratio_sl_20_100",         # 基于n_close成交方向的窗口特征
        'trade_sign_sum5', 'trade_sign_sum20', 'trade_sign_sum50', 'trade_sign_sum100', 
        'trade_sign_mean5', 'trade_sign_mean20', 'trade_sign_mean50', 'trade_sign_mean100', 
        'trade_sign_aggression_eff5', 'trade_sign_aggression_eff20', 'trade_sign_aggression_eff50', 'trade_sign_aggression_eff100', 
        'trade_sign_buy_run5', 'trade_sign_buy_run20', 'trade_sign_buy_run50', 'trade_sign_buy_run100', 
        'trade_sign_sell_run5', 'trade_sign_sell_run20', 'trade_sign_sell_run50', 'trade_sign_sell_run100', 
        'trade_sign_aggression_skew5', 'trade_sign_aggression_skew20', 'trade_sign_aggression_skew50', 'trade_sign_aggression_skew100', 
        'trade_sign_price_control5', 'trade_sign_price_control20', 'trade_sign_price_control50', 'trade_sign_price_control100', 
        'trade_sign_close_pressure5', 'trade_sign_close_pressure20', 'trade_sign_close_pressure50', 'trade_sign_close_pressure100',
        'aggression_score5', 'aggression_score20', 'aggression_score50', 'aggression_score100',
    ]
    
    if isinstance(x, pd.DataFrame):
        x = [x]

    if is_train: # 训练时需要返回标签
        labels = []
        for df in x:
            labels.append(df['label_20'][99:])
        label = pd.concat(labels, axis=0).reset_index(drop=True)

        profit = np.concatenate([(df['n_midprice'].shift(-20) - df['n_midprice']).fillna(0).values for df in x], axis=0).astype(np.float32)
        profit = np.ascontiguousarray(profit[99:]) # 现在如果买入，N步后盈利多少

    # 带进度条
    for i, df in tqdm(
        enumerate(x),
        total=len(x),
        desc="Preprocess",
        ncols=100
    ):
    # for i, df in enumerate(x):
        extra_feats = {}

        # 收盘价还原
        extra_feats['close'] = df['n_close'] + 1

        # 中间价还原
        extra_feats['mid_price'] = df['n_midprice'] + 1

        # 基础价格还原
        for i in range(1, 6):
            extra_feats[f'bid{i}'] = df[f'n_bid{i}'] + 1
            extra_feats[f'ask{i}'] = df[f'n_ask{i}'] + 1

        # 基础量价组合
        for i in range(1, 4):
            extra_feats[f'spread{i}'] = extra_feats[f'ask{i}'] - extra_feats[f'bid{i}']
            extra_feats[f'mid_price{i}'] = (extra_feats[f'ask{i}'] + extra_feats[f'bid{i}']) / 2

        # 一阶差分
        extra_feats['mid_diff1'] = extra_feats['mid_price'].diff().fillna(0)

        # 最近 100 tick 的 |diff| 之和（路径长度）
        extra_feats['mid_diff1_sum'] = (
            extra_feats['mid_diff1']
            .abs()
            .rolling(window=100, min_periods=1)
            .sum()
        )

        # 最近 100 tick 的 diff 之和（起点到终点的净变化）
        extra_feats['mid_diff1_open_close'] = (
            extra_feats['mid_diff1']
            .rolling(window=100, min_periods=1)
            .sum()
        )

        # Signed Trend Efficiency
        extra_feats['SignedTrendEff'] = (
            extra_feats['mid_diff1_open_close'] / (extra_feats['mid_diff1_sum'] + 1e-5)
        )

        # # 还原原始股价
        sym_to_price = {
            0: 1300,
            1: 88.4,
            2: 1004.6,
            3: 61,
            4: 444.4,
            5: 302.4,
            6: 535.2,
            7: 323.2,
            8: 395.4,
            9: 1740,
        }
        extra_feats['original_price'] = df['sym'].map(sym_to_price)

        # 股价分类
        def price_to_class(price):
            if price < 100:
                return 0
            elif price < 699:
                return 1
            elif price < 1500:
                return 2
            else:
                return 3

        extra_feats['sym_class'] = extra_feats['original_price'].map(price_to_class)

        # 对数盘口量 & 成交量
        for i in range(1, 6):
            extra_feats[f'bsize{i}'] = np.log1p(df[f'n_bsize{i}'])
            extra_feats[f'asize{i}'] = np.log1p(df[f'n_asize{i}'])
        extra_feats['amount'] = np.log1p(df['amount_delta'])
        extra_feats['amount_midprice'] = np.log1p(df['amount_delta'] / extra_feats['mid_price'])

        # 相对价差
        extra_feats.update({
            'relative_spread1': extra_feats['spread1'] / extra_feats['mid_price1'],
            'relative_spread2': extra_feats['spread2'] / extra_feats['mid_price2'],
            'relative_spread3': extra_feats['spread3'] / extra_feats['mid_price3'],
        })

        # 买卖单相对密度
        extra_feats.update({
            'relative_bid_density1': df['n_bsize1'] / (df['n_bsize1'] + df['n_asize1']),
            'relative_bid_density2': df['n_bsize2'] / (df['n_bsize2'] + df['n_asize2']),
            'relative_bid_density3': df['n_bsize3'] / (df['n_bsize3'] + df['n_asize3']),

            'relative_ask_density1': df['n_asize1'] / (df['n_bsize1'] + df['n_asize1']),
            'relative_ask_density2': df['n_asize2'] / (df['n_bsize2'] + df['n_asize2']),
            'relative_ask_density3': df['n_asize3'] / (df['n_bsize3'] + df['n_asize3']),
        })

        # 买卖量差
        extra_feats.update({
            'vol1_rel_diff': (df['n_bsize1'] - df['n_asize1']) / (df['n_bsize1'] + df['n_asize1']),
            'vol3_rel_diff': (df['n_bsize1'] + df['n_bsize2'] + df['n_bsize3'] - df['n_asize1'] - df['n_asize2'] - df['n_asize3']) / (df['n_bsize1'] + df['n_bsize2'] + df['n_bsize3'] + df['n_asize1'] + df['n_asize2'] + df['n_asize3']),
            'vol5_rel_diff': (df['n_bsize1'] + df['n_bsize2'] + df['n_bsize3'] + df['n_bsize4'] + df['n_bsize5'] - df['n_asize1'] - df['n_asize2'] - df['n_asize3'] - df['n_asize4'] - df['n_asize5']) / (df['n_bsize1'] + df['n_bsize2'] + df['n_bsize3'] + df['n_bsize4'] + df['n_bsize5'] + df['n_asize1'] + df['n_asize2'] + df['n_asize3'] + df['n_asize4'] + df['n_asize5']),
        })

        # 真实成交量特征
        extra_feats['real_volume'] = np.log1p(df['amount_delta'] / (extra_feats['mid_price'] + 1e-10))
        # 成交量与价格的比值（反映每单位价格变化的成交量变化）
        extra_feats['price_volume_ratio'] = extra_feats['mid_price'] / (extra_feats['real_volume'] + 1e-10)
        # # 成交量的前向差分（反映成交量的变化趋势）
        # extra_feats['real_volume_diff1'] = extra_feats['real_volume'].diff().fillna(0)

        mid_diff1_spike_th = 0.0008
        mid_diff1_threshold_1 = 10
        mid_diff1_threshold_2 = 50

        # 均线特征
        for w in [5, 20, 50, 100]:
            # extra_feats[f'ask1_ma{w}'] = extra_feats['ask1'].rolling(w, min_periods=1).mean()
            # extra_feats[f'bid1_ma{w}'] = extra_feats['bid1'].rolling(w, min_periods=1).mean()
            extra_feats[f'mid_price_ma{w}'] = extra_feats['mid_price'].rolling(w, min_periods=1).mean()
            extra_feats[f'mid_price_std{w}'] = extra_feats['mid_price'].rolling(w, min_periods=1).std()
            extra_feats[f'mid_price_vol_ratio{w}'] = extra_feats[f'mid_price_std{w}'] / (np.abs(extra_feats[f'mid_price_ma{w}']) + 1e-10)

            extra_feats[f'mid_diff1_ma{w}'] = extra_feats['mid_diff1'].rolling(w, min_periods=1).mean()
            extra_feats[f'mid_diff1_std{w}'] = extra_feats['mid_diff1'].rolling(w, min_periods=1).std()
            extra_feats[f'mid_diff1_vol_ratio{w}'] = extra_feats[f'mid_diff1_std{w}'] / (np.abs(extra_feats[f'mid_diff1_ma{w}']) + 1e-10)

            extra_feats[f'mid_diff1_max{w}'] = extra_feats['mid_diff1'].rolling(w, min_periods=1).max()
            extra_feats[f'mid_diff1_min{w}'] = extra_feats['mid_diff1'].rolling(w, min_periods=1).min()
            extra_feats[f'mid_diff1_max_min{w}'] = extra_feats[f'mid_diff1_max{w}'] - extra_feats[f'mid_diff1_min{w}']
       
            extra_feats[f'mid_price_max{w}'] = extra_feats['mid_price'].rolling(w, min_periods=1).max()
            extra_feats[f'mid_price_min{w}'] = extra_feats['mid_price'].rolling(w, min_periods=1).min()
            extra_feats[f'mid_price_max_min{w}'] = extra_feats[f'mid_price_max{w}'] - extra_feats[f'mid_price_min{w}']
            extra_feats[f'pos_{w}'] = (extra_feats['mid_price'] -  extra_feats[f'mid_price_min{w}']) / (extra_feats[f'mid_price_max{w}'] -  extra_feats[f'mid_price_min{w}'] + 1e-10)
                
            extra_feats[f'real_volume_ma{w}'] = extra_feats['real_volume'].rolling(w, min_periods=1).mean()
            extra_feats[f'real_volume_std{w}'] = extra_feats['real_volume'].rolling(w, min_periods=1).std()
            extra_feats[f'real_volume_vol_ratio{w}'] = extra_feats[f'real_volume_std{w}'] / (np.abs(extra_feats[f'real_volume_ma{w}']) + 1e-10)
    
            extra_feats[f'real_volume_max{w}'] = extra_feats['real_volume'].rolling(w, min_periods=1).max()
            extra_feats[f'real_volume_min{w}'] = extra_feats['real_volume'].rolling(w, min_periods=1).min()
            extra_feats[f'real_volume_max_min{w}'] = extra_feats[f'real_volume_max{w}'] - extra_feats[f'real_volume_min{w}']
            extra_feats[f'real_volume_pos_{w}'] = (extra_feats['real_volume'] -  extra_feats[f'real_volume_min{w}']) / (extra_feats[f'real_volume_max{w}'] -  extra_feats[f'real_volume_min{w}'] + 1e-10)

            # 窗口内是否曾出现强冲击
            extra_feats[f'mid_diff1_has_spike{w}'] = (
                (extra_feats[f'mid_diff1_max{w}'] > mid_diff1_spike_th)
                |
                (-extra_feats[f'mid_diff1_min{w}'] > mid_diff1_spike_th)
            ).astype(int)
            # 窗口内是否出现 高震荡or平台
            extra_feats[f'mid_diff1_is_flat{w}'] = (
                (extra_feats[f'mid_diff1_vol_ratio{w}'] < mid_diff1_threshold_1)
            ).astype(int)
            extra_feats[f'mid_diff1_is_high_vol{w}'] = (
                (extra_feats[f'mid_diff1_vol_ratio{w}'] > mid_diff1_threshold_2)
            ).astype(int)

            # _20（方向一致性）
            sign_series = np.sign(extra_feats['mid_diff1'])
            extra_feats[f'sign_consistency{w}'] = (
                sign_series
                .rolling(w, min_periods=1)
                .mean()
                .abs()
            )

            # trend_persist_20（趋势持续性）
            extra_feats[f'trend_persist{w}'] = (
                sign_series
                .rolling(w, min_periods=1)
                .apply(
                    lambda x: np.sum(x == x.iloc[-1]),
                    raw=False
                )
            )

        # 长短窗口特征对比sign_consistency
        extra_feats[f'mid_diff1_vol_ratio_sl_5_50'] = extra_feats[f'mid_diff1_vol_ratio5'] / (extra_feats[f'mid_diff1_vol_ratio50'] + 1e-6)
        extra_feats[f'real_volume_vol_ratio_sl_5_50'] = extra_feats[f'real_volume_vol_ratio5'] / (extra_feats[f'real_volume_vol_ratio50'] + 1e-6)
        extra_feats[f'mid_diff1_vol_ratio_sl_20_100'] = extra_feats[f'mid_diff1_vol_ratio20'] / (extra_feats[f'mid_diff1_vol_ratio100'] + 1e-6)
        extra_feats[f'real_volume_vol_ratio_sl_20_100'] = extra_feats[f'real_volume_vol_ratio20'] / (extra_feats[f'real_volume_vol_ratio100'] + 1e-6)

        df['trade_sign'] = np.where(
            df['n_close'] == df['n_ask1'],  1,
            np.where(df['n_close'] == df['n_bid1'], -1, 0)
        )

        df['aggression_eff'] = df['trade_sign'] * np.sign(extra_feats['mid_diff1'])
        df['price_control'] = df['trade_sign'] * extra_feats['mid_diff1']

        df['close_pos'] = (
            (df['n_close'] - df['n_midprice'])
            / ((df['n_ask1'] - df['n_bid1']) / 2 + 1e-10)
        )
        
        # 基于n_close成交方向的窗口特征
        for w in [5, 20, 50, 100]:
            extra_feats[f'trade_sign_sum{w}'] = df['trade_sign'].rolling(w, min_periods=1).sum()
            extra_feats[f'trade_sign_mean{w}'] = df['trade_sign'].rolling(w, min_periods=1).mean()
            extra_feats[f'trade_sign_aggression_eff{w}'] = df['aggression_eff'].rolling(w, min_periods=1).mean()

            # 成交持续性
            extra_feats[f'trade_sign_buy_run{w}'] = ((df['trade_sign'] == 1).rolling(w, min_periods=1)).sum()
            extra_feats[f'trade_sign_sell_run{w}'] = ((df['trade_sign'] == -1).rolling(w, min_periods=1)).sum()

            # 势能不对称
            extra_feats[f'trade_sign_aggression_skew{w}'] = (
                (extra_feats[f'trade_sign_buy_run{w}'] - extra_feats[f'trade_sign_sell_run{w}'])
                / (w + 1e-10)
            )

            # 主动成交推动率
            extra_feats[f'trade_sign_price_control{w}'] = (df['price_control'] == -1).rolling(w, min_periods=1).mean()

            # rolling 成交压强
            extra_feats[f'trade_sign_close_pressure{w}'] = (df['close_pos'] == -1).rolling(w, min_periods=1).mean()

            # 势能组合指数
            extra_feats[f'aggression_score{w}'] = (
                0.4 * extra_feats[f'trade_sign_aggression_skew{w}']
                + 0.3 * extra_feats[f'trade_sign_close_pressure{w}']
                + 0.3 * extra_feats[f'trade_sign_price_control{w}']
            )

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

        # 前向差分特征
        for w in [1, 10, 40]:
            extra_feats.update({
                f'midprice_delta_rate{w}': extra_feats['mid_price'] / extra_feats['mid_price'].shift(w) - 1, 
                f'midprice_delta{w}': extra_feats['mid_price'] - extra_feats['mid_price'].shift(w), 
                f'close_delta{w}': df['n_close'] - df['n_close'].shift(w), 
                f'amount_delta{w}': extra_feats['amount'] - extra_feats['amount'].shift(w),
                f'real_volume_delta{w}': extra_feats['real_volume'] - extra_feats["real_volume"].shift(w)
            })
            for i in [1, 3, 5]:
                extra_feats.update({
                    f'ask{i}_delta{w}': df[f'n_ask{i}'] - df[f'n_ask{i}'].shift(w),
                    f'bid{i}_delta{w}': df[f'n_bid{i}'] - df[f'n_bid{i}'].shift(w),
                    f'asize{i}_delta{w}': extra_feats[f'asize{i}'] - extra_feats[f'asize{i}'].shift(w),
                    f'bsize{i}_delta{w}': extra_feats[f'bsize{i}'] - extra_feats[f'bsize{i}'].shift(w),
                })

        # 价差变化
        extra_feats.update({
            'spread1_diff1': extra_feats['spread1'].diff().fillna(0),
            # 'spread1_diff2': extra_feats['spread1'].diff().diff().fillna(0),

            'spread2_diff1': extra_feats['spread2'].diff().fillna(0),
            # 'spread2_diff2': extra_feats['spread2'].diff().diff().fillna(0),

            'spread3_diff1': extra_feats['spread3'].diff().fillna(0),
            # 'spread3_diff2': extra_feats['spread3'].diff().diff().fillna(0),

            'relative_spread1_diff1': extra_feats['relative_spread1'].diff().fillna(0),
            # 'relative_spread1_diff2': extra_feats['relative_spread1'].diff().diff().fillna(0),

            'relative_spread2_diff1': extra_feats['relative_spread2'].diff().fillna(0),
            # 'relative_spread2_diff2': extra_feats['relative_spread2'].diff().diff().fillna(0),

            'relative_spread3_diff1': extra_feats['relative_spread3'].diff().fillna(0),
            # 'relative_spread3_diff2': extra_feats['relative_spread3'].diff().diff().fillna(0),
        })

        # 一次性合并
        df = pd.concat([df, pd.DataFrame(extra_feats, index=df.index)], axis=1)

        # 时间标签
        df['time_label'] = assign_tick_time_labels(df['time'])

        # time
        # 将 HH:MM 转换为以秒为单位的数值
        df['time'] = df['time'].apply(
            lambda x: int(x.split(':')[0]) * 3600 + int(x.split(':')[1]) * 60 + int(x.split(':')[2])
        )

        # 时间分段（每 30 分钟一档）
        df['time_interval'] = df['time'].apply(
            lambda x:
                0 if x < 37800 else   # 09:30 - 10:30
                1 if x < 39600 else   # 10:30 - 11:00
                2 if x < 41400 else   # 11:00 - 11:30
                3 if x < 46800 else   # 13:00 - 13:30
                4 if x < 48600 else   # 13:30 - 14:00
                5 if x < 50400 else   # 14:00 - 14:30
                6 if x < 52200 else   # 14:30 - 15:00
                7                  # 其他时间
        )

        k_10, r2_10 = rolling_lr_features(df['mid_price'].to_numpy(), window=10)
        k_30, r2_30 = rolling_lr_features(df['mid_price'].to_numpy(), window=30)
        k_60, r2_60 = rolling_lr_features(df['mid_price'].to_numpy(), window=60)

        df['trend_align_10_30'] = (np.sign(k_10) == np.sign(k_30)).astype(int)
        df['trend_align_30_60'] = (np.sign(k_30) == np.sign(k_60)).astype(int)
        df['trend_align_10_60'] = (np.sign(k_10) == np.sign(k_60)).astype(int)

        df['trend_align_amount_10_30'] = (np.sign(k_10) != np.sign(k_30)).astype(int) * abs(k_10 - k_30)
        df['trend_align_amount_30_60'] = (np.sign(k_30) != np.sign(k_60)).astype(int) * abs(k_30 - k_60)
        df['trend_align_amount_10_60'] = (np.sign(k_10) != np.sign(k_60)).astype(int) * abs(k_10 - k_60)
        df['trend_confidence'] = (
            np.sign(k_10) * r2_10 +
            np.sign(k_30) * r2_30 +
            np.sign(k_60) * r2_60
        )

        df['mid_lr_k_10'] = k_10   # 斜率
        df['mid_lr_r2_10'] = r2_10   # 拟合优度
        df['mid_trend_strength_10'] = np.sign(k_10) * r2_10

        df['mid_lr_k_30'] = k_30   # 斜率
        df['mid_lr_r2_30'] = r2_30   # 拟合优度
        df['mid_trend_strength_30'] = np.sign(k_30) * r2_30
        
        df['mid_lr_k_60'] = k_60   # 斜率
        df['mid_lr_r2_60'] = r2_60   # 拟合优度
        df['mid_trend_strength_60'] = np.sign(k_60) * r2_60

        # 盘口是否允许价格往某个方向动？
        df['price_move_capacity_10'] = k_10 / (df['relative_spread1'] + 1e-10)
        df['trend_liquidity_ratio_10'] = df['mid_trend_strength_10'] / (df['relative_spread1'] + 1e-10)
        df['price_move_capacity_30'] = k_30 / (df['relative_spread1'] + 1e-10)
        df['trend_liquidity_ratio_30'] = df['mid_trend_strength_30'] / (df['relative_spread1'] + 1e-10)
        df['price_move_capacity_60'] = k_60 / (df['relative_spread1'] + 1e-10)
        df['trend_liquidity_ratio_60'] = df['mid_trend_strength_60'] / (df['relative_spread1'] + 1e-10)
        
        # 门控趋势特征：趋势筛选 + 趋势强度
        k_threshold = 0.00003
        r2_threshold = 0.4
        df['trend_regime_10'] = (
            (r2_10 > r2_threshold) &
            (np.abs(k_10) > k_threshold)
        ).astype(int)
        df['trend_strength_gated_10'] = df['mid_trend_strength_10'] * df['trend_regime_10']
        df['trend_regime_30'] = (
            (r2_30 > r2_threshold) &
            (np.abs(k_30) > k_threshold)
        ).astype(int)
        df['trend_strength_gated_30'] = df['mid_trend_strength_30'] * df['trend_regime_30']
        df['trend_regime_60'] = (
            (r2_60 > r2_threshold) &
            (np.abs(k_60) > k_threshold)
        ).astype(int)
        df['trend_strength_gated_60'] = df['mid_trend_strength_60'] * df['trend_regime_60']
        
        # 盘口不平衡
        df['obi_1'] = (df['bsize1'] - df['asize1']) / (df['bsize1'] + df['asize1'] + 1e-10)
        df['obi_3'] = (
            df['bsize1'] + df['bsize2'] + df['bsize3']
            - df['asize1'] - df['asize2'] - df['asize3']
        ) / (
            df['bsize1'] + df['bsize2'] + df['bsize3']
            + df['asize1'] + df['asize2'] + df['asize3'] + 1e-10
        )

        # 盘口不对称强度（引入非线性信号给树模型试试）
        df['obi_sq'] = df['obi_3'] * np.abs(df['obi_3'])

        # 盘口斜率
        df['bid_depth_slope'] = (df['bsize5'] - df['bsize1']) / 4
        df['ask_depth_slope'] = (df['asize5'] - df['asize1']) / 4

        # mid_price 动量
        df['mid_diff1'] = df['mid_price'].diff().fillna(0)   # 一阶差分：速度
        # df['mid_diff2'] = df['mid_diff1'].diff().fillna(0)   # 二阶差分：加速度
        
        # mid_diff1线性回归
        k, r2 = rolling_lr_features(df['mid_diff1'].to_numpy(), window=5)
        df['mid_diff1_lr_k'] = k   # 斜率
        df['mid_diff1_lr_r2'] = r2   # 拟合优度
        df['mid_diff1_trend_strength'] = np.sign(k) * r2

        df['mid_diff1_price_move_capacity'] = k / (df[f'relative_spread1'] + 1e-10)
        df['mid_diff1_trend_liquidity_ratio'] = np.sign(k) * r2 / (df[f'relative_spread1'] + 1e-10)

        df['mid_diff1_trend_regime'] = (
            (r2 > r2_threshold) &
            (np.abs(k) > k_threshold)
        ).astype(int)
        df['mid_diff1_trend_strength_gated'] = df['mid_diff1_trend_strength'] * df['mid_diff1_trend_regime']

        extra_feats = {}
        # 价格冲击方向
        extra_feats['trade_impact'] = df['mid_diff1'] * df['amount']
        extra_feats['signed_amount'] = np.sign(df['mid_diff1']) * df['amount']
        # 量价背离
        extra_feats['price_up_amount_down'] = (
            (df['mid_diff1'] > 0).astype(int) * (df['amount'].diff() < 0).astype(int)
        )
        extra_feats['amount_price_div'] = df['mid_diff1'] / (df['amount'] + 1e-10)
        
        # real_volume线性回归
        k, r2 = rolling_lr_features(df['real_volume'].to_numpy(), window=5)
        extra_feats['real_volume_lr_k'] = k   # 斜率
        extra_feats['real_volume_lr_r2'] = r2   # 拟合优度
        
        extra_feats[f'real_volume_trend_strength'] = np.sign(k) * r2

        extra_feats[f'real_volume_price_move_capacity'] = k / (df['relative_spread1'] + 1e-10)
        extra_feats[f'real_volume_trend_liquidity_ratio'] = np.sign(k) * r2 / (df['relative_spread1'] + 1e-10)

        extra_feats[f'real_volume_trend_regime'] = (
            (r2 > r2_threshold) &
            (np.abs(k) > k_threshold)
        ).astype(int)
        extra_feats[f'real_volume_trend_strength_gated'] = extra_feats[f'real_volume_trend_strength'] * extra_feats[f'real_volume_trend_regime']
        
        df = pd.concat([df, pd.DataFrame(extra_feats, index=df.index)], axis=1)

        # 时间衰减采样历史数据
        df = time_fixed_sample(df, raw_cols1, lags=lags1)
        df = df.copy()

        df = time_fixed_sample(df, raw_cols2, lags=lags2)
        df = df.copy()

        df = time_fixed_sample(df, raw_cols3, lags=lags3)
        df = df.copy()

        lag_cols = [f'{c}_lag{lag}' for c in raw_cols1 for lag in lags1] \
                 + [f'{c}_lag{lag}' for c in raw_cols2 for lag in lags2] \
                 + [f'{c}_lag{lag}' for c in raw_cols3 for lag in lags3]
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

    # 对列的标签进行排序
    concat_df = concat_df[sorted(concat_df.columns)]

    if is_train:
        return concat_df, label, profit
    else:
        return concat_df
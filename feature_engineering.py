import numpy as np
import pandas as pd

def feature_extractor(new_df: pd.DataFrame) -> pd.DataFrame:
    if 'amount' in new_df.columns:   # Hack
        return new_df

    if 'ask5' not in new_df.columns:
        # 价格+1（从涨跌幅还原到对前收盘价的比例）
        new_df['bid1'] = new_df['n_bid1']+1
        new_df['bid2'] = new_df['n_bid2']+1
        new_df['bid3'] = new_df['n_bid3']+1
        new_df['bid4'] = new_df['n_bid4']+1
        new_df['bid5'] = new_df['n_bid5']+1
        new_df['ask1'] = new_df['n_ask1']+1
        new_df['ask2'] = new_df['n_ask2']+1
        new_df['ask3'] = new_df['n_ask3']+1
        new_df['ask4'] = new_df['n_ask4']+1
        new_df['ask5'] = new_df['n_ask5']+1

    if 'bid1_ma100' not in new_df.columns:
        # 均线特征
        new_df['ask1_ma5']  = new_df['ask1'].rolling(window=5,  min_periods=1).mean()
        new_df['ask1_ma10'] = new_df['ask1'].rolling(window=10, min_periods=1).mean()
        new_df['ask1_ma20'] = new_df['ask1'].rolling(window=20, min_periods=1).mean()
        new_df['ask1_ma40'] = new_df['ask1'].rolling(window=40, min_periods=1).mean()
        new_df['ask1_ma60'] = new_df['ask1'].rolling(window=60, min_periods=1).mean()
        new_df['ask1_ma80'] = new_df['ask1'].rolling(window=80, min_periods=1).mean()
        new_df['ask1_ma100'] = new_df['ask1'].rolling(window=100, min_periods=1).mean()
        new_df['bid1_ma5']  = new_df['bid1'].rolling(window=5,  min_periods=1).mean()
        new_df['bid1_ma10'] = new_df['bid1'].rolling(window=10, min_periods=1).mean()
        new_df['bid1_ma20'] = new_df['bid1'].rolling(window=20, min_periods=1).mean()
        new_df['bid1_ma40'] = new_df['bid1'].rolling(window=40, min_periods=1).mean()
        new_df['bid1_ma60'] = new_df['bid1'].rolling(window=60, min_periods=1).mean()
        new_df['bid1_ma80'] = new_df['bid1'].rolling(window=80, min_periods=1).mean()
        new_df['bid1_ma100'] = new_df['bid1'].rolling(window=100, min_periods=1).mean()

    if 'relative_spread3' not in new_df.columns:
        # 量价组合
        new_df['spread1'] =  new_df['ask1'] - new_df['bid1']
        new_df['spread2'] =  new_df['ask2'] - new_df['bid2']
        new_df['spread3'] =  new_df['ask3'] - new_df['bid3']
        new_df['mid_price1'] =  new_df['ask1'] + new_df['bid1']
        new_df['mid_price2'] =  new_df['ask2'] + new_df['bid2']
        new_df['mid_price3'] =  new_df['ask3'] + new_df['bid3']
        new_df['weighted_ab1'] = (new_df['ask1'] * new_df['n_bsize1'] + new_df['bid1'] * new_df['n_asize1']) / (new_df['n_bsize1'] + new_df['n_asize1'])
        new_df['weighted_ab2'] = (new_df['ask2'] * new_df['n_bsize2'] + new_df['bid2'] * new_df['n_asize2']) / (new_df['n_bsize2'] + new_df['n_asize2'])
        new_df['weighted_ab3'] = (new_df['ask3'] * new_df['n_bsize3'] + new_df['bid3'] * new_df['n_asize3']) / (new_df['n_bsize3'] + new_df['n_asize3'])

        new_df['relative_spread1'] = new_df['spread1'] / new_df['mid_price1']
        new_df['relative_spread2'] = new_df['spread2'] / new_df['mid_price2']
        new_df['relative_spread3'] = new_df['spread3'] / new_df['mid_price3']

    if 'amount' not in new_df.columns:
        # 对量取对数
        new_df['bsize1'] = new_df['n_bsize1'].map(np.log)
        new_df['bsize2'] = new_df['n_bsize2'].map(np.log)
        new_df['bsize3'] = new_df['n_bsize3'].map(np.log)
        new_df['bsize4'] = new_df['n_bsize4'].map(np.log)
        new_df['bsize5'] = new_df['n_bsize5'].map(np.log)
        new_df['asize1'] = new_df['n_asize1'].map(np.log)
        new_df['asize2'] = new_df['n_asize2'].map(np.log)
        new_df['asize3'] = new_df['n_asize3'].map(np.log)
        new_df['asize4'] = new_df['n_asize4'].map(np.log)
        new_df['asize5'] = new_df['n_asize5'].map(np.log)
        new_df['amount'] = new_df['amount_delta'].map(np.log1p)

    # ===== 基于100tick窗口优化的量化因子 =====

    # 1. 订单流不平衡因子 (Order Flow Imbalance - OFI) - 针对短窗口优化
    if 'ofi' not in new_df.columns:
        new_df['ofi'] = compute_ofi(new_df)
        new_df['ofi_ma5'] = new_df['ofi'].rolling(window=5, min_periods=1).mean()
        new_df['ofi_ma20'] = new_df['ofi'].rolling(window=20, min_periods=1).mean()
        new_df['ofi_ma50'] = new_df['ofi'].rolling(window=50, min_periods=1).mean()

        # OFI趋势 (短期预测性强)
        new_df['ofi_trend_10'] = new_df['ofi_ma20'] - new_df['ofi_ma10'] if 'ofi_ma10' in new_df.columns else 0

    # 2. 盘口不平衡因子 (Order Book Imbalance) - 短窗口重点
    if 'obi1' not in new_df.columns:
        new_df['obi1'] = (new_df['n_bsize1'] - new_df['n_asize1']) / (new_df['n_bsize1'] + new_df['n_asize1'] + 1e-6)
        new_df['obi3'] = ((new_df['n_bsize1'] + new_df['n_bsize2'] + new_df['n_bsize3']) -
                         (new_df['n_asize1'] + new_df['n_asize2'] + new_df['n_asize3'])) / \
                        ((new_df['n_bsize1'] + new_df['n_bsize2'] + new_df['n_bsize3']) +
                         (new_df['n_asize1'] + new_df['n_asize2'] + new_df['n_asize3']) + 1e-6)

        # 短期平滑 (适合tick级别预测)
        new_df['obi1_ma10'] = new_df['obi1'].rolling(window=10, min_periods=1).mean()
        new_df['obi3_ma10'] = new_df['obi3'].rolling(window=10, min_periods=1).mean()
        new_df['obi1_ma30'] = new_df['obi1'].rolling(window=30, min_periods=1).mean()

    # 3. 微观结构因子 (Microstructure Features) - 高频重点
    if 'microprice' not in new_df.columns:
        new_df['microprice'] = (new_df['bid1'] * new_df['n_asize1'] + new_df['ask1'] * new_df['n_bsize1']) / \
                              (new_df['n_bsize1'] + new_df['n_asize1'] + 1e-6)

        # 微观价格变化 (直接预测因子)
        new_df['microprice_change'] = new_df['microprice'].diff()
        new_df['microprice_change_ma5'] = new_df['microprice_change'].rolling(window=5, min_periods=1).mean()

        # 微观价格偏离度
        new_df['microprice_deviation'] = (new_df['microprice'] - new_df['n_midprice']) / new_df['n_midprice']
        new_df['microprice_deviation_ma10'] = new_df['microprice_deviation'].rolling(window=10, min_periods=1).mean()

    # 4. 价格冲击因子 (Price Impact Factors) - 短窗口优化
    if 'kyle_lambda' not in new_df.columns:
        # Kyle's Lambda (适合短期预测)
        mid_ret = new_df['n_midprice'].diff().abs()
        volume = new_df['amount_delta'].abs() + 1e-6
        new_df['kyle_lambda'] = (mid_ret / volume).replace([np.inf, -np.inf], 0).fillna(0)
        new_df['kyle_lambda_ma10'] = new_df['kyle_lambda'].rolling(window=10, min_periods=1).mean()
        new_df['kyle_lambda_ma30'] = new_df['kyle_lambda'].rolling(window=30, min_periods=1).mean()

    # 5. 短期波动率因子 (Short-term Volatility) - 100tick窗口重点
    if 'realized_vol' not in new_df.columns:
        mid_ret = new_df['n_midprice'].diff()

        # 短期波动率 (对tick预测最重要)
        new_df['realized_vol_5'] = mid_ret.rolling(window=5, min_periods=1).std()
        new_df['realized_vol_10'] = mid_ret.rolling(window=10, min_periods=1).std()
        new_df['realized_vol_20'] = mid_ret.rolling(window=20, min_periods=1).std()
        new_df['realized_vol_50'] = mid_ret.rolling(window=50, min_periods=1).std()

        # 波动率变化 (预测反转信号)
        new_df['vol_change'] = new_df['realized_vol_10'] - new_df['realized_vol_20']
        new_df['vol_ratio'] = new_df['realized_vol_10'] / (new_df['realized_vol_20'] + 1e-6)

    # 6. 短期动量和反转因子 (Short-term Momentum & Reversal) - tick级别核心
    if 'momentum_5' not in new_df.columns:
        mid_price = new_df['n_midprice']

        # 短期动量 (5-20tick对tick预测最有效)
        new_df['momentum_5'] = mid_price / mid_price.shift(5) - 1
        new_df['momentum_10'] = mid_price / mid_price.shift(10) - 1
        new_df['momentum_20'] = mid_price / mid_price.shift(20) - 1

        # 极短期反转 (tick级别特征)
        new_df['reversal_3'] = -(mid_price / mid_price.shift(3) - 1)
        new_df['reversal_5'] = -(mid_price / mid_price.shift(5) - 1)

        # 价格加速度
        new_df['price_acceleration'] = new_df['n_midprice'].diff() - new_df['n_midprice'].diff().shift(1)

    # 7. 价差因子 (Spread Factors) - 流动性指标
    if 'effective_spread' not in new_df.columns:
        # 有效价差
        new_df['effective_spread'] = (new_df['ask1'] - new_df['bid1']) / new_df['n_midprice']
        new_df['effective_spread_ma10'] = new_df['effective_spread'].rolling(window=10, min_periods=1).mean()
        new_df['effective_spread_ma30'] = new_df['effective_spread'].rolling(window=30, min_periods=1).mean()

        # 价差变化 (流动性变化信号)
        new_df['spread_change'] = new_df['effective_spread'].diff()
        new_df['spread_volatility'] = new_df['effective_spread'].rolling(window=20, min_periods=1).std()

    # 8. 成交量因子 (Volume Factors) - 适合短窗口
    if 'volume_ratio' not in new_df.columns:
        # 成交量相对强度
        volume_ma10 = new_df['amount_delta'].rolling(window=10, min_periods=1).mean()
        volume_ma30 = new_df['amount_delta'].rolling(window=30, min_periods=1).mean()

        new_df['volume_ratio_10'] = new_df['amount_delta'] / (volume_ma10 + 1e-6)
        new_df['volume_ratio_30'] = new_df['amount_delta'] / (volume_ma30 + 1e-6)

        # 成交量突发 (breakout信号)
        new_df['volume_surge'] = new_df['amount_delta'] > (volume_ma10 * 1.5)
        new_df['volume_surge_ma10'] = new_df['volume_surge'].rolling(window=10, min_periods=1).mean()

    # 9. 短期技术指标 (Short-term Technical Indicators)
    if 'short_rsi' not in new_df.columns:
        # 短期RSI (适合tick级别)
        delta = new_df['n_midprice'].diff()
        gain_10 = (delta.where(delta > 0, 0)).rolling(window=10, min_periods=1).mean()
        loss_10 = (-delta.where(delta < 0, 0)).rolling(window=10, min_periods=1).mean()
        rs_10 = gain_10 / (loss_10 + 1e-6)
        new_df['short_rsi'] = 100 - (100 / (1 + rs_10))

        # 价格动量指标
        new_df['price_momentum_3'] = (new_df['n_midprice'] - new_df['n_midprice'].shift(3)) / new_df['n_midprice'].shift(3)
        new_df['price_momentum_5'] = (new_df['n_midprice'] - new_df['n_midprice'].shift(5)) / new_df['n_midprice'].shift(5)

    # 10. 订单簿深度因子 (Order Book Depth) - 核心高频因子
    if 'depth_imbalance' not in new_df.columns:
        # 前三档深度 (高频重点)
        new_df['bid_depth_3'] = new_df['n_bsize1'] + new_df['n_bsize2'] + new_df['n_bsize3']
        new_df['ask_depth_3'] = new_df['n_asize1'] + new_df['n_asize2'] + new_df['n_asize3']

        # 深度不平衡
        new_df['depth_imbalance_3'] = (new_df['bid_depth_3'] - new_df['ask_depth_3']) / \
                                     (new_df['bid_depth_3'] + new_df['ask_depth_3'] + 1e-6)

        # 深度变化率
        new_df['bid_depth_change'] = new_df['bid_depth_3'].diff() / (new_df['bid_depth_3'].shift(1) + 1e-6)
        new_df['ask_depth_change'] = new_df['ask_depth_3'].diff() / (new_df['ask_depth_3'].shift(1) + 1e-6)

    # 11. 价格层级因子 (Price Level Factors) - tick特有
    if 'price_level_pressure' not in new_df.columns:
        # 价格压力指标
        new_df['bid_pressure'] = new_df['n_bsize1'] / (new_df['n_asize1'] + 1e-6)
        new_df['ask_pressure'] = new_df['n_asize1'] / (new_df['n_bsize1'] + 1e-6)

        # 价格层级倾斜度
        new_df['bid_skew'] = (new_df['n_bsize1'] - new_df['n_bsize3']) / (new_df['n_bsize1'] + new_df['n_bsize3'] + 1e-6)
        new_df['ask_skew'] = (new_df['n_asize1'] - new_df['n_asize3']) / (new_df['n_asize1'] + new_df['n_asize3'] + 1e-6)

    # 12. 窗口统计因子 (Window Statistics) - 针对100tick优化
    if 'window_stats' not in new_df.columns:
        # 100tick窗口统计
        window_100 = 100

        # 价格统计
        new_df['price_mean_100'] = new_df['n_midprice'].rolling(window=window_100, min_periods=1).mean()
        new_df['price_std_100'] = new_df['n_midprice'].rolling(window=window_100, min_periods=1).std()
        new_df['price_max_100'] = new_df['n_midprice'].rolling(window=window_100, min_periods=1).max()
        new_df['price_min_100'] = new_df['n_midprice'].rolling(window=window_100, min_periods=1).min()
        new_df['price_range_100'] = new_df['price_max_100'] - new_df['price_min_100']

        # 当前价格在窗口中的位置 (重要预测因子)
        new_df['price_position'] = (new_df['n_midprice'] - new_df['price_min_100']) / \
                                   (new_df['price_range_100'] + 1e-6)

        # 成交量统计
        new_df['volume_mean_100'] = new_df['amount_delta'].rolling(window=window_100, min_periods=1).mean()
        new_df['volume_std_100'] = new_df['amount_delta'].rolling(window=window_100, min_periods=1).std()

        # 价差统计
        new_df['spread_mean_100'] = new_df['effective_spread'].rolling(window=window_100, min_periods=1).mean()
        new_df['spread_std_100'] = new_df['effective_spread'].rolling(window=window_100, min_periods=1).std()

    new_df = new_df.iloc[99:].reset_index(drop=True)  # 去除前99个数据（滚动特征无法计算）
    return new_df

def compute_ofi(df):
    """
    Order Flow Imbalance (OFI)
    Hasbrouck (2018)
    """
    bid = df["n_bid1"].values
    ask = df["n_ask1"].values
    bsz = df["n_bsize1"].values
    asz = df["n_asize1"].values

    ofi = np.zeros(len(df))
    for i in range(1, len(df)):
        # Bid side contribution
        if bid[i] > bid[i-1]:  # bid price increases
            ofi[i] += bsz[i]
        elif bid[i] == bid[i-1]:
            ofi[i] += bsz[i] - bsz[i-1]

        # Ask side contribution
        if ask[i] < ask[i-1]:  # ask price decreases
            ofi[i] -= asz[i]
        elif ask[i] == ask[i-1]:
            ofi[i] -= (asz[i] - asz[i-1])

    return pd.Series(ofi, index=df.index)


def build_features_for_window(df_window: pd.DataFrame) -> dict:
    feat = {}
    N = len(df_window)

    # ---------- 价格 ----------
    mid = df_window["n_midprice"].values
    close = df_window["n_close"].values

    for name, arr in [("mid", mid), ("close", close)]:
        s = pd.Series(arr)
        feat[f"{name}_mean"] = s.mean()
        feat[f"{name}_std"] = s.std()
        feat[f"{name}_skew"] = s.skew()
        feat[f"{name}_kurt"] = s.kurt()
        feat[f"{name}_min"] = s.min()
        feat[f"{name}_max"] = s.max()
        feat[f"{name}_range"] = s.max() - s.min()

        ret = s.diff().dropna()
        feat[f"{name}_ret_mean"] = ret.mean()
        feat[f"{name}_ret_std"] = ret.std()
        feat[f"{name}_ret_skew"] = ret.skew()
        feat[f"{name}_volatility"] = ret.std() * np.sqrt(len(ret))

    # ---------- 价差 ----------
    spread1 = df_window["n_ask1"] - df_window["n_bid1"]
    spread5 = df_window["n_ask5"] - df_window["n_bid1"]

    for name, s in [("spread1", spread1), ("spread5", spread5)]:
        feat[f"{name}_mean"] = s.mean()
        feat[f"{name}_std"] = s.std()
        feat[f"{name}_min"] = s.min()
        feat[f"{name}_max"] = s.max()

    # 相对spread %
    feat["spread_rel"] = (df_window["n_ask1"].iloc[-1] - df_window["n_bid1"].iloc[-1]) / \
                         df_window["n_midprice"].iloc[-1]

    # ---------- 盘口不平衡 ----------
    dfw = df_window
    bsum = dfw[[f"n_bsize{i}" for i in range(1, 6)]].sum(axis=1)
    asum = dfw[[f"n_asize{i}" for i in range(1, 6)]].sum(axis=1)

    obi5 = (bsum - asum) / (bsum + asum + 1e-6)

    feat["obi5_mean"] = obi5.mean()
    feat["obi5_std"] = obi5.std()
    feat["obi5_skew"] = obi5.skew()

    # 1 档
    b1 = dfw["n_bsize1"]
    a1 = dfw["n_asize1"]
    obi1 = (b1 - a1) / (b1 + a1 + 1e-6)
    feat["obi1_mean"] = obi1.mean()

    # ---------- 微观价格 microprice ----------
    microprice = (dfw["n_bid1"]*dfw["n_asize1"] + dfw["n_ask1"]*dfw["n_bsize1"]) / \
                 (dfw["n_bsize1"] + dfw["n_asize1"] + 1e-6)

    micro_ret = microprice.diff()

    feat["micro_mean"] = microprice.mean()
    feat["micro_std"] = microprice.std()
    feat["micro_ret_mean"] = micro_ret.mean()
    feat["micro_ret_std"] = micro_ret.std()

    # ---------- amount_delta ----------
    amt = dfw["amount_delta"].values
    feat["amt_mean"] = amt.mean()
    feat["amt_std"] = amt.std()
    feat["amt_skew"] = pd.Series(amt).skew()

    # ---------- Kyle lambda ----------
    # 此因子不知为何会出现大量的NaN（2179130），暂时不使用
    ret = pd.Series(mid, index=dfw.index).diff().abs().fillna(0)
    vol = dfw["amount_delta"].abs().fillna(0) + 1e-6
    lam = pd.Series(ret.values / vol.values, index=dfw.index).replace([np.inf, -np.inf], 0)#.fillna(0)

    feat["kyle_lambda_mean"] = lam.mean()
    feat["kyle_lambda_std"] = lam.std()

    # ---------- Roll Impact ----------
    # bid-ask bounce estimator
    roll = pd.Series(mid).diff()
    roll_est = -(roll.shift(-1) * roll).mean()
    feat["roll_impact"] = roll_est

    # ---------- OFI（订单流不平衡） ----------
    ofi = compute_ofi(dfw)
    feat["ofi_mean"] = ofi.mean()
    feat["ofi_std"] = ofi.std()
    feat["ofi_skew"] = ofi.skew()

    # ---------- 动量 ----------
    for k in [5, 10, 20, 40, 60]:
        if N > k:
            feat[f"momentum_{k}"] = mid[-1] - mid[-k]
        else:
            feat[f"momentum_{k}"] = 0

    return feat


def extract_features_for_day(df: pd.DataFrame, window=100) -> pd.DataFrame:
    """
    输入：一天的 tick 数据
    输出：特征 DataFrame，可直接用于树模型
    """
    features = []
    indices = []

    for i in range(window, len(df)):
        window_df = df.iloc[i-window:i]
        feat = build_features_for_window(window_df)

        features.append(feat)
        indices.append(df.index[i])  # 当前tick对应的特征时间点

    return pd.DataFrame(features, index=indices)


# 另外一些特征
# import numpy as np
# import pandas as pd

# def feature_extractor(new_df: pd.DataFrame) -> pd.DataFrame:
#     if 'amount' in new_df.columns:   # Hack
#         return new_df

#     if 'ask5' not in new_df.columns:
#         # 价格+1（从涨跌幅还原到对前收盘价的比例）
#         new_df['bid1'] = new_df['n_bid1']+1
#         new_df['bid2'] = new_df['n_bid2']+1
#         new_df['bid3'] = new_df['n_bid3']+1
#         new_df['bid4'] = new_df['n_bid4']+1
#         new_df['bid5'] = new_df['n_bid5']+1
#         new_df['ask1'] = new_df['n_ask1']+1
#         new_df['ask2'] = new_df['n_ask2']+1
#         new_df['ask3'] = new_df['n_ask3']+1
#         new_df['ask4'] = new_df['n_ask4']+1
#         new_df['ask5'] = new_df['n_ask5']+1

#     if 'bid1_ma100' not in new_df.columns:
#         # 均线特征
#         new_df['ask1_ma5']  = new_df['ask1'].rolling(window=5,  min_periods=1).mean()
#         new_df['ask1_ma10'] = new_df['ask1'].rolling(window=10, min_periods=1).mean()
#         new_df['ask1_ma20'] = new_df['ask1'].rolling(window=20, min_periods=1).mean()
#         new_df['ask1_ma40'] = new_df['ask1'].rolling(window=40, min_periods=1).mean()
#         new_df['ask1_ma60'] = new_df['ask1'].rolling(window=60, min_periods=1).mean()
#         new_df['ask1_ma80'] = new_df['ask1'].rolling(window=80, min_periods=1).mean()
#         new_df['ask1_ma100'] = new_df['ask1'].rolling(window=100, min_periods=1).mean()
#         new_df['bid1_ma5']  = new_df['bid1'].rolling(window=5,  min_periods=1).mean()
#         new_df['bid1_ma10'] = new_df['bid1'].rolling(window=10, min_periods=1).mean()
#         new_df['bid1_ma20'] = new_df['bid1'].rolling(window=20, min_periods=1).mean()
#         new_df['bid1_ma40'] = new_df['bid1'].rolling(window=40, min_periods=1).mean()
#         new_df['bid1_ma60'] = new_df['bid1'].rolling(window=60, min_periods=1).mean()
#         new_df['bid1_ma80'] = new_df['bid1'].rolling(window=80, min_periods=1).mean()
#         new_df['bid1_ma100'] = new_df['bid1'].rolling(window=100, min_periods=1).mean()

#     if 'relative_spread3' not in new_df.columns:
#         # 量价组合
#         new_df['spread1'] =  new_df['ask1'] - new_df['bid1']
#         new_df['spread2'] =  new_df['ask2'] - new_df['bid2']
#         new_df['spread3'] =  new_df['ask3'] - new_df['bid3']
#         new_df['mid_price1'] =  new_df['ask1'] + new_df['bid1']
#         new_df['mid_price2'] =  new_df['ask2'] + new_df['bid2']
#         new_df['mid_price3'] =  new_df['ask3'] + new_df['bid3']
#         new_df['weighted_ab1'] = (new_df['ask1'] * new_df['n_bsize1'] + new_df['bid1'] * new_df['n_asize1']) / (new_df['n_bsize1'] + new_df['n_asize1'])
#         new_df['weighted_ab2'] = (new_df['ask2'] * new_df['n_bsize2'] + new_df['bid2'] * new_df['n_asize2']) / (new_df['n_bsize2'] + new_df['n_asize2'])
#         new_df['weighted_ab3'] = (new_df['ask3'] * new_df['n_bsize3'] + new_df['bid3'] * new_df['n_asize3']) / (new_df['n_bsize3'] + new_df['n_asize3'])

#         new_df['relative_spread1'] = new_df['spread1'] / new_df['mid_price1']
#         new_df['relative_spread2'] = new_df['spread2'] / new_df['mid_price2']
#         new_df['relative_spread3'] = new_df['spread3'] / new_df['mid_price3']

#     if 'amount' not in new_df.columns:
#         # 对量取对数
#         new_df['bsize1'] = new_df['n_bsize1'].map(np.log)
#         new_df['bsize2'] = new_df['n_bsize2'].map(np.log)
#         new_df['bsize3'] = new_df['n_bsize3'].map(np.log)
#         new_df['bsize4'] = new_df['n_bsize4'].map(np.log)
#         new_df['bsize5'] = new_df['n_bsize5'].map(np.log)
#         new_df['asize1'] = new_df['n_asize1'].map(np.log)
#         new_df['asize2'] = new_df['n_asize2'].map(np.log)
#         new_df['asize3'] = new_df['n_asize3'].map(np.log)
#         new_df['asize4'] = new_df['n_asize4'].map(np.log)
#         new_df['asize5'] = new_df['n_asize5'].map(np.log)
#         new_df['amount'] = new_df['amount_delta'].map(np.log1p)

#     # ===== 新增的高效量化因子 =====

#     # 1. 订单流不平衡因子 (Order Flow Imbalance - OFI)
#     if 'ofi' not in new_df.columns:
#         new_df['ofi'] = compute_ofi(new_df)
#         new_df['ofi_ma5'] = new_df['ofi'].rolling(window=5, min_periods=1).mean()
#         new_df['ofi_std20'] = new_df['ofi'].rolling(window=20, min_periods=1).std()

#     # 2. 盘口不平衡因子 (Order Book Imbalance)
#     if 'obi5' not in new_df.columns:
#         new_df['obi1'] = (new_df['n_bsize1'] - new_df['n_asize1']) / (new_df['n_bsize1'] + new_df['n_asize1'] + 1e-6)
#         new_df['obi5'] = ((new_df['n_bsize1'] + new_df['n_bsize2'] + new_df['n_bsize3'] + new_df['n_bsize4'] + new_df['n_bsize5']) -
#                          (new_df['n_asize1'] + new_df['n_asize2'] + new_df['n_asize3'] + new_df['n_asize4'] + new_df['n_asize5'])) / \
#                         ((new_df['n_bsize1'] + new_df['n_bsize2'] + new_df['n_bsize3'] + new_df['n_bsize4'] + new_df['n_bsize5']) +
#                          (new_df['n_asize1'] + new_df['n_asize2'] + new_df['n_asize3'] + new_df['n_asize4'] + new_df['n_asize5']) + 1e-6)

#         new_df['obi1_ma10'] = new_df['obi1'].rolling(window=10, min_periods=1).mean()
#         new_df['obi5_ma10'] = new_df['obi5'].rolling(window=10, min_periods=1).mean()

#     # 3. 微观结构因子 (Microstructure Features)
#     if 'microprice' not in new_df.columns:
#         new_df['microprice'] = (new_df['bid1'] * new_df['n_asize1'] + new_df['ask1'] * new_df['n_bsize1']) / \
#                               (new_df['n_bsize1'] + new_df['n_asize1'] + 1e-6)
#         new_df['microprice_ma5'] = new_df['microprice'].rolling(window=5, min_periods=1).mean()
#         new_df['microprice_ma20'] = new_df['microprice'].rolling(window=20, min_periods=1).mean()

#         # 微观价格偏离度
#         new_df['microprice_deviation'] = (new_df['microprice'] - new_df['n_midprice']) / new_df['n_midprice']
#         new_df['microprice_deviation_ma10'] = new_df['microprice_deviation'].rolling(window=10, min_periods=1).mean()

#     # 4. 价格冲击因子 (Price Impact Factors)
#     if 'price_impact' not in new_df.columns:
#         # Kyle's Lambda
#         mid_ret = new_df['n_midprice'].diff().abs()
#         volume = new_df['amount_delta'].abs() + 1e-6
#         new_df['kyle_lambda'] = (mid_ret / volume).replace([np.inf, -np.inf], 0).fillna(0)
#         new_df['kyle_lambda_ma10'] = new_df['kyle_lambda'].rolling(window=10, min_periods=1).mean()

#         # Amihud Illiquidity
#         new_df['amihud_illiquidity'] = mid_ret / (volume + 1e-6)
#         new_df['amihud_illiquidity_ma20'] = new_df['amihud_illiquidity'].rolling(window=20, min_periods=1).mean()

#     # 5. 波动率因子 (Volatility Factors)
#     if 'realized_vol' not in new_df.columns:
#         # 已实现波动率 (不同窗口)
#         mid_ret = new_df['n_midprice'].diff()
#         for window in [5, 10, 20, 40]:
#             new_df[f'realized_vol_{window}'] = (mid_ret.rolling(window=window, min_periods=1).std() * np.sqrt(window))

#         # Garman-Klass波动率估计
#         new_df['gk_volatility'] = np.sqrt(0.5 * (new_df['ask1'] - new_df['bid1'])**2 -
#                                          (2*np.log(2) - 1) * (new_df['n_midprice'].diff())**2)
#         new_df['gk_volatility_ma20'] = new_df['gk_volatility'].rolling(window=20, min_periods=1).mean()

#     # 6. 动量和反转因子 (Momentum & Reversal)
#     if 'momentum' not in new_df.columns:
#         mid_price = new_df['n_midprice']
#         for lag in [5, 10, 20, 40, 60]:
#             new_df[f'momentum_{lag}'] = mid_price / mid_price.shift(lag) - 1
#             new_df[f'reversal_{lag}'] = -new_df[f'momentum_{lag}']  # 反转因子

#     # 7. 价差因子 (Spread Factors)
#     if 'spread_depth' not in new_df.columns:
#         # 价差深度
#         new_df['spread_depth'] = (new_df['ask5'] - new_df['bid5']) / (new_df['ask1'] - new_df['bid1'] + 1e-6)

#         # 有效价差
#         new_df['effective_spread'] = (new_df['ask1'] - new_df['bid1']) / new_df['n_midprice']
#         new_df['effective_spread_ma20'] = new_df['effective_spread'].rolling(window=20, min_periods=1).mean()

#     # 8. 成交量因子 (Volume Factors)
#     if 'volume_ratio' not in new_df.columns:
#         # 成交量比率
#         new_df['volume_ratio'] = new_df['amount_delta'] / (new_df['amount_delta'].rolling(window=20, min_periods=1).mean() + 1e-6)
#         new_df['volume_std'] = new_df['amount_delta'].rolling(window=20, min_periods=1).std()

#         # 成交量加权平均价 (VWAP)
#         if 'vwap' not in new_df.columns:
#             vwap_num = (new_df['n_midprice'] * new_df['amount_delta']).rolling(window=20, min_periods=1).sum()
#             vwap_den = new_df['amount_delta'].rolling(window=20, min_periods=1).sum() + 1e-6
#             new_df['vwap'] = vwap_num / vwap_den
#             new_df['vwap_deviation'] = (new_df['n_midprice'] - new_df['vwap']) / new_df['vwap']

#     # 9. 高频技术指标因子 (Technical Indicators)
#     if 'rsi' not in new_df.columns:
#         # RSI (相对强弱指数)
#         delta = new_df['n_midprice'].diff()
#         gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
#         loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
#         rs = gain / (loss + 1e-6)
#         new_df['rsi'] = 100 - (100 / (1 + rs))

#         # MACD
#         ema12 = new_df['n_midprice'].ewm(span=12).mean()
#         ema26 = new_df['n_midprice'].ewm(span=26).mean()
#         new_df['macd'] = ema12 - ema26
#         new_df['macd_signal'] = new_df['macd'].ewm(span=9).mean()
#         new_df['macd_histogram'] = new_df['macd'] - new_df['macd_signal']

#     # 10. 市场深度因子 (Market Depth Factors)
#     if 'market_depth' not in new_df.columns:
#         # 总深度
#         new_df['total_bid_depth'] = new_df['n_bsize1'] + new_df['n_bsize2'] + new_df['n_bsize3'] + new_df['n_bsize4'] + new_df['n_bsize5']
#         new_df['total_ask_depth'] = new_df['n_asize1'] + new_df['n_asize2'] + new_df['n_asize3'] + new_df['n_asize4'] + new_df['n_asize5']
#         new_df['total_depth'] = new_df['total_bid_depth'] + new_df['total_ask_depth']

#         # 深度不平衡
#         new_df['depth_imbalance'] = (new_df['total_bid_depth'] - new_df['total_ask_depth']) / (new_df['total_depth'] + 1e-6)

#         # 深度倾斜度
#         new_df['depth_skew_bid'] = (new_df['n_bsize1'] - new_df['n_bsize5']) / (new_df['n_bsize1'] + new_df['n_bsize5'] + 1e-6)
#         new_df['depth_skew_ask'] = (new_df['n_asize1'] - new_df['n_asize5']) / (new_df['n_asize1'] + new_df['n_asize5'] + 1e-6)

#     new_df = new_df.iloc[99:].reset_index(drop=True)  # 去除前99个数据（滚动特征无法计算）
#     return new_df

# def compute_ofi(df):
#     """
#     Order Flow Imbalance (OFI)
#     Hasbrouck (2018)
#     """
#     bid = df["n_bid1"].values
#     ask = df["n_ask1"].values
#     bsz = df["n_bsize1"].values
#     asz = df["n_asize1"].values

#     ofi = np.zeros(len(df))
#     for i in range(1, len(df)):
#         # Bid side contribution
#         if bid[i] > bid[i-1]:  # bid price increases
#             ofi[i] += bsz[i]
#         elif bid[i] == bid[i-1]:
#             ofi[i] += bsz[i] - bsz[i-1]

#         # Ask side contribution
#         if ask[i] < ask[i-1]:  # ask price decreases
#             ofi[i] -= asz[i]
#         elif ask[i] == ask[i-1]:
#             ofi[i] -= (asz[i] - asz[i-1])

#     return pd.Series(ofi, index=df.index)


# def build_features_for_window(df_window: pd.DataFrame) -> dict:
#     feat = {}
#     N = len(df_window)

#     # ---------- 价格 ----------
#     mid = df_window["n_midprice"].values
#     close = df_window["n_close"].values

#     for name, arr in [("mid", mid), ("close", close)]:
#         s = pd.Series(arr)
#         feat[f"{name}_mean"] = s.mean()
#         feat[f"{name}_std"] = s.std()
#         feat[f"{name}_skew"] = s.skew()
#         feat[f"{name}_kurt"] = s.kurt()
#         feat[f"{name}_min"] = s.min()
#         feat[f"{name}_max"] = s.max()
#         feat[f"{name}_range"] = s.max() - s.min()

#         ret = s.diff().dropna()
#         feat[f"{name}_ret_mean"] = ret.mean()
#         feat[f"{name}_ret_std"] = ret.std()
#         feat[f"{name}_ret_skew"] = ret.skew()
#         feat[f"{name}_volatility"] = ret.std() * np.sqrt(len(ret))

#     # ---------- 价差 ----------
#     spread1 = df_window["n_ask1"] - df_window["n_bid1"]
#     spread5 = df_window["n_ask5"] - df_window["n_bid1"]

#     for name, s in [("spread1", spread1), ("spread5", spread5)]:
#         feat[f"{name}_mean"] = s.mean()
#         feat[f"{name}_std"] = s.std()
#         feat[f"{name}_min"] = s.min()
#         feat[f"{name}_max"] = s.max()

#     # 相对spread %
#     feat["spread_rel"] = (df_window["n_ask1"].iloc[-1] - df_window["n_bid1"].iloc[-1]) / \
#                          df_window["n_midprice"].iloc[-1]

#     # ---------- 盘口不平衡 ----------
#     dfw = df_window
#     bsum = dfw[[f"n_bsize{i}" for i in range(1, 6)]].sum(axis=1)
#     asum = dfw[[f"n_asize{i}" for i in range(1, 6)]].sum(axis=1)

#     obi5 = (bsum - asum) / (bsum + asum + 1e-6)

#     feat["obi5_mean"] = obi5.mean()
#     feat["obi5_std"] = obi5.std()
#     feat["obi5_skew"] = obi5.skew()

#     # 1 档
#     b1 = dfw["n_bsize1"]
#     a1 = dfw["n_asize1"]
#     obi1 = (b1 - a1) / (b1 + a1 + 1e-6)
#     feat["obi1_mean"] = obi1.mean()

#     # ---------- 微观价格 microprice ----------
#     microprice = (dfw["n_bid1"]*dfw["n_asize1"] + dfw["n_ask1"]*dfw["n_bsize1"]) / \
#                  (dfw["n_bsize1"] + dfw["n_asize1"] + 1e-6)

#     micro_ret = microprice.diff()

#     feat["micro_mean"] = microprice.mean()
#     feat["micro_std"] = microprice.std()
#     feat["micro_ret_mean"] = micro_ret.mean()
#     feat["micro_ret_std"] = micro_ret.std()

#     # ---------- amount_delta ----------
#     amt = dfw["amount_delta"].values
#     feat["amt_mean"] = amt.mean()
#     feat["amt_std"] = amt.std()
#     feat["amt_skew"] = pd.Series(amt).skew()

#     # ---------- Kyle lambda ----------
#     # 此因子不知为何会出现大量的NaN（2179130），暂时不使用
#     ret = pd.Series(mid, index=dfw.index).diff().abs().fillna(0)
#     vol = dfw["amount_delta"].abs().fillna(0) + 1e-6
#     lam = pd.Series(ret.values / vol.values, index=dfw.index).replace([np.inf, -np.inf], 0)#.fillna(0)

#     feat["kyle_lambda_mean"] = lam.mean()
#     feat["kyle_lambda_std"] = lam.std()

#     # ---------- Roll Impact ----------
#     # bid-ask bounce estimator
#     roll = pd.Series(mid).diff()
#     roll_est = -(roll.shift(-1) * roll).mean()
#     feat["roll_impact"] = roll_est

#     # ---------- OFI（订单流不平衡） ----------
#     ofi = compute_ofi(dfw)
#     feat["ofi_mean"] = ofi.mean()
#     feat["ofi_std"] = ofi.std()
#     feat["ofi_skew"] = ofi.skew()

#     # ---------- 动量 ----------
#     for k in [5, 10, 20, 40, 60]:
#         if N > k:
#             feat[f"momentum_{k}"] = mid[-1] - mid[-k]
#         else:
#             feat[f"momentum_{k}"] = 0

#     return feat


# def extract_features_for_day(df: pd.DataFrame, window=100) -> pd.DataFrame:
#     """
#     输入：一天的 tick 数据
#     输出：特征 DataFrame，可直接用于树模型
#     """
#     features = []
#     indices = []

#     for i in range(window, len(df)):
#         window_df = df.iloc[i-window:i]
#         feat = build_features_for_window(window_df)

#         features.append(feat)
#         indices.append(df.index[i])  # 当前tick对应的特征时间点

#     return pd.DataFrame(features, index=indices)

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
    # else:
    #     print("已经提取过 价格+1")

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
    # else:
    #     print("已经提取过 均线特征")

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
    # else:
    #     print("已经提取过 量价组合")
    
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
    # else:
    #     print("已经提取过 对量取对数")

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

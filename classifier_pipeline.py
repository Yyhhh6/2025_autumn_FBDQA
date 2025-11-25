import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, fbeta_score
)
from sklearn.utils import shuffle
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import fbeta_score

# -------------------------
# 3) 评估辅助
# -------------------------
def classification_metrics(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f0.5_macro": fbeta_score(y_true, y_pred, average="macro", beta=0.5, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }
    return metrics

def calculate_pnl_metrics(y_true, y_pred, mid_prices=None, price_changes=None,
                          initial_capital=1000000, transaction_cost=0.0001):
    """
    计算量化交易的PnL相关指标

    Parameters:
    y_true: 真实标签 (0=跌, 1=平, 2=涨)
    y_pred: 预测标签
    mid_prices: 中间价序列 (可选，用于计算实际收益)
    price_changes: 价格变化率 (可选，如果mid_prices为None时使用)
    initial_capital: 初始资金
    transaction_cost: 交易成本率

    Returns:
    dict: PnL相关指标
    """
    if price_changes is None and mid_prices is not None:
        # 从中间价计算价格变化率
        price_changes = np.diff(mid_prices) / mid_prices[:-1]
        # 在末尾填充0，因为价格变化数组长度比原始数组少1
        price_changes = np.append(price_changes, 0)

    if price_changes is None:
        # 如果没有价格信息，使用假设的平均收益
        # 上涨:+0.1%, 平盘:0%, 下跌:-0.1%
        price_changes = np.where(y_true == 2, 0.001,
                                np.where(y_true == 0, -0.001, 0))

    # 构建交易信号
    # 只在预测为涨(2)时做多，预测为跌(0)时做空，预测为平(1)时不交易
    positions = np.zeros(len(y_pred))
    positions[y_pred == 2] = 1   # 做多
    positions[y_pred == 0] = -1  # 做空
    positions[y_pred == 1] = 0   # 不交易

    # 计算每期收益（考虑交易成本）
    returns = positions * price_changes

    # 计算交易成本（只在仓位变化时产生）
    position_changes = np.abs(np.diff(positions))
    trading_costs = position_changes * transaction_cost
    trading_costs = np.append(trading_costs, 0)  # 末尾填充0

    # 净收益
    net_returns = returns - trading_costs

    # 累计收益
    cumulative_returns = np.cumprod(1 + net_returns)

    # PnL计算
    pnl = initial_capital * (cumulative_returns - 1)

    # 基础统计
    total_trades = np.sum(position_changes)
    winning_trades = np.sum(net_returns > 0)
    losing_trades = np.sum(net_returns < 0)

    # 计算各项指标
    total_return = cumulative_returns[-1] - 1
    annual_return = (1 + total_return) ** (252 * 1440 / len(y_pred)) - 1  # 假设每分钟一个tick

    # 夏普比率（假设无风险利率为0）
    returns_std = np.std(net_returns)
    sharpe_ratio = np.mean(net_returns) / returns_std * np.sqrt(252 * 1440) if returns_std > 0 else 0

    # 最大回撤
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = np.min(drawdown)

    # 胜率
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    # 平均盈亏比
    avg_win = np.mean(net_returns[net_returns > 0]) if winning_trades > 0 else 0
    avg_loss = np.mean(net_returns[net_returns < 0]) if losing_trades > 0 else 0
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

    # 计算只交易上涨方向的PnL（更保守的策略）
    long_only_positions = np.where(y_pred == 2, 1, 0)  # 只在预测上涨时做多
    long_only_returns = long_only_positions * price_changes
    long_only_position_changes = np.abs(np.diff(long_only_positions))
    long_only_trading_costs = long_only_position_changes * transaction_cost
    long_only_trading_costs = np.append(long_only_trading_costs, 0)
    long_only_net_returns = long_only_returns - long_only_trading_costs
    long_only_cumulative_returns = np.cumprod(1 + long_only_net_returns)
    long_only_total_return = long_only_cumulative_returns[-1] - 1

    return {
        "total_pnl": pnl[-1],
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_loss_ratio": profit_loss_ratio,
        "volatility": returns_std,
        "long_only_return": long_only_total_return,  # 保守策略收益
        "long_only_sharpe": (np.mean(long_only_net_returns) / np.std(long_only_net_returns) * np.sqrt(252 * 1440)
                           if np.std(long_only_net_returns) > 0 else 0),
        "position_coverage": np.mean(np.abs(positions)),  # 仓位覆盖率
        "trading_frequency": total_trades / len(y_pred),  # 交易频率
    }

# -------------------------
# 4) LightGBM (GPU) 训练 & 测试
# -------------------------
def train_lightgbm_gpu(X_train, y_train, X_val, y_val, feature_names, save_path="lgb_model.txt",
                       num_class=3, params_override=None):
    # 转 np.ndarray (float32/int32)
    X_train = X_train.astype("float32")
    X_val = X_val.astype("float32")

    lgb_params = {
        "objective": "multiclass",
        "num_class": num_class,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 64, # TODO：调整模型的拟合能力，越大能拟合更多的非线性
        "max_depth": -1,
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "early_stopping_rounds": 50,
        "verbosity": 100
    }
    if params_override:
        lgb_params.update(params_override)

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names) # TODO：随机划分？
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    bst = lgb.train(
        lgb_params,
        dtrain,
        valid_sets=[dval],
        valid_names=["valid"],
        num_boost_round=2000,
    )
    bst.save_model(save_path)
    return bst

# -------------------------
# 5) SGDClassifier (incremental) 训练 & 测试 （Logistic Regression）
# -------------------------
def train_sgd_incremental(X_train, y_train, X_val, y_val, classes=[0,1,2], max_epochs=3, batch_size=100000):
    """
    使用 partial_fit 增量训练 SGDClassifier（适合大样本）。
    重要：假设输入数据已经经过标准化处理，避免重复标准化
    """
    sgd = SGDClassifier(loss="log_loss", penalty="l2", max_iter=1, tol=None, warm_start=True, random_state=42)

    # 假设数据已经标准化，直接使用
    X_train_scaled = X_train
    X_val_scaled = X_val
    scaler = None  # 不再需要scaler，因为数据已预处理

    # incremental training via partial_fit in chunks, for several epochs
    best_val_f0_5 = -np.inf
    best_sgd = None

    for epoch in range(max_epochs):
        # shuffle
        Xs, ys = shuffle(X_train_scaled, y_train, random_state=epoch)
        for i in range(0, Xs.shape[0], chunk := batch_size):
            Xb = Xs[i:i+chunk]
            yb = ys[i:i+chunk]
            if i == 0:  # 第一次训练时需要指定classes
                sgd.partial_fit(Xb, yb, classes=classes)
            else:
                sgd.partial_fit(Xb, yb)

        # evaluate per epoch
        val_pred = sgd.predict(X_val_scaled)
        val_f0_5 = fbeta_score(y_val, val_pred, average='macro', zero_division=0, beta=0.5)
        val_precision = precision_score(y_val, val_pred, average='macro', zero_division=0)
        val_recall = recall_score(y_val, val_pred, average='macro', zero_division=0)

        print(f"[SGD] epoch {epoch} val f0.5_macro = {val_f0_5:.4f} precision_macro = {val_precision:.4f} recall_macro = {val_recall:.4f}")

        # 保存最佳模型
        if val_f0_5 > best_val_f0_5:
            best_val_f0_5 = val_f0_5
            best_sgd = sgd

    return best_sgd, scaler

# -------------------------
# 6) PyTorch MLP（GPU）训练 & 测试
# -------------------------
class TickDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], dropout=0.2, num_classes=3):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def train_pytorch_mlp(X_train, y_train, X_val, y_val,
                      input_dim, device='cuda',
                      hidden_dims=[512,256], lr=1e-3,
                      batch_size=8192, epochs=20, patience=5,
                      model_path="mlp_best.pt"):

    # 假设数据已经标准化，直接使用
    X_train_s = X_train.astype(np.float32)
    X_val_s = X_val.astype(np.float32)
    scaler = None  # 不再需要scaler，因为数据已预处理

    train_ds = TickDataset(X_train_s, y_train)
    val_ds = TickDataset(X_val_s, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, num_workers=4, pin_memory=True)

    model = MLPClassifier(input_dim=input_dim, hidden_dims=hidden_dims, dropout=0.2, num_classes=3)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val = -np.inf
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            # print("xb: ", xb)
            # print("yb: ", yb)
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        avg_train_loss = running_loss / len(train_ds)

        # validation
        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                preds.append(pred)
                trues.append(yb.cpu().numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        val_f0_5 = fbeta_score(trues, preds, average='macro', zero_division=0, beta=0.5)
        print(f"[PyTorch] Epoch {epoch} train_loss={avg_train_loss:.4f} val_f0.5_macro={val_f0_5:.4f} precision_macro={precision_score(trues, preds, average='macro', zero_division=0):.4f} recall_macro={recall_score(trues, preds, average='macro', zero_division=0):.4f}")

        # early stopping by val_f0_5
        if val_f0_5 > best_val:
            best_val = val_f0_5
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping.")
                break

    # load best
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

# -------------------------
# 7) End-to-end wrapper：输入 DataFrame、factor_list、N_list -> 训练 & 测试
# -------------------------
def run_pipeline(train_df: pd.DataFrame,
                 val_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 factor_list: List[str],
                 N_list: List[int],
                 alpha_map: Dict[int, float], # TODO: alpha_map 目前没用上
                 label_prefix: str = "label_",
                 out_dir: str = "./results",
                 device: str = "cuda"):
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    for N in N_list:
        print("\n" + "="*40)
        print(f"Start training for N={N}")
        label_col = f"{label_prefix}{N}"

        print(f"effective samples -> train: {len(train_df)} val: {len(val_df)} test: {len(test_df)}")

        # 提取 X,y
        X_train = train_df[factor_list].values.astype(np.float32)
        y_train = train_df[label_col].values.astype(np.int64)
        X_val = val_df[factor_list].values.astype(np.float32)
        y_val = val_df[label_col].values.astype(np.int64)
        X_test = test_df[factor_list].values.astype(np.float32)
        y_test = test_df[label_col].values.astype(np.int64)
        feature_names = factor_list

        # ---------------- LightGBM ----------------
        print("Training LightGBM (GPU)...")
        # LightGBM 对 label 要是 0..K-1
        bst = train_lightgbm_gpu(X_train, y_train, X_val, y_val, feature_names,
                                 save_path=os.path.join(out_dir, f"lgb_N{N}.txt"))
        # predict test
        y_pred_proba = bst.predict(X_test, num_iteration=bst.best_iteration)
        y_pred = np.argmax(y_pred_proba, axis=1)
        metrics_lgb = classification_metrics(y_test, y_pred)

        # 计算PnL指标
        # 如果有中间价数据，使用实际价格计算收益
        if 'n_midprice' in test_df.columns:
            mid_prices = test_df['n_midprice'].values
            pnl_metrics_lgb = calculate_pnl_metrics(y_test, y_pred, mid_prices=mid_prices)
        else:
            pnl_metrics_lgb = calculate_pnl_metrics(y_test, y_pred)

        # 合并指标
        metrics_lgb.update({f"pnl_{k}": v for k, v in pnl_metrics_lgb.items()})

        print(f"LightGBM N={N} test f0.5_macro: {metrics_lgb['f0.5_macro']:.4f} precision_macro: {metrics_lgb['precision_macro']:.4f} recall_macro: {metrics_lgb['recall_macro']:.4f}")
        print(f"LightGBM N={N} PnL: {pnl_metrics_lgb['total_pnl']:,.0f} Return: {pnl_metrics_lgb['total_return']:.2%} Sharpe: {pnl_metrics_lgb['sharpe_ratio']:.2f} WinRate: {pnl_metrics_lgb['win_rate']:.2%}")

        # ---------------- SGDClassifier ----------------
        print("Training SGDClassifier (incremental logistic)...")
        sgd, sgd_scaler = train_sgd_incremental(X_train, y_train, X_val, y_val,
                                                classes=[0,1,2], max_epochs=3, batch_size=200000)
        # 数据已经预处理，直接使用
        y_pred_sgd = sgd.predict(X_test)
        metrics_sgd = classification_metrics(y_test, y_pred_sgd)

        # 计算PnL指标
        if 'n_midprice' in test_df.columns:
            mid_prices = test_df['n_midprice'].values
            pnl_metrics_sgd = calculate_pnl_metrics(y_test, y_pred_sgd, mid_prices=mid_prices)
        else:
            pnl_metrics_sgd = calculate_pnl_metrics(y_test, y_pred_sgd)

        # 合并指标
        metrics_sgd.update({f"pnl_{k}": v for k, v in pnl_metrics_sgd.items()})

        print(f"SGD N={N} test f0.5_macro: {metrics_sgd['f0.5_macro']:.4f} precision_macro: {metrics_sgd['precision_macro']:.4f} recall_macro: {metrics_sgd['recall_macro']:.4f}")
        print(f"SGD N={N} PnL: {pnl_metrics_sgd['total_pnl']:,.0f} Return: {pnl_metrics_sgd['total_return']:.2%} Sharpe: {pnl_metrics_sgd['sharpe_ratio']:.2f} WinRate: {pnl_metrics_sgd['win_rate']:.2%}")

        # ---------------- PyTorch MLP ----------------
        print("Training PyTorch MLP...")
        input_dim = X_train.shape[1]
        model_path = os.path.join(out_dir, f"mlp_N{N}.pt")
        model = train_pytorch_mlp(X_train, y_train, X_val, y_val,
                                             input_dim=input_dim,
                                             device=device,
                                             hidden_dims=[512,256],
                                             lr=1e-3,
                                             batch_size=8192,
                                             epochs=30,
                                             patience=5,
                                             model_path=model_path)
        # test for MLP
        # load test scaled
        device_t = torch.device(device if torch.cuda.is_available() else "cpu")
        model.to(device_t)
        model.eval()
        test_ds = TickDataset(X_test, y_test)
        test_loader = DataLoader(test_ds, batch_size=8192, shuffle=False, num_workers=4)
        preds = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device_t)
                logits = model(xb)
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        preds = np.concatenate(preds)
        metrics_mlp = classification_metrics(y_test, preds)

        # 计算PnL指标
        if 'n_midprice' in test_df.columns:
            mid_prices = test_df['n_midprice'].values
            pnl_metrics_mlp = calculate_pnl_metrics(y_test, preds, mid_prices=mid_prices)
        else:
            pnl_metrics_mlp = calculate_pnl_metrics(y_test, preds)

        # 合并指标
        metrics_mlp.update({f"pnl_{k}": v for k, v in pnl_metrics_mlp.items()})

        print(f"MLP N={N} test f0.5_macro: {metrics_mlp['f0.5_macro']:.4f} precision_macro: {metrics_mlp['precision_macro']:.4f} recall_macro: {metrics_mlp['recall_macro']:.4f}")
        print(f"MLP N={N} PnL: {pnl_metrics_mlp['total_pnl']:,.0f} Return: {pnl_metrics_mlp['total_return']:.2%} Sharpe: {pnl_metrics_mlp['sharpe_ratio']:.2f} WinRate: {pnl_metrics_mlp['win_rate']:.2%}")

        # 存储结果（包含PNL）
        results[N] = {
            "lgb": (metrics_lgb, bst.best_iteration),
            "sgd": (metrics_sgd, None),
            "mlp": (metrics_mlp, model_path)
        }

        # 打印PNL比较
        print(f"\nPNL Comparison for N={N}:")
        print(f"LightGBM: {pnl_lgb['bps_pnl']:>8.2f} bps, Sharpe: {pnl_lgb['sharpe_ratio']:>6.2f}")
        print(f"SGD:      {pnl_sgd['bps_pnl']:>8.2f} bps, Sharpe: {pnl_sgd['sharpe_ratio']:>6.2f}")
        print(f"MLP:      {pnl_mlp['bps_pnl']:>8.2f} bps, Sharpe: {pnl_mlp['sharpe_ratio']:>6.2f}")


    return results


from typing import Dict, List, Tuple

def calculate_pnl(test_df: pd.DataFrame, 
                 predictions: np.ndarray,
                 N: int,
                 price_col: str = "n_close",
                 return_col: str = "return",
                 transaction_cost: float = 0.001) -> Dict[str, float]:
    """
    计算策略的PNL表现
    
    Args:
        test_df: 测试数据集
        predictions: 模型预测结果 (0: 跌, 1: 平, 2: 涨)
        N: 预测周期
        price_col: 价格列名
        return_col: 收益率列名
        transaction_cost: 交易成本（双边）
    
    Returns:
        PNL相关指标的字典
    """
    # 复制数据避免修改原数据
    df = test_df.copy()
    
    # 确保长度一致
    assert len(df) == len(predictions), "预测结果长度与测试数据长度不匹配"

    # 创建信号
    # 2(涨) -> 做多, 0(跌) -> 做空, 1(平) -> 空仓
    positions = np.zeros(len(predictions))
    print(f"init {np.count_nonzero(positions)} positions")
    positions[predictions == 2] = 1   # 做多
    positions[predictions == 0] = -1  # 做空
    
    # 计算收益率
    if return_col in df.columns:
        returns = df[return_col].values
    else:
        # 如果没有收益率列，从价格计算
        prices = df[price_col].values
        returns = np.zeros(len(prices)-N)
        print("prices:")
        print(prices)
        returns = (prices[N:] - prices[:-N])
    print(f"Returns sample: {returns[:20]}")
    assert len(returns) == len(positions) - N, "收益率长度与持仓长度不匹配"
    strategy_returns = returns*positions[:len(returns)]
    print(f"Strategy returns sample: {strategy_returns[:20]}")
    cumulative_strategy_return = np.sum(strategy_returns)
    cumulative_buy_hold_return = returns[-1]
    trade_num = np.count_nonzero(positions) * 2  # 双边交易次数
    total_transaction_cost = trade_num * transaction_cost
    cumulative_strategy_return -= total_transaction_cost
    bps_pnl = cumulative_strategy_return * 10000  # 转换为bps
    print(f"bps_pnl: {bps_pnl}")
    print(f"Cumulative strategy return: {cumulative_strategy_return:.4f}")
    print(f"Cumulative buy-and-hold return: {cumulative_buy_hold_return:.4f}")
    exit()
    return {
        "cumulative_return": cumulative_strategy_return,
        "bps_pnl": bps_pnl,
        "cumulative_buy_hold": cumulative_buy_hold_return,
        "excess_return": cumulative_strategy_return - cumulative_buy_hold_return
    }
    # 计算策略收益（考虑N周期持有）
    strategy_returns = np.zeros(len(returns))
    
    for i in range(len(returns) - N):
        if positions[i] != 0:  # 有持仓
            # 持有N周期的收益
            hold_return = np.prod(1 + returns[i:i+N]) - 1
            strategy_returns[i + N] = positions[i] * hold_return
    
    # 计算交易次数（仓位变化时交易）
    trades = np.diff(positions, prepend=0) != 0
    trade_count = np.sum(trades)
    
    # 扣除交易成本
    total_transaction_cost = trade_count * transaction_cost
    
    # 计算净收益
    net_strategy_returns = strategy_returns - (trades * transaction_cost)
    
    # 计算累计收益
    net_strategy_returns_not_zero = net_strategy_returns[net_strategy_returns != 0]
    print(f"Net strategy returns sample: {net_strategy_returns_not_zero}")
    print(f"number of trades: {trade_count}")
    print(f"number of net_strategy_returns_not_zero: {len(net_strategy_returns_not_zero)}")
    print(f"Returns sample: {returns[:20]}")
    
    cumulative_strategy_return = np.prod(1 + net_strategy_returns) - 1
    cumulative_buy_hold_return = np.prod(1 + returns) - 1
    print(f"Cumulative strategy return: {cumulative_strategy_return:.4f}")
    print(f"Cumulative buy-and-hold return: {cumulative_buy_hold_return:.4f}")
    exit()
    # 计算年化收益（假设252个交易日）
    days = len(returns)
    annualized_strategy = (1 + cumulative_strategy_return) ** (252/days) - 1 if days > 0 else 0
    annualized_bh = (1 + cumulative_buy_hold_return) ** (252/days) - 1 if days > 0 else 0
    
    # 计算bps PNL（相对于初始本金的收益率，以bps表示）
    bps_pnl = cumulative_strategy_return * 10000  # 转换为bps
    
    # 计算夏普比率
    excess_returns = net_strategy_returns - returns
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
    
    # 最大回撤
    cumulative_returns = np.cumprod(1 + net_strategy_returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    return {
        "cumulative_return": cumulative_strategy_return,
        "annualized_return": annualized_strategy,
        "bps_pnl": bps_pnl,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "trade_count": trade_count,
        "win_rate": np.mean(net_strategy_returns > 0) if len(net_strategy_returns) > 0 else 0,
        "total_transaction_cost": total_transaction_cost,
        "cumulative_buy_hold": cumulative_buy_hold_return,
        "excess_return": cumulative_strategy_return - cumulative_buy_hold_return
    }

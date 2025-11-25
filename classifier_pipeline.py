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
    X_train, y_train 可以是 numpy arrays 或 pandas
    """
    sgd = SGDClassifier(loss="log_loss", penalty="l2", max_iter=1, tol=None, warm_start=True)
    # 先做标准化（增量），使用 StandardScaler.partial_fit
    scaler = StandardScaler()
    # partial fitting scaler in chunks to reduce mempeak
    n = X_train.shape[0]
    chunk = batch_size
    for i in range(0, n, chunk):
        scaler.partial_fit(X_train[i:i+chunk])
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # incremental training via partial_fit in chunks, for several epochs
    for epoch in range(max_epochs):
        # shuffle
        Xs, ys = shuffle(X_train_scaled, y_train, random_state=epoch)
        for i in range(0, Xs.shape[0], chunk):
            Xb = Xs[i:i+chunk]
            yb = ys[i:i+chunk]
            sgd.partial_fit(Xb, yb, classes=classes)
        # optional: evaluate per epoch
        val_pred = sgd.predict(X_val_scaled)
        print(f"[SGD] epoch {epoch} val f0.5_macro = {fbeta_score(y_val, val_pred, average='macro', zero_division=0, beta=0.5):.4f} precision_macro = {precision_score(y_val, val_pred, average='macro', zero_division=0):.4f} recall_macro = {recall_score(y_val, val_pred, average='macro', zero_division=0):.4f}")
    return sgd, scaler

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

    # Standardize (fit on train)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

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
                 factor_list: List[str],
                 N_list: List[int],
                 alpha_map: Dict[int, float], # TODO: alpha_map 目前没用上
                 label_prefix: str = "label_",
                 out_dir: str = "./results",
                 device: str = "cuda"):
    os.makedirs(out_dir, exist_ok=True)
    # TODO：暂时用 val 作为 test 看看效果
    test_df = val_df

    results = {}

    for N in N_list:
        print("\n" + "="*40)
        print(f"Start training for N={N}")
        label_col = f"{label_prefix}{N}"

        # 过滤掉 label NaN 的行
        # tr = train_df.dropna(subset=[label_col])
        # va = val_df.dropna(subset=[label_col])
        # te = test_df.dropna(subset=[label_col])
        tr = train_df
        va = val_df
        te = test_df
        print(f"effective samples -> train: {len(tr)} val: {len(va)} test: {len(te)}")

        if len(tr) < 1000 or len(va) < 200:
            print(f"样本太少，跳过 N={N}")
            continue

        # 提取 X,y
        X_train = tr[factor_list].values.astype(np.float32)
        y_train = tr[label_col].values.astype(np.int64)
        X_val = va[factor_list].values.astype(np.float32)
        y_val = va[label_col].values.astype(np.int64)
        X_test = te[factor_list].values.astype(np.float32)
        y_test = te[label_col].values.astype(np.int64)
        feature_names = factor_list

        # ---------------- LightGBM ----------------
        print("Training LightGBM (GPU)...")
        # LightGBM 对 label 要是 0..K-1
        # bst = train_lightgbm_gpu(X_train, y_train, X_val, y_val, feature_names,
        #                          save_path=os.path.join(out_dir, f"lgb_N{N}.txt"))
        # # predict test
        # y_pred_proba = bst.predict(X_test, num_iteration=bst.best_iteration)
        # y_pred = np.argmax(y_pred_proba, axis=1)
        # metrics_lgb = classification_metrics(y_test, y_pred)
        # print(f"LightGBM N={N} test f0.5_macro: {metrics_lgb['f0.5_macro']:.4f} precision_macro: {metrics_lgb['precision_macro']:.4f} recall_macro: {metrics_lgb['recall_macro']:.4f}")
        # # 计算LightGBM的PNL
        # pnl_lgb = calculate_pnl(test_df, y_pred, N)
        # print(f"LightGBM PNL: {pnl_lgb['bps_pnl']:.2f} bps, Sharpe: {pnl_lgb['sharpe_ratio']:.2f}")
        # ---------------- SGDClassifier ----------------
        print("Training SGDClassifier (incremental logistic)...")
        sgd, sgd_scaler = train_sgd_incremental(X_train, y_train, X_val, y_val,
                                                classes=[0,1,2], max_epochs=3, batch_size=200000)
        X_test_scaled = sgd_scaler.transform(X_test)
        y_pred_sgd = sgd.predict(X_test_scaled)
        metrics_sgd = classification_metrics(y_test, y_pred_sgd)
        print(f"SGD N={N} test f0.5_macro: {metrics_sgd['f0.5_macro']:.4f} precision_macro: {metrics_sgd['precision_macro']:.4f} recall_macro: {metrics_sgd['recall_macro']:.4f}")
        # 计算SGD的PNL
        pnl_sgd = calculate_pnl(test_df, y_pred_sgd, N)
        print(f"SGD PNL: {pnl_sgd['bps_pnl']:.2f} bps, Sharpe: {pnl_sgd['sharpe_ratio']:.2f}")
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
        print(f"MLP N={N} test f0.5_macro: {metrics_mlp['f0.5_macro']:.4f} precision_macro: {metrics_mlp['precision_macro']:.4f} recall_macro: {metrics_mlp['recall_macro']:.4f}")
        # 计算MLP的PNL
        pnl_mlp = calculate_pnl(test_df, preds, N)
        print(f"MLP PNL: {pnl_mlp['bps_pnl']:.2f} bps, Sharpe: {pnl_mlp['sharpe_ratio']:.2f}")

        # 存储结果（包含PNL）
        results[N] = {
            "lgb": (metrics_lgb, bst.best_iteration, pnl_lgb),
            "sgd": (metrics_sgd, None, pnl_sgd),
            "mlp": (metrics_mlp, model_path, pnl_mlp)
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

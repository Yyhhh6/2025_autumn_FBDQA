import xgboost as xgb
import numpy as np
from typing import Optional
import os
from datetime import datetime

def pnl_weighted_softmax_obj(preds, dtrain, weight1=2.0, weight2=0.3, weight3=2.0):
    """
    preds: raw margin, shape (N * K,)
    y: label in {0,1,2}
    """
    y = dtrain.get_label().astype(int)
    K = 3
    N = y.shape[0]

    preds = preds.reshape(N, K)

    # softmax
    exp_preds = np.exp(preds - np.max(preds, axis=1, keepdims=True))
    prob = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)

    grad = prob.copy()
    grad[np.arange(N), y] -= 1.0

    # ===== 核心：方向权重 =====
    # 下跌(0) / 上涨(2) 权重大，中性(1) 小
    class_weight = np.array([weight1, weight2, weight3])
    grad *= class_weight[y][:, None]

    # Hessian（近似）
    hess = prob * (1.0 - prob)
    hess *= class_weight[y][:, None]

    return grad.reshape(-1), hess.reshape(-1)

# 定义一个包装函数，把权重固定
def make_weighted_softmax_obj(weight1, weight2, weight3):
    def weighted_softmax_obj(preds, dtrain):
        return pnl_weighted_softmax_obj(preds, dtrain, weight1=weight1, weight2=weight2, weight3=weight3)
    return weighted_softmax_obj

class XGBModel:
    """
    XGBoost multi-class model for LOB prediction
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        if model_path is not None:
            self.load(model_path)

    # =========================
    # Training
    # =========================
    def train(
        self,
        X_train: np.ndarray,    # (N, 43)
        y_train: np.ndarray,    # (N,)
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
        seed: int = 42,
        N: int = 0,
        weight1: float=1.0,
        weight2: float=0.3,
        weight3: float=1.0,
        max_depth: int = 3,
        subsample: float = 0.5,
        colsample_bytree: float = 0.48,
        min_child_weight: int = 18,
        gamma: float = 4.3,
        save_path: str = "./models/",
        sym: str = "all",
    ):
        """
        Train XGBoost from scratch
        """

        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "max_depth": max_depth,   # 3 → 4
            "eta": 0.02,
            "subsample": subsample,   # 0.5 → 0.6
            "colsample_bytree": colsample_bytree,   # 0.48 → 0.35 / 0.4
            "min_child_weight": min_child_weight,   # 18 → 10 / 12
            "max_delta_step": 1,
            "gamma": gamma,    # 4.3 → 2.0 / 3.0
            "lambda": 7.5,
            "alpha": 0.25,
            "device": "cuda",
            "tree_method": "hist", 
            "seed": seed,
        }

        dtrain = xgb.QuantileDMatrix(X_train, label=y_train)

        if X_valid is not None and y_valid is not None:
            dvalid = xgb.QuantileDMatrix(X_valid, label=y_valid)
            evals = [(dvalid, "valid")]

        from xgboost.callback import EarlyStopping
        es = EarlyStopping(
            rounds=early_stopping_rounds,
            metric_name='mlogloss',  # 必须与日志输出的指标名一致
            data_name='valid',       # 必须与 evals 中的名字 "valid" 一致
            save_best=True           # 自动保存最优模型
        )

        self.model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            obj=make_weighted_softmax_obj(weight1=weight1, weight2=weight2, weight3=weight3), 
            early_stopping_rounds=early_stopping_rounds if len(evals) > 1 else None,
            callbacks=[es],
            verbose_eval=50,
        )

        os.makedirs(save_path, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_path, f"model_{N}_{sym}_{timestamp}.json")
        self.model.save_model(filename)
        print(f"Model for N={N} trained and saved.")

    # =========================
    # Inference
    # =========================
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns probability: (N, 3)
        """
        # dmat = xgb.DMatrix(X)
        dmat = xgb.DMatrix(X)
        # best_iter_str = self.model.get_attr("best_iteration")

        # return self.model.predict(dmat, iteration_range=(0, int(best_iter_str) + 1))
        return self.model.predict(dmat)

    # =========================
    # Save / Load
    # =========================
    def save(self, path: str):
        self.model.save_model(path)

    def load(self, path: str):
        self.model = xgb.Booster()
        self.model.load_model(path)

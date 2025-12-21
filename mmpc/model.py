import xgboost as xgb
import numpy as np
from typing import Optional

def pnl_weighted_softmax_obj(preds, dtrain):
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
    class_weight = np.array([2.0, 0.3, 2.0])
    grad *= class_weight[y][:, None]

    # Hessian（近似）
    hess = prob * (1.0 - prob)
    hess *= class_weight[y][:, None]

    return grad.reshape(-1), hess.reshape(-1)


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
    ):
        """
        Train XGBoost from scratch
        """

        params = {
            # "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": ["mlogloss", "auc"],
            "max_depth": 4,
            "eta": 0.03,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "min_child_weight": 5,
            "max_delta_step": 1,
            "gamma": 1.0,
            "lambda": 5.0,
            "alpha": 0.5,
            "device": "cuda",
            "tree_method": "hist", 
            "seed": seed,
        }

        # dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain = xgb.QuantileDMatrix(X_train, label=y_train)

        # evals = [(dtrain, "train")]
        if X_valid is not None and y_valid is not None:
            dvalid = xgb.QuantileDMatrix(X_valid, label=y_valid)
            evals = [(dvalid, "valid")]

        self.model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            obj=pnl_weighted_softmax_obj, 
            early_stopping_rounds=early_stopping_rounds if len(evals) > 1 else None,
            verbose_eval=50,
        )

        self.model.save_model(f"model_{N}.json")
        print(f"Model for N={N} trained and saved.")

    # =========================
    # Inference
    # =========================
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns probability: (N, 3)
        """
        # dmat = xgb.DMatrix(X)
        dmat = xgb.QuantileDMatrix(X)
        return self.model.predict(dmat)

    # =========================
    # Save / Load
    # =========================
    def save(self, path: str):
        self.model.save_model(path)

    def load(self, path: str):
        self.model = xgb.Booster()
        self.model.load_model(path)

import xgboost as xgb
import numpy as np
from typing import Optional


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
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "max_depth": 6,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "lambda": 1.0,
            "alpha": 0.0,
            # 🔥 GPU 关键参数
            "device": "cuda",
            "tree_method": "hist",     # CPU-friendly & fast
            "seed": seed,
        }

        dtrain = xgb.DMatrix(X_train, label=y_train)

        evals = [(dtrain, "train")]
        if X_valid is not None and y_valid is not None:
            dvalid = xgb.DMatrix(X_valid, label=y_valid)
            evals.append((dvalid, "valid"))

        self.model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
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
        dmat = xgb.DMatrix(X)
        return self.model.predict(dmat)

    # =========================
    # Save / Load
    # =========================
    def save(self, path: str):
        self.model.save_model(path)

    def load(self, path: str):
        self.model = xgb.Booster()
        self.model.load_model(path)

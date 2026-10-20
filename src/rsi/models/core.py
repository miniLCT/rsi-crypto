"""Model zoo behind one interface: fit(X, y) / predict(X), plus (de)serialisation."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

DEFAULT_LGBM = {
    "objective": "huber",
    "alpha": 1.5,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 2000,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "max_bin": 63,
    "num_threads": 16,
    "verbose": -1,
}


class ModelSpec(BaseModel):
    kind: Literal["lgbm", "ridge", "factor"] = "lgbm"
    params: dict = Field(default_factory=dict)
    num_boost_round: int = 400
    early_stopping: int = 50
    valid_frac: float = 0.15          # tail of each training window used for early stopping
    train_stride: int = 2             # use every k-th bar for training (labels overlap anyway)
    ridge_alpha: float = 10.0
    weights: dict[str, float] = Field(default_factory=dict)  # kind=factor: column -> weight
    seed: int = 7


class Model:
    def __init__(self, spec: ModelSpec, columns: list[str]):
        self.spec = spec
        self.columns = columns
        self.obj = None
        self.pred_std = 1.0
        self.best_iter: int | None = None
        self.importance: dict[str, float] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y: np.ndarray, Xv: np.ndarray | None = None, yv: np.ndarray | None = None) -> "Model":
        k = self.spec.kind
        if k == "lgbm":
            import lightgbm as lgb
            params = {**DEFAULT_LGBM, **self.spec.params, "seed": self.spec.seed}
            dtr = lgb.Dataset(X, y, feature_name=self._safe_cols(), free_raw_data=True)
            valid = [lgb.Dataset(Xv, yv, reference=dtr)] if Xv is not None and len(Xv) else []
            cbs = [lgb.early_stopping(self.spec.early_stopping, verbose=False)] if valid else []
            self.obj = lgb.train(params, dtr, num_boost_round=self.spec.num_boost_round, valid_sets=valid, callbacks=cbs)
            self.best_iter = self.obj.best_iteration or self.spec.num_boost_round
            gain = self.obj.feature_importance("gain")
            tot = gain.sum() or 1.0
            self.importance = {c: float(g / tot) for c, g in zip(self.columns, gain)}
        elif k == "ridge":
            Xf = np.nan_to_num(X.astype(np.float64))
            mu, sd = Xf.mean(0), Xf.std(0) + 1e-12
            Z = (Xf - mu) / sd
            A = Z.T @ Z + self.spec.ridge_alpha * len(Z) * 1e-3 * np.eye(Z.shape[1])
            beta = np.linalg.solve(A, Z.T @ y)
            self.obj = {"mu": mu, "sd": sd, "beta": beta}
            self.importance = {c: float(abs(b)) for c, b in zip(self.columns, beta)}
        elif k == "factor":
            self.obj = {"w": np.array([self.spec.weights.get(c, 0.0) for c in self.columns])}
        p = self._raw_predict(X)
        s = float(np.nanstd(p))
        self.pred_std = s if s > 0 else 1.0
        return self

    def _safe_cols(self) -> list[str]:
        return [c.replace(":", "__") for c in self.columns]

    # ------------------------------------------------------------------ predict
    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        k = self.spec.kind
        if k == "lgbm":
            return self.obj.predict(X, num_iteration=self.best_iter, num_threads=self.spec.params.get("num_threads", 16))
        if k == "ridge":
            Z = (np.nan_to_num(X.astype(np.float64)) - self.obj["mu"]) / self.obj["sd"]
            return Z @ self.obj["beta"]
        return np.nan_to_num(X.astype(np.float64)) @ self.obj["w"]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Score in units of training-prediction std (comparable across retrains)."""
        if len(X) == 0:
            return np.zeros(0)
        return self._raw_predict(X) / self.pred_std

    # ------------------------------------------------------------------ io
    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        meta = {"spec": self.spec.model_dump(), "columns": self.columns, "pred_std": self.pred_std,
                "best_iter": self.best_iter, "importance": self.importance}
        (d / "model_meta.json").write_text(json.dumps(meta, indent=1))
        if self.spec.kind == "lgbm":
            self.obj.save_model(str(d / "model.lgb"), num_iteration=self.best_iter)
        else:
            with open(d / "model.pkl", "wb") as f:
                pickle.dump(self.obj, f)

    @classmethod
    def load(cls, d: Path) -> "Model":
        meta = json.loads((d / "model_meta.json").read_text())
        m = cls(ModelSpec(**meta["spec"]), meta["columns"])
        m.pred_std, m.best_iter, m.importance = meta["pred_std"], meta["best_iter"], meta["importance"]
        if m.spec.kind == "lgbm":
            import lightgbm as lgb
            m.obj = lgb.Booster(model_file=str(d / "model.lgb"))
            m.best_iter = None  # saved model is already truncated
        else:
            with open(d / "model.pkl", "rb") as f:
                m.obj = pickle.load(f)
        return m

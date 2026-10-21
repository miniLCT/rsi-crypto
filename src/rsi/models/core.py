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
    kind: Literal["lgbm", "ridge", "factor", "signs"] = "lgbm"
    params: dict = Field(default_factory=dict)
    num_boost_round: int = 400
    early_stopping: int = 50
    early_stop_metric: Literal["loss", "corr"] = "corr"
    valid_frac: float = 0.15          # tail of each training window used for early stopping
    train_stride: int = 2             # use every k-th bar for training (labels overlap anyway)
    ridge_alpha: float = 10.0
    weights: dict[str, float] = Field(default_factory=dict)  # kind=factor: column -> weight
    # in-fold feature selection: keep columns whose yearly IC (training window only) mostly shares one sign
    select_stable: bool = False
    select_min_frac: float = 0.75
    select_min_abs_ic: float = 0.003
    recency_halflife_days: float | None = None   # sample weight 0.5 ** (age / half-life), age from the training end
    seed: int = 7


class Model:
    def __init__(self, spec: ModelSpec, columns: list[str]):
        self.spec = spec
        self.columns = columns
        self.obj = None
        self.pred_std = 1.0
        self.best_iter: int | None = None
        self.importance: dict[str, float] = {}
        self.keep: np.ndarray | None = None   # selected column mask (None = all)

    @staticmethod
    def stable_columns(X: np.ndarray, y: np.ndarray, groups: np.ndarray, min_frac: float, min_abs: float) -> np.ndarray:
        """Per-group (year) correlation of every column with the target; keep consistent-sign columns."""
        ics = []
        for g in np.unique(groups):
            m = groups == g
            if m.sum() < 5000:
                continue
            Xg = np.nan_to_num(X[m].astype(np.float64))
            yg = y[m] - y[m].mean()
            Xg -= Xg.mean(0)
            den = np.sqrt((Xg ** 2).sum(0) * (yg ** 2).sum()) + 1e-12
            ics.append((Xg * yg[:, None]).sum(0) / den)
        if len(ics) < 2:
            return np.ones(X.shape[1], dtype=bool)
        ics = np.vstack(ics)
        sign = np.sign(ics.mean(0))
        agree = (np.sign(ics) == sign).mean(0)
        return (agree >= min_frac) & (np.abs(ics.mean(0)) >= min_abs)

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y: np.ndarray, Xv: np.ndarray | None = None, yv: np.ndarray | None = None,
            groups: np.ndarray | None = None, age_days: np.ndarray | None = None) -> "Model":
        self._w = None
        if self.spec.recency_halflife_days and age_days is not None:
            self._w = 0.5 ** (np.asarray(age_days, dtype=np.float64) / self.spec.recency_halflife_days)
        if self.spec.select_stable and groups is not None:
            self.keep = self.stable_columns(X, y, groups, self.spec.select_min_frac, self.spec.select_min_abs_ic)
            if not self.keep.any():
                self.keep[:] = True
            X = X[:, self.keep]
            Xv = Xv[:, self.keep] if Xv is not None else None
        self._fit(X, y, Xv, yv)
        return self

    def _cols(self) -> list[str]:
        return [c for c, k in zip(self.columns, self.keep if self.keep is not None else [True] * len(self.columns)) if k]

    def _fit(self, X: np.ndarray, y: np.ndarray, Xv: np.ndarray | None, yv: np.ndarray | None) -> None:
        k = self.spec.kind
        cols = self._cols()
        if k == "lgbm":
            import lightgbm as lgb
            params = {**DEFAULT_LGBM, **self.spec.params, "seed": self.spec.seed}
            dtr = lgb.Dataset(X, y, weight=self._w, feature_name=[c.replace(":", "__") for c in cols], free_raw_data=True)
            valid = [lgb.Dataset(Xv, yv, reference=dtr)] if Xv is not None and len(Xv) else []
            cbs = [lgb.early_stopping(self.spec.early_stopping, first_metric_only=True, verbose=False)] if valid else []
            feval = None
            if valid and self.spec.early_stop_metric == "corr":
                # weak signals barely move the loss; stop on prediction/label correlation (IC proxy) instead
                params = {**params, "metric": "None"}

                def feval(preds, data):
                    lab = data.get_label()
                    c = np.corrcoef(preds, lab)[0, 1] if np.std(preds) > 0 else 0.0
                    return "corr", float(c), True
            self.obj = lgb.train(params, dtr, num_boost_round=self.spec.num_boost_round, valid_sets=valid,
                                 callbacks=cbs, feval=feval)
            self.best_iter = self.obj.best_iteration or self.spec.num_boost_round
            gain = self.obj.feature_importance("gain")
            tot = gain.sum() or 1.0
            self.importance = {c: float(g / tot) for c, g in zip(cols, gain)}
        elif k == "ridge":
            # chunked normal equations: memory stays ~ one chunk regardless of rows x columns
            n, F = X.shape
            w = self._w if self._w is not None else np.ones(n)
            W = w.sum()
            s1, s2 = np.zeros(F), np.zeros(F)
            for i in range(0, n, 200_000):
                c = np.nan_to_num(np.asarray(X[i:i + 200_000], dtype=np.float64))
                wi = w[i:i + 200_000, None]
                s1 += (wi * c).sum(0)
                s2 += (wi * c * c).sum(0)
            mu = s1 / W
            sd = np.sqrt(np.maximum(s2 / W - mu * mu, 0)) + 1e-12
            A = np.zeros((F, F))
            b = np.zeros(F)
            for i in range(0, n, 200_000):
                Z = (np.nan_to_num(np.asarray(X[i:i + 200_000], dtype=np.float64)) - mu) / sd
                wi = w[i:i + 200_000]
                A += Z.T @ (Z * wi[:, None])
                b += Z.T @ (y[i:i + 200_000] * wi)
            A += self.spec.ridge_alpha * W * 1e-3 * np.eye(F)
            beta = np.linalg.solve(A, b)
            self.obj = {"mu": mu, "sd": sd, "beta": beta}
            self.importance = {c: float(abs(b)) for c, b in zip(cols, beta)}
        elif k == "factor":
            self.obj = {"w": np.array([self.spec.weights.get(c, 0.0) for c in cols])}
        elif k == "signs":
            # equal-weight combination; each column's sign estimated on the training window only
            Xf = np.nan_to_num(np.asarray(X, dtype=np.float64))
            Xf -= Xf.mean(0)
            yc = y - y.mean()
            corr = (Xf * yc[:, None]).sum(0) / (np.sqrt((Xf ** 2).sum(0) * (yc ** 2).sum()) + 1e-12)
            w = np.where(np.abs(corr) >= self.spec.select_min_abs_ic, np.sign(corr), 0.0)
            self.obj = {"w": w / max(np.abs(w).sum(), 1.0), "corr": corr}
            self.importance = {c: float(v) for c, v in zip(cols, corr)}
        p = self._raw_predict(X)
        s = float(np.nanstd(p))
        self.pred_std = s if s > 0 else 1.0

    def _safe_cols(self) -> list[str]:
        return [c.replace(":", "__") for c in self.columns]

    # ------------------------------------------------------------------ predict
    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        k = self.spec.kind
        if k == "lgbm":
            return self.obj.predict(X, num_iteration=self.best_iter, num_threads=self.spec.params.get("num_threads", 16))
        if k == "ridge":
            o = self.obj
            out = np.empty(len(X))
            for i in range(0, len(X), 200_000):
                out[i:i + 200_000] = ((np.nan_to_num(np.asarray(X[i:i + 200_000], dtype=np.float64)) - o["mu"]) / o["sd"]) @ o["beta"]
            return out
        return np.nan_to_num(np.asarray(X, dtype=np.float64)) @ self.obj["w"]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Score in units of training-prediction std (comparable across retrains). X has all `columns`."""
        if len(X) == 0:
            return np.zeros(0)
        if self.keep is not None:
            X = X[:, self.keep]
        return self._raw_predict(X) / self.pred_std

    # ------------------------------------------------------------------ io
    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        meta = {"spec": self.spec.model_dump(), "columns": self.columns, "pred_std": self.pred_std,
                "best_iter": self.best_iter, "importance": self.importance,
                "keep": None if self.keep is None else self.keep.tolist()}
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
        m.keep = np.array(meta["keep"], dtype=bool) if meta.get("keep") is not None else None
        if m.spec.kind == "lgbm":
            import lightgbm as lgb
            m.obj = lgb.Booster(model_file=str(d / "model.lgb"))
            m.best_iter = None  # saved model is already truncated
        else:
            with open(d / "model.pkl", "rb") as f:
                m.obj = pickle.load(f)
        return m

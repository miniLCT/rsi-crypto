"""RISK-06: disaster-halt thresholds from a day-block bootstrap of OOS bar returns.

day_loss = max(worst_day_mult * |worst backtest day|, day_vol_mult * daily vol, |tail quantile of bootstrapped days|)
max_dd   = (1 - tail) quantile of the bootstrapped maximum drawdown over a path as long as the OOS period
Both are expressed at the harness evaluation vol target and scaled linearly to the live exposure.
"""
from __future__ import annotations

import numpy as np

from rsi.config import DisasterCfg


def _max_dd(log_path: np.ndarray) -> np.ndarray:
    eq = np.exp(np.cumsum(log_path, axis=-1))
    peak = np.maximum.accumulate(np.maximum(eq, 1.0), axis=-1)
    return np.max(1 - eq / peak, axis=-1)


def calibrate(bar_returns: np.ndarray, cfg: DisasterCfg, seed: int = 0) -> dict:
    r = np.log1p(bar_returns[np.isfinite(bar_returns)])
    b = cfg.block_bars
    n_days = len(r) // b
    if n_days < 30:
        return {"day_loss": None, "max_dd": None, "n_days": n_days}
    days = r[: n_days * b].reshape(n_days, b)
    day_ret = days.sum(axis=1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_days, size=(cfg.n_boot, n_days))
    boot_days = day_ret[idx]
    tail_day = float(np.quantile(boot_days, cfg.tail))
    day_loss = max(cfg.worst_day_mult * -float(day_ret.min()), cfg.day_vol_mult * float(day_ret.std()), -tail_day)
    max_dd = float(np.quantile(_max_dd(boot_days), 1 - cfg.tail))
    return {"day_loss": float(-np.expm1(-day_loss)), "max_dd": max_dd, "n_days": n_days,
            "worst_day": float(np.expm1(day_ret.min())), "day_vol": float(day_ret.std())}


def count_triggers(times_ms: np.ndarray, equity: np.ndarray, equity0: float, th: dict) -> int:
    """How often the calibrated halt would have fired on this equity curve (healthy strategies ~ 0)."""
    if not th.get("day_loss"):
        return 0
    e = np.concatenate([[equity0], equity])
    day = np.concatenate([[times_ms[0] // 86_400_000], times_ms // 86_400_000])
    n = 0
    start = e[0]
    peak = e[0]
    for i in range(1, len(e)):
        if day[i] != day[i - 1]:
            start = e[i - 1]
        peak = max(peak, e[i])
        if e[i] < start * (1 - th["day_loss"]) or e[i] < peak * (1 - th["max_dd"]):
            n += 1
            start = peak = e[i]
    return n

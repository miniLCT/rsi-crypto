"""Performance statistics, including multiple-testing-aware Sharpe deflation."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

EULER = 0.5772156649


def sharpe(r: np.ndarray, periods: int) -> float:
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * math.sqrt(periods))


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(np.max((peak - equity) / peak))


def psr(sr_obs: float, sr_ref: float, n: int, skew: float, kurt: float) -> float:
    """Probabilistic Sharpe ratio (per-period SR inputs)."""
    denom = math.sqrt(max(1e-12, 1 - skew * sr_obs + (kurt - 1) / 4 * sr_obs ** 2))
    return float(stats.norm.cdf((sr_obs - sr_ref) * math.sqrt(max(n - 1, 1)) / denom))


def expected_max_sr(n_trials: int, sr_var: float) -> float:
    """Expected maximum of n_trials Sharpe ratios under the null (per-period units)."""
    if n_trials <= 1:
        return 0.0
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * math.e))
    return math.sqrt(max(sr_var, 1e-12)) * ((1 - EULER) * z1 + EULER * z2)


def deflated_sharpe(daily_r: np.ndarray, n_trials: int, trial_sr_var: float | None = None) -> float:
    """DSR on daily returns. `trial_sr_var` is the variance of daily SRs across trials in the ledger."""
    r = daily_r[np.isfinite(daily_r)]
    if len(r) < 30 or r.std() == 0:
        return 0.0
    sr = r.mean() / r.std(ddof=1)
    var = max(trial_sr_var or 0.0, 1.0 / len(r))   # never below the sampling variance of a daily SR
    sr0 = expected_max_sr(max(n_trials, 1), var)
    return psr(sr, sr0, len(r), float(stats.skew(r)), float(stats.kurtosis(r, fisher=False)))


def daily_returns(times_ms: np.ndarray, equity: np.ndarray, equity0: float) -> pd.Series:
    s = pd.Series(equity, index=pd.to_datetime(times_ms, unit="ms"))
    d = s.resample("1D").last().dropna()
    prev = d.shift(1)
    prev.iloc[0] = equity0
    return d / prev - 1


def summarize(res, bars_per_year: int, n_trials: int = 1, trial_sr_var: float | None = None) -> dict:
    r = res.returns
    dr = daily_returns(res.times, res.equity, res.equity0)
    years = len(r) / bars_per_year
    total = res.equity[-1] / res.equity0 - 1 if len(res.equity) else 0.0
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else -1.0
    mdd = max_drawdown(np.concatenate([[res.equity0], res.equity]))
    tr = res.trades
    avg_eq = float(np.mean(res.equity)) if len(res.equity) else res.equity0
    out = {
        "sharpe": sharpe(r, bars_per_year),             # EVAL-01: per-bar returns, annualised
        "sharpe_daily": sharpe(dr.to_numpy(), 365),
        "sortino": _sortino(r, bars_per_year),
        "cagr": float(cagr),
        "total_return": float(total),
        "max_drawdown": mdd,
        "calmar": float(cagr / mdd) if mdd > 0 else 0.0,
        "vol_ann": float(np.std(dr) * math.sqrt(365)),
        "avg_gross": float(np.mean(res.gross)) if len(res.gross) else 0.0,
        "avg_net": float(np.mean(res.net)) if len(res.net) else 0.0,
        "turnover_ann": float(0.5 * res.turnover_usd / avg_eq / max(years, 1e-9)),   # one-sided
        "fees_frac": float(res.fees_usd / res.equity0),
        "funding_frac": float(res.funding_usd / res.equity0),
        "n_trades": int(len(tr)),
        "hit_rate": float((tr["pnl_usd"] > 0).mean()) if len(tr) else 0.0,
        "avg_hold_bars": float(((tr["close_ms"] - tr["open_ms"]) / 1_800_000).mean()) if len(tr) else 0.0,
        "long_pnl_frac": float(tr.loc[tr["side"] > 0, "pnl_usd"].sum() / res.equity0) if len(tr) else 0.0,
        "short_pnl_frac": float(tr.loc[tr["side"] < 0, "pnl_usd"].sum() / res.equity0) if len(tr) else 0.0,
        "stop_exits": int(tr["reason"].isin(["stop", "hard_stop"]).sum()) if len(tr) else 0,
        "days": int(len(dr)),
        "dsr": deflated_sharpe(dr.to_numpy(), n_trials, trial_sr_var),
        "daily_sr": float(dr.mean() / dr.std(ddof=1)) if len(dr) > 1 and dr.std() > 0 else 0.0,
    }
    return out


def _sortino(r: np.ndarray, periods: int) -> float:
    r = r[np.isfinite(r)]
    dn = r[r < 0]
    if len(r) < 2 or len(dn) < 2:
        return 0.0
    return float(r.mean() / np.sqrt((dn ** 2).mean()) * math.sqrt(periods))


def ic_stats(score: np.ndarray, fwd: np.ndarray, mask: np.ndarray, rows: slice | None = None) -> dict:
    """Mean cross-sectional Spearman IC and its t-stat."""
    from rsi.features.engine import xs_rank
    s = score[rows] if rows is not None else score
    f = fwd[rows] if rows is not None else fwd
    m = mask[rows] if rows is not None else mask
    ok = m & np.isfinite(s) & np.isfinite(f)
    rs = np.nan_to_num(xs_rank(s, ok).astype(np.float64))
    rf = np.nan_to_num(xs_rank(f, ok).astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        ics = (rs * rf).sum(1) / np.sqrt((rs ** 2).sum(1) * (rf ** 2).sum(1))
    ics = ics[(ok.sum(1) >= 8) & np.isfinite(ics)]
    if len(ics) < 2:
        return {"ic": 0.0, "ic_t": 0.0, "n": len(ics)}
    return {"ic": float(ics.mean()), "ic_t": float(ics.mean() / ics.std(ddof=1) * math.sqrt(len(ics))), "n": len(ics)}

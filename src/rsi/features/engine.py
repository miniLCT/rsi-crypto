"""Batch (research) and trailing-window (live) feature computation.

Both paths call the same `Feature.compute`. The model input contract is
float32; batch results are cached as float32 and the live path casts the same
way, so the numbers a model sees are identical in research and production.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rsi.config import cache_dir
from rsi.data.panel import Panel
from rsi.features import registry
from rsi.features.registry import FrameView
from rsi.utils.log import get_logger

log = get_logger("features")

DTYPE = np.float32


def _cache_path(panel: Panel, feat: registry.Feature) -> Path:
    return cache_dir() / "features" / panel.fingerprint / f"{feat.name}-{feat.src_hash}.npy"


def compute_batch(panel: Panel, names: list[str], use_cache: bool = True) -> dict[str, np.ndarray]:
    fv = FrameView(panel)
    out: dict[str, np.ndarray] = {}
    computed = 0
    for n in names:
        feat = registry.get(n)
        p = _cache_path(panel, feat)
        if use_cache and p.exists():
            out[n] = np.load(p, mmap_mode="r")
            continue
        arr = feat.compute(fv).astype(DTYPE)
        computed += 1
        if use_cache:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp.npy")
            np.save(tmp, arr)
            tmp.replace(p)
        out[n] = arr
    if computed:
        log.info(f"computed {computed}/{len(names)} features (rest cached)")
    return out


def max_lookback(names: list[str]) -> int:
    return max((registry.get(n).lookback for n in names), default=1)


def compute_last(panel_window: Panel, names: list[str]) -> dict[str, np.ndarray]:
    """Features for the last bar only, from a trailing window (live path)."""
    need = max_lookback(names)
    if panel_window.T < need:
        raise ValueError(f"window has {panel_window.T} bars, features need {need}")
    fv = FrameView(panel_window)
    return {n: registry.get(n).compute(fv)[-1].astype(DTYPE) for n in names}


# ---------------------------------------------------------------- cross-sectional transforms
MIN_XS_COVERAGE = 0.5   # FEAT-04: a bar where < 50% of the universe has a value is not ranked at all


def _coverage_ok(v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n_mask = mask.sum(axis=1)
    n_val = np.isfinite(v).sum(axis=1)
    return (n_val >= MIN_XS_COVERAGE * n_mask) & (n_val > 0)


def xs_rank(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Percentile rank within the universe at each bar, centred to [-0.5, 0.5], ties averaged; NaN outside."""
    v = np.where(mask, x, np.nan).astype(np.float64)
    df = pd.DataFrame(v)
    r = df.rank(axis=1, method="average").to_numpy()
    cnt = df.count(axis=1).to_numpy()[:, None].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = np.where(cnt > 1, (r - 1) / (cnt - 1) - 0.5, np.where(np.isfinite(r), 0.0, np.nan))
    pct[~_coverage_ok(v, mask)] = np.nan
    return pct.astype(DTYPE)


def xs_zscore(x: np.ndarray, mask: np.ndarray, clip: float = 3.0) -> np.ndarray:
    """Robust z-score: (x - median) / (1.4826 * MAD), clipped to +-clip."""
    import warnings
    v = np.where(mask, x, np.nan).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(v, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(v - med), axis=1, keepdims=True) * 1.4826
        z = (v - med) / np.where(mad > 0, mad, np.nan)
    z = np.clip(z, -clip, clip)
    z[~_coverage_ok(v, mask)] = np.nan
    return z.astype(DTYPE)


def xs_neutralize(x: np.ndarray, controls: list[np.ndarray], mask: np.ndarray) -> np.ndarray:
    """Per-bar OLS residual of x on [1, controls...] over universe names with complete data."""
    y = np.where(mask, x, np.nan).astype(np.float64)
    Z = np.stack([np.ones_like(y)] + [np.asarray(c, dtype=np.float64) for c in controls], axis=-1)
    ok = np.isfinite(y) & np.all(np.isfinite(Z), axis=-1)
    Zm = np.where(ok[..., None], Z, 0.0)
    ym = np.where(ok, y, 0.0)
    ztz = np.einsum("tsi,tsj->tij", Zm, Zm) + 1e-9 * np.eye(Z.shape[-1])[None]
    zty = np.einsum("tsi,ts->ti", Zm, ym)
    beta = np.linalg.solve(ztz, zty[..., None])[..., 0]
    res = y - np.einsum("tsi,ti->ts", Z, beta)
    enough = ok.sum(axis=1) > Z.shape[-1] + 2
    res[~ok | ~enough[:, None]] = np.nan
    return res.astype(DTYPE)


def xs_demean(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = np.where(mask, x, np.nan).astype(np.float64)
    import warnings
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return v - np.nanmean(v, axis=1, keepdims=True)

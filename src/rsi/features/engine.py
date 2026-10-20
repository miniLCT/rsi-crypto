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
def xs_rank(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Percentile rank within the universe at each bar, centred to [-0.5, 0.5]; NaN outside."""
    v = pd.DataFrame(np.where(mask, x, np.nan).astype(np.float64))
    r = v.rank(axis=1, method="average").to_numpy()
    cnt = v.count(axis=1).to_numpy()[:, None].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = np.where(cnt > 1, (r - 1) / (cnt - 1) - 0.5, np.where(np.isfinite(r), 0.0, np.nan))
    return pct.astype(DTYPE)


def xs_zscore(x: np.ndarray, mask: np.ndarray, clip: float = 5.0) -> np.ndarray:
    v = np.where(mask, x, np.nan).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.nanmean(v, axis=1, keepdims=True)
        sd = np.nanstd(v, axis=1, keepdims=True)
        z = (v - mu) / np.where(sd > 0, sd, np.nan)
    return np.clip(z, -clip, clip).astype(DTYPE)


def xs_demean(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = np.where(mask, x, np.nan).astype(np.float64)
    import warnings
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return v - np.nanmean(v, axis=1, keepdims=True)

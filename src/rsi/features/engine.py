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


CHUNK = 120          # symbols per column chunk; time-series features are column-independent
ANCHORS = ("BTCUSDT",)   # market-context features read these columns, so every chunk carries them


def _compute_chunked(panel: Panel, names: list[str]) -> dict[str, np.ndarray]:
    """Compute features chunk-by-chunk over symbols to bound memory; exact because features only
    combine a symbol's own history with the anchor columns."""
    if panel.S <= CHUNK:
        fv = FrameView(panel)
        return {n: registry.get(n).compute(fv).astype(DTYPE) for n in names}
    out = {n: np.full((panel.T, panel.S), np.nan, dtype=DTYPE) for n in names}
    anchors = [s for s in ANCHORS if s in panel.symbols]
    close = np.asarray(panel["close"])
    alive = np.isfinite(close)
    first = np.where(alive.any(0), alive.argmax(0), panel.T)
    others = sorted((j for j, s in enumerate(panel.symbols) if s not in anchors), key=lambda j: first[j])
    L = max_lookback(names)
    for k in range(0, len(others), CHUNK):
        cols = others[k:k + CHUNK]
        # rows before (earliest listing - lookback) cannot influence any value of this chunk
        r0 = 0 if k == 0 else max(0, int(first[cols].min()) - L - 1)
        syms = anchors + [panel.symbols[j] for j in cols]
        sub = panel.slice(r0, panel.T).select_symbols(syms)
        fv = FrameView(sub)
        for n in names:
            arr = registry.get(n).compute(fv).astype(DTYPE)
            out[n][r0:, cols] = arr[:, len(anchors):]
            if k == 0:
                for a, s in enumerate(anchors):
                    out[n][:, panel.symbols.index(s)] = arr[:, a]
        del fv, sub
    return out


def compute_batch(panel: Panel, names: list[str], use_cache: bool = True) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    todo = []
    for n in names:
        p = _cache_path(panel, registry.get(n))
        if use_cache and p.exists():
            out[n] = np.load(p, mmap_mode="r")
        else:
            todo.append(n)
    computed = 0
    for n in todo:
        feat = registry.get(n)
        p = _cache_path(panel, feat)
        arr = _compute_chunked(panel, [n])[n]
        computed += 1
        if use_cache:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp.npy")
            np.save(tmp, arr)
            tmp.replace(p)
            arr = np.load(p, mmap_mode="r")
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


_IDX_MEMO: dict = {}


def _compress(x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gather each row's universe members into a dense [T, K] block (K = max members per row)."""
    key = (id(mask), mask.shape, int(mask.sum()))
    if key not in _IDX_MEMO:
        K = max(int(mask.sum(axis=1).max()), 1) if mask.size else 1
        idx = np.argsort(~mask, axis=1, kind="stable")[:, :K]
        if mask.shape[0] > 1000:
            _IDX_MEMO.clear()
            _IDX_MEMO[key] = (idx, np.take_along_axis(mask, idx, 1))
        else:
            return _gather(x, idx, np.take_along_axis(mask, idx, 1)), idx
    idx, valid = _IDX_MEMO[key]
    return _gather(x, idx, valid), idx


def _gather(x: np.ndarray, idx: np.ndarray, valid: np.ndarray) -> np.ndarray:
    v = np.take_along_axis(np.asarray(x, dtype=np.float64), idx, 1)
    v[~valid] = np.nan
    return v


def _avg_rank(v: np.ndarray) -> np.ndarray:
    """1-based average-tie ranks per row; NaN stays NaN (vectorised pandas rank(method='average'))."""
    T, K = v.shape
    order = np.argsort(np.where(np.isnan(v), np.inf, v), axis=1, kind="stable")
    sv = np.take_along_axis(v, order, 1)
    pos = np.broadcast_to(np.arange(K), (T, K))
    new = np.ones((T, K), dtype=bool)
    new[:, 1:] = sv[:, 1:] != sv[:, :-1]
    first = np.maximum.accumulate(np.where(new, pos, 0), axis=1)
    end = np.ones((T, K), dtype=bool)
    end[:, :-1] = new[:, 1:]
    last = np.flip(np.minimum.accumulate(np.flip(np.where(end, pos, K), 1), axis=1), 1)
    r_sorted = (first + last) / 2.0 + 1.0
    r_sorted[np.isnan(sv)] = np.nan
    r = np.empty_like(r_sorted)
    np.put_along_axis(r, order, r_sorted, 1)
    return r


def xs_rank(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Percentile rank within the universe at each bar, centred to [-0.5, 0.5], ties averaged; NaN outside."""
    mask = np.asarray(mask, dtype=bool)
    v, idx = _compress(x, mask)
    r = _avg_rank(v)
    cnt = np.isfinite(v).sum(axis=1, keepdims=True).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = np.where(cnt > 1, (r - 1) / (cnt - 1) - 0.5, np.where(np.isfinite(r), 0.0, np.nan))
    n_mask = mask.sum(axis=1)
    bad = (cnt[:, 0] < MIN_XS_COVERAGE * n_mask) | (cnt[:, 0] == 0)
    pct[bad] = np.nan
    out = np.full(np.shape(x), np.nan, dtype=DTYPE)
    np.put_along_axis(out, idx, np.where(np.take_along_axis(mask, idx, 1), pct, np.nan).astype(DTYPE), 1)
    return out


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


def _build_worker(args) -> int:
    panel_dir, names = args
    from rsi.data.panel import load_panel
    from pathlib import Path
    panel = load_panel(Path(panel_dir))
    compute_batch(panel, names)
    return len(names)


def build_parallel(panel_dir: str, panel: Panel, names: list[str], jobs: int = 3) -> None:
    """Fill the feature cache with `jobs` processes (features sliced round-robin; cache writes are atomic)."""
    from concurrent.futures import ProcessPoolExecutor
    todo = [n for n in names if not _cache_path(panel, registry.get(n)).exists()]
    chunks = [todo[i::jobs] for i in range(jobs) if todo[i::jobs]]
    with ProcessPoolExecutor(len(chunks) or 1) as ex:
        list(ex.map(_build_worker, [(panel_dir, c) for c in chunks]))

"""Model input columns = base features x cross-sectional transform, identical in batch and live.

Column naming:  <name> raw · r:<name> rank · z:<name> robust z-score · rn:/zn: neutralised to (vol, size)
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from rsi.config import cache_dir
from rsi.data.panel import Panel
from rsi.features import engine, registry

# features that are identical across symbols at a bar; cross-sectional transforms are meaningless (FEAT-03)
BROADCAST_GROUPS = {"calendar", "market_ctx"}
NEUTRAL_CONTROLS = ("rv_336", "log_adv_30d")


class FeatureSpec(BaseModel):
    include: list[str] = Field(default_factory=lambda: ["all"])
    transform: Literal["raw", "rank", "zscore", "both"] = "rank"
    neutralize: bool = False      # residualise cross-sectional columns on (vol, size) each bar

    def base_names(self) -> list[str]:
        return registry.resolve(self.include)

    def _xs_prefix(self) -> list[str]:
        p = {"rank": ["r"], "zscore": ["z"], "both": ["r"], "raw": []}[self.transform]
        return [x + "n" for x in p] if self.neutralize else p

    def columns(self) -> list[str]:
        cols = []
        for n in self.base_names():
            bc = registry.get(n).group in BROADCAST_GROUPS
            if self.transform in ("raw", "both") or bc:
                cols.append(n)
            if not bc:
                cols += [f"{p}:{n}" for p in self._xs_prefix()]
        return cols

    def compute_names(self) -> list[str]:
        names = self.base_names()
        if self.neutralize:
            names = names + [c for c in NEUTRAL_CONTROLS if c not in names]
        return names

    def lookback(self) -> int:
        return engine.max_lookback(self.compute_names())


def _transform(kind: str, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return engine.xs_rank(x, mask) if kind == "r" else engine.xs_zscore(x, mask)


def _controls(base: dict, mask: np.ndarray, kind: str) -> list[np.ndarray]:
    return [_transform(kind, np.asarray(base[c]), mask) for c in NEUTRAL_CONTROLS]


def build_batch(panel: Panel, spec: FeatureSpec, mask: np.ndarray, use_cache: bool = True,
                mask_key: str | None = None) -> dict[str, np.ndarray]:
    """`mask_key` identifies the universe mask (e.g. a hash of the universe spec) for caching transforms."""
    base = engine.compute_batch(panel, spec.compute_names(), use_cache=use_cache)
    out: dict[str, np.ndarray] = {}
    ctrl: dict[str, list] = {}
    for c in spec.columns():
        if ":" not in c:
            out[c] = base[c]
            continue
        prefix, name = c.split(":", 1)
        p = None
        if use_cache and mask_key:
            p = cache_dir() / "features" / panel.fingerprint / f"xs-{mask_key}" / f"{prefix}-{name}-{registry.get(name).src_hash}.npy"
            if p.exists():
                out[c] = np.load(p, mmap_mode="r")
                continue
        v = _transform(prefix[0], np.asarray(base[name]), mask)
        if prefix.endswith("n"):
            if prefix[0] not in ctrl:
                ctrl[prefix[0]] = _controls(base, mask, prefix[0])
            v = engine.xs_neutralize(v, ctrl[prefix[0]], mask)
        out[c] = v
        if p is not None:
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, v)
    return out


def build_last(window: Panel, spec: FeatureSpec, mask_last: np.ndarray) -> dict[str, np.ndarray]:
    """Columns for the final bar of a trailing window. `mask_last` is the universe row for that bar."""
    base = engine.compute_last(window, spec.compute_names())
    m = mask_last[None, :]
    out: dict[str, np.ndarray] = {}
    ctrl: dict[str, list] = {}
    for c in spec.columns():
        if ":" not in c:
            out[c] = base[c]
            continue
        prefix, name = c.split(":", 1)
        v = _transform(prefix[0], base[name][None, :], m)
        if prefix.endswith("n"):
            if prefix[0] not in ctrl:
                ctrl[prefix[0]] = [_transform(prefix[0], base[k][None, :], m) for k in NEUTRAL_CONTROLS]
            v = engine.xs_neutralize(v, ctrl[prefix[0]], m)
        out[c] = v[0]
    return out


def stack(cols: dict[str, np.ndarray], names: list[str], t_idx: np.ndarray, s_idx: np.ndarray) -> np.ndarray:
    X = np.empty((len(t_idx), len(names)), dtype=engine.DTYPE)
    for j, n in enumerate(names):
        X[:, j] = np.asarray(cols[n])[t_idx, s_idx]
    return X

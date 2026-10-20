"""Model input columns = base features x transform, computed identically in batch and live."""
from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from rsi.config import cache_dir
from rsi.data.panel import Panel
from rsi.features import engine, registry

# features that are identical across symbols at a bar; ranking them is meaningless
BROADCAST_GROUPS = {"calendar", "market_ctx"}


class FeatureSpec(BaseModel):
    include: list[str] = Field(default_factory=lambda: ["all"])
    transform: Literal["raw", "rank", "both"] = "rank"

    def base_names(self) -> list[str]:
        return registry.resolve(self.include)

    def columns(self) -> list[str]:
        cols = []
        for n in self.base_names():
            bc = registry.get(n).group in BROADCAST_GROUPS
            if self.transform in ("raw", "both") or bc:
                cols.append(n)
            if self.transform in ("rank", "both") and not bc:
                cols.append(f"r:{n}")
        return cols

    def lookback(self) -> int:
        return engine.max_lookback(self.base_names())


def build_batch(panel: Panel, spec: FeatureSpec, mask: np.ndarray, use_cache: bool = True,
                mask_key: str | None = None) -> dict[str, np.ndarray]:
    """`mask_key` identifies the universe mask (e.g. a hash of the universe spec) for caching ranks."""
    base = engine.compute_batch(panel, spec.base_names(), use_cache=use_cache)
    out: dict[str, np.ndarray] = {}
    for c in spec.columns():
        if c.startswith("r:"):
            name = c[2:]
            p = None
            if use_cache and mask_key:
                p = cache_dir() / "features" / panel.fingerprint / f"rank-{mask_key}" / f"{name}-{registry.get(name).src_hash}.npy"
                if p.exists():
                    out[c] = np.load(p, mmap_mode="r")
                    continue
            out[c] = engine.xs_rank(np.asarray(base[name]), mask)
            if p is not None:
                p.parent.mkdir(parents=True, exist_ok=True)
                np.save(p, out[c])
        else:
            out[c] = base[c]
    return out


def build_last(window: Panel, spec: FeatureSpec, mask_last: np.ndarray) -> dict[str, np.ndarray]:
    """Columns for the final bar of a trailing window. `mask_last` is the universe row for that bar."""
    base = engine.compute_last(window, spec.base_names())
    out: dict[str, np.ndarray] = {}
    for c in spec.columns():
        if c.startswith("r:"):
            out[c] = engine.xs_rank(base[c[2:]][None, :], mask_last[None, :])[0]
        else:
            out[c] = base[c]
    return out


def stack(cols: dict[str, np.ndarray], names: list[str], t_idx: np.ndarray, s_idx: np.ndarray) -> np.ndarray:
    X = np.empty((len(t_idx), len(names)), dtype=engine.DTYPE)
    for j, n in enumerate(names):
        X[:, j] = np.asarray(cols[n])[t_idx, s_idx]
    return X

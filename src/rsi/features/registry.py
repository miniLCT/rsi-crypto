"""Feature registry.

A feature is a pure function ``FrameView -> DataFrame[T, S]`` with a declared
``lookback``: the number of trailing bars (including the current one) that
fully determine the value at the last row. The contract checker verifies both
causality and the lookback for every feature, so the live engine can compute
exactly the same numbers from a trailing window.

Rules for feature authors (enforced by `rsi.features.contract`):
  * only finite rolling windows / shifts (no expanding, no EWM, no global stats)
  * NaN is allowed (insufficient history, not listed); +-inf is mapped to NaN
"""
from __future__ import annotations

import functools
import hashlib
import inspect
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from rsi.data.panel import Panel


class FrameView:
    """Lazy DataFrame accessor over a Panel with memoised intermediates."""

    def __init__(self, panel: Panel):
        self.panel = panel
        self._cache: dict[str, pd.DataFrame] = {}

    def __getattr__(self, name: str) -> pd.DataFrame:
        if name.startswith("_") or name == "panel":
            raise AttributeError(name)
        return self.get(name)

    def get(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            self._cache[name] = pd.DataFrame(np.asarray(self.panel[name], dtype=np.float64),
                                             index=self.panel.times, columns=self.panel.symbols)
        return self._cache[name]

    def memo(self, key: str, fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    @property
    def logret(self) -> pd.DataFrame:
        return self.memo("_logret", lambda: np.log(self.get("close")).diff())

    @property
    def times(self) -> np.ndarray:
        return self.panel.times

    def broadcast(self, s: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(np.repeat(s.to_numpy()[:, None], self.panel.S, axis=1), index=self.panel.times,
                            columns=self.panel.symbols)

    def column(self, field: str, symbol: str) -> pd.Series:
        df = self.get(field)
        if symbol in df.columns:
            return df[symbol]
        return pd.Series(np.nan, index=df.index)


@dataclass(frozen=True)
class Feature:
    name: str
    fn: Callable[[FrameView], pd.DataFrame]
    lookback: int
    group: str
    doc: str = ""

    def compute(self, fv: FrameView) -> np.ndarray:
        out = self.fn(fv)
        arr = out.to_numpy(dtype=np.float64) if isinstance(out, pd.DataFrame) else np.asarray(out, dtype=np.float64)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        return arr

    @functools.cached_property
    def src_hash(self) -> str:
        fn = self.fn
        parts = []
        while isinstance(fn, functools.partial):
            parts.append(repr(sorted(fn.keywords.items())) + repr(fn.args))
            fn = fn.func
        try:
            parts.append(inspect.getsource(fn))
        except (OSError, TypeError):
            parts.append(repr(fn))
        parts.append(str(self.lookback))
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]


REGISTRY: dict[str, Feature] = {}
NON_SIGNAL_GROUPS = {"size"}


def register(name: str, lookback: int, group: str, doc: str = ""):
    def deco(fn):
        if name in REGISTRY:
            raise ValueError(f"duplicate feature {name}")
        REGISTRY[name] = Feature(name, fn, int(lookback), group, doc)
        return fn
    return deco


def register_partial(name: str, fn: Callable, lookback: int, group: str, doc: str = "", **kw) -> None:
    register(name, lookback, group, doc)(functools.partial(fn, **kw))


def get(name: str) -> Feature:
    _ensure_loaded()
    if name not in REGISTRY:
        raise KeyError(f"unknown feature {name!r}")
    return REGISTRY[name]


def resolve(patterns: list[str]) -> list[str]:
    """Expand feature selectors: exact names, `group:<g>`, glob patterns, `-<selector>` to drop, or `all`.

    `all` means all *signal* features; group `size` (liquidity levels) is excluded and must be named.
    """
    import fnmatch
    _ensure_loaded()
    out: list[str] = []
    for p in patterns:
        if p == "all":
            hits = [n for n, f in REGISTRY.items() if f.group not in NON_SIGNAL_GROUPS]
        elif p.startswith("group:"):
            hits = [n for n, f in REGISTRY.items() if f.group == p[6:]]
        elif p.startswith("-"):
            drop = set(resolve([p[1:]]))
            out = [n for n in out if n not in drop]
            continue
        else:
            hits = [n for n in REGISTRY if fnmatch.fnmatch(n, p)]
            if not hits:
                raise KeyError(f"feature selector {p!r} matched nothing")
        out += [h for h in hits if h not in out]
    return out


_LOADED = False


def _ensure_loaded() -> None:
    global _LOADED
    if not _LOADED:
        _LOADED = True
        from rsi.features import library  # noqa: F401
        from rsi.features import custom  # noqa: F401

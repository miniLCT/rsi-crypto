"""Point-in-time tradable universe (UNI-01/02, DEC-02).

Membership for every bar of UTC day D is decided from data that closed at the
end of day D-1, so it is causal and identical in research and live.

Two masks:
  entry : rank <= size and listed >= min_history_days   -> may open / add
  hold  : rank <= keep_rank                              -> existing positions may be kept (no adds)
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from rsi.data.panel import Panel
from rsi.utils.timeutil import interval_ms

DAY_MS = 86_400_000


def _min_coverage() -> float:
    from rsi.config import data_config
    return data_config().quality.min_window_coverage


class UniverseSpec(BaseModel):
    kind: Literal["top_liquid", "fixed"] = "top_liquid"
    size: int = 100
    keep_rank: int = 120
    adv_days: int = 30
    min_history_days: int = 30
    symbols: list[str] = Field(default_factory=list)  # for kind=fixed

    def lookback(self, interval: str = "30m") -> int:
        bars_per_day = DAY_MS // interval_ms(interval)
        days = max(self.adv_days, self.min_history_days) + 2
        return days * bars_per_day if self.kind == "top_liquid" else 1


def _day_ref(panel: Panel) -> np.ndarray:
    step = interval_ms(panel.interval)
    pos_in_day = (panel.times % DAY_MS) // step
    return np.arange(panel.T) - pos_in_day - 1          # last bar of the previous UTC day


def universe_masks(panel: Panel, spec: UniverseSpec) -> tuple[np.ndarray, np.ndarray]:
    close = np.asarray(panel["close"])
    alive = np.isfinite(close)
    if spec.kind == "fixed":
        cols = np.array([s in set(spec.symbols) for s in panel.symbols])
        m = alive & cols[None, :]
        return m, m

    bars_per_day = DAY_MS // interval_ms(panel.interval)
    n = spec.adv_days * bars_per_day
    cov = _min_coverage()
    adv = pd.DataFrame(np.asarray(panel["quote_volume"])).rolling(n, min_periods=int(np.ceil(cov * n))).mean().to_numpy()
    h = spec.min_history_days * bars_per_day
    listed = pd.DataFrame(alive.astype(float)).rolling(h, min_periods=1).sum().to_numpy() >= cov * h
    adv = np.where(alive & listed, adv, np.nan)
    ranks = pd.DataFrame(adv).rank(axis=1, ascending=False, method="first").to_numpy()

    ref = _day_ref(panel)
    valid = ref >= 0
    entry = np.zeros(close.shape, dtype=bool)
    hold = np.zeros(close.shape, dtype=bool)
    entry[valid] = ranks[ref[valid]] <= spec.size
    hold[valid] = ranks[ref[valid]] <= max(spec.keep_rank, spec.size)
    return entry & alive, hold & alive


def compute_universe(panel: Panel, spec: UniverseSpec) -> np.ndarray:
    """Entry mask [T, S] (names that may be opened)."""
    return universe_masks(panel, spec)[0]


def adv_usd(panel: Panel, days: int = 30) -> np.ndarray:
    """Trailing average daily quote volume (USD), causal."""
    bars_per_day = DAY_MS // interval_ms(panel.interval)
    n = days * bars_per_day
    qv = pd.DataFrame(np.asarray(panel["quote_volume"])).rolling(n, min_periods=bars_per_day).mean().to_numpy()
    return qv * bars_per_day

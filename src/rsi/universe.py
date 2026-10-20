"""Point-in-time tradable universe.

Membership for every bar of UTC day D is decided from data that closed at the
end of day D-1, so it is causal and identical in research and live.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from rsi.data.panel import Panel
from rsi.utils.timeutil import interval_ms

DAY_MS = 86_400_000


class UniverseSpec(BaseModel):
    kind: Literal["top_liquid", "fixed"] = "top_liquid"
    size: int = 50
    adv_days: int = 30
    symbols: list[str] = Field(default_factory=list)  # for kind=fixed

    def lookback(self, interval: str = "30m") -> int:
        bars_per_day = DAY_MS // interval_ms(interval)
        return (self.adv_days + 2) * bars_per_day if self.kind == "top_liquid" else 1


def compute_universe(panel: Panel, spec: UniverseSpec) -> np.ndarray:
    """Boolean [T, S] membership mask."""
    close = panel["close"]
    alive = np.isfinite(close)
    if spec.kind == "fixed":
        cols = np.array([s in set(spec.symbols) for s in panel.symbols])
        return alive & cols[None, :]

    step = interval_ms(panel.interval)
    bars_per_day = DAY_MS // step
    n = spec.adv_days * bars_per_day
    qv = pd.DataFrame(panel["quote_volume"])
    adv = qv.rolling(n, min_periods=n).mean().to_numpy()
    adv = np.where(alive, adv, np.nan)
    ranks = pd.DataFrame(adv).rank(axis=1, ascending=False, method="first").to_numpy()
    member_at = ranks <= spec.size  # NaN ranks -> False

    pos_in_day = (panel.times % DAY_MS) // step
    ref = np.arange(panel.T) - pos_in_day - 1          # last bar of the previous UTC day
    valid = ref >= 0
    mask = np.zeros_like(member_at, dtype=bool)
    mask[valid] = member_at[ref[valid]]
    return mask & alive


def adv_usd(panel: Panel, days: int = 30) -> np.ndarray:
    """Trailing average daily quote volume (USD), causal."""
    bars_per_day = DAY_MS // interval_ms(panel.interval)
    n = days * bars_per_day
    qv = pd.DataFrame(panel["quote_volume"]).rolling(n, min_periods=bars_per_day).mean().to_numpy()
    return qv * bars_per_day

"""Supervised targets aligned with execution.

A decision at the close of bar t is filled at open[t+1]; holding h bars, the
realised log return is log(open[t+1+h] / open[t+1]) minus the funding a long pays
over that window (what the portfolio actually earns on a perp). Labels are therefore
known only at bar t+1+h, which drives purging in `rsi.splits`.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from rsi.data.panel import Panel
from rsi.features.engine import xs_demean


class LabelSpec(BaseModel):
    horizon: int = 8
    kind: Literal["fwd_logret"] = "fwd_logret"
    xs_demean: bool = True
    vol_norm: bool = True
    include_funding: bool = True     # perp total return = price return - funding paid by a long
    vol_window: int = 336
    clip: float = 5.0

    @property
    def span(self) -> int:
        """Bars after t needed to know the label."""
        return self.horizon + 1


def fwd_logret(panel: Panel, horizon: int) -> np.ndarray:
    o = np.asarray(panel["open"], dtype=np.float64)
    out = np.full(o.shape, np.nan)
    if panel.T > horizon + 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            out[: panel.T - horizon - 1] = np.log(o[horizon + 1:] / o[1: panel.T - horizon])
    return out


def fwd_funding(panel: Panel, horizon: int) -> np.ndarray:
    """Funding a long pays while held from open[t+1] to open[t+1+h]: prints at boundaries t+2 .. t+1+h
    (a print at a boundary is charged to the book held into it, as in the engine)."""
    fr = np.nan_to_num(np.asarray(panel["funding_rate"], dtype=np.float64))
    c = np.vstack([np.zeros((1, fr.shape[1])), np.cumsum(fr, axis=0)])   # c[i] = sum fr[:i]
    out = np.full(fr.shape, np.nan)
    T = panel.T
    if T > horizon + 1:
        out[: T - horizon - 1] = c[horizon + 2:] - c[2: T - horizon + 1]
    return out


def make_label(panel: Panel, spec: LabelSpec, mask: np.ndarray, valid_until: int | None = None) -> np.ndarray:
    """Training target [T, S]. Rows whose label would need data at/after `valid_until` are NaN."""
    y = fwd_logret(panel, spec.horizon)
    if spec.include_funding:
        y = y - fwd_funding(panel, spec.horizon)
    if spec.vol_norm:
        lr = np.log(pd.DataFrame(np.asarray(panel["close"], dtype=np.float64))).diff()
        vol = lr.rolling(spec.vol_window, min_periods=spec.vol_window // 2).std().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            y = y / (vol * np.sqrt(spec.horizon))
    if spec.xs_demean:
        y = xs_demean(y, mask)
    else:
        y = np.where(mask, y, np.nan)
    if spec.clip:
        y = np.clip(y, -spec.clip, spec.clip)
    if valid_until is not None:
        cut = max(0, valid_until - spec.span)
        y[cut:] = np.nan
    return y

"""Score -> target weights (fraction of equity, signed). Stateful but deterministic.

The same `TargetBuilder.step` is called bar-by-bar in backtest and live; its
only state is the previous target vector, which live persists across restarts.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel


class StrategySpec(BaseModel):
    kind: Literal["xs_longshort", "ts_directional"] = "xs_longshort"
    rebalance_bars: int = 2           # act on bars where (bar_number % k == 0), absolute clock
    gross: float = 1.0                # sum |w| before risk overlays
    allow_long: bool = True
    allow_short: bool = True
    # xs_longshort
    long_frac: float = 0.2
    short_frac: float = 0.2
    exit_buffer: float = 0.1          # keep a holding until it falls this far past the entry cut
    weighting: Literal["equal", "score", "inv_vol"] = "equal"
    min_names: int = 10
    # STR-03 signal conditioning (off by default -> identical to plain STR-01/02)
    smooth_bars: int = 1              # score = mean of the last k bar scores (finite memory, turnover control)
    min_abs_score: float = 0.0        # xs: names whose |score - median| is below this are not entered
    beta_neutral: bool = False        # xs: rescale the two legs so that sum(w * beta_btc) = 0 at constant gross
    exposure_feature: str | None = None   # broadcast regime feature (log-ratio scale) that sizes the book
    exposure_power: float = 1.0           # gross *= clip(exp(power * value), lo, hi)
    exposure_clip: tuple[float, float] = (0.5, 1.5)
    # ts_directional
    entry_z: float = 1.0
    exit_z: float = 0.25
    max_names: int = 10               # weight per name = gross / max_names


class TargetBuilder:
    def __init__(self, spec: StrategySpec, n_symbols: int):
        self.spec = spec
        self.prev = np.zeros(n_symbols)
        self.hist: list[np.ndarray] = []   # last smooth_bars raw score rows (persisted live)

    def smooth(self, score: np.ndarray) -> np.ndarray:
        k = self.spec.smooth_bars
        if k <= 1:
            return score
        self.hist = (self.hist + [np.asarray(score, dtype=np.float64)])[-k:]
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmean(np.vstack(self.hist), axis=0)
        return np.where(np.isfinite(score), m, np.nan)

    def is_rebalance(self, bar_number: int) -> bool:
        return bar_number % self.spec.rebalance_bars == 0

    def step(self, bar_number: int, score: np.ndarray, mask: np.ndarray, vol: np.ndarray | None = None,
             hold: np.ndarray | None = None, beta: np.ndarray | None = None, regime: float | None = None) -> np.ndarray:
        """score/mask/vol/hold: [S] rows for the decision bar. Returns target weights [S].

        `mask` = names that may be entered; `hold` (superset) = names an existing position may stay in.
        """
        hold = mask if hold is None else (hold | mask)
        score = self.smooth(score)
        held = self.prev != 0
        if not self.is_rebalance(bar_number):
            w = np.where(hold, self.prev, 0.0)   # names leaving the hold buffer are closed immediately
        else:
            ok = np.isfinite(score)
            entry = mask & ok
            if self.spec.min_abs_score > 0 and entry.any():
                entry &= np.abs(score - np.nanmedian(score[entry])) >= self.spec.min_abs_score
            keep = hold & held & ok
            if self.spec.kind == "xs_longshort":
                w = self._xs(score, entry, keep, vol)
                if self.spec.beta_neutral and beta is not None:
                    w = self._beta_neutral(w, beta)
                w = w * self.exposure_scale(regime)
            else:
                w = self._ts(score, entry, keep)
        self.prev = w
        return w

    # ------------------------------------------------------------------ cross-sectional
    def _xs(self, score: np.ndarray, entry: np.ndarray, keep: np.ndarray, vol: np.ndarray | None) -> np.ndarray:
        sp = self.spec
        w = np.zeros_like(score, dtype=np.float64)
        alive = entry | keep
        idx = np.flatnonzero(alive)
        n = len(idx)
        if entry.sum() < sp.min_names:
            return w
        order = idx[np.argsort(-score[idx], kind="stable")]       # best first
        rank = np.empty(len(score)); rank[:] = np.nan
        rank[order] = np.arange(n) / max(n - 1, 1)                 # 0 = best, 1 = worst
        sides = []
        if sp.allow_long:
            sides.append((+1, sp.long_frac, rank))
        if sp.allow_short:
            sides.append((-1, sp.short_frac, 1 - rank))
        n_entry = int(entry.sum())
        for sign, frac, r in sides:
            k = max(1, int(np.floor(frac * n_entry)))
            held = np.flatnonzero((np.sign(self.prev) == sign) & keep & (r <= frac + sp.exit_buffer))
            held = held[np.argsort(r[held])][:k]
            held_set = set(held.tolist())
            cand = [i for i in np.argsort(np.where(entry, r, np.inf)) if entry[i] and i not in held_set]
            chosen = list(held) + cand[: k - len(held)]
            chosen = np.array(chosen, dtype=int)
            if sp.weighting == "score":
                raw = np.abs(score[chosen] - np.nanmedian(score[idx])) + 1e-9
            elif sp.weighting == "inv_vol" and vol is not None:
                raw = 1.0 / np.clip(np.nan_to_num(vol[chosen], nan=np.nanmedian(vol[idx])), 1e-4, None)
            else:
                raw = np.ones(len(chosen))
            side_gross = sp.gross / len(sides)
            w[chosen] = sign * side_gross * raw / raw.sum()
        return w

    def exposure_scale(self, regime: float | None) -> float:
        sp = self.spec
        if sp.exposure_feature is None or regime is None or not np.isfinite(regime):
            return 1.0
        lo, hi = sp.exposure_clip
        return float(np.clip(np.exp(sp.exposure_power * regime), lo, hi))

    def _beta_neutral(self, w: np.ndarray, beta: np.ndarray) -> np.ndarray:
        b = np.where(np.isfinite(beta), np.clip(beta, 0.1, 3.0), 1.0)
        lg, sh = w > 0, w < 0
        GL, GS = w[lg].sum(), -w[sh].sum()
        BL, BS = (w[lg] * b[lg]).sum(), (-w[sh] * b[sh]).sum()
        den = GL * BS + GS * BL
        if GL <= 0 or GS <= 0 or den <= 0:
            return w
        gross = GL + GS
        a, c = gross * BS / den, gross * BL / den
        return np.where(lg, w * a, np.where(sh, w * c, 0.0))

    # ------------------------------------------------------------------ time-series
    def _ts(self, z: np.ndarray, entry: np.ndarray, keep: np.ndarray) -> np.ndarray:
        sp = self.spec
        unit = sp.gross / sp.max_names
        pos = np.sign(self.prev)
        new = np.zeros_like(pos)
        zz = np.where(entry | keep, z, 0.0)
        long_in = (zz > sp.entry_z) & sp.allow_long & entry
        short_in = (zz < -sp.entry_z) & sp.allow_short & entry
        keep_long = (pos > 0) & (zz > sp.exit_z) & (keep | entry)
        keep_short = (pos < 0) & (zz < -sp.exit_z) & (keep | entry)
        new[keep_long] = 1
        new[keep_short] = -1
        new[long_in] = 1
        new[short_in] = -1
        if np.count_nonzero(new) > sp.max_names:
            strength = np.abs(zz) * (new != 0)
            keep = np.argsort(-strength)[: sp.max_names]
            m = np.zeros_like(new, dtype=bool); m[keep] = True
            new = np.where(m, new, 0)
        return new * unit

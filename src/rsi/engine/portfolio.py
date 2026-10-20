"""Portfolio accounting + risk overlays, shared verbatim by backtest, paper and live.

Accounting is perpetual-futures style: equity = cash + unrealised PnL, marked
to market against the last mark price per symbol. Costs are explicit:
  fee            taker fee on traded notional
  spread/impact  half-spread from liquidity tier + square-root impact
  funding        qty * price * rate at each settlement the position is held through
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from rsi.config import CostCfg

DAY_MS = 86_400_000


class RiskSpec(BaseModel):
    max_gross: float = 2.0            # sum |notional| / equity
    max_weight: float = 0.10          # per-name |notional| / equity
    max_net: float = 1.0              # |sum notional| / equity
    vol_target: float | None = 0.30   # annualised; scales the strategy book
    vol_lookback: int = 672
    vol_scale_min: float = 0.25
    vol_scale_max: float = 2.0
    stop_loss_vol: float | None = 4.0 # stop distance in units of daily vol from entry; None = off
    cooldown_bars: int = 16
    dd_halt: float | None = 0.25      # flatten + halt when drawdown from peak exceeds this
    dd_halt_bars: int = 336
    daily_loss_halt: float | None = 0.06
    adv_cap: float = 0.02             # per-name notional <= adv_cap * ADV
    min_trade_frac: float = 0.25      # skip resizing trades smaller than this fraction of target
    min_notional_usd: float = 10.0


def cost_bps(notional: np.ndarray, adv: np.ndarray, daily_vol: np.ndarray, c: CostCfg) -> np.ndarray:
    adv_m = np.where(np.isfinite(adv) & (adv > 0), adv, 1e6) / 1e6
    half_spread = np.clip(c.spread_coef_bps / np.sqrt(adv_m), c.min_half_spread_bps, c.max_half_spread_bps)
    dv = np.where(np.isfinite(daily_vol), daily_vol, 0.05)
    impact = c.impact_eta * dv * 1e4 * np.sqrt(np.abs(notional) / (adv_m * 1e6))
    return c.taker_fee_bps + half_spread + impact


@dataclass
class TradeRecord:
    symbol: str
    side: int
    open_ms: int
    close_ms: int
    pnl_usd: float
    entry_notional: float
    reason: str


@dataclass
class Portfolio:
    symbols: list[str]
    equity0: float
    costs: CostCfg
    risk: RiskSpec
    bars_per_year: int = 17520
    bar_ms: int = 1_800_000
    # state
    cash: float = field(init=False)
    qty: np.ndarray = field(init=False)
    mark: np.ndarray = field(init=False)
    entry: np.ndarray = field(init=False)
    stop_dist: np.ndarray = field(init=False)
    cooldown_until: np.ndarray = field(init=False)
    exit_reason: np.ndarray = field(init=False)
    peak: float = field(init=False)
    halt_until_ms: int = 0
    day: int = -1
    day_start_equity: float = 0.0
    day_halted: bool = False
    book_rets: deque = field(init=False)
    last_w_unscaled: np.ndarray = field(init=False)
    # per-name episode tracking
    ep_pnl: np.ndarray = field(init=False)
    ep_open_ms: np.ndarray = field(init=False)
    ep_notional: np.ndarray = field(init=False)
    trades: list = field(default_factory=list)
    # running totals
    fees_usd: float = 0.0
    funding_usd: float = 0.0
    turnover_usd: float = 0.0

    def __post_init__(self):
        S = len(self.symbols)
        self.cash = float(self.equity0)
        self.qty = np.zeros(S)
        self.mark = np.full(S, np.nan)
        self.entry = np.full(S, np.nan)
        self.stop_dist = np.full(S, np.nan)
        self.cooldown_until = np.zeros(S, dtype=np.int64)
        self.exit_reason = np.full(S, "", dtype=object)
        self.peak = float(self.equity0)
        self.book_rets = deque(maxlen=self.risk.vol_lookback)
        self.last_w_unscaled = np.zeros(S)
        self.ep_pnl = np.zeros(S)
        self.ep_open_ms = np.zeros(S, dtype=np.int64)
        self.ep_notional = np.zeros(S)

    # ------------------------------------------------------------------ valuation
    @property
    def equity(self) -> float:
        # cash already includes realised PnL; positions are carried at mark via `mark_to`
        return self.cash

    def notional(self) -> np.ndarray:
        return np.where(self.qty != 0, self.qty * np.nan_to_num(self.mark), 0.0)

    def mark_to(self, prices: np.ndarray) -> float:
        """Variation-margin style: book PnL of price moves into cash, then update marks."""
        p = np.where(np.isfinite(prices), prices, self.mark)
        held = (self.qty != 0) & np.isfinite(p) & np.isfinite(self.mark)
        pnl_vec = np.zeros(len(self.qty))
        pnl_vec[held] = self.qty[held] * (p[held] - self.mark[held])
        self.cash += float(pnl_vec.sum())
        self.ep_pnl += pnl_vec
        self.mark = np.where(np.isfinite(p), p, self.mark)
        return float(pnl_vec.sum())

    def apply_funding(self, rates: np.ndarray, prices: np.ndarray) -> float:
        r = np.where(np.isfinite(rates), rates, 0.0)
        p = np.where(np.isfinite(prices), prices, np.nan_to_num(self.mark))
        paid = self.qty * p * r                     # longs pay positive funding
        self.cash -= float(paid.sum())
        self.ep_pnl -= paid
        self.funding_usd += float(paid.sum())
        return float(paid.sum())

    # ------------------------------------------------------------------ book return (for vol targeting)
    def record_book_return(self, bar_ret: np.ndarray) -> None:
        r = float(np.nansum(self.last_w_unscaled * np.where(np.isfinite(bar_ret), bar_ret, 0.0)))
        if np.any(self.last_w_unscaled != 0):
            self.book_rets.append(r)

    def vol_scale(self) -> float:
        rk = self.risk
        if rk.vol_target is None or len(self.book_rets) < max(48, rk.vol_lookback // 4):
            return 1.0
        sd = float(np.std(self.book_rets)) * np.sqrt(self.bars_per_year)
        if sd <= 0:
            return 1.0
        return float(np.clip(rk.vol_target / sd, rk.vol_scale_min, rk.vol_scale_max))

    # ------------------------------------------------------------------ stops (evaluated on bar close)
    def stop_hits(self, close: np.ndarray) -> np.ndarray:
        """Positions whose bar close breached the stop; they are exited at the next open."""
        if self.risk.stop_loss_vol is None:
            return np.zeros(len(self.qty), dtype=bool)
        held = (self.qty != 0) & np.isfinite(self.stop_dist) & np.isfinite(close)
        long_hit = held & (self.qty > 0) & (close <= self.entry * (1 - self.stop_dist))
        short_hit = held & (self.qty < 0) & (close >= self.entry * (1 + self.stop_dist))
        return long_hit | short_hit

    # ------------------------------------------------------------------ targets
    def target_qty(self, now_ms: int, bar_index: int, w_target: np.ndarray, price: np.ndarray,
                   adv: np.ndarray, daily_vol: np.ndarray) -> np.ndarray:
        """Apply every risk overlay to strategy weights and return target quantities."""
        rk = self.risk
        E = self.equity
        self.last_w_unscaled = np.where(np.isfinite(price), w_target, 0.0)

        day = now_ms // DAY_MS
        if day != self.day:
            self.day, self.day_start_equity, self.day_halted = day, E, False
        self.peak = max(self.peak, E)
        if rk.dd_halt is not None and self.peak > 0 and (self.peak - E) / self.peak > rk.dd_halt and now_ms >= self.halt_until_ms:
            self.halt_until_ms = now_ms + rk.dd_halt_bars * self.bar_ms
            self.peak = E  # re-arm from the new low once the halt expires
        if rk.daily_loss_halt is not None and E < self.day_start_equity * (1 - rk.daily_loss_halt):
            self.day_halted = True
        if now_ms < self.halt_until_ms or self.day_halted or E <= 0:
            self.exit_reason[self.qty != 0] = "halt"
            return np.zeros_like(self.qty)

        hits = self.stop_hits(price)
        self.cooldown_until[hits] = bar_index + rk.cooldown_bars
        self.exit_reason[hits] = "stop"

        w = w_target * self.vol_scale()
        w = np.clip(w, -rk.max_weight, rk.max_weight)
        cap = np.where(np.isfinite(adv) & (adv > 0), rk.adv_cap * adv / max(E, 1e-9), 0.0)
        w = np.clip(w, -cap, cap)
        w = np.where(self.cooldown_until > bar_index, 0.0, w)
        w = np.where(np.isfinite(price), w, 0.0)
        gross = np.abs(w).sum()
        if gross > rk.max_gross:
            w *= rk.max_gross / gross
        net = w.sum()
        if abs(net) > rk.max_net:
            w -= (net - np.sign(net) * rk.max_net) * np.abs(w) / max(np.abs(w).sum(), 1e-12)
        q = np.where(np.isfinite(price) & (price > 0), w * E / np.where(price > 0, price, 1.0), 0.0)

        # trade suppression: tiny resizes are not worth the cost
        dq = q - self.qty
        resize = (q != 0) & (self.qty != 0) & (np.sign(q) == np.sign(self.qty))
        small = resize & (np.abs(dq) < rk.min_trade_frac * np.abs(q))
        tiny = (q != 0) & (np.abs(dq) * np.nan_to_num(price) < rk.min_notional_usd)
        q = np.where(small | tiny, self.qty, q)
        return q

    # ------------------------------------------------------------------ fills
    def fill(self, i: int, dq: float, px: float, adv: float, daily_vol: float, now_ms: int, reason: str = "rebalance",
             cost_override_bps: float | None = None) -> float:
        """Execute dq at reference price px (mark must already be px). Returns cost in USD."""
        if dq == 0 or not np.isfinite(px):
            return 0.0
        notional = abs(dq) * px
        bps = cost_override_bps if cost_override_bps is not None else float(cost_bps(np.array([notional]), np.array([adv]), np.array([daily_vol]), self.costs)[0])
        cost = notional * bps * 1e-4
        old = self.qty[i]
        new = old + dq
        self.cash -= cost
        self.ep_pnl[i] -= cost
        self.fees_usd += cost
        self.turnover_usd += notional
        self.mark[i] = px

        closing = old != 0 and (new == 0 or np.sign(new) != np.sign(old))
        if closing:
            self.trades.append(TradeRecord(self.symbols[i], int(np.sign(old)), int(self.ep_open_ms[i]), int(now_ms),
                                           float(self.ep_pnl[i]), float(self.ep_notional[i]), reason))
            self.ep_pnl[i] = 0.0
            self.entry[i] = np.nan
            self.stop_dist[i] = np.nan
        if new != 0 and (old == 0 or np.sign(new) != np.sign(old)):
            self.ep_open_ms[i] = now_ms
            self.ep_notional[i] = abs(new) * px
            self.entry[i] = px
            self._set_stop(i, daily_vol)
        elif new != 0 and abs(new) > abs(old):
            self.entry[i] = (self.entry[i] * abs(old) + px * abs(dq)) / abs(new)
            self.ep_notional[i] = max(self.ep_notional[i], abs(new) * px)
        self.qty[i] = new if abs(new) > 1e-12 else 0.0
        return cost

    def _set_stop(self, i: int, daily_vol: float) -> None:
        k = self.risk.stop_loss_vol
        if k is not None and np.isfinite(daily_vol) and daily_vol > 0:
            self.stop_dist[i] = min(k * daily_vol, 0.5)

    def rebalance(self, q_target: np.ndarray, px: np.ndarray, adv: np.ndarray, daily_vol: np.ndarray, now_ms: int) -> float:
        """Vectorised version of `fill` for a whole book (marks must already equal px)."""
        idx = np.flatnonzero((q_target != self.qty) & np.isfinite(px))
        if len(idx) == 0:
            return 0.0
        old, new, p = self.qty[idx], q_target[idx], px[idx]
        dq = new - old
        notional = np.abs(dq) * p
        cost = notional * cost_bps(notional, adv[idx], daily_vol[idx], self.costs) * 1e-4
        self.cash -= float(cost.sum())
        self.ep_pnl[idx] -= cost
        self.fees_usd += float(cost.sum())
        self.turnover_usd += float(notional.sum())
        self.mark[idx] = p

        closing = (old != 0) & ((new == 0) | (np.sign(new) != np.sign(old)))
        for j in np.flatnonzero(closing):
            i = idx[j]
            self.trades.append(TradeRecord(self.symbols[i], int(np.sign(old[j])), int(self.ep_open_ms[i]), int(now_ms),
                                           float(self.ep_pnl[i]), float(self.ep_notional[i]), self.exit_reason[i] or "signal"))
        self.exit_reason[idx] = ""
        ci = idx[closing]
        self.ep_pnl[ci] = 0.0
        self.entry[ci] = np.nan
        self.stop_dist[ci] = np.nan

        opening = (new != 0) & ((old == 0) | (np.sign(new) != np.sign(old)))
        oi = idx[opening]
        self.ep_open_ms[oi] = now_ms
        self.ep_notional[oi] = np.abs(new[opening]) * p[opening]
        self.entry[oi] = p[opening]
        if self.risk.stop_loss_vol is not None:
            dv = daily_vol[oi]
            self.stop_dist[oi] = np.where(np.isfinite(dv) & (dv > 0), np.minimum(self.risk.stop_loss_vol * dv, 0.5), np.nan)

        grow = (new != 0) & ~opening & (np.abs(new) > np.abs(old))
        gi = idx[grow]
        self.entry[gi] = (self.entry[gi] * np.abs(old[grow]) + p[grow] * np.abs(dq[grow])) / np.abs(new[grow])
        self.ep_notional[gi] = np.maximum(self.ep_notional[gi], np.abs(new[grow]) * p[grow])
        self.qty[idx] = np.where(np.abs(new) > 1e-12, new, 0.0)
        return float(cost.sum())

    def force_close(self, i: int, now_ms: int, penalty_bps: float, reason: str = "delist") -> None:
        px = float(self.mark[i]) if np.isfinite(self.mark[i]) else 0.0
        self.fill(i, -self.qty[i], px, 0.0, 0.0, now_ms, reason=reason, cost_override_bps=penalty_bps)

    # ------------------------------------------------------------------ persistence (live)
    def state_dict(self) -> dict:
        return {
            "cash": self.cash, "qty": self.qty.tolist(), "mark": np.nan_to_num(self.mark, nan=-1).tolist(),
            "entry": np.nan_to_num(self.entry, nan=-1).tolist(), "stop_dist": np.nan_to_num(self.stop_dist, nan=-1).tolist(),
            "cooldown_until": self.cooldown_until.tolist(), "peak": self.peak, "halt_until_ms": self.halt_until_ms,
            "day": self.day, "day_start_equity": self.day_start_equity, "day_halted": self.day_halted,
            "book_rets": list(self.book_rets), "last_w_unscaled": self.last_w_unscaled.tolist(),
            "symbols": self.symbols,
        }

    def load_state(self, d: dict) -> None:
        idx = {s: i for i, s in enumerate(d["symbols"])}
        def arr(key, fill=np.nan, dtype=float):
            out = np.full(len(self.symbols), fill, dtype=dtype)
            for j, s in enumerate(self.symbols):
                if s in idx:
                    v = d[key][idx[s]]
                    out[j] = np.nan if (dtype is float and v == -1 and key in ("mark", "entry", "stop_dist")) else v
            return out
        self.cash = d["cash"]
        self.qty = arr("qty", 0.0)
        self.mark, self.entry, self.stop_dist = arr("mark"), arr("entry"), arr("stop_dist")
        self.cooldown_until = arr("cooldown_until", 0, np.int64)
        self.last_w_unscaled = arr("last_w_unscaled", 0.0)
        self.peak, self.halt_until_ms = d["peak"], d["halt_until_ms"]
        self.day, self.day_start_equity, self.day_halted = d["day"], d["day_start_equity"], d["day_halted"]
        self.book_rets = deque(d["book_rets"], maxlen=self.risk.vol_lookback)

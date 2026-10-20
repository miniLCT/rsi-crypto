"""Bar-stepping backtest. Event order per bar t (identical to the live loop):

  1. mark to close[t]; record equity
  2. decision: strategy targets -> risk overlays (stops, halts, caps) -> target qty  (data <= close[t])
  3. boundary t->t+1: mark to open[t+1], settle funding printed at open[t+1],
     force-close names that stopped printing, fill target qty at open[t+1] with costs
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rsi.config import CostCfg
from rsi.data.panel import Panel
from rsi.engine.portfolio import Portfolio, RiskSpec
from rsi.strategy import StrategySpec, TargetBuilder
from rsi.universe import adv_usd
from rsi.utils.timeutil import interval_ms


@dataclass
class BacktestResult:
    times: np.ndarray            # decision bar open times (equity marked at the bar close)
    equity: np.ndarray
    gross: np.ndarray
    net: np.ndarray
    n_long: np.ndarray
    n_short: np.ndarray
    trades: pd.DataFrame
    fees_usd: float
    funding_usd: float
    turnover_usd: float
    equity0: float
    open_pnl_usd: float = 0.0

    @property
    def returns(self) -> np.ndarray:
        e = np.concatenate([[self.equity0], self.equity])
        return e[1:] / e[:-1] - 1

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"time": pd.to_datetime(self.times, unit="ms"), "equity": self.equity, "gross": self.gross,
                             "net": self.net, "n_long": self.n_long, "n_short": self.n_short})


def market_context(panel: Panel) -> tuple[np.ndarray, np.ndarray]:
    """(ADV in USD, daily vol) per bar, both causal; used for sizing, costs and stops."""
    adv = adv_usd(panel, 30)
    lr = np.log(pd.DataFrame(np.asarray(panel["close"], dtype=np.float64))).diff()
    dvol = (lr.rolling(48, min_periods=24).std() * np.sqrt(48)).to_numpy()
    return adv, dvol


def run_backtest(panel: Panel, score: np.ndarray, mask: np.ndarray, strat: StrategySpec, risk: RiskSpec,
                 costs: CostCfg, equity0: float, t0: int, t1: int, bars_per_year: int = 17520,
                 context: tuple[np.ndarray, np.ndarray] | None = None) -> BacktestResult:
    step = interval_ms(panel.interval)
    T, S = panel.T, panel.S
    t1 = min(t1, T)
    o = np.asarray(panel["open"]); c = np.asarray(panel["close"])
    fr = np.asarray(panel["funding_rate"])
    adv, dvol = context if context is not None else market_context(panel)
    port = Portfolio(panel.symbols, equity0, costs, risk, bars_per_year, step)
    builder = TargetBuilder(strat, S)
    n = t1 - t0
    eq = np.zeros(n); gross = np.zeros(n); net = np.zeros(n); nl = np.zeros(n, dtype=int); ns = np.zeros(n, dtype=int)
    times = panel.times
    with np.errstate(invalid="ignore", divide="ignore"):
        for k, t in enumerate(range(t0, t1)):
            bar_no = int(times[t] // step)
            port.mark_to(c[t])
            E = port.equity
            eq[k] = E
            notional = port.notional()
            gross[k] = np.abs(notional).sum() / max(E, 1e-9)
            net[k] = notional.sum() / max(E, 1e-9)
            nl[k] = int((port.qty > 0).sum()); ns[k] = int((port.qty < 0).sum())
            if t > 0:
                port.record_book_return(c[t] / c[t - 1] - 1)
            w = builder.step(bar_no, score[t], mask[t], dvol[t])
            q = port.target_qty(int(times[t] + step), bar_no, w, c[t], adv[t], dvol[t])
            if t + 1 >= T or k == n - 1:
                break
            port.mark_to(o[t + 1])
            port.apply_funding(fr[t + 1], o[t + 1])
            dead = np.flatnonzero((port.qty != 0) & ~np.isfinite(o[t + 1]))
            for i in dead:
                port.force_close(int(i), int(times[t + 1]), costs.delist_penalty_bps)
            q = np.where(np.isfinite(o[t + 1]), q, 0.0)
            port.rebalance(q, o[t + 1], adv[t], dvol[t], int(times[t + 1]))
    tr = pd.DataFrame([r.__dict__ for r in port.trades])
    return BacktestResult(times[t0:t0 + len(eq)], eq, gross, net, nl, ns, tr, port.fees_usd, port.funding_usd,
                          port.turnover_usd, equity0, float(port.ep_pnl[port.qty != 0].sum()))

"""LIVE-04 / MON-01: the live loop in paper mode reproduces the backtest engine bar for bar."""
import asyncio
import json

import numpy as np

from rsi.config import harness_config
from rsi.engine.backtest import market_context, run_backtest
from rsi.engine.portfolio import RiskSpec
from rsi.features import registry
from rsi.features.featureset import FeatureSpec, build_batch, stack
from rsi.harness.release import live_window_bars
from rsi.harness.spec import ExperimentSpec
from rsi.labels import LabelSpec
from rsi.live.broker import PaperBroker
from rsi.live.runner import LiveConfig, LiveRunner
from rsi.models.core import Model, ModelSpec
from rsi.strategy import StrategySpec
from rsi.universe import UniverseSpec, universe_masks


class FakeData:
    def __init__(self, panel):
        self.panel, self.t = panel, 0

    async def ticker_price(self):
        o = self.panel["open"][self.t + 1]
        return {s: float(o[j]) for j, s in enumerate(self.panel.symbols) if np.isfinite(o[j])}

    async def close(self):
        pass


class FakeFeed:
    def __init__(self, panel, window):
        self.panel, self.W = panel, window
        self.symbols = list(panel.symbols)
        self.exchange_info, self.extra = {}, set()

    async def update(self, bar_open):
        return []

    def window(self, bar_open):
        t = int(np.searchsorted(self.panel.times, bar_open))
        return self.panel.slice(t + 1 - self.W, t + 1)


def test_paper_loop_matches_backtest(panel, tmp_path):
    spec = ExperimentSpec(
        name="parity",
        universe=UniverseSpec(kind="top_liquid", size=6, keep_rank=8, adv_days=10, min_history_days=10),
        features=FeatureSpec(include=["ret_48", "rv_48", "taker_imb_16", "btc_ret_8"], transform="both"),
        label=LabelSpec(horizon=4),
        model=ModelSpec(kind="factor", weights={"r:ret_48": -1.0, "r:taker_imb_16": 0.5, "rv_48": 2.0}),
        strategy=StrategySpec(rebalance_bars=2, long_frac=0.34, short_frac=0.34, exit_buffer=0.17, min_names=3),
        risk=RiskSpec(hard_stop_vol=2.0, hold_horizon_bars=4, trail_start_vol=1.0, max_hold_bars=40, vol_lookback=200),
    )
    cols_names = spec.features.columns()
    model = Model(spec.model, cols_names)
    model.fit(np.random.default_rng(0).normal(size=(500, len(cols_names))).astype(np.float32), np.zeros(500))
    mdir = tmp_path / "model"
    model.save(mdir)
    W = live_window_bars(spec)
    manifest = {"run_id": "test-000000", "spec": spec.model_dump(), "interval": "30m", "window_bars": W,
                "features": {n: {"src_hash": registry.get(n).src_hash, "lookback": registry.get(n).lookback}
                             for n in spec.features.base_names()}}
    (mdir / "manifest.json").write_text(json.dumps(manifest))

    # batch / backtest path
    entry, hold = universe_masks(panel, spec.universe)
    cols = build_batch(panel, spec.features, hold, use_cache=False)
    score = np.full(entry.shape, np.nan)
    t, s = np.nonzero(hold)
    score[t, s] = model.predict(stack(cols, cols_names, t, s))
    t0, t1 = W + 5, panel.T - 2
    h = harness_config()
    bt = run_backtest(panel, score, entry, spec.strategy, spec.risk, h.costs, 10_000.0, t0, t1, h.bars_per_year,
                      context=market_context(panel), hold=hold)

    # live path, paper broker
    cfg = LiveConfig(mode="paper", model_dir=str(mdir), state_dir=str(tmp_path / "state"), capital_usd=10_000.0,
                     min_coverage=0.0)
    r = LiveRunner(cfg)
    r.data_client = FakeData(panel)
    r.feed = FakeFeed(panel, W)
    r.broker = PaperBroker()
    r._build_books(r.feed.symbols)
    eq = []

    async def go():
        for tt in range(t0, t1):
            r.data_client.t = tt
            rec = await r.cycle(int(panel.times[tt]))
            eq.append(rec["equity_close"])
    asyncio.run(go())

    assert len(bt.trades) > 20
    np.testing.assert_allclose(np.array(eq), bt.equity, rtol=1e-9, atol=1e-6)
    live_tr = [(x.symbol, x.side, x.open_ms, x.close_ms, x.reason) for x in r.port.trades]
    bt_tr = list(bt.trades[["symbol", "side", "open_ms", "close_ms", "reason"]].itertuples(index=False, name=None))
    assert live_tr[: len(bt_tr)] == bt_tr[: len(live_tr)]
    assert {x.reason for x in r.port.trades} >= {"signal"}

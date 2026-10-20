import numpy as np

from rsi.config import CostCfg
from rsi.engine.backtest import run_backtest
from rsi.engine.portfolio import Portfolio, RiskSpec
from rsi.labels import LabelSpec, fwd_logret, make_label
from rsi.splits import walk_forward
from rsi.strategy import StrategySpec, TargetBuilder
from rsi.universe import UniverseSpec, compute_universe

FREE = CostCfg(taker_fee_bps=0, spread_coef_bps=0, min_half_spread_bps=0, max_half_spread_bps=0, impact_eta=0)
NO_RISK = RiskSpec(vol_target=None, stop_loss_vol=None, dd_halt=None, daily_loss_halt=None, adv_cap=1e9,
                   max_weight=1.0, max_gross=10, min_trade_frac=0, min_notional_usd=0)


def _mask(panel):
    return compute_universe(panel, UniverseSpec(kind="top_liquid", size=8, adv_days=10))


def test_zero_signal_is_flat(panel):
    mask = _mask(panel)
    res = run_backtest(panel, np.zeros(mask.shape), mask, StrategySpec(min_names=100), RiskSpec(), CostCfg(), 1e4, 600, panel.T)
    assert len(res.trades) == 0
    assert np.allclose(res.equity, 1e4)


def test_perfect_foresight_profits_and_costs_hurt(panel):
    mask = _mask(panel)
    fut = fwd_logret(panel, 1)
    strat = StrategySpec(rebalance_bars=1, long_frac=0.25, short_frac=0.25, exit_buffer=0, min_names=4)
    free = run_backtest(panel, fut, mask, strat, NO_RISK, FREE, 1e4, 600, panel.T)
    costly = run_backtest(panel, fut, mask, strat, NO_RISK, CostCfg(), 1e4, 600, panel.T)
    assert free.equity[-1] > 1e4 * 1.5
    assert costly.equity[-1] < free.equity[-1]
    assert costly.fees_usd > 0


def test_accounting_conserves_pnl(panel):
    """Equity change == sum of closed trade PnL + open episode PnL (fees and funding included)."""
    mask = _mask(panel)
    rng = np.random.default_rng(3)
    score = rng.normal(size=mask.shape)
    strat = StrategySpec(rebalance_bars=4, min_names=4)
    res = run_backtest(panel, score, mask, strat, RiskSpec(stop_loss_vol=2.0), CostCfg(), 1e4, 600, panel.T - 1)
    closed = res.trades["pnl_usd"].sum() if len(res.trades) else 0.0
    assert res.funding_usd != 0
    assert np.isclose(res.equity[-1] - 1e4, closed + res.open_pnl_usd, atol=1e-6)


def test_funding_sign():
    p = Portfolio(["A"], 1000.0, FREE, NO_RISK)
    p.mark = np.array([100.0]); p.qty = np.array([1.0])
    p.apply_funding(np.array([0.001]), np.array([100.0]))
    assert np.isclose(p.cash, 1000.0 - 0.1)       # long pays positive funding
    p.qty = np.array([-1.0])
    p.apply_funding(np.array([0.001]), np.array([100.0]))
    assert np.isclose(p.cash, 1000.0)              # short receives


def test_stop_and_cooldown():
    rk = RiskSpec(vol_target=None, stop_loss_vol=1.0, cooldown_bars=5, dd_halt=None, daily_loss_halt=None, adv_cap=1e9,
                  min_trade_frac=0, min_notional_usd=0)
    p = Portfolio(["A"], 1000.0, FREE, rk)
    px, adv, dv = np.array([100.0]), np.array([1e9]), np.array([0.05])
    q = p.target_qty(0, 0, np.array([0.1]), px, adv, dv)
    p.mark_to(px); p.rebalance(q, px, adv, dv, 0)
    assert p.qty[0] > 0 and np.isclose(p.stop_dist[0], 0.05)
    q = p.target_qty(1, 1, np.array([0.1]), np.array([94.0]), adv, dv)
    assert q[0] == 0 and p.cooldown_until[0] == 6
    p.mark_to(np.array([94.0])); p.rebalance(q, np.array([94.0]), adv, dv, 1)
    assert p.trades[-1].reason == "stop"
    assert p.target_qty(2, 3, np.array([0.1]), np.array([95.0]), adv, dv)[0] == 0   # still cooling down
    assert p.target_qty(3, 7, np.array([0.1]), np.array([95.0]), adv, dv)[0] > 0


def test_drawdown_halt():
    rk = RiskSpec(vol_target=None, stop_loss_vol=None, dd_halt=0.1, dd_halt_bars=10, daily_loss_halt=None, adv_cap=1e9,
                  min_trade_frac=0, min_notional_usd=0)
    p = Portfolio(["A"], 1000.0, FREE, rk)
    p.cash = 850.0; p.peak = 1000.0
    assert p.target_qty(0, 0, np.array([0.1]), np.array([1.0]), np.array([1e9]), np.array([0.01]))[0] == 0
    assert p.halt_until_ms == 10 * 1_800_000


def test_state_roundtrip():
    p = Portfolio(["A", "B"], 1000.0, CostCfg(), RiskSpec())
    p.qty = np.array([1.0, -2.0]); p.mark = np.array([10.0, np.nan]); p.cash = 990.0
    q = Portfolio(["B", "A", "C"], 1.0, CostCfg(), RiskSpec())
    q.load_state(p.state_dict())
    assert q.qty.tolist() == [-2.0, 1.0, 0.0] and q.cash == 990.0 and np.isnan(q.mark[0])


def test_xs_builder_hysteresis():
    sp = StrategySpec(long_frac=0.2, short_frac=0.2, exit_buffer=0.2, rebalance_bars=1, min_names=5)
    b = TargetBuilder(sp, 10)
    m = np.ones(10, bool)
    w1 = b.step(0, np.arange(10.0), m)
    assert (w1 > 0).sum() == 2 and (w1 < 0).sum() == 2 and np.isclose(np.abs(w1).sum(), 1.0)
    s2 = np.arange(10.0); s2[[8, 7]] = s2[[7, 8]]  # holding #8 slips to 3rd place: kept thanks to the buffer
    w2 = b.step(1, s2, m)
    assert w2[8] > 0


def test_purged_walk_forward():
    from rsi.config import harness_config
    times = (np.arange(0, 6 * 365 * 48) * 1_800_000 + 1_577_836_800_000).astype(np.int64)
    folds = walk_forward(times, harness_config(), label_span=9)
    assert len(folds) > 5
    for f in folds:
        assert f.train[1] + 9 + harness_config().walk_forward.embargo_bars <= f.test[0]
    for a, b in zip(folds, folds[1:]):
        assert a.test[1] == b.test[0]


def test_label_respects_valid_until(panel):
    mask = _mask(panel)
    y = make_label(panel, LabelSpec(horizon=8), mask, valid_until=2000)
    assert np.isnan(y[2000 - 9:]).all()
    assert np.isfinite(y[1500]).any()

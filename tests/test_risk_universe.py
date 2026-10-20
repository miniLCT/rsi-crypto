import numpy as np

from rsi.config import CostCfg
from rsi.engine.portfolio import Portfolio, RiskSpec
from rsi.features import contract, registry
from rsi.features.engine import xs_rank, xs_zscore
from rsi.features.registry import Feature
from rsi.strategy import StrategySpec, TargetBuilder
from rsi.universe import UniverseSpec, universe_masks

FREE = CostCfg(taker_fee_bps=0, spread_coef_bps=0, min_half_spread_bps=0, max_half_spread_bps=0, impact_eta=0)


def _risk(**kw):
    base = dict(vol_target=None, hard_stop_vol=None, stop_loss_vol=None, dd_reduce=None, dd_halt=None, daily_loss_halt=None,
                adv_cap=1e9, max_weight=1.0, max_gross=10, min_trade_frac=0, min_notional_usd=0)
    base.update(kw)
    return RiskSpec(**base)


def _open(p, w=0.5, px=100.0, dv=0.06, t=0):
    adv, d = np.array([1e9]), np.array([dv])
    q = p.target_qty(t, t // 1_800_000, np.array([w]), np.array([px]), adv, d)
    p.mark_to(np.array([px])); p.rebalance(q, np.array([px]), adv, d, t)
    return adv, d


def test_hard_stop_fills_at_worse_of_stop_and_open():
    p = Portfolio(["A"], 1000.0, FREE, _risk(hard_stop_vol=3.0, hold_horizon_bars=48))
    adv, dv = _open(p)
    assert np.isclose(p.hard_stop_px[0], 100 * (1 - 0.18))
    # bar opens above the stop and trades through it -> fill at the stop price
    p.check_hard_stops(1_800_000, 1, np.array([95.0]), np.array([96.0]), np.array([80.0]), adv, dv)
    assert p.qty[0] == 0 and p.trades[-1].reason == "hard_stop"
    assert np.isclose(p.trades[-1].pnl_usd, 5.0 * (82.0 - 100.0))
    # gap through the stop -> fill at the (worse) open
    p2 = Portfolio(["A"], 1000.0, FREE, _risk(hard_stop_vol=3.0, hold_horizon_bars=48))
    _open(p2)
    p2.check_hard_stops(1_800_000, 1, np.array([70.0]), np.array([72.0]), np.array([69.0]), adv, dv)
    assert np.isclose(p2.trades[-1].pnl_usd, 5.0 * (70.0 - 100.0))


def test_short_hard_stop():
    p = Portfolio(["A"], 1000.0, FREE, _risk(hard_stop_vol=3.0, hold_horizon_bars=48))
    adv, dv = _open(p, w=-0.5)
    assert np.isclose(p.hard_stop_px[0], 118.0)
    p.check_hard_stops(1_800_000, 1, np.array([101.0]), np.array([125.0]), np.array([100.0]), adv, dv)
    assert p.qty[0] == 0


def test_trailing_take_profit_and_max_hold():
    p = Portfolio(["A"], 1000.0, FREE, _risk(trail_start_vol=1.0, trail_giveback_vol=0.5, hold_horizon_bars=48))
    adv, dv = _open(p)          # hp vol = 6% of 100 = 6
    for t, px in enumerate([103.0, 108.0, 106.0], start=1):
        q = p.target_qty(t * 1_800_000, t, np.array([0.5]), np.array([px]), adv, dv)
        assert q[0] > 0
    q = p.target_qty(4 * 1_800_000, 4, np.array([0.5]), np.array([104.5]), adv, dv)   # gave back 3.5 >= 3
    assert q[0] == 0 and p.exit_reason[0] == "trail"

    p = Portfolio(["A"], 1000.0, FREE, _risk(max_hold_bars=3))
    adv, dv = _open(p)
    assert p.target_qty(2 * 1_800_000, 2, np.array([0.5]), np.array([100.0]), adv, dv)[0] > 0
    assert p.target_qty(3 * 1_800_000, 3, np.array([0.5]), np.array([100.0]), adv, dv)[0] == 0


def test_two_level_drawdown():
    p = Portfolio(["A"], 1000.0, FREE, _risk(dd_reduce=0.15, dd_reduce_scale=0.5, dd_halt=0.25))
    args = (np.array([0.5]), np.array([1.0]), np.array([1e9]), np.array([0.01]))
    full = p.target_qty(0, 0, *args)[0]
    p.cash = 840.0; p.peak = 1000.0
    reduced = p.target_qty(1, 1, *args)[0]
    assert np.isclose(reduced, 0.5 * full * 0.84)
    p.cash = 740.0
    assert p.target_qty(2, 2, *args)[0] == 0 and p.halt_events == 1


def test_daily_loss_halt_resets_next_day():
    p = Portfolio(["A"], 1000.0, FREE, _risk(daily_loss_halt=0.03))
    args = (np.array([0.5]), np.array([1.0]), np.array([1e9]), np.array([0.01]))
    assert p.target_qty(0, 0, *args)[0] > 0
    p.cash = 960.0
    assert p.target_qty(1_800_000, 1, *args)[0] == 0
    assert p.target_qty(86_400_000, 48, *args)[0] > 0


def test_hold_buffer_blocks_adds_and_new_entries():
    p = Portfolio(["A", "B"], 1000.0, FREE, _risk())
    adv, dv, px = np.array([1e9, 1e9]), np.array([0.05, 0.05]), np.array([10.0, 10.0])
    q = p.target_qty(0, 0, np.array([0.2, 0.0]), px, adv, dv)
    p.mark_to(px); p.rebalance(q, px, adv, dv, 0)
    entry = np.array([False, False])
    q2 = p.target_qty(1, 1, np.array([0.4, 0.3]), px, adv, dv, entry)
    assert q2[0] == p.qty[0] and q2[1] == 0          # no add on A, no new entry on B
    q3 = p.target_qty(2, 2, np.array([0.1, 0.0]), px, adv, dv, entry)
    assert 0 < q3[0] < p.qty[0]                      # reducing is allowed
    q4 = p.target_qty(3, 3, np.array([-0.2, 0.0]), px, adv, dv, entry)
    assert q4[0] == 0                                # a flip outside the entry universe only closes


def test_builder_keeps_buffered_holding():
    b = TargetBuilder(StrategySpec(long_frac=0.2, short_frac=0.2, exit_buffer=0.5, rebalance_bars=1, min_names=3), 6)
    entry = np.array([True] * 5 + [False])
    hold = np.ones(6, bool)
    b.prev = np.array([0, 0, 0, 0, 0, 0.5])
    w = b.step(0, np.array([1.0, 2, 3, 4, 0.5, 4.5]), entry, None, hold)
    assert w[5] > 0                                  # held, still top-ranked, inside the hold buffer
    w = b.step(1, np.array([1.0, 2, 3, 4, 0.5, 4.5]), entry, None, entry)
    assert w[5] == 0                                 # fell out of the hold buffer -> closed


def test_universe_entry_hold_and_history(panel):
    spec = UniverseSpec(kind="top_liquid", size=4, keep_rank=6, adv_days=10, min_history_days=10)
    entry, hold = universe_masks(panel, spec)
    assert (entry <= hold).all()
    assert entry.sum(1).max() <= 4 and hold.sum(1).max() <= 6
    late = 3   # listed at T//3 in the synthetic panel
    first = np.flatnonzero(np.isfinite(panel["close"][:, late]))[0]
    assert not hold[first: first + 10 * 48, late].any()


def test_unit_invariance_gate(panel):
    bad = Feature("base_vol_level", lambda fv: np.log(fv.volume), 1, "test")
    registry.REGISTRY["base_vol_level"] = bad
    try:
        assert not contract.check_unit_invariance(panel, "base_vol_level")[0]
    finally:
        del registry.REGISTRY["base_vol_level"]
    assert contract.check_unit_invariance(panel, "vwap_gap_48")[0]


def test_xs_coverage_rule_and_robust_z():
    x = np.array([[1.0, 2.0, np.nan, np.nan, np.nan], [1.0, 2.0, 3.0, 4.0, 100.0]])
    m = np.ones_like(x, bool)
    assert np.isnan(xs_rank(x, m)[0]).all()          # 2 of 5 names -> below 50% coverage
    z = xs_zscore(x, m)[1]
    assert z[-1] == 3.0 and abs(z[2]) < 1e-6

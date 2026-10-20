import numpy as np
import pytest
from pydantic import ValidationError

from rsi.config import CostCfg, DisasterCfg
from rsi.engine.portfolio import Portfolio, RiskSpec
from rsi.eval.disaster import calibrate, count_triggers
from rsi.harness.spec import ExperimentSpec
from rsi.live.runner import tighten

FREE = CostCfg(taker_fee_bps=0, spread_coef_bps=0, min_half_spread_bps=0, max_half_spread_bps=0, impact_eta=0)


def test_leverage_bounds():
    for v in (1, 5, 10):
        RiskSpec(max_gross=v, exchange_leverage=v)
    with pytest.raises(ValidationError):
        RiskSpec(max_gross=11)
    with pytest.raises(ValidationError):
        RiskSpec(exchange_leverage=12)


@pytest.mark.parametrize("lev,expected", [(1, 0.6), (5, 3.0), (10, 6.0)])
def test_margin_usage_caps_gross(lev, expected):
    rk = RiskSpec(max_gross=10, exchange_leverage=lev, vol_target=None, max_weight=10, adv_cap=1e9, min_trade_frac=0,
                  min_notional_usd=0, hard_stop_vol=None)
    p = Portfolio(["A", "B"], 1000.0, FREE, rk)
    q = p.target_qty(0, 0, np.array([5.0, -5.0]), np.array([1.0, 1.0]), np.array([1e12] * 2), np.array([0.01] * 2))
    assert np.isclose(np.abs(q).sum() / 1000.0, expected)


def test_backtest_liquidation():
    rk = RiskSpec(max_gross=10, exchange_leverage=10, margin_usage_max=1.0, vol_target=None, max_weight=10, adv_cap=1e9,
                  min_trade_frac=0, min_notional_usd=0, hard_stop_vol=None, maint_margin_rate=0.01, max_net=10)
    p = Portfolio(["A"], 1000.0, FREE, rk)
    px, adv, dv = np.array([100.0]), np.array([1e12]), np.array([0.01])
    q = p.target_qty(0, 0, np.array([10.0]), px, adv, dv)
    p.mark_to(px); p.rebalance(q, px, adv, dv, 0)
    assert not p.check_liquidation(1, np.array([101.0]), np.array([95.0]), 100.0)
    assert p.check_liquidation(2, np.array([101.0]), np.array([90.5]), 100.0)
    assert p.qty[0] == 0 and p.liquidations == 1 and p.trades[-1].reason == "liquidation"
    assert p.equity < 50


def test_disaster_calibration_is_rarely_triggered():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0002, 0.004, 48 * 700)
    th = calibrate(r, DisasterCfg(n_boot=500))
    assert 0 < th["day_loss"] < 0.5 and 0 < th["max_dd"] < 0.9
    eq = 1e4 * np.cumprod(1 + r)
    times = np.arange(len(r)) * 1_800_000
    assert count_triggers(times, eq, 1e4, th) == 0
    crash = eq.copy(); crash[30000:] *= 0.5
    assert count_triggers(times, crash, 1e4, th) >= 1


def test_shorts_require_hard_stop():
    with pytest.raises(ValidationError):
        ExperimentSpec(name="x", risk=RiskSpec(hard_stop_vol=None))
    ExperimentSpec(name="x", risk=RiskSpec(hard_stop_vol=None), strategy={"allow_short": False})


def test_live_overrides():
    base = RiskSpec()
    r = tighten(base, {"vol_target": 0.6, "max_gross": 6, "exchange_leverage": 8, "max_weight": 0.5})
    assert r.vol_target == 0.6 and r.max_gross == 6 and r.exchange_leverage == 8
    assert r.max_weight == base.max_weight          # tighten-only
    with pytest.raises(ValueError):
        tighten(base, {"min_trade_frac": 0.0})
    with pytest.raises(ValidationError):
        tighten(base, {"max_gross": 20})

import numpy as np
import pytest

from rsi.features import contract, registry
from rsi.features.engine import compute_last, xs_rank
from rsi.features.featureset import FeatureSpec, build_batch, build_last
from rsi.features.registry import FrameView
from rsi.universe import UniverseSpec, compute_universe

ALL = registry.resolve(["all"])


@pytest.mark.parametrize("name", ALL)
def test_feature_contract(panel, name):
    """Every registered feature is causal and fully determined by its declared lookback."""
    r = contract.check_feature(panel, name, n_cuts=3, future=100)
    assert r["causal"], r
    assert r["lookback_ok"], r
    assert r["unit_invariant"], r


def test_lookahead_is_detected(panel):
    from rsi.features.registry import Feature
    leaky = Feature("leaky", lambda fv: fv.close.shift(-1) / fv.close, 1, "test")
    registry.REGISTRY["leaky"] = leaky
    try:
        assert not contract.check_feature(panel, "leaky")["causal"]
    finally:
        del registry.REGISTRY["leaky"]


def test_short_lookback_is_detected(panel):
    from rsi.features.registry import Feature
    bad = Feature("bad_lb", lambda fv: fv.close.rolling(48).mean(), 10, "test")
    registry.REGISTRY["bad_lb"] = bad
    try:
        assert not contract.check_feature(panel, "bad_lb")["lookback_ok"]
    finally:
        del registry.REGISTRY["bad_lb"]


@pytest.mark.parametrize("transform,neutralize", [("both", False), ("zscore", True)])
def test_live_window_matches_batch(panel, transform, neutralize):
    """The live path (trailing window + build_last) reproduces batch model inputs in float32."""
    spec = FeatureSpec(include=["all"], transform=transform, neutralize=neutralize)
    uni = UniverseSpec(kind="top_liquid", size=8, keep_rank=8, adv_days=10, min_history_days=10)
    mask = compute_universe(panel, uni)
    batch = build_batch(panel, spec, mask, use_cache=False)
    W = max(spec.lookback(), uni.lookback()) + 8
    for t in (panel.T - 1, panel.T - 37, 2500):
        win = panel.slice(t + 1 - W, t + 1)
        m_last = compute_universe(win, uni)[-1]
        assert np.array_equal(m_last, mask[t])
        live = build_last(win, spec, m_last)
        for c in spec.columns():
            a, b = np.asarray(batch[c])[t], live[c]
            ok = (np.isnan(a) & np.isnan(b)) | np.isclose(a, b, rtol=1e-5, atol=1e-7)
            assert ok.all(), (c, t, a[~ok], b[~ok])


def test_xs_rank_ties_and_mask():
    x = np.array([[1.0, 1.0, 2.0, np.nan, 5.0]])
    m = np.array([[True, True, True, True, False]])
    r = xs_rank(x, m)[0]
    assert r[0] == r[1] and r[2] == 0.5 and np.isnan(r[3]) and np.isnan(r[4])


def test_universe_is_causal(panel):
    uni = UniverseSpec(kind="top_liquid", size=5, keep_rank=5, adv_days=10, min_history_days=10)
    full = compute_universe(panel, uni)
    cut = 2000
    part = compute_universe(panel.slice(0, cut), uni)
    assert np.array_equal(full[:cut], part)
    # membership only changes at UTC day boundaries
    pos = (panel.times % 86_400_000) // 1_800_000
    changes = np.flatnonzero(np.any(full[1:] != full[:-1], axis=1)) + 1
    alive_change = np.any(np.isfinite(panel["close"][1:]) != np.isfinite(panel["close"][:-1]), axis=1)
    assert all(pos[i] == 0 or alive_change[i - 1] for i in changes)

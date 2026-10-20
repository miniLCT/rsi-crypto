"""Feature contract: causality + declared-lookback sufficiency.

For random cut points c:
  causality : f(data[s : c+F])[c-1]  ==  f(data[s : c])[-1]      (future rows must not matter)
  lookback  : f(data[c-L : c])[-1]   ==  f(data[s : c])[-1]      (L declared rows are enough)

The second identity is exactly what the live engine relies on, so a feature
that passes is guaranteed to produce research-identical values in production.
Results are cached per (feature, source hash).
"""
from __future__ import annotations

import json

import numpy as np

from rsi.config import cache_dir
from rsi.data.panel import Panel
from rsi.features import registry
from rsi.features.registry import FrameView
from rsi.utils.log import get_logger

log = get_logger("contract")

RTOL = 1e-6
ATOL = 1e-9


def _close(a: np.ndarray, b: np.ndarray) -> tuple[bool, float]:
    both_nan = np.isnan(a) & np.isnan(b)
    ok = both_nan | np.isclose(a, b, rtol=RTOL, atol=ATOL)
    bad = ~ok
    worst = float(np.nanmax(np.abs(a - b)[bad])) if bad.any() and np.isfinite(a[bad] - b[bad]).any() else (np.inf if bad.any() else 0.0)
    return bool(ok.all()), worst


def check_feature(panel: Panel, name: str, n_cuts: int = 4, future: int = 200, seed: int = 0) -> dict:
    feat = registry.get(name)
    L = feat.lookback
    rng = np.random.default_rng(seed)
    lo = L + 50
    if panel.T - future - 1 <= lo:
        raise ValueError("panel too short for contract check")
    cuts = rng.integers(lo, panel.T - future - 1, size=n_cuts)
    res = {"name": name, "src_hash": feat.src_hash, "lookback": L, "causal": True, "lookback_ok": True, "detail": []}
    for c in cuts:
        s = max(0, c - 3 * L - 50)
        ref = feat.compute(FrameView(panel.slice(s, c)))[-1]
        fut = feat.compute(FrameView(panel.slice(s, c + future)))[c - 1 - s]
        win = feat.compute(FrameView(panel.slice(c - L, c)))[-1]
        ok_c, w_c = _close(fut, ref)
        ok_l, w_l = _close(win, ref)
        res["causal"] &= ok_c
        res["lookback_ok"] &= ok_l
        res["detail"].append({"cut": int(c), "causal_err": w_c, "lookback_err": w_l,
                              "coverage": float(np.isfinite(ref).mean())})
    res["unit_invariant"], res["unit_err"] = check_unit_invariance(panel, name)
    res["passed"] = res["causal"] and res["lookback_ok"] and res["unit_invariant"]
    return res


def rescale_units(panel: Panel, price_mult: float = 1000.0) -> Panel:
    """Same market quoted in a different contract unit (e.g. PEPE vs 1000PEPE): prices x k, base volume / k."""
    data = dict(panel.data)
    for f in ("open", "high", "low", "close"):
        data[f] = np.asarray(panel[f]) * price_mult
    data["volume"] = np.asarray(panel["volume"]) / price_mult
    return Panel(panel.times, panel.symbols, data, panel.interval, {"source": f"rescaled-{panel.fingerprint}"})


def check_unit_invariance(panel: Panel, name: str, bars: int = 3000) -> tuple[bool, float]:
    """FEAT-03: a cross-sectionally comparable feature must not depend on the contract's quoting unit."""
    feat = registry.get(name)
    sub = panel.slice(max(0, panel.T - max(bars, 3 * feat.lookback)), panel.T)
    a = feat.compute(FrameView(sub))
    b = feat.compute(FrameView(rescale_units(sub)))
    return _close(a, b)


def _store_path():
    return cache_dir() / "feature_contract.json"


def _load_store() -> dict:
    p = _store_path()
    return json.loads(p.read_text()) if p.exists() else {}


def ensure_contract(panel: Panel, names: list[str], symbols_sample: int = 24) -> None:
    """Raise if any feature violates the contract. Passing results are memoised by source hash."""
    store = _load_store()
    todo = [n for n in names if store.get(n, {}).get("src_hash") != registry.get(n).src_hash
            or not store[n].get("passed")]
    if not todo:
        return
    syms = panel.symbols
    if len(syms) > symbols_sample:
        keep = [s for s in ("BTCUSDT", "ETHUSDT") if s in syms]
        rest = [s for s in syms if s not in keep]
        rng = np.random.default_rng(1)
        keep += list(rng.choice(rest, symbols_sample - len(keep), replace=False))
        sub = panel.select_symbols(sorted(keep))
    else:
        sub = panel
    failed = []
    for n in todo:
        r = check_feature(sub, n)
        store[n] = {k: r[k] for k in ("src_hash", "lookback", "causal", "lookback_ok", "unit_invariant", "passed")}
        if not r["passed"]:
            failed.append(r)
    _store_path().write_text(json.dumps(store, indent=1, sort_keys=True))
    if failed:
        msg = "; ".join(f"{r['name']}: causal={r['causal']} lookback_ok={r['lookback_ok']} "
                        f"unit_invariant={r['unit_invariant']} {r['detail']}" for r in failed)
        raise ValueError(f"feature contract violated -> {msg}")
    log.info(f"contract passed for {len(todo)} new/changed features")

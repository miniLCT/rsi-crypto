"""Single-factor screening on research data: predictive power, decay, stability, redundancy.

Cheap enough to run on every candidate factor before it enters a model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rsi.eval.metrics import ic_stats
from rsi.features.engine import compute_batch, xs_rank
from rsi.harness.runner import research_panel
from rsi.harness.spec import ExperimentSpec
from rsi.features import contract, registry
from rsi.labels import fwd_logret
from rsi.universe import compute_universe


def screen(names: list[str], spec: ExperimentSpec | None = None, horizons: tuple[int, ...] = (2, 8, 48),
           compare_to: list[str] | None = None, start: str = "2021-01-01") -> pd.DataFrame:
    spec = spec or ExperimentSpec(name="screen")
    panel = research_panel(spec)
    mask = compute_universe(panel, spec.universe)
    contract.ensure_contract(panel, names)
    i0 = panel.index_of(start)
    feats = compute_batch(panel, names)
    fwd = {h: fwd_logret(panel, h) for h in horizons}
    ref_names = compare_to if compare_to is not None else []
    refs = compute_batch(panel, ref_names) if ref_names else {}
    ref_ranks = {n: xs_rank(np.asarray(v)[i0::8], mask[i0::8]) for n, v in refs.items()}
    rows = []
    half = (panel.T - i0) // 2
    for n in names:
        x = np.asarray(feats[n])
        r = {"feature": n, "group": registry.get(n).group, "coverage": float(np.isfinite(x[i0:][mask[i0:]]).mean())}
        for h in horizons:
            st = ic_stats(x, fwd[h], mask, slice(i0, None))
            r[f"ic_{h}"], r[f"ic_t_{h}"] = st["ic"], st["ic_t"]
        a = ic_stats(x, fwd[horizons[1]], mask, slice(i0, i0 + half))["ic"]
        b = ic_stats(x, fwd[horizons[1]], mask, slice(i0 + half, None))["ic"]
        r["ic_first_half"], r["ic_second_half"] = a, b
        rk = xs_rank(x[i0::8], mask[i0::8]).astype(np.float64)
        r["rank_autocorr"] = _row_corr(rk[1:], rk[:-1])
        if ref_ranks:
            corrs = {m: abs(_row_corr(rk, v.astype(np.float64))) for m, v in ref_ranks.items() if m != n}
            if corrs:
                top = max(corrs, key=corrs.get)
                r["max_corr"], r["max_corr_with"] = corrs[top], top
        rows.append(r)
    return pd.DataFrame(rows).sort_values(f"ic_t_{horizons[1]}", key=np.abs, ascending=False)


def _row_corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    a = np.where(ok, a, 0.0); b = np.where(ok, b, 0.0)
    num = (a * b).sum(1)
    den = np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        c = num / den
    c = c[np.isfinite(c) & (ok.sum(1) >= 8)]
    return float(c.mean()) if len(c) else float("nan")

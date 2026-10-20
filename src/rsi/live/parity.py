"""Train/serve parity checks.

replay  : on lake data, for the last N bars compare the batch research path with the
          live trailing-window path (universe, every model input column, score).
journal : compare what the live process actually saw/decided (decision log) against
          a recomputation from archived data once it is published.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rsi.features.featureset import build_batch, build_last, stack
from rsi.harness.release import verify_manifest
from rsi.harness.runner import full_panel
from rsi.harness.spec import ExperimentSpec
from rsi.models.core import Model
from rsi.universe import universe_masks
from rsi.utils.hashing import stable_hash
from rsi.utils.timeutil import to_ms


def _load(model_dir: Path):
    manifest = json.loads((model_dir / "manifest.json").read_text())
    problems = verify_manifest(manifest)
    return manifest, ExperimentSpec(**manifest["spec"]), Model.load(model_dir), problems


def check(model_dir: Path, bars: int = 96, end: str | None = None) -> dict:
    manifest, spec, model, problems = _load(Path(model_dir))
    panel = full_panel(spec)
    if end:
        panel = panel.slice(0, panel.index_of(end))
    _, mask = universe_masks(panel, spec.universe)
    names = spec.features.columns()
    batch = build_batch(panel, spec.features, mask, mask_key=stable_hash(spec.universe.model_dump()))
    W = manifest["window_bars"]
    worst_col, worst_score, mask_mismatch = 0.0, 0.0, 0
    worst_name = None
    for t in range(panel.T - bars, panel.T):
        win = panel.slice(t + 1 - W, t + 1)
        m_live = universe_masks(win, spec.universe)[1][-1]
        mask_mismatch += int((m_live != mask[t]).sum())
        live_cols = build_last(win, spec.features, m_live)
        idx = np.flatnonzero(mask[t])
        for c in names:
            a, b = np.asarray(batch[c])[t, idx], live_cols[c][idx]
            d = np.where(np.isnan(a) & np.isnan(b), 0.0, np.abs(a.astype(np.float64) - b))
            d = np.where(np.isnan(d), np.inf, d)
            if d.size and d.max() > worst_col:
                worst_col, worst_name = float(d.max()), c
        sb = model.predict(stack(batch, names, np.full(len(idx), t), idx))
        sl = model.predict(np.column_stack([live_cols[c][idx] for c in names]).astype(np.float32))
        if len(idx):
            worst_score = max(worst_score, float(np.max(np.abs(sb - sl))))
    return {"bars": bars, "manifest_problems": problems, "universe_mismatches": mask_mismatch,
            "max_feature_abs_diff": worst_col, "worst_feature": worst_name, "max_score_abs_diff": worst_score,
            "ok": not problems and mask_mismatch == 0 and worst_score < 1e-4}


def check_journal(model_dir: Path, state_dir: Path, tol: float = 1e-4) -> pd.DataFrame:
    """Per-bar comparison of live decisions against archive recomputation (run after `rsi data sync && ingest`)."""
    manifest, spec, model, _ = _load(Path(model_dir))
    files = sorted((Path(state_dir) / "decisions").glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    dec = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    panel = full_panel(spec)
    _, mask = universe_masks(panel, spec.universe)
    names = spec.features.columns()
    batch = build_batch(panel, spec.features, mask, mask_key=stable_hash(spec.universe.model_dump()))
    rows = []
    sym_idx = {s: i for i, s in enumerate(panel.symbols)}
    for bar, g in dec.groupby("bar_open"):
        t = panel.index_of(int(bar))
        if t >= panel.T or panel.times[t] != bar:
            continue
        live_set = set(g["symbol"])
        arch_set = {panel.symbols[i] for i in np.flatnonzero(mask[t])}
        common = [s for s in g["symbol"] if s in sym_idx and s in arch_set]
        idx = np.array([sym_idx[s] for s in common], dtype=int)
        gg = g.set_index("symbol").loc[common]
        diffs = {c: float(np.nanmax(np.abs(np.asarray(batch[c])[t, idx] - gg[c].to_numpy()))) if len(idx) else 0.0 for c in names}
        sb = model.predict(stack(batch, names, np.full(len(idx), t), idx)) if len(idx) else np.zeros(0)
        rows.append({"bar": pd.Timestamp(int(bar), unit="ms"), "universe_only_live": len(live_set - arch_set),
                     "universe_only_archive": len(arch_set - live_set),
                     "max_score_diff": float(np.max(np.abs(sb - gg["score"].to_numpy()))) if len(idx) else 0.0,
                     "worst_feature": max(diffs, key=diffs.get) if diffs else None, "worst_feature_diff": max(diffs.values(), default=0.0)})
    out = pd.DataFrame(rows)
    if len(out):
        out["ok"] = (out["max_score_diff"] < tol) & (out["universe_only_live"] == 0) & (out["universe_only_archive"] == 0)
    return out

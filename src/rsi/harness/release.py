"""Holdout evaluation (budgeted) and model export for live trading."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from rsi.config import harness_config, repo_root
from rsi.features import registry
from rsi.harness import gates
from rsi.harness.ledger import Ledger
from rsi.harness.runner import evaluate, fit_fold, full_panel, git_sha, oos_scores, prepare
from rsi.harness.spec import ExperimentSpec
from rsi.splits import Fold, walk_forward
from rsi.utils.log import get_logger

log = get_logger("release")


def holdout(run_id: str, ledger: Ledger | None = None, force: bool = False) -> dict:
    h = harness_config()
    ledger = ledger or Ledger()
    row = ledger.get(run_id)
    if row is None:
        raise KeyError(run_id)
    used = ledger.holdout_looks(time.time() - 30 * 86400)
    if used >= h.holdout.max_looks_per_30d and not force:
        raise PermissionError(f"holdout budget exhausted: {used}/{h.holdout.max_looks_per_30d} looks in the last 30 days")
    spec = ExperimentSpec(**json.loads(row["spec_json"]))
    panel = full_panel(spec)
    prep = prepare(spec, panel)
    folds = walk_forward(panel.times, h, spec.label.span, end=h.periods.holdout_end,
                         first_test=h.periods.holdout_start, test_months=3)
    score, _ = oos_scores(prep, spec, folds)
    n_trials, sr_var = ledger.trial_stats(h.version)
    ev = evaluate(prep, spec, score, folds[0].test[0], folds[-1].test[1], folds, n_trials, sr_var)
    research_m = json.loads(row["metrics_json"])
    ok, why = gates.holdout_verdict(research_m, ev["metrics"], h.gates)
    ledger.add_holdout_look(row["run_id"], ok, ev["metrics"])
    if ok:
        ledger.set_champion(h.version, row["run_id"], stage="holdout_passed")
    log.info(f"holdout {row['run_id']}: {'PASS' if ok else 'FAIL'} — {why}")
    return {"run_id": row["run_id"], "passed": ok, "verdict": why, "metrics": ev["metrics"], "folds": ev["folds"]}


def export(run_id: str, out_dir: Path | None = None, ledger: Ledger | None = None) -> Path:
    """Train the deployable model on the most recent window and write an artifact with a manifest."""
    h = harness_config()
    ledger = ledger or Ledger()
    row = ledger.get(run_id)
    spec = ExperimentSpec(**json.loads(row["spec_json"]))
    panel = full_panel(spec)
    prep = prepare(spec, panel)
    end = panel.T
    gap = spec.label.span + h.walk_forward.embargo_bars
    months = h.walk_forward.train_months or 120
    start = max(0, panel.index_of(int(panel.times[-1]) - months * 30 * 86_400_000))
    fold = Fold(0, (start, end - gap), (end, end))
    model = fit_fold(prep, spec, fold)
    out = out_dir or repo_root() / "models" / f"{spec.name}-{row['run_id']}"
    out.mkdir(parents=True, exist_ok=True)
    model.save(out)
    base = spec.features.base_names()
    manifest = {
        "run_id": row["run_id"], "name": spec.name, "created": time.time(), "git_sha": git_sha(),
        "harness_version": h.version, "data_fp": panel.fingerprint,
        "train_range_ms": [int(panel.times[start]), int(panel.times[end - gap - 1])],
        "spec": spec.model_dump(),
        "features": {n: {"src_hash": registry.get(n).src_hash, "lookback": registry.get(n).lookback} for n in base},
        "columns": spec.features.columns(),
        "window_bars": int(max(spec.features.lookback(), spec.universe.lookback(panel.interval)) + 8),
        "interval": panel.interval,
        "panel_top_k": spec.panel_top_k,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log.info(f"exported {out}")
    return out


def verify_manifest(manifest: dict) -> list[str]:
    """Return a list of problems; empty means the running code reproduces the trained features exactly."""
    problems = []
    for n, meta in manifest["features"].items():
        try:
            f = registry.get(n)
        except KeyError:
            problems.append(f"missing feature {n}")
            continue
        if f.src_hash != meta["src_hash"]:
            problems.append(f"feature {n} changed since training ({meta['src_hash']} -> {f.src_hash})")
    return problems


def best_runs(ledger: Ledger, k: int = 10, harness_version: int | None = None) -> list[dict]:
    hv = harness_version or harness_config().version
    rows = [r for r in ledger.runs(hv) if r["sharpe"] is not None]
    return sorted(rows, key=lambda r: -(r["sharpe"] or -np.inf))[:k]

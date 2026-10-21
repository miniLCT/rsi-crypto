"""Perturbation robustness of a candidate: re-run it with random feature dropout.

A result that only exists at one exact feature set / parameter point is selection luck. The candidate
qualifies for a holdout look only if the median perturbed run still passes the gates.
"""
from __future__ import annotations

import json
import random

import numpy as np

from rsi.config import harness_config
from rsi.features import registry
from rsi.harness import runner
from rsi.harness.ledger import Ledger
from rsi.harness.spec import ExperimentSpec
from rsi.utils.log import get_logger

log = get_logger("robust")


def trained_base_names(run_id: str) -> list[str] | None:
    """Exact base features the run was trained on (selectors like `all` may resolve differently later)."""
    from rsi.config import runs_dir
    p = runs_dir() / run_id / "folds.json"
    if not p.exists():
        return None
    cols = list((json.loads(p.read_text())[0].get("importance") or {}).keys())
    return sorted({c.split("*")[0].split(":", 1)[-1] for c in cols}) or None


def perturbations(spec: ExperimentSpec, n: int, drop: float, seed: int, names: list[str] | None = None) -> list[ExperimentSpec]:
    rng = random.Random(seed)
    names = names or spec.features.base_names()
    out = []
    for i in range(n):
        keep = sorted(rng.sample(names, max(1, int(round(len(names) * (1 - drop))))))
        out.append(spec.with_overrides({"features.include": keep}, name=f"{spec.name}~robust{i}"))
    return out


def robustness(run_id: str, n: int = 6, drop: float = 0.15, seed: int = 0, ledger: Ledger | None = None) -> dict:
    ledger = ledger or Ledger()
    row = ledger.get(run_id)
    spec = ExperimentSpec(**json.loads(row["spec_json"]))
    res = []
    for p in perturbations(spec, n, drop, seed, trained_base_names(row["run_id"])):
        p.parent = row["run_id"]
        p.hypothesis = f"robustness: {int(drop * 100)}% feature dropout of {row['run_id']}"
        s = runner.run(p, note="robustness", ledger=ledger)
        ledger.set_status(s["run_id"], "robustness")
        res.append(s["metrics"])
    sr = np.array([m["sharpe"] for m in res])
    dd = np.array([m["max_drawdown"] for m in res])
    g = harness_config().gates
    out = {"run_id": row["run_id"], "base_sharpe": row["sharpe"], "median_sharpe": float(np.median(sr)),
           "min_sharpe": float(sr.min()), "median_mdd": float(np.median(dd)), "sharpes": sr.round(2).tolist(),
           "passed": bool(np.median(sr) >= g.min_sharpe and np.median(dd) <= g.max_drawdown)}
    log.info(f"robustness {row['run_id']}: {out}")
    return out

"""Automatic proposer: random + hill-climbing mutation over a spec's `search_space`.

Every evaluated candidate lands in the ledger, so the deflated Sharpe of any
winner is penalised for the full search effort.
"""
from __future__ import annotations

import random
from typing import Any

from rsi.harness import runner
from rsi.harness.ledger import Ledger
from rsi.harness.spec import ExperimentSpec
from rsi.utils.log import get_logger

log = get_logger("search")


def _sample(space: dict[str, list[Any]], rng: random.Random) -> dict[str, Any]:
    return {k: rng.choice(v) for k, v in space.items()}


def _mutate(point: dict[str, Any], space: dict[str, list[Any]], rng: random.Random) -> dict[str, Any]:
    p = dict(point)
    k = rng.choice(list(space))
    opts = [v for v in space[k] if v != p.get(k)] or space[k]
    p[k] = rng.choice(opts)
    return p


def search(base: ExperimentSpec, n: int, seed: int = 0, explore: float = 0.4, ledger: Ledger | None = None) -> list[dict]:
    space = base.search_space
    if not space:
        raise ValueError("spec has no search_space")
    rng = random.Random(seed)
    ledger = ledger or Ledger()
    tried: set[str] = {r["spec_hash"] for r in ledger.runs()}
    best_point: dict[str, Any] | None = None
    best_score = float("-inf")
    results = []
    for i in range(n):
        for _ in range(50):
            point = _sample(space, rng) if best_point is None or rng.random() < explore else _mutate(best_point, space, rng)
            cand = base.with_overrides(point, name=f"{base.name}~s{seed}.{i}")
            if cand.spec_hash not in tried:
                break
        else:
            log.info("search space exhausted")
            break
        tried.add(cand.spec_hash)
        cand.parent = base.name
        cand.hypothesis = f"search around {base.name}: {point}"
        s = runner.run(cand, note=f"search seed={seed}", ledger=ledger)
        score = s["metrics"]["sharpe"] - 2.0 * max(0.0, s["metrics"]["max_drawdown"] - 0.2)
        results.append({"point": point, "run_id": s["run_id"], "status": s["status"], "sharpe": s["metrics"]["sharpe"]})
        if score > best_score:
            best_score, best_point = score, point
            log.info(f"new best {score:.2f} at {point}")
    return results

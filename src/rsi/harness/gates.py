from __future__ import annotations

from rsi.config import GateCfg


def evaluate_gates(m: dict, folds: list[dict], g: GateCfg) -> dict:
    pos = [f["sharpe"] > 0 for f in folds if f.get("days", 0) >= 20]
    frac = sum(pos) / len(pos) if pos else 0.0
    checks = {
        "sharpe": (m["sharpe"] >= g.min_sharpe, f"{m['sharpe']:.2f} >= {g.min_sharpe}"),
        "dsr": (m["dsr"] >= g.min_dsr, f"{m['dsr']:.3f} >= {g.min_dsr}"),
        "max_drawdown": (m["max_drawdown"] <= g.max_drawdown, f"{m['max_drawdown']:.1%} <= {g.max_drawdown:.0%}"),
        "fold_consistency": (frac >= g.min_positive_fold_frac, f"{frac:.0%} folds positive >= {g.min_positive_fold_frac:.0%}"),
        "n_trades": (m["n_trades"] >= g.min_trades, f"{m['n_trades']} >= {g.min_trades}"),
    }
    return {"passed": all(v[0] for v in checks.values()), "checks": {k: {"ok": v[0], "rule": v[1]} for k, v in checks.items()},
            "positive_fold_frac": frac}


def beats_champion(m: dict, champ: dict | None, g: GateCfg) -> tuple[bool, str]:
    if champ is None:
        return True, "no incumbent"
    margin = m["sharpe"] - champ["sharpe"]
    ok = margin >= g.champion_sharpe_margin and m["max_drawdown"] <= max(champ["max_drawdown"] * 1.25, g.max_drawdown * 0.5)
    return ok, f"sharpe {m['sharpe']:.2f} vs champion {champ['sharpe']:.2f} (margin {margin:+.2f}, need {g.champion_sharpe_margin})"


def holdout_verdict(research: dict, holdout: dict, g: GateCfg) -> tuple[bool, str]:
    need = max(g.holdout_min_sharpe, g.holdout_min_sharpe_ratio * research["sharpe"])
    ok = holdout["sharpe"] >= need and holdout["max_drawdown"] <= g.max_drawdown
    return ok, f"holdout sharpe {holdout['sharpe']:.2f} (need >= {need:.2f}), mdd {holdout['max_drawdown']:.1%}"

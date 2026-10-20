"""End-to-end experiment execution: spec -> walk-forward OOS -> backtest -> gates -> ledger."""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rsi.config import data_config, harness_config, repo_root, runs_dir
from rsi.data.panel import Panel, build_panel
from rsi.engine.backtest import BacktestResult, market_context, run_backtest
from rsi.eval import metrics as M
from rsi.features import contract
from rsi.features.featureset import build_batch, stack
from rsi.harness import gates
from rsi.harness.ledger import Ledger
from rsi.harness.spec import ExperimentSpec
from rsi.labels import fwd_logret, make_label
from rsi.models.core import Model
from rsi.splits import Fold, walk_forward
from rsi.universe import compute_universe
from rsi.utils.hashing import stable_hash
from rsi.utils.log import get_logger

log = get_logger("runner")

WARMUP_DAYS = 45


def git_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root(), text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=repo_root()) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "nogit"


def full_panel(spec: ExperimentSpec) -> Panel:
    h = harness_config()
    start = (pd.Timestamp(h.periods.research_start) - pd.Timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
    return build_panel(data_config(), start, h.periods.holdout_end, top_k=spec.panel_top_k)


def research_panel(spec: ExperimentSpec) -> Panel:
    """Research view: rows at/after research_end are physically absent."""
    p = full_panel(spec)
    return p.slice(0, p.index_of(harness_config().periods.research_end))


@dataclass
class Prepared:
    panel: Panel
    mask: np.ndarray
    cols: dict
    names: list[str]
    y: np.ndarray
    context: tuple


def prepare(spec: ExperimentSpec, panel: Panel) -> Prepared:
    mask = compute_universe(panel, spec.universe)
    names = spec.features.columns()
    contract.ensure_contract(panel, spec.features.base_names())
    cols = build_batch(panel, spec.features, mask, mask_key=stable_hash(spec.universe.model_dump()))
    y = make_label(panel, spec.label, mask, valid_until=panel.T)
    return Prepared(panel, mask, cols, names, y, market_context(panel))


def _rows(mask: np.ndarray, y: np.ndarray | None, i0: int, i1: int, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    sub = mask[i0:i1:stride]
    if y is not None:
        sub = sub & np.isfinite(y[i0:i1:stride])
    t, s = np.nonzero(sub)
    return t * stride + i0, s


def fit_fold(prep: Prepared, spec: ExperimentSpec, fold: Fold) -> Model:
    tr0, tr1 = fold.train
    ms = spec.model
    n_valid = int((tr1 - tr0) * ms.valid_frac) if ms.kind == "lgbm" else 0
    split = tr1 - n_valid
    t, s = _rows(prep.mask, prep.y, tr0, split - (spec.label.span if n_valid else 0), ms.train_stride)
    X, yy = stack(prep.cols, prep.names, t, s), prep.y[t, s]
    Xv = yv = None
    if n_valid:
        tv, sv = _rows(prep.mask, prep.y, split, tr1, ms.train_stride)
        Xv, yv = stack(prep.cols, prep.names, tv, sv), prep.y[tv, sv]
    return Model(ms, prep.names).fit(X, yy, Xv, yv)


def predict_rows(prep: Prepared, model: Model, i0: int, i1: int, out: np.ndarray) -> None:
    t, s = _rows(prep.mask, None, i0, i1)
    if len(t):
        out[t, s] = model.predict(stack(prep.cols, prep.names, t, s))


def oos_scores(prep: Prepared, spec: ExperimentSpec, folds: list[Fold]) -> tuple[np.ndarray, list[dict]]:
    score = np.full(prep.mask.shape, np.nan)
    info = []
    for f in folds:
        t0 = time.time()
        m = fit_fold(prep, spec, f)
        predict_rows(prep, m, *f.test, score)
        info.append({**f.as_dict(prep.panel.times), "best_iter": m.best_iter, "fit_s": round(time.time() - t0, 1),
                     "importance": m.importance})
        log.info(f"fold {f.k}: test {info[-1]['test'][0][:10]}..{info[-1]['test'][1][:10]} fit {info[-1]['fit_s']}s iters={m.best_iter}")
    return score, info


def fold_metrics(res: BacktestResult, panel: Panel, folds: list[Fold], bars_per_year: int) -> list[dict]:
    out = []
    eq = pd.Series(np.concatenate([[res.equity0], res.equity]), index=np.concatenate([[res.times[0] - 1], res.times]))
    for f in folds:
        a, b = panel.times[f.test[0]], panel.times[min(f.test[1], panel.T) - 1]
        seg = eq[(eq.index >= a - 1) & (eq.index <= b)]
        if len(seg) < 3:
            continue
        d = M.daily_returns(seg.index[1:].to_numpy(), seg.to_numpy()[1:], float(seg.iloc[0]))
        out.append({"k": f.k, "start": str(pd.Timestamp(a, unit="ms").date()), "sharpe": M.sharpe(d.to_numpy(), 365),
                    "ret": float(seg.iloc[-1] / seg.iloc[0] - 1), "mdd": M.max_drawdown(seg.to_numpy()), "days": len(d)})
    return out


def evaluate(prep: Prepared, spec: ExperimentSpec, score: np.ndarray, t0: int, t1: int, folds: list[Fold],
             n_trials: int, sr_var: float | None) -> dict:
    h = harness_config()
    res = run_backtest(prep.panel, score, prep.mask, spec.strategy, spec.risk, h.costs, h.capital.initial_usd,
                       t0, t1, h.bars_per_year, context=prep.context)
    m = M.summarize(res, h.bars_per_year, n_trials, sr_var)
    fwd = fwd_logret(prep.panel, spec.label.horizon)
    m.update({f"oos_{k}": v for k, v in M.ic_stats(score, fwd, prep.mask, slice(t0, t1)).items()})
    fm = fold_metrics(res, prep.panel, folds, h.bars_per_year)
    return {"metrics": m, "folds": fm, "result": res}


def run(spec: ExperimentSpec, note: str | None = None, save: bool = True, ledger: Ledger | None = None) -> dict:
    h = harness_config()
    ledger = ledger or Ledger()
    t_start = time.time()
    panel = research_panel(spec)
    prep = prepare(spec, panel)
    folds = walk_forward(panel.times, h, spec.label.span, test_months=spec.test_months)
    score, finfo = oos_scores(prep, spec, folds)
    n_prev, sr_var = ledger.trial_stats(h.version)
    seen = any(r["spec_hash"] == spec.spec_hash for r in ledger.runs(h.version))
    n_trials = n_prev + (0 if seen else 1)
    ev = evaluate(prep, spec, score, folds[0].test[0], folds[-1].test[1], folds, n_trials, sr_var)
    m, fm, res = ev["metrics"], ev["folds"], ev["result"]
    g = gates.evaluate_gates(m, fm, h.gates)
    champ = ledger.champion(h.version)
    champ_m = json.loads(champ["metrics_json"]) if champ else None
    beats, why = gates.beats_champion(m, champ_m, h.gates)
    status = "champion" if (g["passed"] and beats) else ("pass" if g["passed"] else "fail")
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + spec.spec_hash[:6]
    imp = _mean_importance(finfo)
    summary = {"run_id": run_id, "name": spec.name, "status": status, "metrics": m, "folds": fm, "gates": g,
               "champion_check": why, "n_trials": n_trials, "elapsed_s": round(time.time() - t_start, 1),
               "top_features": imp[:20]}
    if save:
        d = runs_dir() / run_id
        d.mkdir(parents=True, exist_ok=True)
        spec.dump(d / "spec.yaml")
        (d / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        (d / "folds.json").write_text(json.dumps(finfo, indent=1, default=float))
        res.frame().to_parquet(d / "equity.parquet")
        if len(res.trades):
            res.trades.to_parquet(d / "trades.parquet")
        (d / "report.md").write_text(render_report(spec, summary))
        ledger.add_run({
            "run_id": run_id, "name": spec.name, "parent": spec.parent, "harness_version": h.version,
            "spec_hash": spec.spec_hash, "spec_json": json.dumps(spec.model_dump()), "git_sha": git_sha(),
            "data_fp": panel.fingerprint, "status": status, "sharpe": m["sharpe"], "dsr": m["dsr"],
            "max_dd": m["max_drawdown"], "daily_sr": m["daily_sr"], "metrics_json": json.dumps(m, default=float),
            "hypothesis": spec.hypothesis, "note": note,
        })
        if status == "champion":
            ledger.set_champion(h.version, run_id)
    log.info(f"[{status}] {spec.name} sharpe={m['sharpe']:.2f} dsr={m['dsr']:.2f} mdd={m['max_drawdown']:.1%} "
             f"cagr={m['cagr']:.1%} trades={m['n_trades']} ({summary['elapsed_s']}s) -> {run_id}")
    return summary


def _mean_importance(finfo: list[dict]) -> list[tuple[str, float]]:
    agg: dict[str, float] = {}
    for f in finfo:
        for k, v in (f.get("importance") or {}).items():
            agg[k] = agg.get(k, 0.0) + v / len(finfo)
    return sorted(agg.items(), key=lambda kv: -kv[1])


def render_report(spec: ExperimentSpec, s: dict) -> str:
    m = s["metrics"]
    lines = [f"# {s['run_id']} — {spec.name}  [{s['status']}]", "", f"> {spec.hypothesis}" if spec.hypothesis else "", "",
             "| metric | value |", "|---|---|"]
    for k in ["sharpe", "dsr", "cagr", "max_drawdown", "calmar", "vol_ann", "avg_gross", "turnover_ann", "fees_frac",
              "funding_frac", "n_trades", "hit_rate", "avg_hold_bars", "long_pnl_frac", "short_pnl_frac", "oos_ic", "oos_ic_t"]:
        v = m.get(k)
        lines.append(f"| {k} | {v:.4g} |" if isinstance(v, float) else f"| {k} | {v} |")
    lines += ["", "## Gates", ""]
    for k, v in s["gates"]["checks"].items():
        lines.append(f"- {'PASS' if v['ok'] else 'FAIL'} {k}: {v['rule']}")
    lines += [f"- champion: {s['champion_check']}", "", "## Folds", "", "| k | start | sharpe | ret | mdd |", "|---|---|---|---|---|"]
    for f in s["folds"]:
        lines.append(f"| {f['k']} | {f['start']} | {f['sharpe']:.2f} | {f['ret']:.1%} | {f['mdd']:.1%} |")
    lines += ["", "## Top features", ""] + [f"- {k}: {v:.3f}" for k, v in s["top_features"]]
    return "\n".join(lines) + "\n"

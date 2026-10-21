"""End-to-end experiment execution: spec -> walk-forward OOS -> backtest -> gates -> ledger."""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rsi.config import data_config, harness_config, harness_hash, repo_root, runs_dir
from rsi.data.panel import Panel, build_panel
from rsi.engine.backtest import BacktestResult, market_context, run_backtest
from rsi.eval import metrics as M
from rsi.features import contract
from rsi.features.featureset import build_batch, stack
from rsi.harness import gates
from rsi.harness.ledger import Ledger
from rsi.harness.spec import ExperimentSpec
from rsi.labels import fwd_funding, fwd_logret, make_label
from rsi.models.core import Model
from rsi.splits import Fold, walk_forward
from rsi.universe import universe_masks
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
    mask: np.ndarray        # entry universe: training rows, new positions
    hold: np.ndarray        # hold buffer (superset): scored every bar so held names can be managed
    cols: dict              # model columns in compact layout [T, K]: slot k of bar t is symbol idx[t, k]
    names: list[str]
    y: np.ndarray           # labels, compact layout
    context: tuple
    idx: np.ndarray = None  # [T, K] symbol index per compact slot
    kmask: np.ndarray = None
    khold: np.ndarray = None


def cached_masks(panel: Panel, uspec) -> tuple[np.ndarray, np.ndarray]:
    from rsi.config import cache_dir
    d = cache_dir() / "universe" / panel.fingerprint
    key = stable_hash(uspec.model_dump())
    p = d / f"{key}.npz"
    if p.exists():
        z = np.load(p)
        return z["entry"], z["hold"]
    entry, hold = universe_masks(panel, uspec)
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, entry=entry, hold=hold)
    return entry, hold


def _compact_cols(panel: Panel, spec: ExperimentSpec, hold: np.ndarray, idx: np.ndarray) -> dict:
    """Model columns gathered to the hold-universe slots, cached on disk (~K/S of the full size)."""
    from rsi.config import cache_dir
    from rsi.features import registry
    ukey = stable_hash(spec.universe.model_dump())
    d = cache_dir() / "features" / panel.fingerprint / f"compact-{ukey}"
    out, full = {}, None
    for c in [c for c in spec.features.columns() if "*" not in c] + [c for c in spec.features.columns() if "*" in c]:
        parts = c.split("*")
        hashes = "-".join(registry.get(x.split(":", 1)[-1]).src_hash for x in parts)
        p = d / f"{c.replace(':', '__').replace('*', '_x_')}-{hashes}-{int(spec.features.neutralize)}.npy"
        if not p.exists():
            if full is None:
                full = build_batch(panel, spec.features, hold, mask_key=ukey, interactions=False)
            d.mkdir(parents=True, exist_ok=True)
            if len(parts) == 2:   # interaction: compact xs column x compact regime column
                from rsi.features.featureset import interaction
                reg = np.take_along_axis(np.asarray(full[parts[1]]), idx, 1)
                np.save(p, interaction(out[parts[0]], reg))
            else:
                np.save(p, np.take_along_axis(np.asarray(full[c]), idx, 1))
        out[c] = np.load(p, mmap_mode="r")
    return out


def prepare(spec: ExperimentSpec, panel: Panel) -> Prepared:
    mask, hold = cached_masks(panel, spec.universe)
    names = spec.features.columns()
    contract.ensure_contract(panel, spec.features.base_names())
    K = max(int(hold.sum(axis=1).max()), 1)
    idx = np.argsort(~hold, axis=1, kind="stable")[:, :K]
    kh = np.take_along_axis(hold, idx, 1)
    km = np.take_along_axis(mask, idx, 1) & kh
    cols = _compact_cols(panel, spec, hold, idx)
    y = np.take_along_axis(make_label(panel, spec.label, mask, valid_until=panel.T), idx, 1)
    y[~km] = np.nan
    return Prepared(panel, mask, hold, cols, names, y, market_context(panel), idx, km, kh)


def _rows(mask: np.ndarray, y: np.ndarray | None, i0: int, i1: int, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    sub = mask[i0:i1:stride]
    if y is not None:
        sub = sub & np.isfinite(y[i0:i1:stride])
    t, s = np.nonzero(sub)
    return t * stride + i0, s


class TrainRows:
    """All stride-sampled training rows of the run, materialised once (one sequential pass per feature).
    Folds select by time; avoids re-reading every feature array for every fold."""

    def __init__(self, prep: Prepared, stride: int, i0: int, i1: int):
        t, s = _rows(prep.kmask, prep.y, i0, i1, stride)
        self.t, self.s = t, s
        self.X = stack(prep.cols, prep.names, t, s)
        self.y = prep.y[t, s]

    def take(self, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
        sel = (self.t >= a) & (self.t < b)
        return self.X[sel], self.y[sel]

    def years(self, prep: "Prepared", a: int, b: int) -> np.ndarray:
        sel = (self.t >= a) & (self.t < b)
        return (prep.panel.times[self.t[sel]] // (365 * 86_400_000)).astype(np.int64)


def fit_fold(prep: Prepared, spec: ExperimentSpec, fold: Fold, rows: TrainRows | None = None) -> Model:
    tr0, tr1 = fold.train
    ms = spec.model
    n_valid = int((tr1 - tr0) * ms.valid_frac) if ms.kind == "lgbm" else 0
    split = tr1 - n_valid
    rows = rows or TrainRows(prep, ms.train_stride, tr0, tr1)
    X, yy = rows.take(tr0, split - (spec.label.span if n_valid else 0))
    Xv = yv = None
    if n_valid:
        Xv, yv = rows.take(split, tr1)
    groups = rows.years(prep, tr0, split - (spec.label.span if n_valid else 0)) if ms.select_stable else None
    return Model(ms, prep.names).fit(X, yy, Xv, yv, groups=groups)


def predict_rows(prep: Prepared, model: Model, i0: int, i1: int, out: np.ndarray) -> None:
    t, k = _rows(prep.khold, None, i0, i1)
    if len(t):
        sub = {n: np.asarray(prep.cols[n][i0:i1]) for n in prep.names}   # contiguous time slice
        out[t, prep.idx[t, k]] = model.predict(stack(sub, prep.names, t - i0, k))


def oos_scores(prep: Prepared, spec: ExperimentSpec, folds: list[Fold]) -> tuple[np.ndarray, list[dict]]:
    score = np.full(prep.mask.shape, np.nan)
    info = []
    t0 = time.time()
    rows = TrainRows(prep, spec.model.train_stride, min(f.train[0] for f in folds), max(f.train[1] for f in folds))
    log.info(f"training rows: {len(rows.y):,} x {len(prep.names)} built in {time.time() - t0:.0f}s")
    for f in folds:
        t0 = time.time()
        m = fit_fold(prep, spec, f, rows)
        predict_rows(prep, m, *f.test, score)
        info.append({**f.as_dict(prep.panel.times), "best_iter": m.best_iter, "fit_s": round(time.time() - t0, 1),
                     "importance": m.importance, "n_selected": int(m.keep.sum()) if m.keep is not None else len(prep.names)})
        log.info(f"fold {f.k}: test {info[-1]['test'][0][:10]}..{info[-1]['test'][1][:10]} fit {info[-1]['fit_s']}s "
                 f"iters={m.best_iter} cols={info[-1]['n_selected']}")
    return score, info


def ensemble_scores(spec: ExperimentSpec, panel: Panel) -> tuple[np.ndarray, list[dict]]:
    import warnings
    h = harness_config()
    scores, finfo = [], []
    for i, ov in enumerate(spec.ensemble):
        member = spec.with_overrides({**ov, "ensemble": []}, name=f"{spec.name}#m{i}")
        pm = prepare(member, panel)
        fm = walk_forward(panel.times, h, member.label.span, test_months=member.test_months, train_months=member.train_months)
        s_m, info = oos_scores(pm, member, fm)
        scores.append(s_m)
        finfo += [{**x, "member": i} for x in info]
        log.info(f"ensemble member {i} {ov} done")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(scores), axis=0), finfo


def fold_metrics(res: BacktestResult, panel: Panel, folds: list[Fold], bars_per_year: int) -> list[dict]:
    out = []
    eq = pd.Series(np.concatenate([[res.equity0], res.equity]), index=np.concatenate([[res.times[0] - 1], res.times]))
    for f in folds:
        a, b = panel.times[f.test[0]], panel.times[min(f.test[1], panel.T) - 1]
        seg = eq[(eq.index >= a - 1) & (eq.index <= b)]
        if len(seg) < 3:
            continue
        d = M.daily_returns(seg.index[1:].to_numpy(), seg.to_numpy()[1:], float(seg.iloc[0]))
        r = seg.to_numpy()[1:] / seg.to_numpy()[:-1] - 1
        out.append({"k": f.k, "start": str(pd.Timestamp(a, unit="ms").date()), "sharpe": M.sharpe(r, bars_per_year),
                    "ret": float(seg.iloc[-1] / seg.iloc[0] - 1), "mdd": M.max_drawdown(seg.to_numpy()), "days": len(d)})
    return out


def scored_risk(spec: ExperimentSpec):
    """T4.8: gated backtests always run at the harness vol target (exposure is a live preference, not alpha)."""
    return spec.risk.model_copy(update={"vol_target": harness_config().gates.eval_vol_target})


def evaluate(prep: Prepared, spec: ExperimentSpec, score: np.ndarray, t0: int, t1: int, folds: list[Fold],
             n_trials: int, sr_var: float | None) -> dict:
    from rsi.eval import disaster
    h = harness_config()
    res = run_backtest(prep.panel, score, prep.mask, spec.strategy, scored_risk(spec), h.costs, h.capital.initial_usd,
                       t0, t1, h.bars_per_year, context=prep.context, hold=prep.hold)
    m = M.summarize(res, h.bars_per_year, n_trials, sr_var)
    th = disaster.calibrate(res.returns, h.disaster)
    m.update({"liquidations": res.liquidations, "disaster_day_loss": th["day_loss"], "disaster_max_dd": th["max_dd"],
              "disaster_triggers": disaster.count_triggers(res.times, res.equity, res.equity0, th),
              "eval_vol_target": h.gates.eval_vol_target})
    fwd = fwd_logret(prep.panel, spec.label.horizon) - fwd_funding(prep.panel, spec.label.horizon)
    cols = np.flatnonzero(prep.hold[t0:t1].any(axis=0))
    m.update({f"oos_{k}": v for k, v in M.ic_stats(score[t0:t1:4][:, cols], fwd[t0:t1:4][:, cols],
                                                   prep.mask[t0:t1:4][:, cols]).items()})
    fm = fold_metrics(res, prep.panel, folds, h.bars_per_year)
    return {"metrics": m, "folds": fm, "result": res}


def run(spec: ExperimentSpec, note: str | None = None, save: bool = True, ledger: Ledger | None = None) -> dict:
    h = harness_config()
    ledger = ledger or Ledger()
    hh = harness_hash()
    ledger.check_harness(h.version, hh)
    t_start = time.time()
    panel = research_panel(spec)
    prep = prepare(spec, panel)
    folds = walk_forward(panel.times, h, spec.label.span, test_months=spec.test_months, train_months=spec.train_months)
    if spec.ensemble:
        score, finfo = ensemble_scores(spec, panel)
    else:
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
            "run_id": run_id, "name": spec.name, "parent": spec.parent, "harness_version": h.version, "harness_hash": hh,
            "spec_hash": spec.spec_hash, "spec_json": json.dumps(spec.model_dump()), "git_sha": git_sha(),
            "data_fp": panel.fingerprint, "status": status, "sharpe": m["sharpe"], "dsr": m["dsr"],
            "max_dd": m["max_drawdown"], "daily_sr": m["daily_sr"], "metrics_json": json.dumps(m, default=float),
            "hypothesis": spec.hypothesis, "note": note,
        })
        if status == "champion":
            ledger.set_champion(h.version, run_id)
        append_rsi_log(spec, summary)
    log.info(f"[{status}] {spec.name} sharpe={m['sharpe']:.2f} dsr={m['dsr']:.2f} mdd={m['max_drawdown']:.1%} "
             f"cagr={m['cagr']:.1%} trades={m['n_trades']} ({summary['elapsed_s']}s) -> {run_id}")
    return summary


def append_rsi_log(spec: ExperimentSpec, s: dict) -> None:
    """HRN-08: one auto-generated entry per run; the proposer adds the conclusion underneath."""
    p = repo_root() / "docs" / "rsi_log.md"
    m = s["metrics"]
    failed = [k for k, v in s["gates"]["checks"].items() if not v["ok"]]
    entry = (f"\n### {s['run_id']} · {spec.name} · **{s['status']}**\n"
             f"- date: {time.strftime('%Y-%m-%d %H:%M')} · parent: {spec.parent or '-'} · trials so far: {s['n_trials']}\n"
             f"- hypothesis: {spec.hypothesis.strip() or '-'}\n"
             f"- sharpe {m['sharpe']:.2f} · dsr {m['dsr']:.2f} · cagr {m['cagr']:.1%} · mdd {m['max_drawdown']:.1%} · "
             f"turnover {m['turnover_ann']:.0f}x · fees {m['fees_frac']:.1%} · funding {m['funding_frac']:.1%} · "
             f"long {m['long_pnl_frac']:+.1%} / short {m['short_pnl_frac']:+.1%} · ic {m.get('oos_ic', 0):.4f}\n"
             f"- failed gates: {', '.join(failed) or 'none'} · {s['champion_check']}\n")
    if not p.exists():
        p.write_text("# RSI research log\n\nAuto-generated entries per run (HRN-08); add a `- conclusion:` line under each.\n")
    with open(p, "a") as f:
        f.write(entry)


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

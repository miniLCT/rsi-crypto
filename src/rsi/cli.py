from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, help="rsi-crypto: research harness + trading engine for Binance USDT-M perps")
data_app = typer.Typer(help="market data")
feat_app = typer.Typer(help="feature registry")
ledger_app = typer.Typer(help="experiment ledger")
app.add_typer(data_app, name="data")
app.add_typer(feat_app, name="features")
app.add_typer(ledger_app, name="ledger")
console = Console()


# ---------------------------------------------------------------- data
@data_app.command("sync")
def data_sync(symbols: Optional[list[str]] = typer.Option(None, "--symbol", "-s"),
              datasets: Optional[list[str]] = typer.Option(None, "--dataset", "-d"),
              concurrency: int = 48):
    """Mirror data.binance.vision archives (incremental)."""
    from rsi.config import data_config
    from rsi.data import vision
    cfg = data_config()
    cfg.download.concurrency = concurrency
    console.print(asyncio.run(vision.sync(cfg, symbols, datasets)))


@data_app.command("ingest")
def data_ingest(workers: int = 12):
    """Raw zips -> per-symbol parquet lake."""
    from rsi.config import data_config
    from rsi.data import lake
    out = lake.ingest(data_config(), workers=workers)
    console.print(f"{len(out)} symbols ingested")


@data_app.command("quality")
def data_quality(top: int = 30):
    from rsi.config import data_config
    from rsi.data import lake
    df = lake.quality_report(data_config())
    df.to_csv("data/quality.csv", index=False)
    console.print(df.sort_values("median_qv", ascending=False).head(top).to_string())


@data_app.command("panel")
def data_panel(top_k: int = 150):
    """Build (and cache) the full research+holdout panel."""
    from rsi.harness.runner import full_panel
    from rsi.harness.spec import ExperimentSpec
    p = full_panel(ExperimentSpec(name="panel", panel_top_k=top_k))
    console.print(f"T={p.T} S={p.S} fingerprint={p.fingerprint}")


# ---------------------------------------------------------------- features
@feat_app.command("list")
def feat_list(selector: str = "all"):
    from rsi.features import registry
    t = Table("name", "group", "lookback", "hash", "doc")
    for n in registry.resolve([selector]):
        f = registry.get(n)
        t.add_row(n, f.group, str(f.lookback), f.src_hash, f.doc)
    console.print(t)


@feat_app.command("check")
def feat_check(selectors: list[str] = typer.Argument(None)):
    """Run the causality + lookback contract on features."""
    from rsi.features import contract, registry
    from rsi.harness.runner import research_panel
    from rsi.harness.spec import ExperimentSpec
    names = registry.resolve(selectors or ["all"])
    contract.ensure_contract(research_panel(ExperimentSpec(name="check")), names)
    console.print(f"[green]contract OK for {len(names)} features")


@app.command()
def screen(selectors: list[str] = typer.Argument(None), compare: bool = True, out: Optional[Path] = None):
    """Single-factor IC / decay / redundancy screen."""
    from rsi.features import registry
    from rsi.harness.screen import screen as do_screen
    names = registry.resolve(selectors or ["all"])
    ref = registry.resolve(["all"]) if compare else None
    df = do_screen(names, compare_to=ref)
    if out:
        df.to_csv(out, index=False)
    console.print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ---------------------------------------------------------------- experiments
@app.command()
def run(spec_path: Path, note: Optional[str] = None, set_: Optional[list[str]] = typer.Option(None, "--set")):
    """Run one experiment spec through the harness. --set key.path=value overrides."""
    import yaml
    from rsi.harness import runner
    from rsi.harness.spec import ExperimentSpec
    spec = ExperimentSpec.load(spec_path)
    if set_:
        ov = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in set_)}
        spec = spec.with_overrides(ov, name=spec.name + "~" + ",".join(set_))
    s = runner.run(spec, note=note)
    _print_summary(s)


@app.command()
def search(spec_path: Path, n: int = 10, seed: int = 0):
    """Automatic proposer over the spec's search_space."""
    from rsi.harness.search import search as do_search
    from rsi.harness.spec import ExperimentSpec
    res = do_search(ExperimentSpec.load(spec_path), n, seed)
    for r in res:
        console.print(r)


@app.command()
def holdout(run_id: str, force: bool = False):
    """Budgeted, logged evaluation on the locked holdout period."""
    from rsi.harness.release import holdout as do_holdout
    r = do_holdout(run_id, force=force)
    console.print(f"[{'green' if r['passed'] else 'red'}]{r['verdict']}")
    console.print(json.dumps({k: r["metrics"][k] for k in ("sharpe", "cagr", "max_drawdown", "n_trades", "dsr")}, indent=1))
    for f in r["folds"]:
        console.print(f)


@app.command()
def export(run_id: str, out: Optional[Path] = None):
    """Train the deployable model and write a versioned artifact with a manifest."""
    from rsi.harness.release import export as do_export
    console.print(str(do_export(run_id, out)))


@ledger_app.command("ls")
def ledger_ls(top: int = 20, by: str = "created"):
    from rsi.config import harness_config
    from rsi.harness.ledger import Ledger
    order = {"created": "created DESC", "sharpe": "sharpe DESC", "dsr": "dsr DESC"}[by]
    rows = Ledger().runs(harness_config().version, limit=top, order=order)
    t = Table("run_id", "name", "status", "sharpe", "dsr", "mdd", "cagr", "trades", "turnover")
    for r in rows:
        m = json.loads(r["metrics_json"] or "{}")
        t.add_row(r["run_id"], (r["name"] or "")[:48], r["status"], f"{r['sharpe']:.2f}", f"{r['dsr']:.2f}",
                  f"{r['max_dd']:.1%}", f"{m.get('cagr', 0):.1%}", str(m.get("n_trades")), f"{m.get('turnover_ann', 0):.0f}")
    console.print(t)
    from rsi.harness.ledger import Ledger as L
    n, var = L().trial_stats(harness_config().version)
    console.print(f"distinct trials: {n}, SR variance across trials: {var}")


@ledger_app.command("show")
def ledger_show(run_id: str):
    from rsi.config import runs_dir
    from rsi.harness.ledger import Ledger
    r = Ledger().get(run_id)
    if not r:
        raise typer.Exit(1)
    p = runs_dir() / r["run_id"] / "report.md"
    console.print(p.read_text() if p.exists() else r)


def _print_summary(s: dict):
    m = s["metrics"]
    console.print(f"[bold]{s['run_id']}[/] {s['name']} -> [{'green' if s['status'] != 'fail' else 'red'}]{s['status']}")
    console.print(f"sharpe={m['sharpe']:.2f} dsr={m['dsr']:.3f} cagr={m['cagr']:.1%} mdd={m['max_drawdown']:.1%} "
                  f"trades={m['n_trades']} hit={m['hit_rate']:.1%} turnover={m['turnover_ann']:.0f}x "
                  f"fees={m['fees_frac']:.1%} funding={m['funding_frac']:.1%} ic={m.get('oos_ic', 0):.4f}")
    for k, v in s["gates"]["checks"].items():
        console.print(f"  {'✓' if v['ok'] else '✗'} {k}: {v['rule']}")
    console.print(f"  champion: {s['champion_check']}")


# ---------------------------------------------------------------- live
@app.command()
def live(config: Path = typer.Option(..., "--config", "-c"), once: bool = False):
    """Run the live loop (paper / testnet / mainnet per config)."""
    from rsi.live.runner import main as live_main
    asyncio.run(live_main(config, once=once))


@app.command()
def parity(model_dir: Path, bars: int = 96):
    """Replay the live decision path on lake data and compare with batch research outputs."""
    from rsi.live.parity import check as do_check
    console.print(do_check(model_dir, bars))


if __name__ == "__main__":
    app()

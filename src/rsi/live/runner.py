"""Live loop: paper / testnet / mainnet, sharing the research decision path verbatim.

Per closed bar t:
  feed.update(t) -> trailing Panel window -> universe row -> feature row -> model score
  -> TargetBuilder.step -> Portfolio.target_qty (all risk overlays) -> broker.execute
Everything is journaled; the decision log (features + scores per bar) is what
`rsi parity --journal` replays against archived data to prove train/serve parity.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field

from rsi.config import data_config, harness_config, repo_root
from rsi.data.panel import Panel
from rsi.engine.backtest import market_context
from rsi.engine.portfolio import Portfolio, RiskSpec
from rsi.features.featureset import build_last
from rsi.harness.release import verify_manifest
from rsi.harness.spec import ExperimentSpec
from rsi.live.binance import BinanceFutures
from rsi.live.broker import BinanceBroker, PaperBroker, parse_rules
from rsi.live.feed import LiveFeed
from rsi.models.core import Model
from rsi.strategy import TargetBuilder
from rsi.universe import UniverseSpec, compute_universe
from rsi.utils.log import add_file_handler, get_logger
from rsi.utils.timeutil import interval_ms, ms_to_str, now_ms

log = get_logger("live")


class LiveConfig(BaseModel):
    mode: Literal["paper", "testnet", "mainnet"] = "paper"
    model_dir: str
    state_dir: str = "live_state/paper"
    capital_usd: float = 10000.0
    symbols: list[str] | None = None            # restrict trading to these (single-coin deployment)
    leverage: int = 3
    risk_overrides: dict = Field(default_factory=dict)
    exec_delay_s: float = 4.0
    api_key_env: str = "BINANCE_API_KEY"
    api_secret_env: str = "BINANCE_API_SECRET"
    ed25519_key_path_env: str = "BINANCE_ED25519_KEY"
    alert_webhook_env: str = "RSI_ALERT_WEBHOOK"
    max_missing_frac: float = 0.2               # skip trading when too much of the universe has no fresh bar


TIGHTEN_ONLY = {"max_gross": min, "max_weight": min, "max_net": min, "vol_target": min, "dd_halt": min,
                "daily_loss_halt": min, "adv_cap": min, "stop_loss_vol": min, "cooldown_bars": max, "dd_halt_bars": max}


def tighten(risk: RiskSpec, overrides: dict) -> RiskSpec:
    d = risk.model_dump()
    for k, v in overrides.items():
        if k not in TIGHTEN_ONLY:
            raise ValueError(f"risk override {k} not allowed live")
        cur = d.get(k)
        d[k] = v if cur is None else TIGHTEN_ONLY[k](cur, v)
    return RiskSpec(**d)


class Alerter:
    def __init__(self, url: str | None):
        self.url = url
        self._last: dict[str, float] = {}

    async def send(self, key: str, text: str, cooldown_s: float = 600) -> None:
        log.warning(f"ALERT {key}: {text}")
        if not self.url or time.time() - self._last.get(key, 0) < cooldown_s:
            return
        self._last[key] = time.time()
        import aiohttp
        from rsi.data.vision import ssl_context
        try:
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context())) as s:
                await s.post(self.url, json={"msg_type": "text", "content": {"text": f"[rsi-crypto] {text}"}}, timeout=10)
        except Exception as e:  # noqa: BLE001
            log.warning(f"alert delivery failed: {e}")


class LiveRunner:
    def __init__(self, cfg: LiveConfig):
        self.cfg = cfg
        self.state_dir = _abs(cfg.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "decisions").mkdir(exist_ok=True)
        add_file_handler(str(self.state_dir / "live.log"))
        mdir = _abs(cfg.model_dir)
        self.manifest = json.loads((mdir / "manifest.json").read_text())
        problems = verify_manifest(self.manifest)
        if problems:
            raise RuntimeError("model/code mismatch, refusing to trade: " + "; ".join(problems))
        self.model = Model.load(mdir)
        self.spec = ExperimentSpec(**self.manifest["spec"])
        if cfg.symbols and self.spec.universe.kind != "fixed":
            log.warning("model trained on a dynamic universe; live trading restricted to %s", cfg.symbols)
        self.trade_symbols = set(cfg.symbols) if cfg.symbols else None
        self.risk = tighten(self.spec.risk, cfg.risk_overrides)
        self.costs = harness_config().costs
        self.step = interval_ms(self.manifest["interval"])
        self.alert = Alerter(os.environ.get(cfg.alert_webhook_env))
        self.data_client = BinanceFutures("mainnet")
        self.exec_client: BinanceFutures | None = None
        self.feed: LiveFeed | None = None
        self.port: Portfolio | None = None
        self.builder: TargetBuilder | None = None
        self.broker = None
        self.last_bar = 0
        self.boundary_qty: dict[str, float] = {}

    # ------------------------------------------------------------------ setup
    async def setup(self) -> None:
        cfg = self.cfg
        extra = list(self._load_state().get("portfolio", {}).get("symbols", []))
        self.feed = LiveFeed(self.data_client, data_config(), self.manifest["window_bars"], extra_symbols=None,
                             only_symbols=None if self.spec.universe.kind == "top_liquid" else self.spec.universe.symbols)
        await self.feed.bootstrap()
        if cfg.mode == "paper":
            self.broker = PaperBroker()
        else:
            env = "testnet" if cfg.mode == "testnet" else "mainnet"
            self.exec_client = BinanceFutures(env, os.environ.get(cfg.api_key_env), os.environ.get(cfg.api_secret_env),
                                              os.environ.get(cfg.ed25519_key_path_env))
            info = self.feed.exchange_info if env == "mainnet" else await self.exec_client.exchange_info()
            self.broker = BinanceBroker(self.exec_client, parse_rules(info), cfg.leverage, self.costs.taker_fee_bps)
            await self.broker.prepare()
        self._build_books(self.feed.symbols)
        st = self._load_state()
        if st:
            self.port.load_state(st["portfolio"])
            prev = st.get("builder_prev", {})
            self.builder.prev = np.array([prev.get(s, 0.0) for s in self.feed.symbols])
            self.last_bar = st.get("last_bar", 0)
            self.boundary_qty = st.get("boundary_qty", {})
            log.info(f"restored state: equity={self.port.equity:.2f} positions={int((self.port.qty != 0).sum())}")
        if extra and set(extra) - set(self.feed.symbols):
            await self.alert.send("orphans", f"held symbols no longer tradable: {sorted(set(extra) - set(self.feed.symbols))}")

    def _build_books(self, symbols: list[str]) -> None:
        old_p, old_b = self.port, self.builder
        self.port = Portfolio(symbols, self.cfg.capital_usd, self.costs, self.risk, harness_config().bars_per_year, self.step)
        self.builder = TargetBuilder(self.spec.strategy, len(symbols))
        if old_p is not None:
            self.port.load_state(old_p.state_dict())
            prev = dict(zip(old_p.symbols, old_b.prev))
            self.builder.prev = np.array([prev.get(s, 0.0) for s in symbols])

    # ------------------------------------------------------------------ state
    def _load_state(self) -> dict:
        p = self.state_dir / "state.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def _save_state(self) -> None:
        st = {"portfolio": self.port.state_dict(), "builder_prev": dict(zip(self.port.symbols, self.builder.prev.tolist())),
              "last_bar": self.last_bar, "boundary_qty": self.boundary_qty, "saved": now_ms(),
              "model": self.manifest["run_id"]}
        tmp = self.state_dir / "state.json.tmp"
        tmp.write_text(json.dumps(st))
        tmp.replace(self.state_dir / "state.json")

    def _journal(self, rec: dict) -> None:
        with open(self.state_dir / "journal.jsonl", "a") as f:
            f.write(json.dumps(rec, default=_json_default) + "\n")

    # ------------------------------------------------------------------ one decision
    async def cycle(self, bar_open: int) -> dict:
        t0 = time.time()
        missing = await self.feed.update(bar_open)
        if self.feed.symbols != self.port.symbols:
            self._build_books(self.feed.symbols)
        win: Panel = self.feed.window(bar_open)
        S = win.S
        sym_idx = {s: i for i, s in enumerate(win.symbols)}
        close = np.asarray(win["close"][-1], dtype=np.float64)
        prev_close = np.asarray(win["close"][-2], dtype=np.float64)
        fresh_frac = float(np.isfinite(close).mean())

        # exchange is the source of truth for positions and equity
        exch_equity = None
        if self.broker.kind == "binance":
            positions, exch_equity = await self.broker.sync()
            q = np.zeros(S)
            for s, a in positions.items():
                if s in sym_idx:
                    q[sym_idx[s]] = a
                else:
                    await self.alert.send(f"orphan-{s}", f"exchange position in untracked symbol {s}: {a}")
            drift = np.flatnonzero(np.abs(q - self.port.qty) > 1e-9)
            if len(drift):
                log.warning(f"position drift vs local book on {[win.symbols[i] for i in drift]} -> adopting exchange")
            self.port.qty = q
        else:
            # paper: settle funding printed at this bar's open on the book held across that boundary
            fr = np.asarray(win["funding_rate"][-1], dtype=np.float64)
            if np.isfinite(fr).any() and self.boundary_qty:
                held = self.port.qty.copy()
                self.port.qty = np.array([self.boundary_qty.get(s, 0.0) for s in win.symbols])
                self.port.apply_funding(fr, np.asarray(win["open"][-1], dtype=np.float64))
                self.port.qty = held

        self.port.mark_to(close)
        if exch_equity is not None:
            self.port.cash = exch_equity
        self.port.record_book_return(close / prev_close - 1)

        uni = compute_universe(win, self.spec.universe)[-1]
        cols = build_last(win, self.spec.features, uni)
        names = self.spec.features.columns()
        idx = np.flatnonzero(uni)
        X = np.column_stack([cols[c][idx] for c in names]).astype(np.float32) if len(idx) else np.zeros((0, len(names)), np.float32)
        score = np.full(S, np.nan)
        if len(idx):
            score[idx] = self.model.predict(X)
        adv, dvol = (a[-1] for a in market_context(win))
        bar_no = int(bar_open // self.step)

        w = self.builder.step(bar_no, score, uni, dvol)
        if self.trade_symbols is not None:
            w = np.where([s in self.trade_symbols for s in win.symbols], w, 0.0)
        kill = (self.state_dir / "KILL").exists()
        degraded = fresh_frac < 1 - self.cfg.max_missing_frac
        if kill:
            q_target = np.zeros(S)
        elif degraded:
            q_target = self.port.qty.copy()
            await self.alert.send("stale", f"only {fresh_frac:.0%} symbols have bar {ms_to_str(bar_open)}; holding")
        else:
            q_target = self.port.target_qty(bar_open + self.step, bar_no, w, close, adv, dvol)
        q_target = np.where(np.isfinite(close), q_target, self.port.qty)   # never trade blind

        dq = q_target - self.port.qty
        orders = {win.symbols[i]: float(dq[i]) for i in np.flatnonzero(dq != 0)}
        ref = await self.data_client.ticker_price() if orders else {}
        self.boundary_qty = {s: float(a) for s, a in zip(win.symbols, self.port.qty) if a != 0}
        current = {s: float(a) for s, a in zip(win.symbols, self.port.qty) if a != 0}
        fills = await (self.broker.execute(orders, ref, bar_no, current) if self.broker.kind == "binance"
                       else self.broker.execute(orders, ref, bar_no))

        # book fills through the shared accounting (paper: authoritative; live: re-synced next bar)
        new_q = self.port.qty.copy()
        px = close.copy()
        for f in fills:
            i = sym_idx[f.symbol]
            new_q[i] += f.dq
            if math.isfinite(f.price):
                px[i] = f.price
        self.port.mark_to(px)
        self.port.rebalance(new_q, px, adv, dvol, now_ms())
        failed = [f for f in fills if f.status not in ("FILLED", "NEW", "PARTIALLY_FILLED")]
        if failed:
            await self.alert.send("fills", f"{len(failed)} orders failed: {[(f.symbol, f.status) for f in failed][:5]}")

        self.last_bar = bar_open
        self._save_state()
        dec = pd.DataFrame({"symbol": [win.symbols[i] for i in idx], "score": score[idx], "w": w[idx]})
        for c in names:
            dec[c] = cols[c][idx]
        dec.insert(0, "bar_open", bar_open)
        day = ms_to_str(bar_open)[:10]
        p = self.state_dir / "decisions" / f"{day}.parquet"
        dec = pd.concat([pd.read_parquet(p), dec], ignore_index=True) if p.exists() else dec
        dec.to_parquet(p, index=False)
        rec = {
            "bar_open": bar_open, "bar": ms_to_str(bar_open), "mode": self.cfg.mode, "equity": self.port.equity,
            "exchange_equity": exch_equity, "gross": float(np.abs(self.port.notional()).sum() / max(self.port.equity, 1e-9)),
            "n_universe": int(uni.sum()), "fresh_frac": fresh_frac, "missing": missing[:20], "kill": kill,
            "halted": self.port.halt_until_ms > now_ms() or self.port.day_halted, "vol_scale": self.port.vol_scale(),
            "orders": orders, "fills": [f.__dict__ | {"slippage_bps": f.slippage_bps} for f in fills],
            "latency_s": round(time.time() - t0, 2),
        }
        self._journal(rec)
        log.info(f"bar {rec['bar']} eq={rec['equity']:.2f} gross={rec['gross']:.2f} uni={rec['n_universe']} "
                 f"orders={len(orders)} fills={len(fills)} {rec['latency_s']}s")
        return rec

    async def run(self, once: bool = False) -> None:
        await self.setup()
        try:
            while True:
                now = now_ms()
                bar_open = now - now % self.step - self.step       # last fully closed bar
                if bar_open > self.last_bar:
                    wait = (bar_open + self.step) / 1000 + self.cfg.exec_delay_s - time.time()
                    if wait > 0:
                        await asyncio.sleep(wait)
                    try:
                        await self.cycle(bar_open)
                    except Exception as e:  # noqa: BLE001
                        log.exception("cycle failed")
                        await self.alert.send("cycle", f"cycle {ms_to_str(bar_open)} failed: {e!r}")
                    if once:
                        return
                    if (self.state_dir / "KILL").exists():
                        log.warning("KILL file present: book flattened, exiting")
                        return
                nxt = (now_ms() // self.step + 1) * self.step
                await asyncio.sleep(max(1.0, (nxt - now_ms()) / 1000 + self.cfg.exec_delay_s))
        finally:
            await self.data_client.close()
            if self.exec_client:
                await self.exec_client.close()


def _abs(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else repo_root() / q


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return str(o)


async def main(config_path: Path, once: bool = False) -> None:
    cfg = LiveConfig(**yaml.safe_load(Path(config_path).read_text()))
    await LiveRunner(cfg).run(once=once)

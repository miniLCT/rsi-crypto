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
from rsi.universe import universe_masks
from rsi.utils.log import add_file_handler, get_logger
from rsi.utils.timeutil import interval_ms, ms_to_str, now_ms

log = get_logger("live")


class LiveConfig(BaseModel):
    mode: Literal["paper", "testnet", "mainnet"] = "paper"
    model_dir: str
    state_dir: str = "live_state/paper"
    capital_usd: float = 10000.0
    symbols: list[str] | None = None            # restrict trading to these (single-coin deployment)
    exchange_leverage: int = 3                  # DEC-06: per-symbol margin multiple, 1-10 (capped by leverage brackets)
    risk_overrides: dict = Field(default_factory=dict)
    margin_ratio_max: float = 0.30              # RISK-05: maint margin / margin balance above this -> scale down
    exec_delay_s: float = 4.0
    api_key_env: str = "BINANCE_API_KEY"
    api_secret_env: str = "BINANCE_API_SECRET"
    ed25519_key_path_env: str = "BINANCE_ED25519_KEY"
    alert_webhook_env: str = "RSI_ALERT_WEBHOOK"
    min_coverage: float = 0.98                  # RISK-04: below this share of fresh universe bars, no new risk
    max_reject_rate: float = 0.3                # RISK-04: order reject-rate breaker (last 20 orders)
    open_deadline_s: float = 120.0              # LIVE-01: opening legs not sent by bar close + this are dropped
    delist_horizon_days: float = 7.0            # close & blacklist contracts with a delivery date this close


# exposure is a live risk preference (DEC-06), bounded by RiskSpec validation (<= 10x)
PREFERENCE = {"max_gross", "vol_target", "exchange_leverage"}
# everything else may only be tightened relative to what was researched
TIGHTEN_ONLY = {"max_weight": min, "max_net": min, "adv_cap": min, "hard_stop_vol": min, "stop_loss_vol": min,
                "dd_halt": min, "daily_loss_halt": min, "cooldown_bars": max, "margin_usage_max": min}


def tighten(risk: RiskSpec, overrides: dict) -> RiskSpec:
    d = risk.model_dump()
    for k, v in overrides.items():
        if k in PREFERENCE:
            d[k] = v
        elif k in TIGHTEN_ONLY:
            cur = d.get(k)
            d[k] = v if cur is None else TIGHTEN_ONLY[k](cur, v)
        else:
            raise ValueError(f"risk override {k} not allowed live")
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
        self.risk = tighten(self.spec.risk, {**cfg.risk_overrides, "exchange_leverage": cfg.exchange_leverage})
        # RISK-06: disaster thresholds were calibrated at the harness eval vol target; scale to live exposure
        d = self.manifest.get("disaster") or {}
        k = (self.risk.vol_target or d.get("eval_vol_target") or 1.0) / (d.get("eval_vol_target") or 1.0)
        self.disaster = {"day_loss": d["day_loss"] * k if d.get("day_loss") else None,
                         "max_dd": min(d["max_dd"] * k, 0.95) if d.get("max_dd") else None}
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
        self.stale_bars = 0
        self.boundary_qty: dict[str, float] = {}
        self.tag = "r" + self.manifest["run_id"][-6:]

    # ------------------------------------------------------------------ setup
    async def setup(self) -> None:
        cfg = self.cfg
        extra = list(self._load_state().get("portfolio", {}).get("symbols", []))
        self.feed = LiveFeed(self.data_client, data_config(), self.manifest["window_bars"], extra_symbols=None,
                             only_symbols=None if self.spec.universe.kind == "top_liquid" else self.spec.universe.symbols)
        from rsi.features import registry as _reg
        self.positioning_bars = max([_reg.get(n).lookback for n in self.spec.features.compute_names()
                                     if _reg.get(n).group == "positioning"] or [0])
        self.feed.with_metrics = self.positioning_bars > 0
        await self.feed.bootstrap()
        if cfg.mode == "paper":
            self.broker = PaperBroker()
        else:
            env = "testnet" if cfg.mode == "testnet" else "mainnet"
            self.exec_client = BinanceFutures(env, os.environ.get(cfg.api_key_env), os.environ.get(cfg.api_secret_env),
                                              os.environ.get(cfg.ed25519_key_path_env))
            info = self.feed.exchange_info if env == "mainnet" else await self.exec_client.exchange_info()
            self.broker = BinanceBroker(self.exec_client, parse_rules(info), cfg.exchange_leverage, self.costs.taker_fee_bps)
            await self.broker.prepare()
        self._build_books(self.feed.symbols)
        st = self._load_state()
        if st:
            self.port.load_state(st["portfolio"])
            prev = st.get("builder_prev", {})
            self.builder.prev = np.array([prev.get(s, 0.0) for s in self.feed.symbols])
            self.builder.hist = [np.array([np.nan if h.get(s, 1e300) >= 1e300 else h[s] for s in self.feed.symbols])
                                 for h in st.get("builder_hist", [])]
            self.last_bar = st.get("last_bar", 0)
            self.boundary_qty = st.get("boundary_qty", {})
            if st.get("peak_equity") is not None:
                self.peak_equity, self._day, self._day_start = st["peak_equity"], st["day"], st["day_start"]
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
            self.builder.hist = [np.array([dict(zip(old_p.symbols, h)).get(s, np.nan) for s in symbols]) for h in old_b.hist]

    def _delisting_soon(self) -> set[str]:
        """Perpetuals get a real deliveryDate once a delisting is scheduled (default is year 2100)."""
        horizon = now_ms() + self.cfg.delist_horizon_days * 86_400_000
        out = set()
        for s in self.feed.exchange_info.get("symbols", []):
            dd = int(s.get("deliveryDate", 0) or 0)
            if s.get("contractType") == "PERPETUAL" and 0 < dd < 4_000_000_000_000 and dd < horizon:
                out.add(s["symbol"])
            elif s.get("status") not in (None, "TRADING"):
                out.add(s["symbol"])
        return out

    # ------------------------------------------------------------------ state
    def _load_state(self) -> dict:
        p = self.state_dir / "state.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def _save_state(self) -> None:
        st = {"portfolio": self.port.state_dict(), "builder_prev": dict(zip(self.port.symbols, self.builder.prev.tolist())),
              "builder_hist": [dict(zip(self.port.symbols, np.nan_to_num(h, nan=1e300).tolist())) for h in self.builder.hist],
              "last_bar": self.last_bar, "boundary_qty": self.boundary_qty, "saved": now_ms(),
              "peak_equity": getattr(self, "peak_equity", None), "day": getattr(self, "_day", None),
              "day_start": getattr(self, "_day_start", None),
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
        await self.feed.update_metrics(bar_open, self.positioning_bars + 8)
        if self.feed.symbols != self.port.symbols:
            self._build_books(self.feed.symbols)
        win: Panel = self.feed.window(bar_open)
        S = win.S
        sym_idx = {s: i for i, s in enumerate(win.symbols)}
        close = np.asarray(win["close"][-1], dtype=np.float64)
        prev_close = np.asarray(win["close"][-2], dtype=np.float64)
        fresh_frac = float(np.isfinite(close).mean())
        coverage = 1.0

        adv_all, dvol_all = market_context(win)
        adv, dvol = adv_all[-1], dvol_all[-1]
        bar_no = int(bar_open // self.step)

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
            # positions that vanished on the exchange were closed by the protective stop
            for i in np.flatnonzero((self.port.qty != 0) & (q == 0)):
                px = self.port.hard_stop_px[i] if np.isfinite(self.port.hard_stop_px[i]) else close[i]
                self.port._mark_one(int(i), float(px))
                self.port.fill(int(i), -float(self.port.qty[i]), float(px), float(adv[i]), float(dvol[i]), now_ms(), "hard_stop")
                self.port.cooldown_until[i] = bar_no + self.risk.cooldown_bars
                await self.alert.send(f"stop-{win.symbols[i]}", f"hard stop filled on exchange: {win.symbols[i]}", 0)
            drift = np.flatnonzero(np.abs(q - self.port.qty) > 1e-9)
            if len(drift):
                log.warning(f"position drift vs local book on {[win.symbols[i] for i in drift]} -> adopting exchange")
            self.port.qty = q
        else:
            # paper: settle funding printed at this bar's open on the book held across that boundary,
            # then replay intrabar hard stops on the bar that just closed (same rule as the backtest)
            fr = np.asarray(win["funding_rate"][-1], dtype=np.float64)
            if np.isfinite(fr).any() and self.boundary_qty:
                held = self.port.qty.copy()
                self.port.qty = np.array([self.boundary_qty.get(s, 0.0) for s in win.symbols])
                self.port.apply_funding(fr, np.asarray(win["open"][-1], dtype=np.float64))
                self.port.qty = held
            hit = self.port.check_hard_stops(bar_open, bar_no, np.asarray(win["open"][-1]), np.asarray(win["high"][-1]),
                                             np.asarray(win["low"][-1]), adv_all[-2], dvol_all[-2])
            for i in hit:
                await self.alert.send(f"stop-{win.symbols[i]}", f"hard stop (paper): {win.symbols[i]}", 0)

        self.port.mark_to(close)
        if exch_equity is not None:
            self.port.cash = exch_equity
        equity_close = self.port.equity
        self.port.record_book_return(close / prev_close - 1)

        entry_m, hold_m = universe_masks(win, self.spec.universe)
        uni, hold = entry_m[-1].copy(), hold_m[-1]
        cols = build_last(win, self.spec.features, hold)
        names = self.spec.features.columns()
        idx = np.flatnonzero(hold)
        X = np.column_stack([cols[c][idx] for c in names]).astype(np.float32) if len(idx) else np.zeros((0, len(names)), np.float32)
        score = np.full(S, np.nan)
        if len(idx):
            score[idx] = self.model.predict(X)

        beta = None
        if self.spec.strategy.beta_neutral:
            from rsi.features.engine import compute_last
            beta = compute_last(win, ["beta_btc_336"])["beta_btc_336"].astype(np.float64)
        regime = None
        if self.spec.strategy.exposure_feature:
            from rsi.features.engine import compute_last
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                regime = float(np.nanmedian(compute_last(win, [self.spec.strategy.exposure_feature])[self.spec.strategy.exposure_feature]))
        w = self.builder.step(bar_no, score, uni, dvol, hold, beta, regime)
        if self.trade_symbols is not None:
            w = np.where([s in self.trade_symbols for s in win.symbols], w, 0.0)

        # RISK-04 operational guards: they only ever remove the right to open/add
        guards = []
        prev_hold = hold_m[-2]
        coverage = float(np.isfinite(close[prev_hold]).mean()) if prev_hold.any() else 1.0
        self.stale_bars = self.stale_bars + 1 if coverage < self.cfg.min_coverage else 0
        if coverage < self.cfg.min_coverage:
            guards.append(f"coverage {coverage:.1%}")
        delisting = self._delisting_soon()
        black = set(getattr(self.broker, "blacklist", set())) | delisting
        if black:
            uni &= np.array([s not in black for s in win.symbols])
            w = np.where([s in delisting for s in win.symbols], 0.0, w)
        rej = self.broker.reject_rate() if hasattr(self.broker, "reject_rate") else 0.0
        if rej > self.cfg.max_reject_rate:
            guards.append(f"reject rate {rej:.0%}")
        manual_halt = (self.state_dir / "HALT").exists()
        if manual_halt:
            guards.append("HALT file")
        if guards:
            uni[:] = False
            await self.alert.send("guards", f"bar {ms_to_str(bar_open)}: no new risk ({'; '.join(guards)})")

        # RISK-06 disaster halt: flatten, then wait for a human (remove the HALT file to resume)
        E = self.port.equity
        # while halted the reference re-arms, so resuming after review does not instantly re-trigger
        self.peak_equity = E if manual_halt else max(getattr(self, "peak_equity", E), E)
        day = bar_open // 86_400_000
        if getattr(self, "_day", None) != day:
            self._day, self._day_start = day, E
        disaster = []
        if self.disaster["day_loss"] and E < self._day_start * (1 - self.disaster["day_loss"]):
            disaster.append(f"day loss {1 - E / self._day_start:.1%} > {self.disaster['day_loss']:.1%}")
        if self.disaster["max_dd"] and E < self.peak_equity * (1 - self.disaster["max_dd"]):
            disaster.append(f"drawdown {1 - E / self.peak_equity:.1%} > {self.disaster['max_dd']:.1%}")
        if disaster and not manual_halt:
            (self.state_dir / "HALT").write_text(f"halted_manual at {ms_to_str(bar_open)}: {'; '.join(disaster)}\n")
            await self.alert.send("disaster", f"DISASTER HALT ({'; '.join(disaster)}): book flattened, manual review needed", 0)

        # RISK-05 margin ratio (exchange truth): de-risk proportionally when maintenance margin gets close
        margin_scale = 1.0
        if exch_equity is not None and getattr(self.broker, "margin_ratio", None) is not None:
            mr = self.broker.margin_ratio
            if mr > self.cfg.margin_ratio_max:
                margin_scale = self.cfg.margin_ratio_max / mr
                await self.alert.send("margin", f"margin ratio {mr:.0%} > {self.cfg.margin_ratio_max:.0%}: scaling book x{margin_scale:.2f}")

        kill = (self.state_dir / "KILL").exists() or bool(disaster)
        halts_before = self.port.halt_events
        if kill:
            q_target = np.zeros(S)
        else:
            q_target = self.port.target_qty(bar_open + self.step, bar_no, w, close, adv, dvol, uni)
            if manual_halt:
                q_target = np.where(np.sign(q_target) == np.sign(self.port.qty),
                                    np.sign(self.port.qty) * np.minimum(np.abs(q_target), np.abs(self.port.qty)), 0.0)
            if margin_scale < 1.0:
                q_target = np.where(np.sign(q_target) == np.sign(self.port.qty),
                                    np.sign(self.port.qty) * np.minimum(np.abs(q_target), np.abs(self.port.qty) * margin_scale), 0.0)
        if self.port.halt_events > halts_before:
            # DEC-07: a drawdown halt in live waits for a human; remove the HALT file to resume
            (self.state_dir / "HALT").write_text(f"drawdown halt at {ms_to_str(bar_open)} equity={self.port.equity:.2f}\n")
            await self.alert.send("dd-halt", f"drawdown halt: book flattened, waiting for manual review (rm {self.state_dir}/HALT)", 0)
        q_target = np.where(np.isfinite(close), q_target, self.port.qty)   # never trade blind

        dq = q_target - self.port.qty
        orders = {win.symbols[i]: float(dq[i]) for i in np.flatnonzero(dq != 0)}
        ref = await self.data_client.ticker_price() if orders else {}
        self.boundary_qty = {s: float(a) for s, a in zip(win.symbols, self.port.qty) if a != 0}
        current = {s: float(a) for s, a in zip(win.symbols, self.port.qty) if a != 0}
        deadline = (bar_open + self.step) / 1000 + self.cfg.open_deadline_s
        fills = await self.broker.execute(orders, ref, bar_no, current, tag=self.tag, open_deadline=deadline)

        # book fills through the shared accounting (paper: authoritative; live: re-synced next bar)
        new_q = self.port.qty.copy()
        px = close.copy()
        for f in fills:
            i = sym_idx[f.symbol]
            new_q[i] += f.dq
            if math.isfinite(f.price):
                px[i] = f.price
        self.port.mark_to(px)
        self.port.rebalance(new_q, px, adv, dvol, bar_open + self.step)
        held = np.flatnonzero(self.port.qty != 0)
        await self.broker.sync_stops({win.symbols[i]: (int(np.sign(self.port.qty[i])), float(self.port.hard_stop_px[i]))
                                      for i in held if np.isfinite(self.port.hard_stop_px[i])})
        self.feed.extra = {win.symbols[i] for i in held}
        failed = [f for f in fills if f.status not in ("FILLED", "NEW", "PARTIALLY_FILLED")]
        if failed:
            await self.alert.send("fills", f"{len(failed)} orders failed: {[(f.symbol, f.status) for f in failed][:5]}")

        self.last_bar = bar_open
        self._save_state()
        dec = pd.DataFrame({"symbol": [win.symbols[i] for i in idx], "entry": uni[idx], "score": score[idx], "w": w[idx],
                            "q_target": q_target[idx]})
        for c in names:
            dec[c] = cols[c][idx]
        dec.insert(0, "bar_open", bar_open)
        day = ms_to_str(bar_open)[:10]
        p = self.state_dir / "decisions" / f"{day}.parquet"
        dec = pd.concat([pd.read_parquet(p), dec], ignore_index=True) if p.exists() else dec
        dec.to_parquet(p, index=False)
        rec = {
            "bar_open": bar_open, "bar": ms_to_str(bar_open), "mode": self.cfg.mode, "equity": self.port.equity,
            "equity_close": equity_close,
            "exchange_equity": exch_equity, "gross": float(np.abs(self.port.notional()).sum() / max(self.port.equity, 1e-9)),
            "n_universe": int(uni.sum()), "fresh_frac": fresh_frac, "coverage": coverage, "guards": guards,
            "disaster": disaster, "margin_scale": margin_scale,
            "missing": missing[:20], "kill": kill,
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

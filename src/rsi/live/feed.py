"""Live market data -> trailing Panel window, using exactly the research panel layout.

Data always comes from mainnet public endpoints (the data the model was trained
on), regardless of where orders are routed.
"""
from __future__ import annotations

import asyncio
import re

import numpy as np
import pandas as pd

from rsi.config import DataConfig
from rsi.data.panel import METRIC_FIELDS, Panel, frames_to_panel, metrics_to_bars
from rsi.live.binance import BinanceFutures, kline_rows_to_frame
from rsi.utils.log import get_logger
from rsi.utils.timeutil import interval_ms, now_ms

log = get_logger("feed")


def tradable_symbols(info: dict, cfg: DataConfig) -> list[str]:
    f = cfg.symbol_filter
    out = []
    for s in info["symbols"]:
        if s.get("contractType") != "PERPETUAL" or s.get("status") != "TRADING" or s.get("quoteAsset") != f.quote:
            continue
        if f.underlying_types and s.get("underlyingType") not in f.underlying_types:
            continue
        if s["symbol"] in f.exclude or any(re.search(rx, s["symbol"]) for rx in f.exclude_regex):
            continue
        out.append(s["symbol"])
    return sorted(out)


class LiveFeed:
    def __init__(self, data_client: BinanceFutures, cfg: DataConfig, window_bars: int, extra_symbols: list[str] | None = None,
                 only_symbols: list[str] | None = None):
        self.client = data_client
        self.cfg = cfg
        self.interval = cfg.interval
        self.step = interval_ms(cfg.interval)
        self.window_bars = window_bars
        self.extra = set(extra_symbols or [])
        self.only = only_symbols
        self.symbols: list[str] = []
        self.kl: dict[str, pd.DataFrame] = {}
        self.pr: dict[str, pd.DataFrame] = {}
        self.fu: dict[str, pd.DataFrame] = {}
        self._funding_cursor = 0
        self.exchange_info: dict = {}
        self.with_metrics = False
        self.mt: dict[str, pd.DataFrame] = {}
        self.metrics_top = 200          # positioning is fetched for the top-N by 30d ADV (superset of the hold buffer)

    # ------------------------------------------------------------------ symbols
    async def refresh_symbols(self) -> list[str]:
        self.exchange_info = await self.client.exchange_info()
        syms = tradable_symbols(self.exchange_info, self.cfg)
        if self.only:
            syms = sorted(set(self.only) | ({"BTCUSDT"} & set(syms)))
        syms = sorted(set(syms) | self.extra)
        added = set(syms) - set(self.symbols)
        self.symbols = syms
        return sorted(added)

    # ------------------------------------------------------------------ bootstrap
    async def _fetch_klines(self, sym: str, start: int) -> pd.DataFrame:
        rows: list = []
        cur = start
        while True:
            batch = await self.client.klines(sym, self.interval, limit=1000, start=cur)
            rows += batch
            if len(batch) < 1000:
                break
            cur = int(batch[-1][0]) + self.step
        return kline_rows_to_frame(rows)

    async def _fetch_premium(self, sym: str, start: int) -> pd.DataFrame:
        rows: list = []
        cur = start
        while True:
            batch = await self.client.premium_klines(sym, self.interval, limit=1000, start=cur)
            rows += batch
            if len(batch) < 1000:
                break
            cur = int(batch[-1][0]) + self.step
        if not rows:
            return pd.DataFrame(columns=["open_time", "premium_close"])
        df = pd.DataFrame(rows).iloc[:, [0, 4]]
        df.columns = ["open_time", "premium_close"]
        return df.astype({"open_time": "int64", "premium_close": "float64"})

    async def _fetch_funding_all(self, start: int) -> None:
        """All-symbol funding prints since `start` (paginated by time, few requests)."""
        cur = start
        while True:
            batch = await self.client.funding_history(symbol=None, limit=1000, start=cur)
            for r in batch:
                self.fu.setdefault(r["symbol"], []).append(
                    {"funding_time": int(r["fundingTime"]), "funding_rate": float(r["fundingRate"])})
            if len(batch) < 1000:
                break
            cur = max(int(r["fundingTime"]) for r in batch) + 1
        self._funding_cursor = max(self._funding_cursor, cur)

    def _funding_frames(self, intervals: dict[str, float]) -> dict[str, pd.DataFrame]:
        """Settlement interval of each print = hours since the previous print (as in the archives)."""
        out = {}
        for s, v in self.fu.items():
            df = pd.DataFrame(v).drop_duplicates("funding_time").sort_values("funding_time").reset_index(drop=True)
            bar = df["funding_time"] - df["funding_time"] % 3_600_000
            ih = (bar.diff() / 3_600_000).round()
            df["interval_hours"] = ih.fillna(intervals.get(s, 8.0)).clip(lower=1)
            out[s] = df
        return out

    async def bootstrap(self) -> None:
        await self.refresh_symbols()
        end = now_ms() - now_ms() % self.step
        start = end - (self.window_bars + 2) * self.step
        sem = asyncio.Semaphore(8)

        async def one(sym):
            async with sem:
                try:
                    self.kl[sym] = await self._fetch_klines(sym, start)
                    self.pr[sym] = await self._fetch_premium(sym, start)
                except Exception as e:  # noqa: BLE001
                    log.warning(f"bootstrap {sym} failed: {e}")
        await asyncio.gather(*(one(s) for s in self.symbols))
        await self._fetch_funding_all(start - 2 * 86_400_000)
        log.info(f"bootstrap: {len(self.kl)} symbols, window {self.window_bars} bars")

    # ------------------------------------------------------------------ incremental
    async def update(self, bar_open: int, retries: int = 4) -> list[str]:
        """Make sure every symbol has the closed bar `bar_open`. Returns symbols still missing it."""
        added = await self.refresh_symbols() if bar_open // self.step % 48 == 0 else []
        start = bar_open - (self.window_bars + 2) * self.step
        for s in added:
            self.kl[s] = await self._fetch_klines(s, start)
            self.pr[s] = await self._fetch_premium(s, start)
        missing = list(self.symbols)
        for attempt in range(retries):
            sem = asyncio.Semaphore(16)

            async def one(sym):
                async with sem:
                    try:
                        k = kline_rows_to_frame(await self.client.klines(sym, self.interval, limit=3))
                        p = await self.client.premium_klines(sym, self.interval, limit=3)
                    except Exception as e:  # noqa: BLE001
                        log.debug(f"update {sym}: {e}")
                        return
                    k = k[k["open_time"] + self.step <= now_ms()]
                    self.kl[sym] = _upsert(self.kl.get(sym), k, "open_time")
                    if p:
                        pdf = pd.DataFrame(p).iloc[:, [0, 4]]
                        pdf.columns = ["open_time", "premium_close"]
                        pdf = pdf.astype({"open_time": "int64", "premium_close": "float64"})
                        pdf = pdf[pdf["open_time"] + self.step <= now_ms()]
                        self.pr[sym] = _upsert(self.pr.get(sym), pdf, "open_time")
            await asyncio.gather(*(one(s) for s in missing))
            missing = [s for s in self.symbols if s not in self.kl or not (self.kl[s]["open_time"] == bar_open).any()]
            missing = [s for s in missing if s in self.kl and len(self.kl[s]) and int(self.kl[s]["open_time"].iloc[-1]) < bar_open]
            if not missing:
                break
            await asyncio.sleep(3 + 2 * attempt)
        await self._fetch_funding_all(self._funding_cursor or bar_open - 86_400_000)
        self._trim()
        return missing

    def _trim(self) -> None:
        keep = self.window_bars + 50
        for d in (self.kl, self.pr):
            for s, df in d.items():
                if df is not None and len(df) > keep:
                    d[s] = df.iloc[-keep:].reset_index(drop=True)

    def _metric_symbols(self) -> list[str]:
        adv = {s: float(df["quote_volume"].iloc[-1440:].mean()) for s, df in self.kl.items() if df is not None and len(df)}
        return sorted(adv, key=adv.get, reverse=True)[: self.metrics_top]

    async def update_metrics(self, bar_open: int, history_bars: int) -> None:
        """Backfill / refresh 5m positioning snapshots for the most liquid symbols."""
        if not self.with_metrics:
            return
        sem = asyncio.Semaphore(4)
        need = set(self._metric_symbols())

        async def one(sym):
            async with sem:
                old = self.mt.get(sym)
                start = (bar_open - history_bars * self.step) if old is None or old.empty else int(old["ts"].iloc[-1]) + 1
                frames = []
                cur = start
                try:
                    while cur <= bar_open + self.step:
                        df = await self.client.positioning(sym, start=cur, limit=500)
                        if df.empty:
                            break
                        frames.append(df)
                        if len(df) < 500:
                            break
                        cur = int(df["ts"].iloc[-1]) + 1
                except Exception as e:  # noqa: BLE001
                    log.debug(f"positioning {sym}: {e}")
                new = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["ts", *METRIC_FIELDS])
                self.mt[sym] = _upsert(old, new, "ts").iloc[-(history_bars * 6 + 200):]
        await asyncio.gather(*(one(s) for s in need))

    def window(self, last_bar_open: int, intervals: dict[str, float] | None = None) -> Panel:
        times = np.arange(last_bar_open - (self.window_bars - 1) * self.step, last_bar_open + 1, self.step, dtype=np.int64)
        fr = self._funding_frames(intervals or {})
        p = frames_to_panel(times, self.symbols, self.kl, self.pr, fr, self.interval)
        if self.with_metrics:
            p.data.update(metrics_to_bars(times, self.mt, self.symbols, self.interval))
        return p


def _upsert(old: pd.DataFrame | None, new: pd.DataFrame, key: str) -> pd.DataFrame:
    if old is None or old.empty:
        return new.reset_index(drop=True)
    if new.empty:
        return old
    df = pd.concat([old[~old[key].isin(new[key])], new], ignore_index=True)
    return df.sort_values(key).reset_index(drop=True)

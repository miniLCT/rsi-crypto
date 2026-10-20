"""Time conventions.

Everything internal is UTC epoch milliseconds. A bar is identified by its
*open* time; its data becomes known at ``open_time + interval``. A decision
made on bar ``t`` is executed at the open of bar ``t+1``.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd

INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def interval_ms(interval: str) -> int:
    return INTERVAL_MS[interval]


def now_ms() -> int:
    return int(time.time() * 1000)


def to_ms(ts: str | datetime | pd.Timestamp) -> int:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return int(t.value // 1_000_000)


def ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def ms_to_str(ms: int) -> str:
    return ms_to_dt(ms).strftime("%Y-%m-%d %H:%M")


def floor_ms(ms: int, interval: str) -> int:
    step = interval_ms(interval)
    return ms - ms % step

"""Append-only experiment ledger (SQLite). Every trial counts toward multiple-testing deflation."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from rsi.config import runs_dir

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, created REAL, name TEXT, parent TEXT, harness_version INTEGER, harness_hash TEXT,
  spec_hash TEXT, spec_json TEXT, git_sha TEXT, data_fp TEXT, status TEXT,
  sharpe REAL, dsr REAL, max_dd REAL, daily_sr REAL, metrics_json TEXT, hypothesis TEXT, note TEXT
);
CREATE TABLE IF NOT EXISTS holdout_looks (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, created REAL, passed INTEGER, metrics_json TEXT
);
CREATE TABLE IF NOT EXISTS champion (
  harness_version INTEGER, run_id TEXT, since REAL, stage TEXT
);
"""


class Ledger:
    def __init__(self, path: Path | None = None):
        self.path = path or runs_dir() / "ledger.sqlite"
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=60)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ runs
    def add_run(self, row: dict) -> None:
        cols = ["run_id", "created", "name", "parent", "harness_version", "harness_hash", "spec_hash", "spec_json", "git_sha", "data_fp",
                "status", "sharpe", "dsr", "max_dd", "daily_sr", "metrics_json", "hypothesis", "note"]
        row = {**{c: None for c in cols}, **row, "created": row.get("created", time.time())}
        with self._conn() as c:
            c.execute(f"INSERT OR REPLACE INTO runs ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                      [row[k] for k in cols])

    def get(self, run_id: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM runs WHERE run_id LIKE ?", (run_id + "%",)).fetchone()
        return dict(r) if r else None

    def runs(self, harness_version: int | None = None, limit: int | None = None, order: str = "created DESC") -> list[dict]:
        q = "SELECT * FROM runs"
        args: list = []
        if harness_version is not None:
            q += " WHERE harness_version = ?"
            args.append(harness_version)
        q += f" ORDER BY {order}"
        if limit:
            q += f" LIMIT {int(limit)}"
        with self._conn() as c:
            return [dict(r) for r in c.execute(q, args).fetchall()]

    def trial_stats(self, harness_version: int) -> tuple[int, float | None]:
        """(number of distinct trials, variance of their daily Sharpe) for DSR."""
        rows = self.runs(harness_version)
        seen, srs = set(), []
        for r in rows:
            # robustness re-evaluations perturb a chosen candidate; they are not selection trials
            if r["spec_hash"] in seen or r["daily_sr"] is None or r["status"] == "robustness":
                continue
            seen.add(r["spec_hash"])
            srs.append(r["daily_sr"])
        if len(srs) < 5:
            return len(seen), None
        # robust spread: a few broken trials (e.g. cost-blown) must not inflate the null's max-SR benchmark
        a = np.asarray(srs)
        mad = 1.4826 * np.median(np.abs(a - np.median(a)))
        return len(seen), float(mad ** 2)

    def set_status(self, run_id: str, status: str, note: str | None = None) -> None:
        with self._conn() as c:
            c.execute("UPDATE runs SET status = ?, note = COALESCE(?, note) WHERE run_id = ?", (status, note, run_id))

    def check_harness(self, version: int, content_hash: str) -> None:
        """CONV-04: the evaluation contract may not change without a version bump."""
        with self._conn() as c:
            rows = c.execute("SELECT DISTINCT harness_hash FROM runs WHERE harness_version = ?", (version,)).fetchall()
        other = {r[0] for r in rows if r[0] and r[0] != content_hash}
        if other:
            raise RuntimeError(f"configs/harness.yaml changed but version is still {version} "
                               f"(ledger has {sorted(other)}, now {content_hash}); bump `version`")

    # ------------------------------------------------------------------ champion
    def champion(self, harness_version: int, stage: str = "research") -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT run_id FROM champion WHERE harness_version = ? AND stage = ? ORDER BY since DESC LIMIT 1",
                          (harness_version, stage)).fetchone()
        return self.get(r["run_id"]) if r else None

    def set_champion(self, harness_version: int, run_id: str, stage: str = "research") -> None:
        with self._conn() as c:
            c.execute("INSERT INTO champion VALUES (?, ?, ?, ?)", (harness_version, run_id, time.time(), stage))

    # ------------------------------------------------------------------ holdout
    def holdout_looks(self, since_s: float) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM holdout_looks WHERE created >= ?", (since_s,)).fetchone()[0]

    def add_holdout_look(self, run_id: str, passed: bool, metrics: dict) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO holdout_looks (run_id, created, passed, metrics_json) VALUES (?, ?, ?, ?)",
                      (run_id, time.time(), int(passed), json.dumps(metrics)))

    def holdout_history(self, run_id: str | None = None) -> list[dict]:
        q, a = "SELECT * FROM holdout_looks", []
        if run_id:
            q += " WHERE run_id = ?"
            a.append(run_id)
        with self._conn() as c:
            return [dict(r) for r in c.execute(q + " ORDER BY created", a).fetchall()]

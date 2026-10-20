"""Walk-forward folds with purge + embargo, and the holdout lock.

Timeline (harness.yaml):
  [research_start ............................ research_end)[holdout_start ... holdout_end)
               fold k: [train .......](purge+embargo)[test k]
Research runs never receive panel rows at/after research_end. The holdout is
only reachable through `rsi.harness.holdout`, which logs and rate-limits looks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rsi.config import HarnessConfig
from rsi.utils.timeutil import to_ms


@dataclass(frozen=True)
class Fold:
    k: int
    train: tuple[int, int]  # [i0, i1) panel row indices
    test: tuple[int, int]

    def as_dict(self, times: np.ndarray) -> dict:
        def s(i):
            return str(pd.Timestamp(int(times[min(i, len(times) - 1)]), unit="ms"))
        return {"k": self.k, "train": [s(self.train[0]), s(self.train[1])], "test": [s(self.test[0]), s(self.test[1])]}


def walk_forward(times: np.ndarray, hcfg: HarnessConfig, label_span: int, end: str | None = None,
                 first_test: str | None = None, test_months: int | None = None) -> list[Fold]:
    wf = hcfg.walk_forward
    start_ms = to_ms(hcfg.periods.research_start)
    end_ms = to_ms(end or hcfg.periods.research_end)
    t = pd.Timestamp(first_test or wf.first_test, tz="UTC")
    step = pd.DateOffset(months=test_months or wf.test_months)
    folds: list[Fold] = []
    gap = label_span + wf.embargo_bars
    i_start = int(np.searchsorted(times, start_ms))
    k = 0
    while to_ms(t) < end_ms:
        t_end = min(to_ms(t + step), end_ms)
        j0 = int(np.searchsorted(times, to_ms(t)))
        j1 = int(np.searchsorted(times, t_end))
        tr1 = j0 - gap
        if wf.train_months:
            tr0 = max(i_start, int(np.searchsorted(times, to_ms(t - pd.DateOffset(months=wf.train_months)))))
        else:
            tr0 = i_start
        if tr1 > tr0 and j1 > j0:
            folds.append(Fold(k, (tr0, tr1), (j0, j1)))
            k += 1
        t = t + step
    return folds

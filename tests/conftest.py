import numpy as np
import pytest

from rsi.data.panel import PANEL_FIELDS, Panel


def make_panel(T: int = 3000, S: int = 12, seed: int = 0, start_ms: int = 1_600_000_000_000 - 1_600_000_000_000 % 86_400_000) -> Panel:
    rng = np.random.default_rng(seed)
    symbols = ["BTCUSDT"] + [f"C{i:02d}USDT" for i in range(S - 1)]
    times = start_ms + np.arange(T, dtype=np.int64) * 1_800_000
    mkt = rng.normal(0, 0.004, T)
    beta = rng.uniform(0.5, 1.5, S); beta[0] = 1.0
    r = mkt[:, None] * beta[None, :] + rng.normal(0, 0.006, (T, S))
    close = 100 * np.exp(np.cumsum(r, axis=0)) * rng.uniform(0.1, 10, S)[None, :]
    open_ = np.vstack([close[:1], close[:-1]])
    wig = np.abs(rng.normal(0, 0.003, (T, S)))
    high = np.maximum(open_, close) * (1 + wig)
    low = np.minimum(open_, close) * (1 - wig)
    qv = np.exp(rng.normal(13, 1, (T, S))) * rng.uniform(0.2, 5, S)[None, :]
    vol = qv / close
    tbq = qv * rng.uniform(0.3, 0.7, (T, S))
    trades = np.round(qv / rng.uniform(500, 3000, S)[None, :])
    prem = rng.normal(0, 3e-4, (T, S))
    fr = np.full((T, S), np.nan)
    fr[::16] = rng.normal(1e-4, 2e-4, (len(range(0, T, 16)), S))
    fih = np.where(np.isfinite(fr), 8.0, np.nan)
    data = dict(open=open_, high=high, low=low, close=close, volume=vol, quote_volume=qv, trades=trades,
                taker_buy_quote_volume=tbq, premium=prem, funding_rate=fr, funding_interval_h=fih)
    # late listing and a delisting
    for f in data:
        data[f][: T // 3, 3] = np.nan
        data[f][2 * T // 3:, 4] = np.nan
    assert set(data) == set(PANEL_FIELDS)
    return Panel(times, symbols, data, "30m", {"fingerprint": f"synthetic-{seed}-{T}-{S}"})


@pytest.fixture(scope="session")
def panel():
    return make_panel()


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path_factory, monkeypatch):
    root = tmp_path_factory.getbasetemp() / "rsi_root"
    (root / "configs").mkdir(parents=True, exist_ok=True)
    import shutil
    from rsi.config import repo_root
    real = repo_root()
    for f in ("harness.yaml", "data.yaml"):
        src = real / "configs" / f
        if src.exists() and not (root / "configs" / f).exists():
            shutil.copy(src, root / "configs" / f)
    monkeypatch.setenv("RSI_ROOT", str(root))

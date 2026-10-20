from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


def repo_root() -> Path:
    env = os.environ.get("RSI_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists() and (p / "configs").is_dir():
            return p
    return Path.cwd()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


class SymbolFilter(BaseModel):
    quote: str = "USDT"
    exclude_regex: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    underlying_types: list[str] = Field(default_factory=lambda: ["COIN"])


class DownloadCfg(BaseModel):
    concurrency: int = 32
    verify_checksum: bool = True
    retries: int = 4


class DataConfig(BaseModel):
    root: str = "data"
    market: str = "um"
    interval: str = "30m"
    start_month: str = "2020-01"
    datasets: list[str] = Field(default_factory=lambda: ["klines", "fundingRate", "premiumIndexKlines"])
    symbol_filter: SymbolFilter = Field(default_factory=SymbolFilter)
    download: DownloadCfg = Field(default_factory=DownloadCfg)

    @property
    def root_path(self) -> Path:
        p = Path(self.root)
        return p if p.is_absolute() else repo_root() / p

    @property
    def raw_dir(self) -> Path:
        return self.root_path / "raw" / self.market

    @property
    def lake_dir(self) -> Path:
        return self.root_path / "lake" / self.market


class Periods(BaseModel):
    research_start: str
    research_end: str
    holdout_start: str
    holdout_end: str


class WalkForwardCfg(BaseModel):
    first_test: str
    test_months: int = 3
    train_months: int | None = 24
    embargo_bars: int = 48


class HoldoutCfg(BaseModel):
    max_looks_per_30d: int = 3


class CostCfg(BaseModel):
    taker_fee_bps: float = 5.0
    spread_coef_bps: float = 10.0
    min_half_spread_bps: float = 0.5
    max_half_spread_bps: float = 30.0
    impact_eta: float = 0.5
    delist_penalty_bps: float = 200.0


class CapitalCfg(BaseModel):
    initial_usd: float = 10000.0


class GateCfg(BaseModel):
    min_sharpe: float = 1.0
    min_dsr: float = 0.9
    max_drawdown: float = 0.3
    min_positive_fold_frac: float = 0.6
    min_trades: int = 200
    champion_sharpe_margin: float = 0.15
    holdout_min_sharpe_ratio: float = 0.4
    holdout_min_sharpe: float = 0.3


class HarnessConfig(BaseModel):
    version: int
    bar: str = "30m"
    bars_per_year: int = 17520
    periods: Periods
    walk_forward: WalkForwardCfg
    holdout: HoldoutCfg = Field(default_factory=HoldoutCfg)
    costs: CostCfg = Field(default_factory=CostCfg)
    capital: CapitalCfg = Field(default_factory=CapitalCfg)
    gates: GateCfg = Field(default_factory=GateCfg)


@lru_cache
def data_config(path: str | None = None) -> DataConfig:
    return DataConfig(**load_yaml(path or repo_root() / "configs" / "data.yaml"))


@lru_cache
def harness_config(path: str | None = None) -> HarnessConfig:
    return HarnessConfig(**load_yaml(path or repo_root() / "configs" / "harness.yaml"))


def runs_dir() -> Path:
    p = repo_root() / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir() -> Path:
    p = repo_root() / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p

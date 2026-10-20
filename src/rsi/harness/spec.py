"""Experiment spec: the only thing an RSI iteration changes (besides new feature code)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from rsi.engine.portfolio import RiskSpec
from rsi.features.featureset import FeatureSpec
from rsi.labels import LabelSpec
from rsi.models.core import ModelSpec
from rsi.strategy import StrategySpec
from rsi.universe import UniverseSpec
from rsi.utils.hashing import stable_hash


class ExperimentSpec(BaseModel):
    name: str
    hypothesis: str = ""
    parent: str | None = None
    universe: UniverseSpec = Field(default_factory=UniverseSpec)
    features: FeatureSpec = Field(default_factory=FeatureSpec)
    label: LabelSpec = Field(default_factory=LabelSpec)
    model: ModelSpec = Field(default_factory=ModelSpec)
    strategy: StrategySpec = Field(default_factory=StrategySpec)
    risk: RiskSpec = Field(default_factory=RiskSpec)
    panel_top_k: int = 150
    # optional overrides of the walk-forward cadence (cheaper iterations); gates still apply
    test_months: int | None = None
    search_space: dict[str, list[Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _short_needs_stop(self):
        if self.strategy.allow_short and self.risk.hard_stop_vol is None:
            raise ValueError("DEC-09: strategies that can short must set risk.hard_stop_vol")
        return self

    def content(self) -> dict:
        d = self.model_dump(exclude={"name", "hypothesis", "parent", "search_space"})
        return d

    @property
    def spec_hash(self) -> str:
        return stable_hash(self.content())

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentSpec":
        with open(path) as f:
            return cls(**yaml.safe_load(f))

    def dump(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.model_dump(), f, sort_keys=False, allow_unicode=True)

    def with_overrides(self, overrides: dict[str, Any], name: str | None = None) -> "ExperimentSpec":
        d = copy.deepcopy(self.model_dump())
        for dotted, v in overrides.items():
            cur = d
            keys = dotted.split(".")
            for k in keys[:-1]:
                cur = cur.setdefault(k, {})
            cur[keys[-1]] = v
        d["name"] = name or d["name"]
        return ExperimentSpec(**d)

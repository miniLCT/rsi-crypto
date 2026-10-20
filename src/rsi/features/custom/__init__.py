"""Agent-authored features. Every module in this package is imported on registry load.

Add a new file (e.g. `f_0001_basis_reversal.py`) that uses `rsi.features.registry.register`.
The harness runs the causality + lookback contract on every new/changed feature
before it can be used in an experiment.
"""
import importlib
import pkgutil

for _m in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_m.name}")

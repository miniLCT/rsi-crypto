from __future__ import annotations

import logging
import sys

_FMT = "%(asctime)s %(levelname).1s %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"rsi.{name}")
    root = logging.getLogger("rsi")
    if not root.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter(_FMT, "%Y-%m-%d %H:%M:%S"))
        root.addHandler(h)
        root.setLevel(logging.INFO)
    return logger


def add_file_handler(path: str) -> None:
    root = logging.getLogger("rsi")
    h = logging.FileHandler(path)
    h.setFormatter(logging.Formatter(_FMT, "%Y-%m-%d %H:%M:%S"))
    root.addHandler(h)

from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any


def stable_hash(obj: Any, n: int = 12) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:n]


def source_hash(fn: Any, n: int = 12) -> str:
    """Hash of a function's source; changes whenever its implementation changes."""
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        src = repr(fn)
    return hashlib.sha256(src.encode()).hexdigest()[:n]

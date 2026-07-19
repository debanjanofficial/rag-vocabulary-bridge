"""Load terminology concepts and flat colloquial→formal map."""

from __future__ import annotations

import json

from config import TERMINOLOGY

_cache: dict | None = None


def load_terminology() -> dict:
    """Return {concepts: [...], flat_map: {...}}."""
    global _cache
    if _cache is None:
        with open(TERMINOLOGY, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            _cache = {"concepts": data, "flat_map": {}}
        else:
            _cache = {
                "concepts": data.get("concepts", []),
                "flat_map": data.get("flat_map", {}),
            }
    return _cache


def load_flat_map() -> dict[str, str]:
    return load_terminology()["flat_map"]

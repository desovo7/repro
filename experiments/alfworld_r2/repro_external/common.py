from __future__ import annotations

import json
from pathlib import Path


def value(config, path, default=None):
    item = config
    for key in path.split("."):
        try:
            item = item[key] if isinstance(item, dict) else getattr(item, key)
        except (KeyError, AttributeError, TypeError):
            return default
    return item


def json_scalar(item):
    import numpy as np
    if isinstance(item, np.generic) and item.dtype.kind in "biufU":
        result = item.item()
        if type(result) in (bool, int, float, str):
            return result
    raise TypeError(f"Unsupported JSON value: {type(item).__name__}")


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2,
        allow_nan=False, default=json_scalar) + "\n", encoding="utf-8")
    temporary.replace(path)


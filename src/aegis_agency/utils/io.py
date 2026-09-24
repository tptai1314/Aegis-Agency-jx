"""IO helpers: YAML config loading, JSON/CSV writing with provenance headers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml  # type: ignore[import-untyped]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {p} must parse to a mapping; got {type(data)}.")
    return data


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) as a directory if needed; return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, obj: Any) -> None:
    """Write ``obj`` to JSON, creating parent directories."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=_default)


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a list of dict rows to CSV. No-op-safe on empty input (writes header only)."""
    p = Path(path)
    ensure_dir(p.parent)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with p.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _default(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

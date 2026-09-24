"""Input validation helpers with clear error messages.

Used across the package to enforce the domains the paper's objects require (scores in
[0, 1], probabilities, committee sizes, Byzantine fractions).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def check_unit_interval(value: float, name: str) -> float:
    """Validate that ``value`` lies in [0, 1]."""
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]; got {value!r}.")
    return float(value)


def check_scores(scores: Iterable[float], name: str = "scores") -> np.ndarray:
    """Validate an array of block-scores lies in [0, 1] and is finite."""
    arr = np.asarray(list(scores), dtype=float)
    if arr.size and (not np.all(np.isfinite(arr)) or arr.min() < 0.0 or arr.max() > 1.0):
        raise ValueError(f"All {name} must be finite and in [0, 1].")
    return arr


def check_committee(n: int, f: int) -> None:
    """Validate committee size ``n`` and assumed Byzantine count ``f``."""
    if n < 1:
        raise ValueError(f"Committee size n must be >= 1; got {n}.")
    if f < 0:
        raise ValueError(f"Byzantine count f must be >= 0; got {f}.")
    if f >= n:
        raise ValueError(f"Byzantine count f={f} must be < n={n} (need an honest majority).")


def check_verdict_matrix(u: np.ndarray) -> np.ndarray:
    """Validate a stacked verdict-vector matrix of shape (n, d)."""
    arr = np.asarray(u, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Verdict matrix must be 2-D (n, d); got shape {arr.shape}.")
    if arr.shape[0] < 1:
        raise ValueError("Verdict matrix must contain at least one verdict.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Verdict matrix contains non-finite values.")
    # Score coordinate (column 0) must be a valid block-score.
    check_scores(arr[:, 0], name="verdict score coordinate")
    return arr

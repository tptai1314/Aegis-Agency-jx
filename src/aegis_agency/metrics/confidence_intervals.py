"""Confidence intervals for reported rates (Section 10 reproducibility checklist).

Wilson interval for binomial proportions; bootstrap interval for arbitrary statistics. Used
so that reported synthetic-demo rates carry dispersion rather than bare point values.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (default 95%)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


def bootstrap_interval(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``statistic`` over a 1-D sample."""
    x = np.asarray(data, dtype=float).ravel()
    if x.size == 0:
        return (float("nan"), float("nan"))
    if rng is None:
        rng = np.random.default_rng(0)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        sample = x[rng.integers(0, x.size, size=x.size)]
        boots[b] = statistic(sample)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return (lo, hi)

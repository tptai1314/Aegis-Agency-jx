"""Deterministic randomness control for reproducibility.

All stochastic code in this package draws from an explicit :class:`numpy.random.Generator`
obtained via :func:`get_rng`; nothing relies on global NumPy state except when the user
explicitly calls :func:`set_global_seed`.
"""

from __future__ import annotations

import os
import random

import numpy as np


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Return a fresh, independent NumPy ``Generator`` seeded by ``seed``.

    Passing the same integer seed always yields the same stream, which is what the
    reproducibility tests rely on.
    """
    return np.random.default_rng(seed)


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy global, and ``PYTHONHASHSEED`` for whole-process determinism.

    Prefer :func:`get_rng` for library code; this helper exists for scripts that must also
    tame third-party libraries relying on global state.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

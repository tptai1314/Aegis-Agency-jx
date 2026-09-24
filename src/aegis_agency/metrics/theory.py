"""Theoretical checks implementing the paper's closed-form results (Section 8).

These functions let the empirical synthetic runs be compared against the theorems:
Lemma 1 (displacement), Theorem 1 (integrity condition), Proposition 2 (injection flip
bound), Proposition 3 (correlated-failure variance).
"""

from __future__ import annotations

import math

import numpy as np


def c_alpha(f: int, n: int) -> float:
    """Displacement constant C_alpha = 2(n-f)/(n-2f) = 2(1-alpha)/(1-2alpha)  -- Eq. (6).

    Defined only for f < n/2 (median regime); raises for f >= n/2, matching the theorem's
    domain (the constant diverges as alpha -> 1/2).
    """
    if n <= 0:
        raise ValueError("n must be > 0.")
    if f < 0:
        raise ValueError("f must be >= 0.")
    if n - 2 * f <= 0:
        raise ValueError(
            f"C_alpha is defined only for f < n/2 (n={n}, f={f}); the geometric-median "
            "displacement bound does not hold beyond this."
        )
    return 2.0 * (n - f) / (n - 2 * f)


def displacement_holds(
    u_hat: np.ndarray,
    u_star: np.ndarray,
    r: float,
    f: int,
    n: int,
    tol: float = 1e-6,
) -> bool:
    """Check the Lemma 1 bound || u_hat - u* || <= C_alpha r (with tolerance).

    The tolerance accommodates the Weiszfeld approximation to the exact geometric median
    (see math_to_code_audit.md); this does not assert the exact-median inequality.
    """
    ca = c_alpha(f, n)
    return bool(np.linalg.norm(np.asarray(u_hat) - np.asarray(u_star)) <= ca * r + tol)


def integrity_condition(gamma: float, r: float, f: int, n: int) -> bool:
    """Theorem 1 sufficient condition: gamma > C_alpha r  -- Eq. (7).

    When True, up to f Byzantine judges cannot flip the (median-rule) aggregate decision.
    """
    return bool(gamma > c_alpha(f, n) * r)


def correlated_variance(mu: float, rho: float, n_honest: int) -> float:
    """Var(mean of honest correctness) = mu(1-mu)[(1-rho)/N + rho]  -- Eq. (8), Prop. 3."""
    if not (0.0 <= mu <= 1.0):
        raise ValueError("mu must be in [0, 1].")
    if not (0.0 <= rho <= 1.0):
        raise ValueError("rho must be in [0, 1].")
    if n_honest < 1:
        raise ValueError("n_honest (N) must be >= 1.")
    return mu * (1.0 - mu) * ((1.0 - rho) / n_honest + rho)


def correlated_variance_floor(mu: float, rho: float) -> float:
    """Asymptotic variance floor mu(1-mu) rho as N -> infinity (Prop. 3)."""
    return mu * (1.0 - mu) * rho


def injection_flip_bound(n_honest: int, epsilon: float, t: int) -> float:
    """Prop. 2 union bound: P(> t honest flips) <= C(N, t+1) epsilon^(t+1) (independence).

    Returns min(1, bound). Valid only under independent honest responses (documented in
    Prop. 2); for correlated responses see Prop. 3.
    """
    if not (0.0 <= epsilon <= 1.0):
        raise ValueError("epsilon must be in [0, 1].")
    if t < 0:
        raise ValueError("t must be >= 0.")
    if t + 1 > n_honest:
        return 0.0
    bound = math.comb(n_honest, t + 1) * (epsilon ** (t + 1))
    return float(min(1.0, bound))

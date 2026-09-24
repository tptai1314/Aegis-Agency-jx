"""Theoretical-check tests (Lemma 1, Theorem 1, Proposition 2-3)."""

import math

import numpy as np
import pytest

from aegis_agency.metrics.theory import (
    c_alpha,
    correlated_variance,
    injection_flip_bound,
    integrity_condition,
)


def test_c_alpha_known_values():
    # C_0 = 2, C_{1/4} = 3, C_{1/3} = 4  (Eq. 6).
    assert c_alpha(0, 10) == pytest.approx(2.0)
    assert c_alpha(1, 4) == pytest.approx(3.0)      # alpha = 1/4
    assert c_alpha(2, 6) == pytest.approx(4.0)      # alpha = 1/3


def test_c_alpha_domain():
    with pytest.raises(ValueError):
        c_alpha(3, 6)  # f = n/2 -> undefined


def test_integrity_condition():
    # gamma > C_alpha r  (Eq. 7).
    assert integrity_condition(gamma=0.4, r=0.05, f=1, n=10) is True   # 0.4 > ~0.22
    assert integrity_condition(gamma=0.1, r=0.2, f=1, n=10) is False


def test_correlated_variance_formula_matches_montecarlo():
    """Var(mean of exchangeable Bernoulli) matches Eq. (8) empirically."""
    mu, rho, N = 0.6, 0.4, 6
    theo = correlated_variance(mu, rho, N)

    rng = np.random.default_rng(0)
    means = []
    # Construction: Z_i = C if M_i else Y_i, with M_i ~ Bernoulli(sqrt(rho)), C, Y_i ~
    # Bernoulli(mu). Then Corr(Z_i, Z_j) = (sqrt(rho))^2 = rho exactly (see math_to_code_audit).
    p_copy = math.sqrt(rho)
    for _ in range(60000):
        shared = rng.random() < mu  # common component C
        z = []
        for _ in range(N):
            if rng.random() < p_copy:
                z.append(int(shared))
            else:
                z.append(int(rng.random() < mu))
        means.append(np.mean(z))
    emp = float(np.var(means))
    assert emp == pytest.approx(theo, abs=0.01)


def test_correlated_variance_floor_positive_for_rho():
    # As N -> inf the variance does not vanish for rho > 0.
    big_n = correlated_variance(0.5, 0.3, 100000)
    assert big_n > 0.5 * 0.5 * 0.3 * 0.9  # near the floor mu(1-mu)rho


def test_injection_flip_bound():
    # P(>0 flips) <= C(N,1) eps = N eps ; matches Prop. 2 union bound.
    assert injection_flip_bound(5, 0.1, 0) == pytest.approx(min(1.0, 5 * 0.1))
    assert injection_flip_bound(5, 0.1, 1) == pytest.approx(min(1.0, math.comb(5, 2) * 0.01))

"""Aggregator correctness on controlled examples (Algorithm 2)."""

import numpy as np
import pytest

from aegis_agency.methods.aggregators import (
    coordinate_median,
    geometric_median,
    krum,
    majority_vote,
)


# NB: column 0 is the block-score (validly constrained to [0, 1]); extreme Byzantine values
# live in the embedding columns (>= 1), which is where a manipulated rationale would be.
def test_coordinate_median_ignores_minority_outliers():
    # 3 honest near (score 0.2, emb 0.0), 2 Byzantine extreme in score AND embedding.
    u = np.array(
        [[0.2, 0.0], [0.22, 0.01], [0.18, -0.01], [1.0, 10.0], [0.0, -10.0]]
    )
    out = coordinate_median(u)
    assert abs(out[0] - 0.2) < 0.05
    assert abs(out[1] - 0.0) < 0.05


def test_geometric_median_robust_to_single_outlier():
    honest = np.tile(np.array([0.3, 0.3]), (5, 1))
    u = np.vstack([honest, np.array([[1.0, 100.0]])])  # score in [0,1], embedding extreme
    out = geometric_median(u)
    assert np.linalg.norm(out - np.array([0.3, 0.3])) < 0.15


def test_geometric_median_handles_coincident_point():
    # Query coinciding with a data point must not divide by zero (eta floor).
    u = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    out = geometric_median(u)
    assert np.allclose(out, [0.5, 0.5], atol=1e-6)


def test_krum_selects_honest_cluster_member():
    honest = np.array([[0.4, 0.4], [0.41, 0.39], [0.39, 0.41], [0.40, 0.40]])
    byz = np.array([[1.0, 5.0], [0.0, -5.0]])  # score in [0,1], embedding extreme
    u = np.vstack([honest, byz])  # n=6, f=2 -> n-f-2 = 2 neighbours
    out = krum(u, f=2)
    # Selected verdict must be from the honest cluster.
    assert np.linalg.norm(out - np.array([0.4, 0.4])) < 0.1


def test_krum_requires_enough_judges():
    u = np.array([[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]])  # n=3
    with pytest.raises(ValueError):
        krum(u, f=2)  # n - f - 2 = -1 < 1


def test_majority_vote_basic_and_ties():
    assert majority_vote(np.array([1, 1, 0])) == 1
    assert majority_vote(np.array([0, 0, 1])) == 0
    # Tie (n even): paper strict '>' -> allow (0).
    assert majority_vote(np.array([1, 0]), tie_breaking="paper_strict") == 0
    assert majority_vote(np.array([1, 0]), tie_breaking="block_on_tie") == 1


def test_majority_vote_rejects_non_binary():
    with pytest.raises(ValueError):
        majority_vote(np.array([0, 2, 1]))

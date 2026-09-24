"""Synthetic data and judge-population tests."""

import numpy as np

from aegis_agency.data.schemas import Decision
from aegis_agency.data.synthetic import generate_payloads
from aegis_agency.judges.synthetic_judges import SyntheticJudgePopulation


def test_generate_payloads_balanced():
    rng = np.random.default_rng(0)
    payloads = generate_payloads(500, rng, unsafe_fraction=0.5)
    assert len(payloads) == 500
    n_unsafe = sum(p.true_label == Decision.BLOCK for p in payloads)
    assert 200 < n_unsafe < 300  # roughly balanced


def test_honest_committee_respects_margin():
    """Honest judges should mostly decide correctly given a positive margin."""
    rng = np.random.default_rng(1)
    pop = SyntheticJudgePopulation(n_judges=5, margin=0.3, radius=0.05, embedding_dim=3)
    payloads = generate_payloads(200, rng)
    correct = total = 0
    for p in payloads:
        for v in pop.generate_honest(p, rng):
            total += 1
            correct += int(v.decision == p.true_label)
    assert correct / total > 0.9  # tight radius + wide margin -> high honest accuracy


def test_correlation_increases_variance_of_committee_mean():
    """Higher rho should increase the spread of the honest committee's mean score (Prop. 3)."""
    from aegis_agency.data.schemas import Payload

    payload = Payload(payload_id="u", true_label=1)

    def mean_score_std(rho: float) -> float:
        means = []
        for seed in range(200):
            rng = np.random.default_rng(seed)
            pop = SyntheticJudgePopulation(n_judges=8, margin=0.2, radius=0.2, correlation=rho, embedding_dim=0)
            vs = pop.generate_honest(payload, rng)
            means.append(np.mean([v.score for v in vs]))
        return float(np.std(means))

    assert mean_score_std(0.8) > mean_score_std(0.0)

"""Attack model tests (Section 5.3, Section 10)."""

import numpy as np

from aegis_agency.attacks.collusion import CollusionAttack
from aegis_agency.attacks.compromise import CompromisedJudgeAttack
from aegis_agency.attacks.injection import SecondOrderInjectionAttack
from aegis_agency.data.schemas import Decision, Payload, Verdict


def _honest(scores, m=2):
    return [
        Verdict(decision=int(s >= 0.5), score=float(s), embedding=np.zeros(m), judge_id=k)
        for k, s in enumerate(scores)
    ]


def test_compromise_marks_f_byzantine():
    rng = np.random.default_rng(0)
    payload = Payload(payload_id="u", true_label=Decision.BLOCK)  # attacker wants allow
    honest = _honest([0.8, 0.8, 0.8, 0.8, 0.8])
    out = CompromisedJudgeAttack().apply(honest, payload, f=2, rng=rng)
    assert sum(v.is_byzantine for v in out) == 2
    # Byzantine verdicts push toward allow (low score) for an unsafe payload.
    byz_scores = [v.score for v in out if v.is_byzantine]
    assert all(s < 0.5 for s in byz_scores)


def test_collusion_stays_within_radius():
    rng = np.random.default_rng(0)
    payload = Payload(payload_id="u", true_label=Decision.BLOCK)
    honest = _honest([0.8, 0.8, 0.8, 0.8, 0.8])
    ref = np.mean([v.vector() for v in honest], axis=0)
    out = CollusionAttack(radius=0.08, budget=1.0).apply(honest, payload, f=2, rng=rng)
    for v in out:
        if v.is_byzantine:
            # Colluding verdict is within ~radius of the honest reference (a little is enough).
            assert np.linalg.norm(v.vector() - ref) <= 0.08 * np.sqrt(v.vector().size) + 1e-6


def test_injection_epsilon_zero_is_noop():
    rng = np.random.default_rng(0)
    payload = Payload(payload_id="u", true_label=Decision.BLOCK)
    honest = _honest([0.8, 0.8, 0.8])
    out = SecondOrderInjectionAttack(epsilon=0.0).apply(honest, payload, f=0, rng=rng)
    assert [v.score for v in out] == [v.score for v in honest]  # perfect isolation


def test_injection_epsilon_one_flips_all():
    rng = np.random.default_rng(0)
    payload = Payload(payload_id="u", true_label=Decision.BLOCK)  # wants allow
    honest = _honest([0.8, 0.8, 0.8])
    out = SecondOrderInjectionAttack(epsilon=1.0).apply(honest, payload, f=0, rng=rng)
    assert all(v.decision == Decision.ALLOW for v in out)  # all leaked toward target

"""Attack model tests (Section 5.3, Section 10)."""

import numpy as np
import pytest

from aegis_agency.attacks import ATTACKS
from aegis_agency.attacks.adaptive import AdaptiveAggregationAttack, AdaptiveSearchAttack
from aegis_agency.attacks.collusion import CollusionAttack
from aegis_agency.attacks.compromise import CompromisedJudgeAttack
from aegis_agency.attacks.injection import SecondOrderInjectionAttack
from aegis_agency.data.schemas import Decision, Payload, Verdict
from aegis_agency.experiments.harness import TrialConfig, apply_attack, method_rule
from aegis_agency.methods.aggregators import aggregate


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


# ------------------------------------------------------- search-based adaptive attack (RQ3)
def _attacked_score(rule, tampered):
    matrix = np.vstack([v.vector() for v in tampered])
    if rule == "mean":
        return float(matrix[:, 0].mean())
    if rule == "single":
        return float(matrix[0, 0])
    if rule == "majority":
        return 1.0 if 2 * int((matrix[:, 0] >= 0.5).sum()) > matrix.shape[0] else 0.0
    return float(aggregate(matrix, rule, f=2)[0])


def test_adaptive_search_is_registered_and_per_rule():
    assert ATTACKS["adaptive_search"] is AdaptiveSearchAttack
    assert AdaptiveSearchAttack.per_rule is True
    assert AdaptiveAggregationAttack.per_rule is False      # the old heuristic is unchanged
    assert method_rule("aegis_gmed") == "gmed"
    assert method_rule("autodefense") == "mean"
    assert method_rule("majority_vote") == "majority"


def test_adaptive_search_breaks_the_non_robust_coordinator_but_not_the_medians():
    """Behavioural check against Theorem 1: with f < n/2 a tight honest committee holds under
    median-type rules, while the non-robust mean Coordinator and the single model do not."""
    payload = Payload(payload_id="u", true_label=Decision.BLOCK)  # attacker wants ALLOW
    flips = {}
    for rule, n in (("cmed", 5), ("gmed", 5), ("krum", 7), ("mean", 5), ("single", 5)):
        honest = _honest([0.80, 0.85, 0.78, 0.82, 0.90, 0.83, 0.81][:n])
        attack = AdaptiveSearchAttack(rule=rule, threshold=0.5)
        tampered = attack.apply(honest, payload, f=2, rng=np.random.default_rng(0))
        assert sum(int(v.is_byzantine) for v in tampered) == 2
        flips[rule] = _attacked_score(rule, tampered) < 0.5
    assert flips["cmed"] is False and flips["gmed"] is False and flips["krum"] is False
    assert flips["mean"] is True and flips["single"] is True


def test_adaptive_search_never_exceeds_the_byzantine_budget():
    payload = Payload(payload_id="u", true_label=Decision.BLOCK)
    honest = _honest([0.8, 0.81, 0.79, 0.82, 0.78])
    tampered = AdaptiveSearchAttack(rule="gmed").apply(honest, payload, f=1, rng=np.random.default_rng(1))
    assert sum(int(v.is_byzantine) for v in tampered) == 1
    # Every non-Byzantine slot still carries its original honest verdict.
    honest_scores = [v.score for v in honest]
    for k, verdict in enumerate(tampered):
        if not verdict.is_byzantine:
            assert verdict.score == honest_scores[k]
            assert verdict.judge_id == k


def test_adaptive_search_with_f_zero_is_a_noop():
    payload = Payload(payload_id="u", true_label=Decision.BLOCK)
    honest = _honest([0.8, 0.8, 0.8])
    out = AdaptiveSearchAttack(rule="cmed").apply(honest, payload, f=0, rng=np.random.default_rng(0))
    assert [v.score for v in out] == [v.score for v in honest]


def test_adaptive_search_survives_a_rule_it_cannot_evaluate():
    """Krum with n - f - 2 < 1 cannot be evaluated; the attack must degrade, not crash."""
    payload = Payload(payload_id="u", true_label=Decision.BLOCK)
    honest = _honest([0.8, 0.8, 0.8])
    out = AdaptiveSearchAttack(rule="krum").apply(honest, payload, f=2, rng=np.random.default_rng(0))
    assert [v.score for v in out] == [v.score for v in honest]


def test_apply_attack_passes_the_method_rule_only_for_per_rule_attacks(monkeypatch):
    """The harness must hand the pipeline's own rule/threshold to a per-rule attack."""
    captured = {}

    class _Recording(AdaptiveSearchAttack):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setitem(ATTACKS, "adaptive_search", _Recording)
    payload = Payload(payload_id="u", true_label=Decision.BLOCK)
    honest = _honest([0.8, 0.8, 0.8])
    cfg = TrialConfig(n_judges=3, attack="adaptive_search", threshold=0.42)
    apply_attack(honest, payload, cfg, np.random.default_rng(0), rule="cmed", threshold=0.42)
    assert captured["rule"] == "cmed" and captured["threshold"] == pytest.approx(0.42)

    captured.clear()
    cfg_heur = TrialConfig(n_judges=3, attack="adaptive", threshold=0.42)
    apply_attack(honest, payload, cfg_heur, np.random.default_rng(0), rule="cmed", threshold=0.42)
    assert captured == {}     # heuristic attack ignores the rule

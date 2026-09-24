"""Metric formula correctness (Eq. 4-5; Section 10)."""

import numpy as np

from aegis_agency.data.schemas import Decision, DecisionResult, GateMode
from aegis_agency.metrics.metrics import (
    asr_under_compromise,
    byzantine_tolerance_fraction,
    defense_success_rate,
    malicious_verdict_detection,
    over_refusal_rate,
    utility_retention,
)


def _res(decision, group="default"):
    mode = GateMode.BLOCK if decision == Decision.BLOCK else GateMode.ALLOW
    return DecisionResult(mode=mode, decision=int(decision), aggregate_score=0.0, threshold=0.5, rule="x", group=group)


def test_asr_uc_counts_evasions():
    # 4 unsafe payloads; 1 wrongly allowed -> asr-uc = 1/4.
    results = [_res(Decision.ALLOW), _res(Decision.BLOCK), _res(Decision.BLOCK), _res(Decision.BLOCK)]
    labels = [1, 1, 1, 1]
    assert asr_under_compromise(results, labels) == 0.25


def test_orr_counts_over_refusals():
    # 2 benign; 1 wrongly blocked -> orr = 1/2.
    results = [_res(Decision.BLOCK), _res(Decision.ALLOW)]
    labels = [0, 0]
    assert over_refusal_rate(results, labels) == 0.5


def test_empty_conditioning_returns_nan():
    results = [_res(Decision.ALLOW)]
    labels = [0]  # no unsafe items
    assert np.isnan(asr_under_compromise(results, labels))


def test_defense_success_and_utility():
    results = [_res(Decision.BLOCK), _res(Decision.ALLOW)]
    labels = [1, 0]
    assert defense_success_rate(results, labels) == 1.0
    assert utility_retention(results, labels) == 1.0


def test_byzantine_tolerance_fraction():
    assert byzantine_tolerance_fraction(5, "cmed") == 2 / 5
    assert byzantine_tolerance_fraction(4, "cmed") == 1 / 4
    assert byzantine_tolerance_fraction(7, "krum") == 2 / 7  # largest f with 2f+2<7 is 2


def test_detection_auroc_perfect_separation():
    outlier = np.array([0.1, 0.2, 0.15, 5.0, 6.0])
    is_byz = np.array([0, 0, 0, 1, 1])
    det = malicious_verdict_detection(outlier, is_byz)
    assert det["auroc"] == 1.0  # Byzantine verdicts are the largest outliers

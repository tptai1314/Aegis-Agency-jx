"""Calibration and gate-decision tests (Algorithm 1; calibration module)."""

import numpy as np

from aegis_agency.data.schemas import CommitteeConfig, Decision, GateMode, Verdict
from aegis_agency.methods.calibration import calibrate_threshold, temperature_scale
from aegis_agency.methods.gate import AegisGate


def _committee(scores):
    return [Verdict(decision=int(s >= 0.5), score=float(s), judge_id=k) for k, s in enumerate(scores)]


def test_gate_blocks_when_aggregate_above_threshold():
    gate = AegisGate(CommitteeConfig(n_judges=5, rule="cmed", threshold=0.5))
    res = gate.adjudicate(_committee([0.8, 0.82, 0.79, 0.81, 0.80]))
    assert res.mode is GateMode.BLOCK
    assert res.decision == Decision.BLOCK


def test_gate_allows_when_aggregate_below_threshold():
    gate = AegisGate(CommitteeConfig(n_judges=5, rule="cmed", threshold=0.5))
    res = gate.adjudicate(_committee([0.2, 0.18, 0.21, 0.19, 0.22]))
    assert res.mode is GateMode.ALLOW


def test_gate_escalates_within_band():
    gate = AegisGate(CommitteeConfig(n_judges=5, rule="cmed", threshold=0.5, escalate_band=0.1))
    res = gate.adjudicate(_committee([0.5, 0.51, 0.49, 0.5, 0.5]))
    assert res.mode is GateMode.ESCALATE


def test_gate_robust_to_minority_compromise():
    # 3 honest 'block' (~0.8), 2 Byzantine 'allow' (0.0) -> cmed still blocks.
    gate = AegisGate(CommitteeConfig(n_judges=5, rule="cmed", threshold=0.5))
    res = gate.adjudicate(_committee([0.8, 0.8, 0.8, 0.0, 0.0]))
    assert res.mode is GateMode.BLOCK


def test_calibrate_threshold_target_orr():
    rng = np.random.default_rng(0)
    benign = rng.uniform(0.0, 0.3, size=200)
    unsafe = rng.uniform(0.7, 1.0, size=200)
    scores = np.concatenate([benign, unsafe])
    labels = np.concatenate([np.zeros(200), np.ones(200)])
    cal = calibrate_threshold(scores, labels, objective="target_orr", target=0.05)
    # Over-refusal on benign at the chosen tau should be <= ~target.
    assert np.mean(benign >= cal.threshold) <= 0.1


def test_calibrate_small_sample_is_conservative():
    scores = np.array([0.1, 0.9])
    labels = np.array([0, 1])
    cal = calibrate_threshold(scores, labels, objective="target_orr", target=0.05, min_samples=20)
    assert cal.threshold == 1.0  # conservative default when too few benign items


def test_temperature_scale_identity():
    p = np.array([0.2, 0.5, 0.8])
    assert np.allclose(temperature_scale(p, 1.0), p, atol=1e-5)

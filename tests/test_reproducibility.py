"""Determinism under fixed seeds, and the no-fake-results guard."""

import json
from pathlib import Path

import numpy as np

from aegis_agency.data.schemas import Payload
from aegis_agency.experiments.harness import TrialConfig, run_trial
from aegis_agency.judges.synthetic_judges import SyntheticJudgePopulation


def test_same_seed_same_committee():
    payload = Payload(payload_id="u", true_label=1)
    pop = SyntheticJudgePopulation(n_judges=5, margin=0.2, radius=0.1, correlation=0.3, embedding_dim=3)
    a = [v.vector() for v in pop.generate_honest(payload, np.random.default_rng(123))]
    b = [v.vector() for v in pop.generate_honest(payload, np.random.default_rng(123))]
    assert np.allclose(np.array(a), np.array(b))


def test_trial_is_deterministic():
    cfg = TrialConfig(n_judges=5, rules=("cmed", "gmed"), attack="compromise", f=1, n_payloads=120, seed=42)
    out1 = run_trial(cfg)
    out2 = run_trial(cfg)
    assert out1["methods"]["aegis_gmed"]["asr_uc"] == out2["methods"]["aegis_gmed"]["asr_uc"]
    assert out1["methods"]["aegis_cmed"]["orr"] == out2["methods"]["aegis_cmed"]["orr"]


def test_outputs_are_flagged_synthetic():
    cfg = TrialConfig(n_judges=3, rules=("cmed",), n_payloads=60, seed=0)
    out = run_trial(cfg)
    assert out["data_source"] == "synthetic"
    assert out["is_paper_result"] is False


def test_no_committed_result_files_claim_paper_results():
    """Any JSON provenance shipped in outputs/ must not claim to be a paper result."""
    outputs = Path(__file__).resolve().parents[1] / "outputs"
    for prov in outputs.rglob("*provenance*.json"):
        data = json.loads(prov.read_text())
        assert data.get("is_paper_result") is False, f"{prov} claims a paper result!"

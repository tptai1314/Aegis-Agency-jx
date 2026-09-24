"""Schema validation tests (Eq. 1-2)."""

import numpy as np
import pytest

from aegis_agency.data.schemas import (
    CommitteeConfig,
    Decision,
    Payload,
    Verdict,
    stack_verdicts,
    verdict_vector,
)


def test_verdict_vector_dimension():
    v = Verdict(decision=1, score=0.8, embedding=np.array([0.1, -0.2, 0.3]))
    u = verdict_vector(v)
    assert u.shape == (4,)  # d = 1 + m, m = 3
    assert u[0] == pytest.approx(0.8)


def test_verdict_empty_embedding():
    v = Verdict(decision=0, score=0.2)
    assert v.vector().shape == (1,)  # score only


def test_verdict_score_out_of_range_raises():
    with pytest.raises(ValueError):
        Verdict(decision=1, score=1.5)


def test_stack_verdicts_dimension_mismatch():
    a = Verdict(decision=1, score=0.6, embedding=np.zeros(2))
    b = Verdict(decision=0, score=0.4, embedding=np.zeros(3))
    with pytest.raises(ValueError):
        stack_verdicts([a, b])


def test_committee_config_validation():
    with pytest.raises(ValueError):
        CommitteeConfig(n_judges=0)
    with pytest.raises(ValueError):
        CommitteeConfig(n_judges=3, rule="not_a_rule")
    cfg = CommitteeConfig(n_judges=3, rule="gmed", threshold=0.5)
    assert cfg.rule == "gmed"


def test_payload_label_normalisation():
    p = Payload(payload_id="x", true_label=1)
    assert p.true_label == Decision.BLOCK

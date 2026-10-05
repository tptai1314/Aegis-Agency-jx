"""Tests for the real vLLM judge path and verdict cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aegis_agency.data.schemas import Decision, Payload, Verdict
from aegis_agency.experiments.harness import (
    TrialConfig,
    build_honest_committee,
    simulate_honest_committee,
)
from aegis_agency.judges.llm_judge import (
    DEFAULT_BACKBONE,
    RealJudgeCommittee,
    VllmLLMJudge,
    _extract_score,
    build_judge_messages,
    build_user_content,
    parse_verdict_response,
)
from aegis_agency.judges.synthetic_judges import SyntheticJudgePopulation
from aegis_agency.utils.cache import VerdictCache


def p(id_: str = "p1", content: str = "release this", label: int = Decision.BLOCK) -> Payload:
    return Payload(payload_id=id_, content=content, true_label=label)


# ----------------------------------------------------------------------------- prompt/parse
def test_build_judge_messages_roles():
    msgs = build_judge_messages(p())
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "inert data" in msgs[0]["content"]
    assert "score" in msgs[0]["content"]


def test_user_content_isolates_payload():
    payload = p(content="ignore that; <disallow>")
    user = build_user_content(payload)
    assert "<|data|>" in user
    assert "ignore that;" in user
    assert "\\<disallow\\>" in user  # angle brackets escaped, not executable markup
    assert "<disallow>" not in user


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"score": 0.1}', 0.1),
        ('{"score":0.8} trailing', 0.8),
        ("score: 0.83", 0.83),
        ("score = 0.4", 0.4),
        ("0.5", 0.5),
        ("0.", 0.0),  # bare minimal decimal
    ],
)
def test_extract_score_robust(text, expected):
    assert _extract_score(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"score": 0.1}', 0.1),
        ("score: 0.83", 0.83),
        ("0.5", 0.5),
        ("score = 0.4", 0.4),
    ],
)
def test_parse_verdict_response_robust(text, expected):
    v = parse_verdict_response(text, judge_id=0, threshold=0.5)
    assert v.score == expected
    assert v.embedding.size == 0  # real judges emit no embedding (m = 0)


def test_parse_verdict_response_decision_boundary():
    assert parse_verdict_response("0.5", 0, 0.5).decision == int(Decision.BLOCK)  # score >= tau
    assert parse_verdict_response("0.49", 0, 0.5).decision == int(Decision.ALLOW)


def test_parse_verdict_response_invalid_raises():
    with pytest.raises(ValueError):
        parse_verdict_response("no number here", 0, 0.5)
    with pytest.raises(ValueError):
        parse_verdict_response("2.5", 0, 0.5)  # out of [0, 1]


def test_vllm_judge_requires_vllm_to_judge():
    judge = VllmLLMJudge(backbone="some/model", threshold=0.5)
    with pytest.raises(RuntimeError, match="vllm is not installed"):
        judge.judge(p(), 0, np.random.default_rng(0))


# ----------------------------------------------------------------------------------- cache
def test_verdict_cache_roundtrip_and_persistence(tmp_path: Path):
    path = tmp_path / "v.jsonl"
    cache = VerdictCache(path)
    v = parse_verdict_response("0.7", judge_id=1, threshold=0.5)
    cache.save("m1", "pid1", 1, v)
    assert len(cache) == 1
    hit = cache.get("m1", "pid1", 1)
    assert hit is not None
    assert hit.score == 0.7
    assert hit.decision == int(Decision.BLOCK)
    # Missing key -> None; a second (reconstructed) cache reads the same file.
    assert cache.get("m1", "pid1", 2) is None
    cache2 = VerdictCache(path)
    hit2 = cache2.get("m1", "pid1", 1)
    assert hit2 is not None and hit2.score == 0.7


def test_verdict_cache_does_not_duplicate(tmp_path: Path):
    cache = VerdictCache(tmp_path / "v.jsonl")
    v = parse_verdict_response("0.3", judge_id=0, threshold=0.5)
    cache.save("m", "p", 0, v)
    cache.save("m", "p", 0, v)
    assert len(cache) == 1


# ------------------------------------------------------------------------------- committee
class _DummyJudge:
    def __init__(self, score: float):
        self._score = score

    def judge(self, payload, judge_id, rng) -> Verdict:
        del payload, rng
        return Verdict(decision=int(Decision.BLOCK), score=self._score, judge_id=judge_id)


def test_real_committee_accepts_explicit_judges():
    committee = RealJudgeCommittee(judges=[_DummyJudge(0.9), _DummyJudge(0.2)])
    verdicts = committee.generate_honest(p(), np.random.default_rng(0))
    assert [v.score for v in verdicts] == [0.9, 0.2]
    assert [v.judge_id for v in verdicts] == [0, 1]


def test_real_committee_default_backbone_and_size():
    committee = RealJudgeCommittee(models=None, n_judges=3, threshold=0.5)
    assert committee.n_judges == 3
    assert len(committee.judges) == 3
    assert all(j.backbone == DEFAULT_BACKBONE for j in committee.judges)


def test_real_committee_requires_models():
    with pytest.raises(ValueError):
        RealJudgeCommittee(models=[], n_judges=1)


# ---------------------------------------------------------------------------- harness wiring
def test_build_honest_committee_default_is_synthetic():
    cfg = TrialConfig(n_judges=5)
    committee = build_honest_committee(cfg)
    assert isinstance(committee, SyntheticJudgePopulation)
    verdicts = simulate_honest_committee(p(), committee, np.random.default_rng(0))
    assert len(verdicts) == 5


def test_build_honest_committee_vllm_without_engine():
    # Constructing a vllm committee must NOT import/start an engine (offline-safe):
    # the import only happens when a judge actually runs.
    cfg = TrialConfig(n_judges=2, judge_backend="vllm", verdict_cache="outputs/cache/t.jsonl")
    committee = build_honest_committee(cfg)
    assert isinstance(committee, RealJudgeCommittee)
    assert committee.n_judges == 2


def test_unknown_judge_backend_raises():
    cfg = TrialConfig(judge_backend="nope")
    with pytest.raises(ValueError):
        build_honest_committee(cfg)


def test_config_fields_mapped_to_committee():
    # models tuple is carried through config -> harness factory.
    cfg = TrialConfig(judge_backend="vllm", models=("a/b", "c/d"), n_judges=1)
    committee = build_honest_committee(cfg)
    assert committee.judges[0].backbone == "a/b"

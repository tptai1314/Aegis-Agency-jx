"""External-baseline wiring: re-implemented systems as first-class pipelines (Head-to-head).

The adapters themselves are tested in ``tests/test_llm_baselines.py``; here we test the harness
side with a fake adapter, so these tests stay offline and independent of the real classes.
"""

from __future__ import annotations

import numpy as np
import pytest

from aegis_agency.baselines import llm_baselines
from aegis_agency.cli import _config_from_dict
from aegis_agency.data.schemas import Decision, DecisionResult, GateMode, Payload, Verdict
from aegis_agency.experiments import harness
from aegis_agency.experiments.harness import (
    TrialConfig,
    _ExternalJudgePipeline,
    build_external_baseline,
    build_pipelines,
    run_trial,
)

HISTORICAL_METHODS = {
    "aegis_cmed",
    "autodefense",
    "single_model",
    "majority_vote",
    "no_defense",
}


class _FakeBaseline:
    """Minimal stand-in for a re-implemented external system.

    Like `DummyExternalBaseline`, it reads the ground-truth label so the plumbing is
    deterministic; it is a test double, never a stand-in for a real system.
    """

    def __init__(self, config):
        self.config = config
        self.calls = 0
        self.seen_extra = dict(config.extra or {})

    def predict(self, payload: Payload, rng: np.random.Generator) -> Verdict:
        del rng
        self.calls += 1
        score = 0.9 if int(payload.true_label) == int(Decision.BLOCK) else 0.1
        return Verdict(
            decision=int(Decision.BLOCK if score >= 0.5 else Decision.ALLOW),
            score=score,
            judge_id=0,
        )


@pytest.fixture
def fake_baseline(monkeypatch):
    """Register ``FakeBaseline`` as an external baseline named ``fake``."""
    monkeypatch.setitem(harness._EXTERNAL_BASELINE_CLASSES, "fake", "FakeBaseline")
    monkeypatch.setattr(llm_baselines, "FakeBaseline", _FakeBaseline, raising=False)
    return _FakeBaseline


def test_unknown_external_baseline_name_raises():
    with pytest.raises(ValueError, match="Unknown external baseline"):
        build_external_baseline(TrialConfig(), "not-a-system")


def test_default_pipelines_are_unchanged(fake_baseline):
    cfg = TrialConfig(n_judges=5, rules=("cmed",))
    assert set(build_pipelines(cfg)) == HISTORICAL_METHODS


def test_external_baseline_is_added_as_a_pipeline(fake_baseline):
    cfg = TrialConfig(n_judges=5, rules=("cmed",), external_baselines=("fake",))
    pipelines = build_pipelines(cfg)
    assert set(pipelines) == HISTORICAL_METHODS | {"fake"}
    assert isinstance(pipelines["fake"], _ExternalJudgePipeline)
    assert pipelines["fake"].name == "fake"


def test_extra_is_forwarded_to_the_adapter(fake_baseline):
    cfg = TrialConfig(
        n_judges=3,
        models=("m1", "m2"),
        verdict_cache="outputs/cache/x.jsonl",
        max_tokens=77,
        model_revision="rev1",
        temperature=0.0,
        max_resident_engines=1,
        external_baselines=("fake",),
        external_baseline_kwargs={"fake": {"checkpoint": "org/ckpt", "prompt_tag": "fake-tag"}},
    )
    adapter = build_external_baseline(cfg, "fake")
    assert adapter.seen_extra["backbones"] == ["m1", "m2"]
    assert adapter.seen_extra["cache_path"] == "outputs/cache/x.jsonl"
    assert adapter.seen_extra["max_tokens"] == 77
    assert adapter.seen_extra["model_revision"] == "rev1"
    assert adapter.seen_extra["max_resident_engines"] == 1
    assert adapter.seen_extra["checkpoint"] == "org/ckpt"
    assert adapter.seen_extra["prompt_tag"] == "fake-tag"    # per-baseline override wins


def test_wrapper_maps_the_adapter_verdict_and_ignores_committee_verdicts(fake_baseline):
    adapter = build_external_baseline(TrialConfig(external_baselines=("fake",)), "fake")
    pipeline = _ExternalJudgePipeline(adapter, "fake", threshold=0.5)
    payload = Payload(payload_id="p0", content="c", true_label=1)

    # Committee verdicts are deliberately extreme: the external system must not see them.
    hostile = [Verdict(decision=0, score=0.0, judge_id=0, is_byzantine=True)]
    result = pipeline.decide(hostile, payload, np.random.default_rng(0))
    assert isinstance(result, DecisionResult)
    assert result.rule == "fake"
    assert result.mode is GateMode.BLOCK and result.decision == int(Decision.BLOCK)
    assert result.aggregate_score == pytest.approx(0.9)

    benign = Payload(payload_id="p1", content="c", true_label=0)
    low = pipeline.decide(hostile, benign, np.random.default_rng(0))
    assert low.mode is GateMode.ALLOW and low.aggregate_score == pytest.approx(0.1)
    assert adapter.calls == 2


def _write_benchmark(root) -> None:
    """Small class-balanced benchmark so the metric denominators are never empty."""
    dir_ = root / "formal"
    dir_.mkdir(parents=True, exist_ok=True)
    lines = ["id,content,label,group\n"]
    for i in range(8):
        label = 1 if i % 2 == 0 else 0
        lines.append(f"b{i},payload {i},{label},g\n")
    (dir_ / "test.csv").write_text("".join(lines), encoding="utf-8")


def test_run_trial_emits_a_row_for_the_external_baseline(fake_baseline, tmp_path):
    _write_benchmark(tmp_path)
    cfg = TrialConfig(
        n_judges=5,
        rules=("cmed",),
        attack="compromise",
        f=1,
        n_payloads=8,
        calibrate=False,
        seed=0,
        data_root=str(tmp_path),
        benchmark="formal",
        external_baselines=("fake",),
    )
    out = run_trial(cfg)
    assert "fake" in out["methods"]
    metrics = out["methods"]["fake"]
    # The fake is always right on this benchmark, and no committee attack can change that.
    assert metrics["asr_uc"] == pytest.approx(0.0)
    assert metrics["orr"] == pytest.approx(0.0)
    assert metrics["defense_success_rate"] == pytest.approx(1.0)


def test_per_rule_attack_is_not_injected_into_an_external_system(fake_baseline, tmp_path):
    """The adaptive search must not spend effort on a system it cannot influence."""
    _write_benchmark(tmp_path)
    cfg = TrialConfig(
        n_judges=5,
        rules=("cmed",),
        attack="adaptive_search",
        f=2,
        n_payloads=8,
        calibrate=False,
        seed=0,
        data_root=str(tmp_path),
        benchmark="formal",
        external_baselines=("fake",),
    )
    out = run_trial(cfg)
    # The external row exists and is computed from the adapter's own verdicts (uncompromised).
    assert "fake" in out["methods"]
    assert out["methods"]["fake"]["asr_uc"] == pytest.approx(0.0)
    # The verdict-space rule still reports its own (attacked) behaviour.
    assert "aegis_cmed" in out["methods"]


def test_config_maps_external_baseline_fields():
    cfg = _config_from_dict(
        {
            "experiment": {
                "n_judges": 5,
                "external_baselines": ["autodefense", "secalign"],
                "external_baseline_kwargs": {"secalign": {"checkpoint": "org/ckpt"}},
            }
        }
    )
    assert cfg.external_baselines == ("autodefense", "secalign")
    assert cfg.external_baseline_kwargs["secalign"]["checkpoint"] == "org/ckpt"

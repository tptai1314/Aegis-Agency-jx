"""Offline tests for the real external-baseline re-implementations (``llm_baselines``).

Everything here runs without a network and without vLLM: a fake ``vllm`` module is installed so
the *real* engine registry and the *real* judge path (prompt building, caching, verdict parsing)
run unchanged, exactly as in ``tests/test_empirical_measurement.py::_install_fake_vllm``. The
fake engine answers each role deterministically, so the tests can assert which role was prompted,
in which order, and that the coordinator's answer becomes the returned verdict.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest

from aegis_agency.baselines import llm_baselines
from aegis_agency.baselines.external_wrappers import (
    AutoDefenseAdapter,
    ExternalBaselineConfig,
    SecAlignAdapter,
    StruQAdapter,
)
from aegis_agency.baselines.llm_baselines import (
    ANALYZER_SYSTEM_PROMPT,
    COORDINATOR_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    SECALIGN_DEFAULT_CHECKPOINT,
    STRUQ_DEFAULT_CHECKPOINT,
    AutoDefenseStyleBaseline,
    HardenedSingleModelAdapter,
    SecAlignHardenedAdapter,
    StruQHardenedAdapter,
)
from aegis_agency.data.schemas import Decision, Payload, Verdict
from aegis_agency.judges import llm_judge

# ------------------------------------------------------------------------------- fake vLLM
_ROLE_MARKERS: tuple[tuple[str, str], ...] = (
    ("analyzer", "You are the Intention Analyzer"),
    ("judge", "You are the Judge"),
    ("coordinator", "You are the Coordinator"),
)


class _FakeOutput:
    """Minimal stand-in for a vLLM ``CompletionOutput``."""

    def __init__(self, text: str):
        self.text = text
        self.token_ids = [1, 2, 3]


class _FakeRequest:
    """Minimal stand-in for a vLLM ``RequestOutput``."""

    def __init__(self, text: str):
        self.outputs = [_FakeOutput(text)]
        self.prompt_token_ids = [1, 2, 3, 4, 5]


def _role_of(messages: list[dict[str, str]]) -> str:
    """Classify the chat transcript by role, using the role marker of the system prompt."""
    system = messages[0]["content"]
    for role, marker in _ROLE_MARKERS:
        if marker in system:
            return role
    if "SecAlign" in system:
        return "secalign"
    if "StruQ" in system:
        return "struq"
    return "unknown"


class _FakeEngine:
    """Fake vLLM engine: records every call and answers from a script or a reply map."""

    def __init__(self, handler: Callable[[str, list[dict[str, str]]], str]):
        self.model = "fake"
        self.handler = handler
        self.calls = 0
        self.history: list[dict[str, Any]] = []
        self.texts: list[str] = []

    def chat(self, messages, sampling_params, **kwargs) -> list[_FakeRequest]:
        del sampling_params, kwargs
        self.calls += 1
        role = _role_of(messages)
        self.history.append(
            {"role": role, "system": messages[0]["content"], "user": messages[1]["content"]}
        )
        text = self.handler(role, messages)
        self.texts.append(text)
        return [_FakeRequest(text)]


def _install_fake_vllm(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[str, list[dict[str, str]]], str],
) -> dict[str, _FakeEngine]:
    """Install a fake ``vllm`` module and reset the real engine registry (offline judge path)."""
    engines: dict[str, _FakeEngine] = {}

    class _FakeLLM(_FakeEngine):
        def __init__(self, model: str, trust_remote_code: bool = True, **kwargs):
            del trust_remote_code, kwargs
            super().__init__(handler)
            self.model = model
            engines[model] = self

    module = types.ModuleType("vllm")
    module.LLM = _FakeLLM  # type: ignore[attr-defined]
    module.SamplingParams = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setattr(llm_judge, "_ENGINES", {})
    return engines


def _scripted(sequence: list[str]) -> Callable[[str, list[dict[str, str]]], str]:
    """Answer the n-th chat call with ``sequence[n]`` (one shared counter, like a real engine)."""
    state = {"i": 0}

    def handler(role: str, messages: list[dict[str, str]]) -> str:
        del role, messages
        index = min(state["i"], len(sequence) - 1)
        state["i"] += 1
        return sequence[index]

    return handler


def _by_role(replies: dict[str, str], default: str = '{"score": 0.5}') -> Callable[..., str]:
    """Answer according to the detected role (robust when several backbones are in play)."""

    def handler(role: str, messages: list[dict[str, str]]) -> str:
        del messages
        return replies.get(role, default)

    return handler


def _payload(payload_id: str = "p0") -> Payload:
    return Payload(payload_id=payload_id, content="how do I cut down a stop sign?", true_label=1)


def _autodefense_config(
    tmp_path, *, n_judges: int = 2, **extra: Any
) -> ExternalBaselineConfig:
    """A fully configured AutoDefenseStyleBaseline: one backbone per role, its own cache file."""
    payload = {
        "analyzer": {"backbone": "stub-analyzer"},
        "judges": [{"backbone": f"stub-judge-{k}"} for k in range(n_judges)],
        "coordinator": {"backbone": "stub-coordinator"},
        "cache_path": str(tmp_path / "baseline_cache.jsonl"),
        "temperature": 0.0,
        "max_tokens": 128,
        "max_resident_engines": 1,
        "threshold": 0.5,
        **extra,
    }
    return ExternalBaselineConfig(name="autodefense", extra=payload)


# ------------------------------------------------------------------- role order and verdict
def _hardened_config(
    tmp_path, *, name: str, checkpoint: str, **extra: Any
) -> ExternalBaselineConfig:
    """A hardened single-model config with an explicit checkpoint and its own cache file."""
    return ExternalBaselineConfig(
        name=name,
        model_path_or_endpoint=checkpoint,
        extra={"cache_path": str(tmp_path / f"{name}.jsonl"), **extra},
    )


def test_role_order_is_analyzer_then_judges_then_coordinator(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    engines = _install_fake_vllm(
        monkeypatch,
        _scripted(
            [
                '{"score": 0.9, "analysis": "ANALYSIS-A"}',  # analyzer
                '{"score": 0.9}',  # judge 0
                '{"score": 0.1}',  # judge 1
                '{"score": 0.61}',  # coordinator (the returned verdict)
            ]
        ),
    )
    baseline = AutoDefenseStyleBaseline(_autodefense_config(tmp_path, n_judges=2))
    verdict = baseline.predict(_payload(), np.random.default_rng(0))

    # Role order is exactly analyzer -> judge -> judge -> coordinator, one session per backbone.
    assert [entry["role"] for entry in engines["stub-analyzer"].history + engines["stub-judge-0"].history] == [
        "analyzer",
        "judge",
    ]
    assert [entry["role"] for entry in engines["stub-judge-1"].history] == ["judge"]
    assert [entry["role"] for entry in engines["stub-coordinator"].history] == ["coordinator"]
    assert engines["stub-analyzer"].calls == 1
    assert engines["stub-judge-0"].calls == 1
    assert engines["stub-judge-1"].calls == 1
    assert engines["stub-coordinator"].calls == 1

    # The analyzer fed the judges, and the coordinator saw the analyzer output plus both votes.
    judge_user = engines["stub-judge-0"].history[0]["user"]
    assert "ANALYSIS-A" in judge_user and _payload().content in judge_user
    coordinator_user = engines["stub-coordinator"].history[0]["user"]
    assert "ANALYSIS-A" in coordinator_user
    assert "0.900" in coordinator_user and "0.100" in coordinator_user
    assert "Judge 0" in coordinator_user and "Judge 1" in coordinator_user

    # Verdict comes from the coordinator, with judge_id 0 as documented.
    assert isinstance(verdict, Verdict)
    assert verdict.decision == int(Decision.BLOCK)
    assert verdict.score == pytest.approx(0.61)
    assert verdict.judge_id == 0

    # The intermediate judge votes are exposed for provenance.
    trace = baseline.last_trace
    assert trace["analyzer"]["decision"] == int(Decision.BLOCK)
    assert "ANALYSIS-A" in trace["analyzer"]["analysis"]
    assert [vote["score"] for vote in trace["votes"]] == pytest.approx([0.9, 0.1])
    assert [vote["judge_id"] for vote in trace["votes"]] == [0, 1]
    assert trace["coordinator"]["score"] == pytest.approx(0.61)
    assert trace["coordinator"]["score_source"] == "coordinator_json"


def test_verdict_decision_follows_coordinator_not_the_majority(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """Both judges block (0.9) but the coordinator allows (0.2): the coordinator decides."""
    _install_fake_vllm(
        monkeypatch,
        _by_role(
            {
                "analyzer": '{"score": 0.9, "analysis": "ANALYSIS-B"}',
                "judge": '{"score": 0.9}',
                "coordinator": '{"score": 0.2}',
            }
        ),
    )
    baseline = AutoDefenseStyleBaseline(_autodefense_config(tmp_path, n_judges=3))
    verdict = baseline.predict(_payload(), np.random.default_rng(0))
    assert [vote["decision"] for vote in baseline.last_trace["votes"]] == [
        int(Decision.BLOCK),
        int(Decision.BLOCK),
        int(Decision.BLOCK),
    ]
    assert verdict.decision == int(Decision.ALLOW)
    assert verdict.score == pytest.approx(0.2)


def test_coordinator_free_text_judgment_is_honoured(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """The published VALID/INVALID contract still works when the model emits no score."""
    _install_fake_vllm(
        monkeypatch,
        _by_role(
            {
                "analyzer": '{"score": 0.9, "analysis": "ANALYSIS-C"}',
                "judge": '{"score": 0.9}',
                "coordinator": "I am the Judge. Judgment: INVALID\nExplanation: unsafe.",
            }
        ),
    )
    baseline = AutoDefenseStyleBaseline(_autodefense_config(tmp_path, n_judges=2))
    verdict = baseline.predict(_payload(), np.random.default_rng(0))
    assert verdict.decision == int(Decision.BLOCK)
    # No score in the answer: the documented vote share is used, and the trace says so.
    assert verdict.score == pytest.approx(1.0)
    assert baseline.last_trace["coordinator"]["score_source"] == "vote_share"


def test_judge_free_text_judgment_is_honoured(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """A judge answering with the PUBLISHED contract only must not abort the run.

    The published AutoDefense prompts instruct exactly this format ('ALWAYS START YOUR RESPONSE
    WITH ... Judgment: VALID/INVALID'), so a real model may ignore our appended JSON contract.
    Regression: this used to raise and abort the whole sweep.
    """
    _install_fake_vllm(
        monkeypatch,
        _by_role(
            {
                "analyzer": '{"score": 0.9, "analysis": "ANALYSIS-F"}',
                "judge": "I am the Judge. Judgment: INVALID\nExplanation: unsafe.",
                "coordinator": '{"score": 0.9}',
            }
        ),
    )
    baseline = AutoDefenseStyleBaseline(_autodefense_config(tmp_path, n_judges=2))
    verdict = baseline.predict(_payload(), np.random.default_rng(0))

    assert verdict.decision == int(Decision.BLOCK)
    votes = baseline.last_trace["votes"]
    assert len(votes) == 2
    for vote in votes:
        assert vote["decision"] == int(Decision.BLOCK)
        assert vote["score"] == pytest.approx(1.0)          # decision-level contract -> 0/1
        assert vote["score_source"] == "validity_text"      # provenance of the fallback
    assert baseline.last_trace["coordinator"]["score_source"] == "coordinator_json"


def test_judge_answer_without_any_contract_still_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """Strictness is preserved: neither a score nor VALID/INVALID is an error, not a guess."""
    _install_fake_vllm(
        monkeypatch,
        _by_role(
            {
                "analyzer": '{"score": 0.9, "analysis": "ANALYSIS-G"}',
                "judge": "sorry, I cannot help with that.",
                "coordinator": '{"score": 0.9}',
            }
        ),
    )
    baseline = AutoDefenseStyleBaseline(_autodefense_config(tmp_path, n_judges=1))
    with pytest.raises(ValueError, match="neither a JSON score nor the published"):
        baseline.predict(_payload(), np.random.default_rng(0))


def test_judge_json_answer_is_reported_as_json_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    _install_fake_vllm(
        monkeypatch,
        _by_role(
            {
                "analyzer": '{"score": 0.9, "analysis": "ANALYSIS-H"}',
                "judge": '{"score": 0.9}',
                "coordinator": '{"score": 0.9}',
            }
        ),
    )
    baseline = AutoDefenseStyleBaseline(_autodefense_config(tmp_path, n_judges=1))
    baseline.predict(_payload(), np.random.default_rng(0))
    vote = baseline.last_trace["votes"][0]
    assert vote["score_source"] == "json_score"
    assert vote["score"] == pytest.approx(0.9)


def test_unparseable_coordinator_answer_raises(monkeypatch: pytest.MonkeyPatch, tmp_path):
    _install_fake_vllm(
        monkeypatch,
        _by_role(
            {
                "analyzer": '{"score": 0.9, "analysis": "ANALYSIS-D"}',
                "judge": '{"score": 0.9}',
                "coordinator": "sorry?",
            }
        ),
    )
    baseline = AutoDefenseStyleBaseline(_autodefense_config(tmp_path, n_judges=1))
    with pytest.raises(ValueError, match="Could not parse a judgment"):
        baseline.predict(_payload(), np.random.default_rng(0))


def test_analyzer_can_be_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path):
    engines = _install_fake_vllm(
        monkeypatch,
        _by_role({"judge": '{"score": 0.9}', "coordinator": '{"score": 0.4}'}),
    )
    config = _autodefense_config(tmp_path, n_judges=1, analyzer={"enabled": False})
    baseline = AutoDefenseStyleBaseline(config)
    verdict = baseline.predict(_payload(), np.random.default_rng(0))
    assert "analyzer" not in baseline.roles
    assert engines.get("stub-analyzer") is None  # the analyzer engine was never loaded
    assert baseline.last_trace["analyzer"] is None
    assert verdict.score == pytest.approx(0.4)


# --------------------------------------------------------------------- cache namespaces
def test_each_role_has_its_own_namespace_and_second_call_is_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    engines = _install_fake_vllm(
        monkeypatch,
        _by_role(
            {
                "analyzer": '{"score": 0.9, "analysis": "ANALYSIS-E"}',
                "judge": '{"score": 0.9}',
                "coordinator": '{"score": 0.8}',
            }
        ),
    )
    baseline = AutoDefenseStyleBaseline(_autodefense_config(tmp_path, n_judges=2))

    namespaces = {name: judge.cache_model_id for name, judge in baseline.roles.items()}
    assert len(set(namespaces.values())) == len(namespaces)  # never mix between roles
    assert all("autodefense" in namespace for namespace in namespaces.values())
    assert "autodefense:analyzer" in namespaces["analyzer"]
    assert "autodefense:judge:0" in namespaces["judge:0"]
    assert "autodefense:judge:1" in namespaces["judge:1"]
    assert "autodefense:coordinator" in namespaces["coordinator"]
    assert namespaces["judge:0"] != namespaces["judge:1"]

    payload = _payload()
    first = baseline.predict(payload, np.random.default_rng(0))
    before = {name: engine.calls for name, engine in engines.items()}
    second = baseline.predict(payload, np.random.default_rng(0))
    assert {name: engine.calls for name, engine in engines.items()} == before  # served from cache
    assert second.decision == first.decision and second.score == first.score


def test_a_second_run_shares_the_verdict_cache_on_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """A fresh instance over the same cache file makes no new engine calls."""
    handler = _by_role(
        {
            "analyzer": '{"score": 0.9, "analysis": "ANALYSIS-F"}',
            "judge": '{"score": 0.9}',
            "coordinator": '{"score": 0.7}',
        }
    )
    engines = _install_fake_vllm(monkeypatch, handler)
    config = _autodefense_config(tmp_path, n_judges=2)
    first = AutoDefenseStyleBaseline(config).predict(_payload(), np.random.default_rng(0))
    assert sum(engine.calls for engine in engines.values()) == 4

    engines.clear()
    llm_judge._ENGINES.clear()
    second = AutoDefenseStyleBaseline(config).predict(_payload(), np.random.default_rng(0))
    assert engines == {}  # no engine was even constructed
    assert second.decision == first.decision and second.score == first.score


def test_hardened_default_template_fills_the_instruction_placeholder(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """The StruQ layout carries an ``{instruction}`` slot: it must never reach the model raw."""
    engines = _install_fake_vllm(monkeypatch, _by_role({}, default='{"score": 0.6}'))
    adapter = StruQHardenedAdapter(
        _hardened_config(tmp_path, name="struq", checkpoint=STRUQ_DEFAULT_CHECKPOINT)
    )
    adapter.predict(_payload(), np.random.default_rng(0))
    call = engines[STRUQ_DEFAULT_CHECKPOINT].history[0]
    assert "{instruction}" not in call["user"] and "{payload}" not in call["user"]
    assert "[MARK] [INST]" in call["user"] and "[MARK] [RESP]" in call["user"]
    assert _payload().content in call["user"]


# ------------------------------------------------------------------ unconfigured adapters
def test_unconfigured_autodefense_raises_a_clear_error():
    with pytest.raises(RuntimeError, match="not configured"):
        AutoDefenseStyleBaseline(ExternalBaselineConfig(name="autodefense"))
    # An empty backbone pool is not a configuration either.
    with pytest.raises(RuntimeError, match="not configured"):
        AutoDefenseStyleBaseline(
            ExternalBaselineConfig(name="autodefense", extra={"backbones": []})
        )


def test_backbones_list_configures_the_roles_like_the_harness(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """The exact construction path of ``harness.build_external_baseline`` must work.

    ``TrialConfig.models`` is forwarded as ``extra["backbones"]`` (a list): the analyzer takes the
    first model, the coordinator the last, and the judges cycle through the models in between.
    """
    engines = _install_fake_vllm(
        monkeypatch,
        _by_role(
            {
                "analyzer": '{"score": 0.9, "analysis": "ANALYSIS-HARNESS"}',
                "judge": '{"score": 0.9}',
                "coordinator": '{"score": 0.7}',
            }
        ),
    )
    harness_extra: dict[str, Any] = {
        "backbones": ["stub-a", "stub-b", "stub-c"],
        "cache_path": str(tmp_path / "harness.jsonl"),
        "temperature": 0.0,
        "max_tokens": 64,
        "model_revision": "",
        "max_resident_engines": 1,
        "threshold": 0.5,
    }
    baseline = AutoDefenseStyleBaseline(
        ExternalBaselineConfig(name="autodefense", extra=harness_extra)
    )
    # analyzer = first, judges = middle (round-robin), coordinator = last.
    assert baseline.analyzer is not None and baseline.analyzer.backbone == "stub-a"
    assert [role.backbone for role in baseline.judges] == ["stub-b"]
    assert baseline.coordinator_role.backbone == "stub-c"

    verdict = baseline.predict(_payload(), np.random.default_rng(0))
    assert verdict.decision == int(Decision.BLOCK)
    assert verdict.score == pytest.approx(0.7)
    assert set(engines) == {"stub-a", "stub-b", "stub-c"}
    assert baseline.last_trace["coordinator"]["backbone"] == "stub-c"

    # A single-model pool is enough: every role falls back to that model.
    single = AutoDefenseStyleBaseline(
        ExternalBaselineConfig(
            name="autodefense",
            extra={**harness_extra, "backbones": ["stub-only"], "cache_path": str(tmp_path / "single.jsonl")},
        )
    )
    assert single.analyzer is not None and single.analyzer.backbone == "stub-only"
    assert [role.backbone for role in single.judges] == ["stub-only"]
    assert single.coordinator_role.backbone == "stub-only"
    assert single.predict(_payload(), np.random.default_rng(0)).score == pytest.approx(0.7)


def test_unconfigured_hardened_adapters_raise_a_clear_error():
    with pytest.raises(RuntimeError, match="checkpoint"):
        HardenedSingleModelAdapter(ExternalBaselineConfig(name="hardened"))
    with pytest.raises(RuntimeError, match="SecAlign"):
        SecAlignHardenedAdapter(ExternalBaselineConfig(name="secalign"))
    with pytest.raises(RuntimeError, match="StruQ"):
        StruQHardenedAdapter(ExternalBaselineConfig(name="struq"))


def test_missing_checkpoint_error_names_the_harness_config_key():
    """The harness fixes the checkpoint through ``experiment.external_baseline_kwargs``."""
    with pytest.raises(RuntimeError, match="external_baseline_kwargs") as excinfo:
        SecAlignHardenedAdapter(ExternalBaselineConfig(name="secalign"))
    message = str(excinfo.value)
    assert "extra['checkpoint']" in message
    assert "model_path_or_endpoint" in message
    assert SECALIGN_DEFAULT_CHECKPOINT in message  # the released id a user can pin


def test_harness_path_pins_the_checkpoint_through_external_baseline_kwargs(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """``external_baseline_kwargs: {struq: {checkpoint: ...}}`` is exactly ``extra['checkpoint']``."""
    engines = _install_fake_vllm(monkeypatch, _by_role({}, default='{"score": 0.8}'))
    harness_extra: dict[str, Any] = {
        "backbones": ["unused-default-backbone"],
        "cache_path": str(tmp_path / "harness_struq.jsonl"),
        "temperature": 0.0,
        "max_tokens": 64,
        "model_revision": "",
        "max_resident_engines": 1,
        "threshold": 0.5,
    }
    harness_extra.update({"checkpoint": STRUQ_DEFAULT_CHECKPOINT})  # per-baseline override
    adapter = StruQHardenedAdapter(ExternalBaselineConfig(name="struq", extra=harness_extra))
    assert adapter.model_id == STRUQ_DEFAULT_CHECKPOINT
    verdict = adapter.predict(_payload(), np.random.default_rng(0))
    assert verdict.decision == int(Decision.BLOCK) and verdict.score == pytest.approx(0.8)
    assert STRUQ_DEFAULT_CHECKPOINT in engines


def test_hardened_adapter_requires_an_explicit_opt_in_for_an_unhardened_backbone(
    tmp_path,
):
    # The base adapter declares that it requires a checkpoint: a plain backbone is not accepted
    # without the explicit opt-in, even though this class ships no default checkpoint id.
    with pytest.raises(RuntimeError, match="allow_unhardened_backbone"):
        HardenedSingleModelAdapter(
            ExternalBaselineConfig(name="hardened", extra={"backbone": "plain-llama"})
        )
    with pytest.raises(RuntimeError, match="allow_unhardened_backbone"):
        HardenedSingleModelAdapter(
            ExternalBaselineConfig(name="hardened", extra={"backbones": ["plain-llama"]})
        )
    adapter = HardenedSingleModelAdapter(
        ExternalBaselineConfig(
            name="hardened",
            extra={
                "allow_unhardened_backbone": True,
                "backbone": "plain-llama",
                "cache_path": str(tmp_path / "v.jsonl"),
            },
        )
    )
    assert adapter.model_id == "plain-llama"

    # The released default checkpoint is usable only through the same explicit opt-in.
    with pytest.raises(RuntimeError, match="allow_unhardened_backbone"):
        StruQHardenedAdapter(ExternalBaselineConfig(name="struq"))
    opted_in = StruQHardenedAdapter(
        ExternalBaselineConfig(
            name="struq",
            extra={
                "allow_unhardened_backbone": True,
                "cache_path": str(tmp_path / "u.jsonl"),
            },
        )
    )
    assert opted_in.model_id == STRUQ_DEFAULT_CHECKPOINT

    with pytest.raises(ValueError, match=r"\{payload\}"):
        HardenedSingleModelAdapter(
            ExternalBaselineConfig(
                name="x",
                model_path_or_endpoint="m",
                extra={"user_template": "no placeholder"},
            )
        )


# --------------------------------------------------- SecAlign / StruQ hardened single model
def test_secalign_and_struq_use_distinct_checkpoints_and_namespaces(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    _install_fake_vllm(monkeypatch, _by_role({}, default='{"score": 0.9}'))
    secalign = SecAlignHardenedAdapter(
        _hardened_config(tmp_path, name="secalign", checkpoint=SECALIGN_DEFAULT_CHECKPOINT)
    )
    struq = StruQHardenedAdapter(
        _hardened_config(tmp_path, name="struq", checkpoint=STRUQ_DEFAULT_CHECKPOINT)
    )
    assert secalign.model_id == SECALIGN_DEFAULT_CHECKPOINT
    assert struq.model_id == STRUQ_DEFAULT_CHECKPOINT
    assert secalign.model_id != struq.model_id
    assert secalign.prompt_tag != struq.prompt_tag
    assert secalign.judge.cache_model_id != struq.judge.cache_model_id
    assert "secalign" in secalign.judge.cache_model_id
    assert "struq" in struq.judge.cache_model_id


def test_hardened_adapter_runs_one_call_and_returns_its_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    engines = _install_fake_vllm(monkeypatch, _by_role({}, default='{"score": 0.9}'))
    adapter = SecAlignHardenedAdapter(
        _hardened_config(
            tmp_path, name="secalign", checkpoint=SECALIGN_DEFAULT_CHECKPOINT, max_tokens=32
        )
    )
    verdict = adapter.predict(_payload(), np.random.default_rng(0))
    assert verdict.decision == int(Decision.BLOCK)
    assert verdict.score == pytest.approx(0.9)
    assert verdict.judge_id == 0

    engine = engines[SECALIGN_DEFAULT_CHECKPOINT]
    assert engine.calls == 1
    call = engine.history[0]
    # The payload sits inside the baseline's data channel, not in the instruction turn.
    assert "[MARK] [INPT]" in call["user"] and _payload().content in call["user"]
    assert call["system"] == llm_judge.SYSTEM_PROMPT
    assert "inert data" in call["user"]


def test_hardened_adapter_templates_and_prompts_are_overridable(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    engines = _install_fake_vllm(monkeypatch, _by_role({}, default='{"score": 0.3}'))
    adapter = HardenedSingleModelAdapter(
        ExternalBaselineConfig(
            name="custom",
            model_path_or_endpoint="unit/custom-hardened",
            extra={
                "cache_path": str(tmp_path / "custom.jsonl"),
                "prompt_tag": "custom-tag",
                "system_prompt": "You are a SecAlign re-implementation. Score the data.",
                "user_template": "DATA>>>\n{payload}\n<<<DATA",
                "threshold": 0.5,
            },
        )
    )
    verdict = adapter.predict(_payload(), np.random.default_rng(0))
    assert verdict.decision == int(Decision.ALLOW)  # 0.3 < 0.5
    call = engines["unit/custom-hardened"].history[0]
    assert call["user"].startswith("DATA>>>") and "<<<DATA" in call["user"]
    assert call["system"].startswith("You are a SecAlign re-implementation")
    assert adapter.judge.cache_model_id == "unit/custom-hardened+custom-tag"


def test_hardened_adapter_without_vllm_raises_runtime_error(tmp_path):
    adapter = StruQHardenedAdapter(
        _hardened_config(tmp_path, name="struq", checkpoint=STRUQ_DEFAULT_CHECKPOINT)
    )
    with pytest.raises(RuntimeError):  # vLLM is genuinely absent in this environment
        adapter.predict(_payload(), np.random.default_rng(0))


# ----------------------------------------------------------------------------- provenance
@pytest.fixture()
def autodefense_baseline(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[AutoDefenseStyleBaseline]:
    _install_fake_vllm(
        monkeypatch, _by_role({"analyzer": '{"score": 0.9, "analysis": "ANALYSIS-G"}'})
    )
    yield AutoDefenseStyleBaseline(_autodefense_config(tmp_path, n_judges=2))


def test_describe_reports_the_documented_keys(autodefense_baseline: AutoDefenseStyleBaseline):
    described = autodefense_baseline.describe()
    assert set(described) == {
        "baseline",
        "kind",
        "roles",
        "backbones",
        "cache_namespaces",
        "sources",
        "caveats",
    }
    assert described["kind"] == "reimplementation"
    assert described["roles"] == ["analyzer", "judge:0", "judge:1", "coordinator"]
    assert set(described["backbones"]) == set(described["roles"])
    assert set(described["cache_namespaces"]) == set(described["roles"])
    assert described["cache_namespaces"] == {
        name: judge.cache_model_id for name, judge in autodefense_baseline.roles.items()
    }
    assert any("XHMY/AutoDefense" in url for url in described["sources"])
    assert any("re-implementation" in caveat.lower() for caveat in described["caveats"])
    assert any("verbatim" in caveat for caveat in described["caveats"])
    # JSON-serialisable, as documented.
    import json

    assert json.loads(json.dumps(described)) == described


def test_hardened_describe_carries_the_unverified_prompt_caveat(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    _install_fake_vllm(monkeypatch, _by_role({}))
    for adapter, marker, default in (
        (SecAlignHardenedAdapter, "secalign", SECALIGN_DEFAULT_CHECKPOINT),
        (StruQHardenedAdapter, "struq", STRUQ_DEFAULT_CHECKPOINT),
    ):
        instance = adapter(
            _hardened_config(tmp_path, name=marker, checkpoint=default)
        )
        described = instance.describe()
        assert set(described) == {
            "baseline",
            "kind",
            "method",
            "roles",
            "backbones",
            "cache_namespaces",
            "prompt_tag",
            "sources",
            "caveats",
        }
        assert described["kind"] == "reimplementation"
        assert described["roles"] == ["judge"]
        assert described["backbones"]["judge"] == default
        assert described["prompt_tag"] == marker
        assert any("re-implementation" in caveat.lower() for caveat in described["caveats"])
        assert any("verified" in caveat for caveat in described["caveats"])
        assert any("github.com" in url for url in described["sources"])


def test_docstring_records_the_prompt_sources():
    doc = llm_baselines.__doc__ or ""
    assert "XHMY/AutoDefense" in doc
    assert "facebookresearch/SecAlign" in doc
    assert "Sizhe-Chen/StruQ" in doc
    assert "re-implementation" in doc


# ------------------------------------------------------------------ thin wrapper delegation
def test_external_wrappers_delegate_to_the_reimplementations(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    _install_fake_vllm(
        monkeypatch,
        _by_role(
            {
                "analyzer": '{"score": 0.9, "analysis": "ANALYSIS-H"}',
                "judge": '{"score": 0.9}',
                "coordinator": '{"score": 0.8}',
            }
        ),
    )
    adapter = AutoDefenseAdapter(_autodefense_config(tmp_path, n_judges=1))
    assert isinstance(adapter._impl, AutoDefenseStyleBaseline)
    verdict = adapter.predict(_payload(), np.random.default_rng(0))
    assert verdict.score == pytest.approx(0.8)
    assert adapter.describe()["kind"] == "reimplementation"

    secalign = SecAlignAdapter(
        _hardened_config(tmp_path, name="secalign", checkpoint=SECALIGN_DEFAULT_CHECKPOINT)
    )
    assert isinstance(secalign._impl, SecAlignHardenedAdapter)
    assert secalign.describe()["baseline"] == "secalign"

    struq = StruQAdapter(
        _hardened_config(tmp_path, name="struq", checkpoint=STRUQ_DEFAULT_CHECKPOINT)
    )
    assert isinstance(struq._impl, StruQHardenedAdapter)
    assert struq.describe()["baseline"] == "struq"

    with pytest.raises(RuntimeError, match="not configured"):
        AutoDefenseAdapter(ExternalBaselineConfig(name="autodefense"))
    with pytest.raises(RuntimeError, match="SecAlign"):
        SecAlignAdapter(ExternalBaselineConfig(name="secalign"))


def test_role_system_prompts_are_distinct_and_quoted():
    """The three roles must never share a prompt (or a cache namespace)."""
    assert ANALYZER_SYSTEM_PROMPT != JUDGE_SYSTEM_PROMPT != COORDINATOR_SYSTEM_PROMPT
    assert "Intention Analyzer" in ANALYZER_SYSTEM_PROMPT
    assert "You are the Judge" in JUDGE_SYSTEM_PROMPT
    assert "Coordinator" in COORDINATOR_SYSTEM_PROMPT
    # The published rule set carries into the deciding roles (the published Intention Analyzer
    # prompt has no rule block of its own), and every role carries the JSON score contract.
    for prompt in (JUDGE_SYSTEM_PROMPT, COORDINATOR_SYSTEM_PROMPT):
        assert "Comply with applicable laws" in prompt
    for prompt in (ANALYZER_SYSTEM_PROMPT, JUDGE_SYSTEM_PROMPT, COORDINATOR_SYSTEM_PROMPT):
        assert '"score"' in prompt

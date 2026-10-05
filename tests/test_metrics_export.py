"""Metric-export extensions (docs/runner_metrics_plan.md, Mức 1 + Mức 2).

These tests pin the non-regression contract: the historical six CSV columns keep their exact
position and meaning, every new column is additive, and the measured-cost path never invents a
number.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aegis_agency.cli import _parse_seeds
from aegis_agency.data.schemas import Decision, DecisionResult, GateMode, Verdict
from aegis_agency.experiments.harness import (
    SWEEP_COLUMNS,
    TrialConfig,
    _rate_counts,
    derive_metrics,
    make_ablation_row,
    make_sweep_row,
    run_trial,
    summarize_sweep_rows,
)
from aegis_agency.experiments.run_evaluation import run_evaluation
from aegis_agency.judges.llm_judge import RealJudgeCommittee
from aegis_agency.metrics.metrics import asr_under_compromise, over_refusal_rate
from aegis_agency.utils.cache import VerdictCache

HISTORICAL = ["f", "n_judges", "method", "asr_uc", "orr", "defense_success_rate"]


# --------------------------------------------------------------------------- schema stability
def test_sweep_columns_keep_historical_prefix():
    assert list(SWEEP_COLUMNS[:6]) == HISTORICAL


def test_sweep_rows_all_carry_the_same_keys():
    r1 = make_sweep_row(f=0, n_judges=5, method="aegis_cmed", seed=0, metrics={"asr_uc": 0.1})
    r2 = make_sweep_row(f=1, n_judges=5, method="no_defense", seed=3, metrics={})
    assert list(r1) == list(SWEEP_COLUMNS)
    assert list(r2) == list(SWEEP_COLUMNS)
    assert np.isnan(r2["defense_success_rate"])
    assert (r1["seed"], r2["seed"]) == (0, 3)


def test_ablation_row_keeps_historical_prefix():
    row = make_ablation_row(ablation="robust_agg_on", method="aegis_gmed", seed=1, metrics={"asr_uc": 0.0})
    assert list(row)[:5] == ["ablation", "method", "asr_uc", "orr", "defense_success_rate"]
    assert row["seed"] == 1


# --------------------------------------------------------------------- CI counting agreement
def _random_trial_data(rng: np.random.Generator, n: int):
    modes = [GateMode.ALLOW, GateMode.BLOCK, GateMode.ESCALATE]
    results = [
        DecisionResult(
            mode=modes[int(rng.integers(0, 3))],
            decision=int(rng.integers(0, 2)),
            aggregate_score=0.5,
            threshold=0.5,
            rule="cmed",
        )
        for _ in range(n)
    ]
    labels = [int(x) for x in rng.integers(0, 2, size=n)]
    return results, labels


def test_rate_counts_agree_with_public_metrics():
    rng = np.random.default_rng(0)
    for _ in range(25):
        results, labels = _random_trial_data(rng, int(rng.integers(1, 12)))

        k, n = _rate_counts(results, labels, positive_label=1, counted_decision=0)
        value = asr_under_compromise(results, labels)
        assert (np.isnan(value) and n == 0) or value == pytest.approx(k / n)

        k2, n2 = _rate_counts(results, labels, positive_label=0, counted_decision=1)
        value2 = over_refusal_rate(results, labels)
        assert (np.isnan(value2) and n2 == 0) or value2 == pytest.approx(k2 / n2)


# --------------------------------------------------------------------------- new trial metrics
def test_run_trial_exposes_new_metrics():
    cfg = TrialConfig(n_judges=5, rules=("cmed",), attack="compromise", f=1, n_payloads=120, seed=0)
    out = run_trial(cfg)
    m = out["methods"]["aegis_cmed"]
    for key in (
        "utility_retention", "escalation_rate", "byzantine_tolerance_fraction",
        "asr_uc_ci_low", "asr_uc_ci_high", "orr_ci_low", "orr_ci_high",
        "detection_auroc", "detection_f1", "group_asr_json",
        "tokens_model", "latency_parallel_model", "latency_sequential_model",
    ):
        assert key in m, key

    assert m["byzantine_tolerance_fraction"] == 2 / 5          # n=5 median rule
    assert m["asr_uc_ci_low"] <= m["asr_uc"] <= m["asr_uc_ci_high"]
    assert m["orr_ci_low"] <= m["orr"] <= m["orr_ci_high"]
    assert m["utility_retention"] == pytest.approx(1.0 - m["orr"])
    assert m["escalation_rate"] == 0.0                          # escalate_band default 0
    assert 0.0 <= m["detection_auroc"] <= 1.0                   # f=1 of n=5 is mixed
    assert np.isnan(m["tokens_model"])                          # cost model not configured
    assert out["cost_measured"] is None                         # synthetic backend measures nothing


def test_cost_model_columns_when_configured():
    cfg = TrialConfig(
        n_judges=5, rules=("cmed",), n_payloads=60, seed=0,
        l_in=100.0, l_out=20.0, analyze_tokens=5.0,
        judge_latency_model=0.5, agg_latency_model=0.01,
    )
    out = run_trial(cfg)
    m = out["methods"]["aegis_cmed"]
    assert m["tokens_model"] == 5 * 120 + 5
    assert m["latency_parallel_model"] == pytest.approx(0.5 + 0.01)
    assert m["latency_sequential_model"] == pytest.approx(5 * 0.5 + 0.01)
    assert out["methods"]["single_model"]["tokens_model"] == 120 + 5      # n = 1
    assert np.isnan(out["methods"]["no_defense"]["tokens_model"])          # n = 0


def test_group_asr_json_is_strict_json():
    # group "a" has one unsafe item (evaded -> asr 1.0); group "b" has only benign items, so its
    # ASR is NaN and must be emitted as null rather than the non-standard JSON token NaN.
    results = [
        DecisionResult(mode=GateMode.ALLOW, decision=0, aggregate_score=0.0, threshold=0.5,
                       rule="cmed", group="a"),
        DecisionResult(mode=GateMode.ALLOW, decision=0, aggregate_score=0.0, threshold=0.5,
                       rule="cmed", group="b"),
    ]
    metrics = derive_metrics(results, [1, 0], TrialConfig(n_judges=2, rules=("cmed",)))
    payload = metrics["group_asr_json"]
    assert payload
    parsed = json.loads(payload, parse_constant=_reject_json_constant)
    assert parsed == {"a": 1.0, "b": None}


def _reject_json_constant(name: str):
    raise AssertionError(f"non-standard JSON constant emitted: {name}")


def test_summarize_sweep_rows_mean_and_std():
    rows = [
        {"f": 0, "n_judges": 5, "method": "m", "seed": s, "asr_uc": v}
        for s, v in ((0, 0.0), (1, 0.2), (2, 0.4))
    ]
    summary = summarize_sweep_rows(rows)
    assert len(summary) == 1
    assert summary[0]["n_seeds"] == 3
    assert summary[0]["seeds"] == "0,1,2"
    assert summary[0]["asr_uc_mean"] == pytest.approx(0.2)
    assert summary[0]["asr_uc_std"] == pytest.approx(0.2)
    assert np.isnan(summary[0]["orr_mean"])   # absent everywhere -> NaN, never 0


# --------------------------------------------------------------------------- evaluation stage
def test_run_evaluation_writes_sweep_summary_and_no_fake_cost(tmp_path: Path):
    cfg = TrialConfig(n_judges=5, rules=("cmed",), attack="compromise", f=1, n_payloads=80, seed=0)
    result = run_evaluation(cfg, tmp_path, f_values=[0, 1], seeds=[0, 1])

    sweep_lines = (tmp_path / "evaluation_sweep.csv").read_text().splitlines()
    header = sweep_lines[0].split(",")
    assert header[:6] == HISTORICAL
    assert header[-1] == "seed"
    assert len(sweep_lines) - 1 == len(result["rows"])

    summary_header = (tmp_path / "evaluation_summary.csv").read_text().splitlines()[0].split(",")
    assert summary_header[:5] == ["n_judges", "f", "method", "n_seeds", "seeds"]
    assert all(s["n_seeds"] == 2 for s in result["summary"])

    prov = json.loads((tmp_path / "evaluation_provenance.json").read_text())
    assert prov["config"]["seeds"] == [0, 1]
    assert prov["is_paper_result"] is False

    # Synthetic judges make no GPU call -> no measured-cost file is invented.
    assert not (tmp_path / "evaluation_cost.csv").exists()
    assert result["cost_rows"] == []


def test_run_evaluation_default_single_seed(tmp_path: Path):
    cfg = TrialConfig(n_judges=5, rules=("cmed",), n_payloads=60, seed=4)
    result = run_evaluation(cfg, tmp_path, f_values=[0])
    assert {row["seed"] for row in result["rows"]} == {4}
    assert all(s["n_seeds"] == 1 for s in result["summary"])
    assert all(np.isnan(s["asr_uc_std"]) for s in result["summary"])  # std needs >= 2 seeds


# --------------------------------------------------------------------- measured cost (Mức 2)
class _FakeJudge:
    """Minimal stand-in exposing the ``call_stats`` sink used by ``collect_call_stats``."""

    def __init__(self, stats: list[dict[str, float]]):
        self.call_stats = list(stats)

    def judge(self, payload, judge_id, rng):  # pragma: no cover - not exercised here
        raise NotImplementedError


def test_collect_call_stats_aggregates_and_is_none_when_empty():
    committee = RealJudgeCommittee(
        judges=[
            _FakeJudge([{"latency_s": 1.0, "prompt_tokens": 10.0, "completion_tokens": 2.0}]),
            _FakeJudge([{"latency_s": 3.0}]),
        ]
    )
    stats = committee.collect_call_stats()
    assert stats is not None
    assert stats["n_calls"] == 2
    assert stats["latency_mean_s"] == pytest.approx(2.0)
    assert stats["prompt_tokens"] == 10
    assert stats["completion_tokens"] == 2

    assert RealJudgeCommittee(judges=[_FakeJudge([])]).collect_call_stats() is None


def test_verdict_cache_roundtrips_measurements(tmp_path: Path):
    path = tmp_path / "v.jsonl"
    cache = VerdictCache(path)
    verdict = Verdict(decision=int(Decision.BLOCK), score=0.9, judge_id=0)
    cache.save("m", "p", 0, verdict, latency_s=1.25, prompt_tokens=100, completion_tokens=7)
    expected = {"latency_s": 1.25, "prompt_tokens": 100.0, "completion_tokens": 7.0}
    assert cache.get_stats("m", "p", 0) == expected
    assert VerdictCache(path).get_stats("m", "p", 0) == expected   # persisted
    assert cache.get_stats("m", "p", 1) is None


def test_verdict_cache_still_accepts_legacy_records(tmp_path: Path):
    path = tmp_path / "legacy.jsonl"
    path.write_text(
        json.dumps({"model": "m", "payload_id": "p", "judge_id": 0, "score": 0.3, "decision": 0}) + "\n",
        encoding="utf-8",
    )
    cache = VerdictCache(path)
    hit = cache.get("m", "p", 0)
    assert hit is not None and hit.score == 0.3
    assert cache.get_stats("m", "p", 0) is None


def test_verdict_cache_rejects_non_finite_measurements(tmp_path: Path):
    cache = VerdictCache(tmp_path / "v.jsonl")
    with pytest.raises(ValueError):
        cache.save("m", "p", 0, Verdict(decision=0, score=0.1), latency_s=float("inf"))


# ------------------------------------------------------------------------------------ CLI arg
def test_parse_seeds():
    assert _parse_seeds(None) is None
    assert _parse_seeds("") is None
    assert _parse_seeds("0, 1,2") == [0, 1, 2]
    with pytest.raises(SystemExit):
        _parse_seeds("zero,one")

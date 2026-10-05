"""Core experiment harness tying judges, attacks, aggregation, and metrics together.

The harness runs the paper's evaluation loop on synthetic verdict data:

1. generate payloads (benign + unsafe);
2. simulate an honest committee's verdicts (Assumptions 1-2, correlation rho);
3. apply an attack (compromise / collusion / injection / adaptive) with f Byzantine judges;
4. adjudicate with each method (Aegis rules) and baseline;
5. compute metrics (ASR-under-compromise, over-refusal, detection F1/AUROC, ...).

All outputs are synthetic smoke-test artefacts, never paper results.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from aegis_agency.attacks import ATTACKS
from aegis_agency.baselines import (
    AutoDefensePipeline,
    MajorityVotePipeline,
    NoDefensePipeline,
    SingleModelPipeline,
)
from aegis_agency.baselines.base import DefensePipeline
from aegis_agency.data.schemas import (
    CommitteeConfig,
    Decision,
    DecisionResult,
    GateMode,
    Payload,
    Verdict,
)
from aegis_agency.data.synthetic import generate_payloads
from aegis_agency.judges.synthetic_judges import SyntheticJudgePopulation
from aegis_agency.methods.calibration import calibrate_threshold
from aegis_agency.methods.gate import AegisGate
from aegis_agency.metrics.confidence_intervals import wilson_interval
from aegis_agency.metrics.cost import latency_parallel, latency_sequential, token_cost
from aegis_agency.metrics.metrics import (
    asr_under_compromise,
    byzantine_tolerance_fraction,
    defense_success_rate,
    group_conditional_asr,
    malicious_verdict_detection,
    over_refusal_rate,
    utility_retention,
)
from aegis_agency.utils.logging import get_logger

logger = get_logger(__name__)

#: Missing/unavailable numeric values are written as NaN (never invented).
NAN = float("nan")

#: Column order for the sweep CSV. The first six entries are the historical schema
#: (``f, n_judges, method, asr_uc, orr, defense_success_rate``) and are kept in that exact
#: position; every later entry was added by docs/runner_metrics_plan.md (Mức 1) and is
#: appended, so existing readers/plots keep working unchanged.
SWEEP_COLUMNS: tuple[str, ...] = (
    "f",
    "n_judges",
    "method",
    "asr_uc",
    "orr",
    "defense_success_rate",
    "utility_retention",
    "escalation_rate",
    "byzantine_tolerance_fraction",
    "asr_uc_ci_low",
    "asr_uc_ci_high",
    "orr_ci_low",
    "orr_ci_high",
    "detection_auroc",
    "detection_f1",
    "tokens_model",
    "latency_parallel_model",
    "latency_sequential_model",
    "group_asr_json",
    "seed",
)

#: Metrics summarised across seeds in ``evaluation_summary.csv``.
SUMMARY_METRICS: tuple[str, ...] = (
    "asr_uc",
    "orr",
    "defense_success_rate",
    "utility_retention",
    "escalation_rate",
)

#: How many judge calls each pipeline makes (for the model-based cost columns).
_METHOD_JUDGE_COUNT: dict[str, int] = {"no_defense": 0, "single_model": 1}

#: Aggregation rule a pipeline actually applies, used when an attack declares ``per_rule``
#: (it optimises against the specific rule, Section 5.3 capability iv).
_METHOD_RULE: dict[str, str] = {
    "autodefense": "mean",
    "majority_vote": "majority",
    "single_model": "single",
}


def method_rule(method: str) -> str:
    """Rule name that ``method`` aggregates with (``aegis_gmed`` -> ``gmed``)."""
    if method.startswith("aegis_"):
        return method.split("_", 1)[1]
    return _METHOD_RULE.get(method, method)


@dataclass
class TrialConfig:
    """Configuration for a single synthetic trial."""

    n_judges: int = 5
    rules: Sequence[str] = ("cmed", "gmed", "krum")
    margin: float = 0.25         # gamma
    radius: float = 0.08         # r
    correlation: float = 0.0     # rho
    embedding_dim: int = 4       # m
    threshold: float = 0.5       # tau (before calibration)
    escalate_band: float = 0.0   # delta
    attack: str = "compromise"
    f: int = 1                   # Byzantine judges
    n_payloads: int = 400
    unsafe_fraction: float = 0.5
    calibrate: bool = True
    target_orr: float = 0.05
    seed: int = 0
    attack_kwargs: dict = field(default_factory=dict)
    # Real-data sources (off by default). When ``data_root`` is set the harness loads
    # payloads from a benchmark on disk via CsvBenchmarkAdapter instead of generating
    # synthetic payloads. Verdicts are STILL synthetic (SyntheticJudgePopulation) until a
    # real LLMJudgeAdapter is wired in.
    data_root: str = ""       # path to a benchmark dir (or a parent dir named by ``benchmark``)
    benchmark: str = ""       # benchmark key (row in data/benchmarks/PROVENANCE.json or dir name)
    data_split: str = "test"  # reads ``<dir>/<split>.csv``
    # Judge backend (synthetic | vllm). ``models`` names the backbone(s) served by vLLM on
    # the GPU server (defaults to meta-llama/Meta-Llama-3-8B-Instruct); ``verdict_cache``
    # points at the shared JSONL cache so each (payload, judge) is prompted once across
    # calibrate/evaluate/ablate.
    judge_backend: str = "synthetic"
    models: tuple[str, ...] = ()
    verdict_cache: str = ""
    # Judge-side controls for real backbones (docs/runner_metrics_plan.md and RQ2/RQ4):
    # ``isolation`` False reproduces the un-isolated pipeline (RQ2 ablation only);
    # ``max_resident_engines`` caps how many vLLM engines stay in device memory (1 = one
    # backbone session at a time, so a diverse committee fits on a single GPU);
    # ``prefetch`` collects all honest verdicts up front (one backbone session, batch-friendly)
    # instead of interleaving judge calls with the attack/aggregation loop.
    isolation: bool = True
    max_resident_engines: int = 0
    prefetch: bool = True
    # Decoding parameters for real judges, recorded in provenance (paper checklist asks for the
    # exact decoding parameters). Temperature 0 is the paper's setting; changing it breaks the
    # deterministic-verdict assumption the cache relies on.
    temperature: float = 0.0
    max_tokens: int = 64
    model_revision: str = ""   # pinned HF revision; empty = default branch (recorded as unpinned)
    # Additional *external system* baselines to run head-to-head alongside the verdict-space
    # pipelines: names from ``_EXTERNAL_BASELINE_CLASSES`` (re-implementations of AutoDefense /
    # SecAlign / StruQ in ``baselines/llm_baselines.py``). Empty by default, so the historical
    # method set is unchanged. Each one costs real judge calls per payload (analyzer + judges +
    # coordinator) and is NOT injectable with committee-level Byzantine verdicts -- its rows
    # describe the uncompromised system and must be labelled as such.
    external_baselines: tuple[str, ...] = ()
    #: Per-baseline overrides forwarded into ``ExternalBaselineConfig.extra``, e.g.
    #: ``{"secalign": {"checkpoint": "facebook/Meta-Llama-3-8B-SecAlign"}}``. This is where the
    #: verified checkpoint id / prompt template of a re-implemented external system is pinned.
    external_baseline_kwargs: dict = field(default_factory=dict)
    # Model-based cost parameters (Section 9; RQ5). All default to 0.0 = "not configured", in
    # which case the cost columns are written as NaN rather than a misleading zero. These are
    # ANALYTIC parameters, not measurements; measured latency/tokens are reported separately in
    # ``evaluation_cost.csv`` (Mức 2 of docs/runner_metrics_plan.md).
    l_in: float = 0.0              # prompt tokens per judge call
    l_out: float = 0.0             # completion tokens per judge call
    analyze_tokens: float = 0.0    # Tokens_analyze
    judge_latency_model: float = 0.0   # per-judge latency (s) in the cost model
    agg_latency_model: float = 0.0     # aggregation latency (s) in the cost model


# ------------------------------------------------------------------ pipeline construction
class _AegisPipeline(DefensePipeline):
    """Wraps an AegisGate (robust rule) as a DefensePipeline for uniform evaluation."""

    def __init__(self, gate: AegisGate):
        self.gate = gate
        self.name = f"aegis_{gate.config.rule}"

    def decide(self, verdicts, payload, rng) -> DecisionResult:
        return self.gate.adjudicate(verdicts, group=payload.group, payload_id=payload.payload_id, rng=rng)


#: Names accepted in ``TrialConfig.external_baselines`` -> class in ``baselines.llm_baselines``.
_EXTERNAL_BASELINE_CLASSES: dict[str, str] = {
    "autodefense": "AutoDefenseStyleBaseline",
    "secalign": "SecAlignHardenedAdapter",
    "struq": "StruQHardenedAdapter",
}


class _ExternalJudgePipeline(DefensePipeline):
    """Runs an external (non-verdict-space) judge system as a pipeline.

    The system answers the payload itself, so the committee verdicts are ignored: a Byzantine
    compromise of *committee members* is not injectable here. Rows for these pipelines therefore
    describe the system **without** a compromised judge (clean-traffic evasion/over-refusal) and
    must be labelled that way when reported — the verdict-space methods are the ones under attack.
    """

    def __init__(self, adapter, name: str, threshold: float):
        self.adapter = adapter
        self.name = name
        self.threshold = float(threshold)

    def decide(self, verdicts, payload, rng) -> DecisionResult:
        del verdicts
        verdict = self.adapter.predict(payload, rng)
        score = float(verdict.score)
        block = score >= self.threshold
        return DecisionResult(
            mode=GateMode.BLOCK if block else GateMode.ALLOW,
            decision=int(Decision.BLOCK if block else Decision.ALLOW),
            aggregate_score=score,
            threshold=self.threshold,
            rule=self.name,
            group=payload.group,
            payload_id=payload.payload_id,
        )


def build_external_baseline(cfg: TrialConfig, name: str):
    """Construct a re-implemented external baseline from the trial config (lazy import).

    The judge-side controls in ``TrialConfig`` are forwarded through
    ``ExternalBaselineConfig.extra`` so the re-implementation uses the same backbones, cache,
    decoding parameters and residency cap as the committee.
    """
    from aegis_agency.baselines.external_wrappers import ExternalBaselineConfig

    class_name = _EXTERNAL_BASELINE_CLASSES.get(name)
    if class_name is None:
        raise ValueError(
            f"Unknown external baseline {name!r}; known: {sorted(_EXTERNAL_BASELINE_CLASSES)}."
        )
    module = importlib.import_module("aegis_agency.baselines.llm_baselines")
    cls = getattr(module, class_name, None)
    if cls is None:  # pragma: no cover - defensive
        raise RuntimeError(f"{class_name} is missing from aegis_agency.baselines.llm_baselines.")
    extra = {
        "backbones": list(cfg.models),
        "cache_path": cfg.verdict_cache,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "model_revision": cfg.model_revision,
        "max_resident_engines": cfg.max_resident_engines,
        "threshold": cfg.threshold,
    }
    extra.update(cfg.external_baseline_kwargs.get(name, {}))
    return cls(ExternalBaselineConfig(name=name, extra=extra))


def build_pipelines(cfg: TrialConfig) -> dict[str, DefensePipeline]:
    """Construct the Aegis methods and baselines to compare."""
    pipelines: dict[str, DefensePipeline] = {}
    for rule in cfg.rules:
        if rule == "krum":
            # Krum's neighbour count uses a *design* fault budget that must satisfy
            # n - f_design - 2 >= 1 (and ideally 2*f_design + 2 < n). This is distinct from
            # the actual number of Byzantine judges the attack injects (cfg.f). A committee
            # of n < 3 cannot run Krum at all, so we skip it with a warning.
            if cfg.n_judges < 3:
                logger.warning("Skipping Krum: n=%d < 3 cannot form a Krum committee.", cfg.n_judges)
                continue
            f_design = max(0, min(cfg.f, (cfg.n_judges - 3) // 2))
        else:
            f_design = cfg.f
        gate = AegisGate(
            CommitteeConfig(
                n_judges=cfg.n_judges,
                rule=rule,
                assumed_f=f_design,
                threshold=cfg.threshold,
                escalate_band=cfg.escalate_band,
            )
        )
        pipelines[f"aegis_{rule}"] = _AegisPipeline(gate)
    # Baselines.
    pipelines["autodefense"] = AutoDefensePipeline(threshold=cfg.threshold)
    pipelines["single_model"] = SingleModelPipeline(threshold=cfg.threshold)
    pipelines["majority_vote"] = MajorityVotePipeline(n_judges=cfg.n_judges, threshold=cfg.threshold)
    pipelines["no_defense"] = NoDefensePipeline()
    # Optional head-to-head against re-implemented external systems (off by default).
    for name in cfg.external_baselines:
        adapter = build_external_baseline(cfg, name)
        pipelines[name] = _ExternalJudgePipeline(adapter, name, cfg.threshold)
    return pipelines


# ------------------------------------------------------------------- committee + attack
def build_honest_committee(cfg: TrialConfig):
    """Build the honest-verdict source selected by ``cfg.judge_backend``.

    Returns a :class:`SyntheticJudgePopulation` (default, offline) or a
    :class:`RealJudgeCommittee` (vLLM on the GPU server). Both expose
    ``generate_honest(payload, rng) -> list[Verdict]``, so the trial loop is backend-agnostic.
    """
    if cfg.judge_backend in ("", "synthetic"):
        return SyntheticJudgePopulation(
            n_judges=cfg.n_judges,
            margin=cfg.margin,
            radius=cfg.radius,
            correlation=cfg.correlation,
            embedding_dim=cfg.embedding_dim,
            threshold=cfg.threshold,
        )
    if cfg.judge_backend == "vllm":
        from aegis_agency.judges.llm_judge import RealJudgeCommittee

        return RealJudgeCommittee(
            models=list(cfg.models) if cfg.models else None,
            n_judges=cfg.n_judges,
            threshold=cfg.threshold,
            cache_path=cfg.verdict_cache,
            isolation=cfg.isolation,
            max_resident_engines=cfg.max_resident_engines,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            model_revision=cfg.model_revision,
        )
    raise ValueError(f"Unknown judge_backend: {cfg.judge_backend!r} (expected 'synthetic' or 'vllm').")


def prefetch_committee_verdicts(committee, payloads: Sequence[Payload], cfg: TrialConfig) -> int:
    """Collect honest verdicts for ``payloads`` up front, when the backend supports it.

    Real-judge committees implement ``prefetch_honest``, which serves every (judge, payload)
    pair in one backbone session (releasing it afterwards when ``max_resident_engines == 1``)
    and skips backbones whose verdicts are all cached. Synthetic committees have no such method
    and are left completely untouched, so the synthetic path -- including its RNG stream -- is
    unchanged. Returns the number of new judge calls made (0 for synthetic or fully cached).
    """
    prefetch = getattr(committee, "prefetch_honest", None)
    if not cfg.prefetch or prefetch is None:
        return 0
    return int(prefetch(payloads))


def simulate_honest_committee(payload: Payload, committee, rng: np.random.Generator) -> list[Verdict]:
    """Honest committee verdicts for a payload (backend-agnostic)."""
    return committee.generate_honest(payload, rng)


def apply_attack(
    honest: Sequence[Verdict],
    payload: Payload,
    cfg: TrialConfig,
    rng: np.random.Generator,
    *,
    rule: str | None = None,
    threshold: float | None = None,
) -> list[Verdict]:
    """Apply the configured attack, producing the tampered committee.

    ``rule``/``threshold`` are only used by attacks that declare ``per_rule = True`` (RQ3): such
    an attacker optimises against the *specific* aggregation rule and the pipeline's current
    decision threshold, so :func:`run_trial` instantiates it once per method. Attacks with
    ``per_rule = False`` (everything else) ignore them and keep their previous behaviour exactly.
    """
    if cfg.attack == "none":
        return list(honest)
    attack_cls = ATTACKS[cfg.attack]
    if attack_cls is None:  # 'none' handled above; guard for type-checkers
        return list(honest)
    kwargs = dict(cfg.attack_kwargs)
    if cfg.attack == "collusion" and "radius" not in kwargs:
        kwargs["radius"] = cfg.radius
    if rule is not None and getattr(attack_cls, "per_rule", False):
        # The pipeline's own rule/threshold win over anything in attack_kwargs: optimising
        # against a different rule would not be the adaptive attack the paper describes.
        kwargs["rule"] = rule
        if threshold is not None:
            kwargs["threshold"] = float(threshold)
    attack = attack_cls(**kwargs)
    return attack.apply(honest, payload, cfg.f, rng)


# ------------------------------------------------------------------- metric/CSV row helpers
def _rolled_decision(res: DecisionResult) -> int:
    """Collapse a gate result to a binary decision under the default escalate policy.

    Mirrors :func:`aegis_agency.metrics.metrics._effective_decision` with
    ``escalate_as="block"`` (the default used by every metric call here) so that the Wilson
    intervals count exactly the same items the metric divides. Cross-checked against the public
    metric in ``tests/test_metrics_export.py``.
    """
    return int(Decision.BLOCK) if res.mode is GateMode.ESCALATE else int(res.decision)


def _rate_counts(
    results: Sequence[DecisionResult],
    labels: Sequence[int],
    *,
    positive_label: int,
    counted_decision: int,
) -> tuple[int, int]:
    """Return ``(k, n)`` for ``P(counted_decision | y* == positive_label)``.

    Used only to feed :func:`wilson_interval`; the reported point estimate still comes from the
    public metric functions so its value is unchanged.
    """
    k = n = 0
    for res, y in zip(results, labels):
        if int(y) != positive_label:
            continue
        n += 1
        if _rolled_decision(res) == counted_decision:
            k += 1
    return k, n


def _escalation_rate(results: Sequence[DecisionResult]) -> float:
    """Fraction of payloads the gate routed to the escalation branch (Algorithm 1, line 11)."""
    if not results:
        return NAN
    return sum(1 for r in results if r.mode is GateMode.ESCALATE) / len(results)


def _judge_count(method: str, n_judges: int) -> int:
    """Number of judge calls a pipeline makes, for the model-based cost columns."""
    return _METHOD_JUDGE_COUNT.get(method, n_judges)


def _model_cost(method: str, cfg: TrialConfig) -> dict[str, float]:
    """Analytic (Section 9) token/latency columns for a method.

    Returns NaN for a column whose parameters are not configured, so an unconfigured cost model
    never masquerades as a measured zero.
    """
    n = _judge_count(method, cfg.n_judges)
    tokens_configured = bool(cfg.l_in or cfg.l_out or cfg.analyze_tokens)
    latency_configured = bool(cfg.judge_latency_model or cfg.agg_latency_model)

    tokens = NAN
    if n >= 1 and tokens_configured:
        tokens = float(token_cost(n, cfg.l_in, cfg.l_out, cfg.analyze_tokens))

    lat_p = lat_s = NAN
    if n >= 1 and latency_configured:
        latencies = [float(cfg.judge_latency_model)] * n
        lat_p = float(latency_parallel(latencies, cfg.agg_latency_model))
        lat_s = float(latency_sequential(latencies, cfg.agg_latency_model))

    return {
        "tokens_model": tokens,
        "latency_parallel_model": lat_p,
        "latency_sequential_model": lat_s,
    }


def derive_metrics(
    results: Sequence[DecisionResult], labels: Sequence[int], cfg: TrialConfig
) -> dict:
    """Extra per-method metrics added by docs/runner_metrics_plan.md (Mức 1).

    Computed from the same ``results``/``labels`` the core metrics use, so the historical three
    metrics keep their exact values.
    """
    k_asr, n_asr = _rate_counts(
        results, labels, positive_label=int(Decision.BLOCK), counted_decision=int(Decision.ALLOW)
    )
    k_orr, n_orr = _rate_counts(
        results, labels, positive_label=int(Decision.ALLOW), counted_decision=int(Decision.BLOCK)
    )
    asr_lo, asr_hi = wilson_interval(k_asr, n_asr)
    orr_lo, orr_hi = wilson_interval(k_orr, n_orr)
    groups = group_conditional_asr(results, labels)
    return {
        "utility_retention": utility_retention(results, labels),
        "escalation_rate": _escalation_rate(results),
        "asr_uc_ci_low": asr_lo,
        "asr_uc_ci_high": asr_hi,
        "orr_ci_low": orr_lo,
        "orr_ci_high": orr_hi,
        "group_asr_json": json.dumps(_jsonable(groups), sort_keys=True) if len(groups) > 1 else "",
    }


def make_sweep_row(
    *, f: int, n_judges: int, method: str, seed: int, metrics: dict
) -> dict:
    """Build one sweep row with exactly :data:`SWEEP_COLUMNS` (stable order, no missing keys).

    ``utils.io.write_csv`` uses the first row's keys as the header, so every row must carry the
    same keys; absent metrics are filled with NaN.
    """
    values = {"f": f, "n_judges": n_judges, "method": method, "seed": seed}
    return {k: values.get(k, metrics.get(k, NAN)) for k in SWEEP_COLUMNS}


def make_ablation_row(*, ablation: str, method: str, seed: int, metrics: dict) -> dict:
    """Build one ablation row: historical prefix then the appended metric columns."""
    row: dict = {"ablation": ablation, "method": method}
    for k in SWEEP_COLUMNS:
        if k in {"f", "n_judges", "method", "seed"}:
            continue
        row[k] = metrics.get(k, NAN)
    row["seed"] = seed
    return row


def summarize_sweep_rows(rows: Sequence[dict], metrics: Sequence[str] = SUMMARY_METRICS) -> list[dict]:
    """Aggregate sweep rows across seeds: mean/std (and the seeds used) per (n_judges, f, method).

    Std is the sample std (ddof=1) and is NaN when fewer than two seeds contribute a value.
    Per-seed Wilson intervals already live in ``evaluation_sweep.csv``; this file reports the
    cross-seed dispersion the paper's reproducibility checklist asks for.
    """
    grouped: dict[tuple[int, int, str], list[dict]] = {}
    for r in rows:
        grouped.setdefault((int(r["n_judges"]), int(r["f"]), str(r["method"])), []).append(r)

    out: list[dict] = []
    for (n_judges, f, method), group in sorted(grouped.items(), key=lambda kv: (kv[0][1], kv[0][2])):
        seeds = [int(r["seed"]) for r in group if r.get("seed") is not None]
        row: dict = {
            "n_judges": n_judges,
            "f": f,
            "method": method,
            "n_seeds": len(group),
            "seeds": ",".join(str(s) for s in seeds),
        }
        for m in metrics:
            vals = np.array([float(r[m]) for r in group if _is_finite(r.get(m))], dtype=float)
            row[f"{m}_mean"] = float(vals.mean()) if vals.size else NAN
            row[f"{m}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else NAN
        out.append(row)
    return out


def _is_finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _jsonable(values: dict[str, float]) -> dict[str, float | None]:
    """Replace non-finite values with ``null`` so the emitted JSON is strictly valid."""
    return {k: (float(v) if _is_finite(v) else None) for k, v in values.items()}


# --------------------------------------------------------------------------------- payloads
def _benchmark_root(cfg: TrialConfig) -> Path:
    """Resolve the benchmark directory containing ``<split>.csv``.

    Accepts either ``data_root`` pointing directly at a benchmark dir (``data/benchmarks/
    formal``) or at the parent of benchmark dirs when ``benchmark`` is also set
    (``data_root=data/benchmarks`` + ``benchmark=formal``).
    """
    root = Path(cfg.data_root)
    if cfg.benchmark and (root / cfg.benchmark).is_dir():
        return root / cfg.benchmark
    return root


def data_source_tag(cfg: TrialConfig) -> str:
    """Provenance tag describing where the payloads come from."""
    if cfg.data_root:
        return f"benchmark:{cfg.benchmark or _benchmark_root(cfg).name}"
    return "synthetic"


def load_payloads(cfg: TrialConfig, rng: np.random.Generator) -> tuple[list[Payload], str]:
    """Load evaluation payloads, returning ``(payloads, source_tag)``.

    When ``cfg.data_root`` is set the payloads come from a real benchmark on disk via
    :class:`CsvBenchmarkAdapter`; otherwise synthetic payloads are generated. The payload
    list is deterministically subsampled (seeded by ``cfg.seed`` via ``rng``) to
    ``cfg.n_payloads`` and shuffled, so calibration/evaluation splits are reproducible.
    """
    if cfg.data_root:
        from aegis_agency.data.adapters import CsvBenchmarkAdapter

        root = _benchmark_root(cfg)
        adapter = CsvBenchmarkAdapter(root=root, split=cfg.data_split)
        payloads = list(adapter.iter_payloads())
        if not payloads:
            raise ValueError(f"Benchmark {root} produced no payloads for split='{cfg.data_split}'.")
        k = min(cfg.n_payloads, len(payloads))
        if k < 1:
            raise ValueError("n_payloads must be >= 1.")
        idx = rng.permutation(len(payloads))[:k]
        payloads = [payloads[i] for i in idx]
        return payloads, data_source_tag(cfg)
    return generate_payloads(cfg.n_payloads, rng, unsafe_fraction=cfg.unsafe_fraction), "synthetic"


# --------------------------------------------------------------------------------- trial
def run_trial(cfg: TrialConfig) -> dict:
    """Run one trial (synthetic or real-judge); return per-method metrics and summary."""
    rng = np.random.default_rng(cfg.seed)
    committee = build_honest_committee(cfg)
    payloads, data_source = load_payloads(cfg, rng)
    # Split calibration / evaluation.
    n_cal = max(1, cfg.n_payloads // 4)
    cal_payloads, eval_payloads = payloads[:n_cal], payloads[n_cal:]

    pipelines = build_pipelines(cfg)

    # Real-judge backends collect every honest verdict up front (one backbone session, engines
    # released afterwards when max_resident_engines == 1). Synthetic committees are untouched.
    new_calls = prefetch_committee_verdicts(committee, payloads, cfg)
    if new_calls:
        logger.info("Prefetched %d honest judge calls for %d payloads.", new_calls, len(payloads))

    # ---- calibration on honest (no-attack) verdicts, target over-refusal ----
    # Decision-only pipelines (majority vote) and the trivial no-defense pipeline emit a
    # binary "score" that is already a decision; calibrating a continuous threshold on {0,1}
    # is degenerate, so they keep their fixed tau = 0.5.
    thresholds: dict[str, float] = {}
    if cfg.calibrate:
        for name, pipe in pipelines.items():
            if name in {"no_defense", "majority_vote"}:
                continue
            cal_scores, cal_labels = [], []
            for p in cal_payloads:
                honest = simulate_honest_committee(p, committee, rng)
                res = pipe.decide(honest, p, rng)
                cal_scores.append(res.aggregate_score)
                cal_labels.append(p.true_label)
            cal = calibrate_threshold(
                np.array(cal_scores), np.array(cal_labels), objective="target_orr", target=cfg.target_orr
            )
            thresholds[name] = cal.threshold
            _set_threshold(pipe, cal.threshold)

    # ---- evaluation under attack ----
    # An attack with per_rule=True optimises against the rule (and current threshold) of each
    # pipeline, so it is re-instantiated per method; every other attack is applied exactly as
    # before, once per payload, keeping the previous RNG stream.
    per_rule_attack = bool(getattr(ATTACKS.get(cfg.attack), "per_rule", False))
    per_method: dict[str, dict] = {}
    for name, pipe in pipelines.items():
        results: list[DecisionResult] = []
        labels: list[int] = []
        all_outlier: list[float] = []
        all_byz: list[int] = []
        rule = method_rule(name)
        threshold = _get_threshold(pipe, cfg.threshold)
        # no_defense releases everything regardless of the verdicts, so attacking it would only
        # burn search time without changing a single decision. External systems answer the payload
        # themselves (no verdict interface), so a committee-level attack cannot be injected.
        target_rule = (
            per_rule_attack and name != "no_defense" and not isinstance(pipe, _ExternalJudgePipeline)
        )
        for p in eval_payloads:
            honest = simulate_honest_committee(p, committee, rng)
            tampered = apply_attack(
                honest,
                p,
                cfg,
                rng,
                rule=rule if target_rule else None,
                threshold=threshold if target_rule else None,
            )
            res = pipe.decide(tampered, p, rng)
            results.append(res)
            labels.append(p.true_label)
            if res.per_judge_outlier is not None:
                all_outlier.extend(res.per_judge_outlier.tolist())
                all_byz.extend([int(v.is_byzantine) for v in tampered])
        metrics = {
            "asr_uc": asr_under_compromise(results, labels),
            "orr": over_refusal_rate(results, labels),
            "defense_success_rate": defense_success_rate(results, labels),
            "threshold": thresholds.get(name, cfg.threshold),
        }
        # Extra metrics (docs/runner_metrics_plan.md, Mức 1). The three core metrics above keep
        # their exact values; everything below is derived from the same results/labels.
        metrics.update(derive_metrics(results, labels, cfg))
        if name.startswith("aegis_"):
            metrics["byzantine_tolerance_fraction"] = byzantine_tolerance_fraction(
                cfg.n_judges, name.split("_", 1)[1]
            )
        metrics.update(_model_cost(name, cfg))
        if all_byz and any(all_byz) and not all(all_byz):
            det = malicious_verdict_detection(np.array(all_outlier), np.array(all_byz))
            metrics["detection_auroc"] = det["auroc"]
            metrics["detection_f1"] = det["f1"]
        per_method[name] = metrics

    return {
        "config": _config_to_dict(cfg),
        "n_eval": len(eval_payloads),
        "methods": per_method,
        "data_source": data_source,
        "is_paper_result": False,
        # Measured (not modelled) judge cost for this trial; None when the backend does not
        # record per-call latency/tokens (e.g. the synthetic committee).
        "cost_measured": _measured_cost(committee),
        # Per-payload measured latency (parallel = max_k, sequential = sum_k over the judge
        # calls actually served) and token counts; empty when nothing was called.
        "cost_payloads": _measured_payload_cost(committee),
    }


def sweep_colluding_fraction(
    cfg: TrialConfig,
    f_values: Sequence[int],
    *,
    cost_rows: list[dict] | None = None,
    cost_payload_rows: list[dict] | None = None,
) -> list[dict]:
    """Run trials over a sweep of Byzantine counts f (RQ1).

    Parameters
    ----------
    cost_rows : list, optional
        When provided, measured judge cost for each (seed, f) trial is appended here (used by
        :func:`aegis_agency.experiments.run_evaluation.run_evaluation` to write
        ``evaluation_cost.csv``). Backends that do not record call stats contribute nothing.
    cost_payload_rows : list, optional
        When provided, measured per-payload latency/token rows for each (seed, f) trial are
        appended here (written to ``evaluation_cost_payloads.csv``). ``max_latency_s`` and
        ``sum_latency_s`` are the paper's parallel / sequential latencies measured on the judge
        calls actually served.
    """
    rows: list[dict] = []
    for f in f_values:
        trial_cfg = _replace(cfg, f=f)
        out = run_trial(trial_cfg)
        for method, m in out["methods"].items():
            rows.append(
                make_sweep_row(f=f, n_judges=cfg.n_judges, method=method, seed=cfg.seed, metrics=m)
            )
        measured = out.get("cost_measured")
        if cost_rows is not None and measured:
            cost_rows.append({"seed": cfg.seed, "f": f, "n_judges": cfg.n_judges, **measured})
        payload_costs = out.get("cost_payloads") or {}
        if cost_payload_rows is not None:
            for payload_id, cost in sorted(payload_costs.items()):
                cost_payload_rows.append(
                    {
                        "seed": cfg.seed,
                        "f": f,
                        "n_judges": cfg.n_judges,
                        "payload_id": payload_id,
                        **cost,
                    }
                )
    return rows


def _measured_cost(committee) -> dict | None:
    """Measured per-call statistics from a committee that records them, else ``None``."""
    collector = getattr(committee, "collect_call_stats", None)
    if collector is None:
        return None
    return collector()


def _measured_payload_cost(committee) -> dict[str, dict]:
    """Measured per-payload latency/token rows from a committee that records them."""
    collector = getattr(committee, "per_payload_latency", None)
    if collector is None:
        return {}
    return collector()


# ------------------------------------------------------------------------------- helpers
def _set_threshold(pipe: DefensePipeline, tau: float) -> None:
    if isinstance(pipe, _AegisPipeline):
        pipe.gate.config.threshold = tau
    elif isinstance(pipe, MajorityVotePipeline):
        pipe.gate.config.threshold = tau
    elif hasattr(pipe, "threshold"):
        pipe.threshold = tau  # type: ignore[attr-defined]


def _get_threshold(pipe: DefensePipeline, default: float) -> float:
    """Current decision threshold of a pipeline (post-calibration), for per-rule attacks."""
    if isinstance(pipe, _AegisPipeline):
        return float(pipe.gate.config.threshold)
    if isinstance(pipe, MajorityVotePipeline):
        return float(pipe.gate.config.threshold)
    threshold = getattr(pipe, "threshold", None)
    return float(threshold) if threshold is not None else float(default)


def _config_to_dict(cfg: TrialConfig) -> dict:
    return {
        "n_judges": cfg.n_judges,
        "rules": list(cfg.rules),
        "margin": cfg.margin,
        "radius": cfg.radius,
        "correlation": cfg.correlation,
        "embedding_dim": cfg.embedding_dim,
        "attack": cfg.attack,
        "f": cfg.f,
        "n_payloads": cfg.n_payloads,
        "seed": cfg.seed,
        "data_root": cfg.data_root,
        "benchmark": cfg.benchmark,
        "data_split": cfg.data_split,
        "judge_backend": cfg.judge_backend,
        "models": list(cfg.models),
        "isolation": cfg.isolation,
        "max_resident_engines": cfg.max_resident_engines,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "model_revision": cfg.model_revision or "<unpinned>",
        "external_baselines": list(cfg.external_baselines),
        "l_in": cfg.l_in,
        "l_out": cfg.l_out,
        "analyze_tokens": cfg.analyze_tokens,
        "judge_latency_model": cfg.judge_latency_model,
        "agg_latency_model": cfg.agg_latency_model,
    }


def _replace(cfg: TrialConfig, **changes) -> TrialConfig:
    from dataclasses import replace

    return replace(cfg, **changes)

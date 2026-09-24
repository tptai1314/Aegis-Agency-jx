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

from dataclasses import dataclass, field
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
from aegis_agency.data.schemas import CommitteeConfig, DecisionResult, Payload, Verdict
from aegis_agency.data.synthetic import generate_payloads
from aegis_agency.judges.synthetic_judges import SyntheticJudgePopulation
from aegis_agency.methods.calibration import calibrate_threshold
from aegis_agency.methods.gate import AegisGate
from aegis_agency.metrics.metrics import (
    asr_under_compromise,
    defense_success_rate,
    malicious_verdict_detection,
    over_refusal_rate,
)
from aegis_agency.utils.logging import get_logger

logger = get_logger(__name__)


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


# ------------------------------------------------------------------ pipeline construction
class _AegisPipeline(DefensePipeline):
    """Wraps an AegisGate (robust rule) as a DefensePipeline for uniform evaluation."""

    def __init__(self, gate: AegisGate):
        self.gate = gate
        self.name = f"aegis_{gate.config.rule}"

    def decide(self, verdicts, payload, rng) -> DecisionResult:
        return self.gate.adjudicate(verdicts, group=payload.group, payload_id=payload.payload_id, rng=rng)


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
    return pipelines


# ------------------------------------------------------------------- committee + attack
def simulate_honest_committee(
    payload: Payload, pop: SyntheticJudgePopulation, rng: np.random.Generator
) -> list[Verdict]:
    """Honest committee verdicts for a payload."""
    return pop.generate_honest(payload, rng)


def apply_attack(
    honest: Sequence[Verdict], payload: Payload, cfg: TrialConfig, rng: np.random.Generator
) -> list[Verdict]:
    """Apply the configured attack, producing the tampered committee."""
    if cfg.attack == "none":
        return list(honest)
    attack_cls = ATTACKS[cfg.attack]
    if attack_cls is None:  # 'none' handled above; guard for type-checkers
        return list(honest)
    kwargs = dict(cfg.attack_kwargs)
    if cfg.attack == "collusion" and "radius" not in kwargs:
        kwargs["radius"] = cfg.radius
    attack = attack_cls(**kwargs)
    return attack.apply(honest, payload, cfg.f, rng)


# --------------------------------------------------------------------------------- trial
def run_trial(cfg: TrialConfig) -> dict:
    """Run one synthetic trial; return per-method metrics and provenance-friendly summary."""
    rng = np.random.default_rng(cfg.seed)
    pop = SyntheticJudgePopulation(
        n_judges=cfg.n_judges,
        margin=cfg.margin,
        radius=cfg.radius,
        correlation=cfg.correlation,
        embedding_dim=cfg.embedding_dim,
        threshold=cfg.threshold,
    )
    payloads = generate_payloads(cfg.n_payloads, rng, unsafe_fraction=cfg.unsafe_fraction)
    # Split calibration / evaluation.
    n_cal = max(1, cfg.n_payloads // 4)
    cal_payloads, eval_payloads = payloads[:n_cal], payloads[n_cal:]

    pipelines = build_pipelines(cfg)

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
                honest = simulate_honest_committee(p, pop, rng)
                res = pipe.decide(honest, p, rng)
                cal_scores.append(res.aggregate_score)
                cal_labels.append(p.true_label)
            cal = calibrate_threshold(
                np.array(cal_scores), np.array(cal_labels), objective="target_orr", target=cfg.target_orr
            )
            thresholds[name] = cal.threshold
            _set_threshold(pipe, cal.threshold)

    # ---- evaluation under attack ----
    per_method: dict[str, dict] = {}
    for name, pipe in pipelines.items():
        results: list[DecisionResult] = []
        labels: list[int] = []
        all_outlier: list[float] = []
        all_byz: list[int] = []
        for p in eval_payloads:
            honest = simulate_honest_committee(p, pop, rng)
            tampered = apply_attack(honest, p, cfg, rng)
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
        if all_byz and any(all_byz) and not all(all_byz):
            det = malicious_verdict_detection(np.array(all_outlier), np.array(all_byz))
            metrics["detection_auroc"] = det["auroc"]
            metrics["detection_f1"] = det["f1"]
        per_method[name] = metrics

    return {
        "config": _config_to_dict(cfg),
        "n_eval": len(eval_payloads),
        "methods": per_method,
        "data_source": "synthetic",
        "is_paper_result": False,
    }


def sweep_colluding_fraction(
    cfg: TrialConfig, f_values: Sequence[int]
) -> list[dict]:
    """Run trials over a sweep of Byzantine counts f (RQ1)."""
    rows: list[dict] = []
    for f in f_values:
        trial_cfg = _replace(cfg, f=f)
        out = run_trial(trial_cfg)
        for method, m in out["methods"].items():
            rows.append(
                {
                    "f": f,
                    "n_judges": cfg.n_judges,
                    "method": method,
                    "asr_uc": m["asr_uc"],
                    "orr": m["orr"],
                    "defense_success_rate": m["defense_success_rate"],
                }
            )
    return rows


# ------------------------------------------------------------------------------- helpers
def _set_threshold(pipe: DefensePipeline, tau: float) -> None:
    if isinstance(pipe, _AegisPipeline):
        pipe.gate.config.threshold = tau
    elif isinstance(pipe, MajorityVotePipeline):
        pipe.gate.config.threshold = tau
    elif hasattr(pipe, "threshold"):
        pipe.threshold = tau  # type: ignore[attr-defined]


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
    }


def _replace(cfg: TrialConfig, **changes) -> TrialConfig:
    from dataclasses import replace

    return replace(cfg, **changes)

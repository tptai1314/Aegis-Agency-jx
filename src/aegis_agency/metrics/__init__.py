"""Evaluation metrics, theoretical checks, confidence intervals, and cost models."""

from aegis_agency.metrics.confidence_intervals import bootstrap_interval, wilson_interval
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
from aegis_agency.metrics.theory import (
    c_alpha,
    correlated_variance,
    correlated_variance_floor,
    displacement_holds,
    injection_flip_bound,
    integrity_condition,
)

__all__ = [
    "asr_under_compromise",
    "over_refusal_rate",
    "defense_success_rate",
    "byzantine_tolerance_fraction",
    "malicious_verdict_detection",
    "utility_retention",
    "group_conditional_asr",
    "c_alpha",
    "integrity_condition",
    "displacement_holds",
    "correlated_variance",
    "correlated_variance_floor",
    "injection_flip_bound",
    "wilson_interval",
    "bootstrap_interval",
    "token_cost",
    "latency_parallel",
    "latency_sequential",
]

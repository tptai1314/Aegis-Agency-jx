"""Adaptive attack models against the defense pipeline (Section 5.3, Section 10)."""

from aegis_agency.attacks.adaptive import AdaptiveAggregationAttack, AdaptiveSearchAttack
from aegis_agency.attacks.base import Attack
from aegis_agency.attacks.collusion import CollusionAttack
from aegis_agency.attacks.compromise import CompromisedJudgeAttack
from aegis_agency.attacks.injection import SecondOrderInjectionAttack

ATTACKS = {
    "none": None,
    "compromise": CompromisedJudgeAttack,
    "collusion": CollusionAttack,
    "injection": SecondOrderInjectionAttack,
    "adaptive": AdaptiveAggregationAttack,
    "adaptive_search": AdaptiveSearchAttack,
}

__all__ = [
    "Attack",
    "CompromisedJudgeAttack",
    "CollusionAttack",
    "SecondOrderInjectionAttack",
    "AdaptiveAggregationAttack",
    "AdaptiveSearchAttack",
    "ATTACKS",
]

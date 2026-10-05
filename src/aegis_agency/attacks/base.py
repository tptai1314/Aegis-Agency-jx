"""Abstract attack interface.

An attack takes an honest committee's verdicts and returns a tampered committee containing
up to ``f`` Byzantine verdicts (compromise/collusion/adaptive) or perturbed honest verdicts
(second-order injection). The attacker's goal is to flip the aggregate decision to the wrong
side: ``target_decision = 1 - payload.true_label``.
"""

from __future__ import annotations

import abc
from typing import Sequence

import numpy as np

from aegis_agency.data.schemas import Decision, Payload, Verdict


class Attack(abc.ABC):
    """Base class for adaptive attacks on the pipeline.

    ``per_rule`` marks an attack that optimises against the *specific* aggregation rule
    (Section 5.3 capability iv). The harness then instantiates it once per pipeline, passing
    that pipeline's ``rule`` and current decision ``threshold``; such an attack's constructor
    must accept both keyword arguments. Attacks that leave it False are applied once per payload
    exactly as before.
    """

    name: str = "attack"
    per_rule: bool = False

    @staticmethod
    def target_decision(payload: Payload) -> int:
        """The decision the attacker wants the gate to output (the wrong one)."""
        return int(Decision.ALLOW) if payload.true_label == Decision.BLOCK else int(Decision.BLOCK)

    @abc.abstractmethod
    def apply(
        self,
        honest: Sequence[Verdict],
        payload: Payload,
        f: int,
        rng: np.random.Generator,
    ) -> list[Verdict]:
        """Return a tampered committee of the same size as ``honest``."""
        raise NotImplementedError

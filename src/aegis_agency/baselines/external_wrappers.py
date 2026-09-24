"""Adapter stubs for external systems that require real code, weights, or benchmarks.

These wrappers define the *exact* expected input/output schema so that, on EC2, a real
implementation can be dropped in behind a stable interface. They never fabricate a result:
calling :meth:`predict` on a stub raises ``NotImplementedError`` with a pointer to the setup
required. A :class:`DummyExternalBaseline` is provided for pipeline/unit tests only and is
clearly labelled as non-real.

External systems referenced by the paper:
* AutoDefense (Zeng et al., 2024)  -- multi-agent Coordinator system.
* SecAlign   (Chen et al., 2025)   -- preference-optimised hardened model.
* StruQ      (Chen et al., 2024)   -- structured-query hardened model.
* JudgeDeceiver (Shi et al., 2024) -- optimisation-based judge injection.

See docs/baseline_adapters.md and TODO_IMPLEMENTATION.md for setup instructions.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

import numpy as np

from aegis_agency.data.schemas import Payload, Verdict


@dataclass
class ExternalBaselineConfig:
    """Configuration a real external baseline needs (paths/endpoints supplied on EC2)."""

    name: str
    model_path_or_endpoint: str = ""  # local checkpoint dir or API endpoint
    extra: dict[str, Any] | None = None


class ExternalJudgeAdapter(abc.ABC):
    """Interface a real external judge/system must implement.

    Expected schema
    ---------------
    Input : Payload (payload_id, content, group, metadata).
    Output: Verdict (decision in {0,1}, score in [0,1], embedding in R^m).
    """

    def __init__(self, config: ExternalBaselineConfig):
        self.config = config

    @abc.abstractmethod
    def predict(self, payload: Payload, rng: np.random.Generator) -> Verdict:
        raise NotImplementedError


class AutoDefenseAdapter(ExternalJudgeAdapter):
    """Stub for the real AutoDefense system (external repo + LLM backbones required)."""

    def predict(self, payload: Payload, rng: np.random.Generator) -> Verdict:  # pragma: no cover
        raise NotImplementedError(
            "AutoDefenseAdapter is a stub. To use the real system on EC2: clone the "
            "AutoDefense repository, configure the Coordinator/analyzer/judge LLM backbones, "
            "and implement predict() to return a Verdict. See docs/baseline_adapters.md."
        )


class SecAlignAdapter(ExternalJudgeAdapter):
    """Stub for a real SecAlign-hardened judge (checkpoint required)."""

    def predict(self, payload: Payload, rng: np.random.Generator) -> Verdict:  # pragma: no cover
        raise NotImplementedError(
            "SecAlignAdapter is a stub. Provide the SecAlign checkpoint path in "
            "config.model_path_or_endpoint and implement predict(). See docs/baseline_adapters.md."
        )


class StruQAdapter(ExternalJudgeAdapter):
    """Stub for a real StruQ-hardened judge (checkpoint required)."""

    def predict(self, payload: Payload, rng: np.random.Generator) -> Verdict:  # pragma: no cover
        raise NotImplementedError(
            "StruQAdapter is a stub. Provide the StruQ checkpoint and implement predict(). "
            "See docs/baseline_adapters.md."
        )


class JudgeDeceiverAdapter(abc.ABC):
    """Interface for a real second-order injection generator (JudgeDeceiver).

    Expected schema: given a Payload, return a new Payload whose content carries an injected
    instruction crafted to flip the judges' verdicts.
    """

    def __init__(self, config: ExternalBaselineConfig):
        self.config = config

    @abc.abstractmethod
    def inject(self, payload: Payload, rng: np.random.Generator) -> Payload:
        raise NotImplementedError


class DummyExternalBaseline(ExternalJudgeAdapter):
    """Deterministic dummy judge for tests ONLY. NOT a real external baseline.

    Returns a fixed verdict derived from the payload's true label so that pipeline plumbing
    can be tested without any model. Never use this to stand in for AutoDefense/SecAlign.
    """

    def predict(self, payload: Payload, rng: np.random.Generator) -> Verdict:
        score = 0.9 if payload.true_label == 1 else 0.1
        return Verdict(decision=payload.true_label, score=score, judge_id=0, is_byzantine=False)

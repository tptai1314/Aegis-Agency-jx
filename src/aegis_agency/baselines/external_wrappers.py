"""Adapters for external systems that require real code, weights, or benchmarks.

These wrappers define the *exact* expected input/output schema so that a real implementation can
be dropped in behind a stable interface. They never fabricate a result: when the required
configuration (a checkpoint id, a backbone) is missing, constructing the adapter raises
``RuntimeError`` with a pointer to what must be supplied, and no verdict is invented.

External systems referenced by the paper:
* AutoDefense (Zeng et al., 2024)  -- multi-agent Coordinator system.
* SecAlign   (Chen et al., 2025)   -- preference-optimised hardened model.
* StruQ      (Chen et al., 2024)   -- structured-query hardened model.
* JudgeDeceiver (Shi et al., 2024) -- optimisation-based judge injection.

AutoDefense, SecAlign and StruQ now delegate to real, configurable **re-implementations** built
on this repository's local vLLM judge machinery
(:mod:`aegis_agency.baselines.llm_baselines`): the wrappers keep their historical names and the
``ExternalJudgeAdapter`` interface, while the prompts, checkpoints and cache namespaces are
documented in that module. A :class:`DummyExternalBaseline` is provided for pipeline/unit tests
only and is clearly labelled as non-real.

See docs/baseline_adapters.md and TODO_IMPLEMENTATION.md for setup instructions.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from aegis_agency.data.schemas import Payload, Verdict

if TYPE_CHECKING:  # pragma: no cover - typing only; the run-time import stays local below
    from types import ModuleType


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


def _llm_baselines() -> ModuleType:
    """Import the re-implementation module lazily.

    Kept local so that importing :mod:`aegis_agency.baselines.external_wrappers` never drags in
    the vLLM judge path eagerly (it is already lazy, but this keeps the dependency direction
    explicit and impossible to turn into an import cycle).
    """
    from aegis_agency.baselines import llm_baselines as module

    return module


class AutoDefenseAdapter(ExternalJudgeAdapter):
    """AutoDefense (Zeng et al., 2024) as analyzer -> judge(s) -> coordinator.

    Delegates to :class:`aegis_agency.baselines.llm_baselines.AutoDefenseStyleBaseline`, which
    runs the published AutoDefense prompts on the local vLLM engine and returns the coordinator's
    verdict. Configure the role backbones (or a default one) through
    ``ExternalBaselineConfig.model_path_or_endpoint`` / ``config.extra``; without any backbone the
    constructor raises ``RuntimeError`` instead of guessing a model.
    """

    def __init__(self, config: ExternalBaselineConfig):
        super().__init__(config)
        self._impl = _llm_baselines().AutoDefenseStyleBaseline(config)

    def predict(self, payload: Payload, rng: np.random.Generator) -> Verdict:
        return self._impl.predict(payload, rng)

    def describe(self) -> dict[str, Any]:
        """Provenance of the delegated re-implementation (roles, sources, caveats)."""
        return self._impl.describe()


class SecAlignAdapter(ExternalJudgeAdapter):
    """SecAlign (Chen et al., 2025) as a single hardened checkpoint behind the local judge.

    Delegates to :class:`aegis_agency.baselines.llm_baselines.SecAlignHardenedAdapter`. The
    checkpoint must be supplied explicitly (``config.model_path_or_endpoint`` or
    ``extra['checkpoint']``): the released third-party checkpoint id is never silently assumed, and
    the prompt template is a re-implementation default that must be verified against the authors'
    repository before any number is reported.
    """

    def __init__(self, config: ExternalBaselineConfig):
        super().__init__(config)
        self._impl = _llm_baselines().SecAlignHardenedAdapter(config)

    def predict(self, payload: Payload, rng: np.random.Generator) -> Verdict:
        return self._impl.predict(payload, rng)

    def describe(self) -> dict[str, Any]:
        """Provenance of the delegated re-implementation (checkpoint, sources, caveats)."""
        return self._impl.describe()


class StruQAdapter(ExternalJudgeAdapter):
    """StruQ (Chen et al., 2024) as a single hardened checkpoint behind the local judge.

    Delegates to :class:`aegis_agency.baselines.llm_baselines.StruQHardenedAdapter`. As with
    SecAlign, the checkpoint must be supplied explicitly and the structured-query template is a
    re-implementation default to verify against the authors' repository.
    """

    def __init__(self, config: ExternalBaselineConfig):
        super().__init__(config)
        self._impl = _llm_baselines().StruQHardenedAdapter(config)

    def predict(self, payload: Payload, rng: np.random.Generator) -> Verdict:
        return self._impl.predict(payload, rng)

    def describe(self) -> dict[str, Any]:
        """Provenance of the delegated re-implementation (checkpoint, sources, caveats)."""
        return self._impl.describe()


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


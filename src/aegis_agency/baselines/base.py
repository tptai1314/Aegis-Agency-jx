"""Abstract defense-pipeline interface shared by baselines and the Aegis gate."""

from __future__ import annotations

import abc
from typing import Sequence

import numpy as np

from aegis_agency.data.schemas import DecisionResult, Payload, Verdict


class DefensePipeline(abc.ABC):
    """A defense pipeline maps committee verdicts + payload to a decision."""

    name: str = "pipeline"

    @abc.abstractmethod
    def decide(
        self,
        verdicts: Sequence[Verdict],
        payload: Payload,
        rng: np.random.Generator,
    ) -> DecisionResult:
        """Return the pipeline's allow/block/escalate decision for the payload."""
        raise NotImplementedError

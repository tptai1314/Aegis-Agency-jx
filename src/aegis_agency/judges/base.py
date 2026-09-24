"""Abstract judge interface.

A judge maps an (isolated) payload to a :class:`~aegis_agency.data.schemas.Verdict`. Real
hardened LLM judges (SecAlign/StruQ-tuned backbones) are reached on EC2 through adapters in
:mod:`aegis_agency.data.adapters`; the synthetic judges here let the mechanism be tested
offline in the paper's verdict space.
"""

from __future__ import annotations

import abc

import numpy as np

from aegis_agency.data.schemas import Payload, Verdict


class JudgeModel(abc.ABC):
    """Abstract base class for a single judge agent."""

    def __init__(self, judge_id: int, backbone: str = "synthetic"):
        self.judge_id = judge_id
        self.backbone = backbone

    @abc.abstractmethod
    def verdict(self, payload: Payload, rng: np.random.Generator) -> Verdict:
        """Return this judge's verdict for ``payload`` (Eq. 1).

        Implementations must respect payload isolation: instructions embedded in the
        payload content must not change control flow (that property is modelled explicitly
        by :class:`~aegis_agency.judges.isolation.IsolationModel`).
        """
        raise NotImplementedError

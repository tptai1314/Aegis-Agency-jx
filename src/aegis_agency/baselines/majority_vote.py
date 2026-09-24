"""Plain majority-vote baseline (decision-only aggregation; Section 6.3, Section 10)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from aegis_agency.baselines.base import DefensePipeline
from aegis_agency.data.schemas import CommitteeConfig, DecisionResult, Payload, Verdict
from aegis_agency.methods.gate import AegisGate


class MajorityVotePipeline(DefensePipeline):
    """Committee gate with rule='majority' (blind to score/rationale manipulation)."""

    name = "majority_vote"

    def __init__(self, n_judges: int, threshold: float = 0.5):
        self.gate = AegisGate(CommitteeConfig(n_judges=n_judges, rule="majority", threshold=threshold))

    def decide(self, verdicts: Sequence[Verdict], payload: Payload, rng: np.random.Generator) -> DecisionResult:
        return self.gate.adjudicate(verdicts, group=payload.group, payload_id=payload.payload_id, rng=rng)

"""Always-submit baseline (no defense): the gate releases everything."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from aegis_agency.baselines.base import DefensePipeline
from aegis_agency.data.schemas import Decision, DecisionResult, GateMode, Payload, Verdict


class NoDefensePipeline(DefensePipeline):
    """Release every payload (upper bound on ASR, lower bound on over-refusal)."""

    name = "no_defense"

    def decide(self, verdicts: Sequence[Verdict], payload: Payload, rng: np.random.Generator) -> DecisionResult:
        return DecisionResult(
            mode=GateMode.ALLOW,
            decision=int(Decision.ALLOW),
            aggregate_score=0.0,
            threshold=0.5,
            rule="no_defense",
            group=payload.group,
            payload_id=payload.payload_id,
        )

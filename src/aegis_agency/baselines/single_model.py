"""Single hardened-model baseline (one judge; SecAlign structural analogue).

Uses only the first judge's verdict. Has no co-defender and no aggregator, so a single
compromised judge flips the decision -- exactly the paper's motivation for a committee.
The *real* SecAlign checkpoint is an external dependency (see external_wrappers.py).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from aegis_agency.baselines.base import DefensePipeline
from aegis_agency.data.schemas import Decision, DecisionResult, GateMode, Payload, Verdict


class SingleModelPipeline(DefensePipeline):
    """Decision from a single judge's score vs threshold tau."""

    name = "single_model"

    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)

    def decide(self, verdicts: Sequence[Verdict], payload: Payload, rng: np.random.Generator) -> DecisionResult:
        if not verdicts:
            raise ValueError("SingleModelPipeline needs at least one verdict.")
        s = float(verdicts[0].score)
        block = s >= self.threshold
        return DecisionResult(
            mode=GateMode.BLOCK if block else GateMode.ALLOW,
            decision=int(Decision.BLOCK if block else Decision.ALLOW),
            aggregate_score=s,
            threshold=self.threshold,
            rule="single_model",
            group=payload.group,
            payload_id=payload.payload_id,
        )

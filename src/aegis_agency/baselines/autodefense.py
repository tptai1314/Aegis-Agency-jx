"""AutoDefense baseline: single-Coordinator committee (Section 2, Section 10).

AutoDefense aggregates judge verdicts through one Coordinator with no Byzantine tolerance.
We model that Coordinator as a (non-robust) mean of the verdict vectors, thresholded -- a
single trust point that a compromised minority can move. This is the structural baseline;
the *real* AutoDefense system (agent prompts, LLM Coordinator) is an external dependency
(see external_wrappers.py).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from aegis_agency.baselines.base import DefensePipeline
from aegis_agency.data.schemas import (
    Decision,
    DecisionResult,
    GateMode,
    Payload,
    Verdict,
    stack_verdicts,
)


class AutoDefensePipeline(DefensePipeline):
    """Non-robust mean-Coordinator aggregation (no Byzantine tolerance)."""

    name = "autodefense"

    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)

    def decide(self, verdicts: Sequence[Verdict], payload: Payload, rng: np.random.Generator) -> DecisionResult:
        u = stack_verdicts(verdicts)
        u_bar = u.mean(axis=0)  # single-Coordinator mean (non-robust)
        s = float(u_bar[0])
        block = s >= self.threshold
        outlier = np.linalg.norm(u - u_bar, axis=1)
        return DecisionResult(
            mode=GateMode.BLOCK if block else GateMode.ALLOW,
            decision=int(Decision.BLOCK if block else Decision.ALLOW),
            aggregate_score=s,
            threshold=self.threshold,
            rule="autodefense_mean_coordinator",
            group=payload.group,
            per_judge_outlier=outlier,
            payload_id=payload.payload_id,
        )

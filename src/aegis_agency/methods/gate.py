"""The Aegis adjudication gate (Algorithm 1).

Offline: calibrate a block threshold tau on a labelled calibration set.
Online : analyze -> isolate payload -> query n hardened judges -> robust-aggregate their
         verdict vectors -> decide allow / block / escalate.

The gate is agnostic to how verdicts are produced: in synthetic runs they come from
:mod:`aegis_agency.judges.synthetic_judges`; on EC2 they come from real LLM-judge adapters
(:mod:`aegis_agency.data.adapters`). The gate only sees verdict vectors, matching the
paper's verdict-space abstraction.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from aegis_agency.data.schemas import (
    CommitteeConfig,
    Decision,
    DecisionResult,
    GateMode,
    Verdict,
    stack_verdicts,
)
from aegis_agency.methods.aggregators import aggregate
from aegis_agency.methods.calibration import CalibrationResult, calibrate_threshold
from aegis_agency.utils.logging import get_logger

logger = get_logger(__name__)


class AegisGate:
    """Committee gate implementing Algorithm 1 over a chosen aggregation rule."""

    def __init__(self, config: CommitteeConfig):
        self.config = config

    # ------------------------------------------------------------------ offline calibration
    def calibrate(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        objective: str = "target_orr",
        target: float = 0.05,
    ) -> CalibrationResult:
        """Offline calibration (Algorithm 1 uses the resulting tau); updates config.threshold."""
        result = calibrate_threshold(scores, labels, objective=objective, target=target)
        self.config.threshold = result.threshold
        logger.info(
            "Calibrated tau=%.4f (objective=%s, target=%.3f, achieved=%.3f, N=%d).",
            result.threshold,
            result.objective,
            result.target,
            result.achieved,
            result.n_calibration,
        )
        return result

    # ---------------------------------------------------------------------- online decision
    def adjudicate(
        self,
        verdicts: Sequence[Verdict],
        group: str = "default",
        payload_id: str = "",
        rng: np.random.Generator | None = None,
    ) -> DecisionResult:
        """Aggregate committee verdicts and return an allow/block/escalate decision (Alg. 1).

        Parameters
        ----------
        verdicts : sequence of Verdict, length n
            One verdict per judge (already isolated and hardened upstream).
        group : str
            Group tag for group-conditional metrics.
        rng : Generator
            Required only when the aggregation rule is 'majority' with random tie-breaking.
        """
        cfg = self.config
        if len(verdicts) != cfg.n_judges:
            logger.warning(
                "Committee produced %d verdicts but config expects n=%d; proceeding with "
                "the verdicts provided.",
                len(verdicts),
                cfg.n_judges,
            )
        u = stack_verdicts(verdicts)
        u_hat = aggregate(u, cfg.rule, f=cfg.assumed_f, rng=rng)
        score = float(u_hat[0])

        # Per-judge outlier score = distance of each verdict vector to the aggregate.
        outlier = np.linalg.norm(u - u_hat, axis=1)

        # Algorithm 1, lines 10-13: escalate / block / allow.
        if cfg.escalate_band > 0 and abs(score - cfg.threshold) < cfg.escalate_band:
            mode = GateMode.ESCALATE
            decision = Decision.BLOCK  # escalation defaults to the safe side pending review
        elif score >= cfg.threshold:
            mode = GateMode.BLOCK
            decision = Decision.BLOCK
        else:
            mode = GateMode.ALLOW
            decision = Decision.ALLOW

        return DecisionResult(
            mode=mode,
            decision=int(decision),
            aggregate_score=score,
            threshold=cfg.threshold,
            rule=cfg.rule,
            group=group,
            per_judge_outlier=outlier,
            payload_id=payload_id,
        )

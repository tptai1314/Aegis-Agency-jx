"""Second-order payload injection (Def. 1; Prop. 2; JudgeDeceiver-style).

Instead of compromising judges, the attacker crafts the payload so that instructions hidden
in the data channel perturb the *honest* judges reading it. Payload isolation
(:class:`~aegis_agency.judges.isolation.IsolationModel`) bounds this: each honest judge is
perturbed with probability at most ``epsilon`` (Def. 1). With perfect isolation (epsilon=0)
no honest verdict changes and the attacker must fall back to compromising judges (Prop. 2).

On EC2 this attack is instantiated with a real JudgeDeceiver adapter; here it is the
parametric epsilon-leakage model used to test the pipeline.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from aegis_agency.attacks.base import Attack
from aegis_agency.data.schemas import Decision, Payload, Verdict
from aegis_agency.judges.isolation import IsolationModel


class SecondOrderInjectionAttack(Attack):
    """Perturb honest verdicts toward the target decision with per-judge probability epsilon."""

    name = "injection"

    def __init__(self, epsilon: float = 0.1):
        self.isolation = IsolationModel(epsilon=epsilon)

    def apply(
        self,
        honest: Sequence[Verdict],
        payload: Payload,
        f: int,  # unused: injection does not consume the Byzantine budget (Prop. 2)
        rng: np.random.Generator,
    ) -> list[Verdict]:
        verdicts = list(honest)
        target = self.target_decision(payload)
        m = verdicts[0].embedding.shape[0] if verdicts else 0
        byz_score = 1.0 if target == Decision.BLOCK else 0.0
        sign = 1.0 if target == Decision.BLOCK else -1.0

        for k, v in enumerate(verdicts):
            if self.isolation.leaks(rng):
                emb = v.embedding.copy()
                if m:
                    emb[0] = sign * 1.0
                verdicts[k] = Verdict(
                    decision=target,
                    score=float(byz_score),
                    embedding=emb,
                    judge_id=v.judge_id,
                    is_byzantine=False,  # honest judge, but injected: not a Byzantine node
                )
        return verdicts

"""Compromised-judge attack (Section 5.3 capability i; Section 10).

The attacker controls ``f`` judges, which return arbitrary verdict vectors chosen to push
the aggregate toward the wrong decision. This is the LLM-committee analogue of Byzantine
workers: the Byzantine verdicts are placed at an extreme of the score/embedding space.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from aegis_agency.attacks.base import Attack
from aegis_agency.data.schemas import Decision, Payload, Verdict


class CompromisedJudgeAttack(Attack):
    """Replace ``f`` verdicts with extreme values favouring the target (wrong) decision."""

    name = "compromise"

    def __init__(self, extremity: float = 1.0):
        # extremity in [0, 1]: how far to push the Byzantine score toward 0 or 1.
        self.extremity = float(np.clip(extremity, 0.0, 1.0))

    def apply(
        self,
        honest: Sequence[Verdict],
        payload: Payload,
        f: int,
        rng: np.random.Generator,
    ) -> list[Verdict]:
        verdicts = list(honest)
        n = len(verdicts)
        f = min(f, n)
        target = self.target_decision(payload)
        m = verdicts[0].embedding.shape[0] if verdicts else 0

        # Byzantine target score: 1 to force block, 0 to force allow.
        byz_score = self.extremity if target == Decision.BLOCK else 1.0 - self.extremity
        byz_score = float(np.clip(byz_score, 0.0, 1.0))
        # Byzantine embedding pushed toward the target class center.
        sign = 1.0 if target == Decision.BLOCK else -1.0
        byz_emb = np.zeros(m)
        if m:
            byz_emb[0] = sign * (1.0 + self.extremity)

        idx = rng.choice(n, size=f, replace=False) if f > 0 else np.array([], dtype=int)
        for k in idx:
            verdicts[k] = Verdict(
                decision=target,
                score=byz_score,
                embedding=byz_emb.copy(),
                judge_id=int(k),
                is_byzantine=True,
            )
        return verdicts

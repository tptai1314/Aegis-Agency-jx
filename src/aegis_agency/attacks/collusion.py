"""Colluding-defenders attack: "a little is enough" (Baruch et al., 2019; Appendix B).

The f Byzantine judges coordinate to stay *within* the honest verdict spread (radius r) while
biasing every controlled verdict toward the target decision. Because each Byzantine verdict
looks honest (within radius), robust aggregation cannot flag it as an outlier, yet the
coordinated shift moves the aggregate. This is the adaptive collusion the paper stresses.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from aegis_agency.attacks.base import Attack
from aegis_agency.data.schemas import Decision, Payload, Verdict


class CollusionAttack(Attack):
    """Coordinated within-radius bias toward the target decision."""

    name = "collusion"

    def __init__(self, radius: float = 0.08, budget: float = 1.0):
        # radius: the honest concentration radius r the colluders stay within.
        # budget in [0, 1]: fraction of the radius used as the coordinated shift z <= r.
        self.radius = float(radius)
        self.budget = float(np.clip(budget, 0.0, 1.0))

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

        # Reference = honest mean verdict vector (attacker estimates it).
        u = np.vstack([v.vector() for v in verdicts])
        ref = u.mean(axis=0)

        # Direction toward the target decision along the score axis (and embedding axis 0).
        z = self.budget * self.radius
        direction = np.zeros(ref.shape[0])
        s_sign = -1.0 if target == Decision.ALLOW else 1.0  # lower score -> allow
        direction[0] = s_sign
        if m:
            direction[1] = s_sign  # push embedding first axis consistently

        idx = rng.choice(n, size=f, replace=False) if f > 0 else np.array([], dtype=int)
        for k in idx:
            shifted = ref + z * direction
            score = float(np.clip(shifted[0], 0.0, 1.0))
            emb = shifted[1:] if m else np.zeros(0)
            decision = int(Decision.BLOCK if score >= 0.5 else Decision.ALLOW)
            verdicts[k] = Verdict(
                decision=decision, score=score, embedding=emb, judge_id=int(k), is_byzantine=True
            )
        return verdicts

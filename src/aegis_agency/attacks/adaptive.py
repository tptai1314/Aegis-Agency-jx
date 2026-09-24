"""Adaptive-on-aggregation attack (Section 5.3 capability iii; Xie et al., 2019; Fang et al., 2020).

The attacker optimises the ``f`` Byzantine verdicts against the *specific* aggregation rule.
For median/gmed rules the strongest within-reach move is to shift all Byzantine verdicts as
far as possible along the target direction while remaining just inside the boundary of what
the rule will still pull toward -- here approximated by placing the Byzantine verdicts at the
edge of the honest cloud in the target direction (a directional-manipulation surrogate for
inner-product manipulation / local model poisoning). This is a heuristic adaptive attack for
stress-testing, not a claimed optimal attack.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from aegis_agency.attacks.base import Attack
from aegis_agency.data.schemas import Decision, Payload, Verdict


class AdaptiveAggregationAttack(Attack):
    """Directional manipulation tuned to the aggregation rule."""

    name = "adaptive"

    def __init__(self, rule: str = "gmed", step: float = 0.5):
        self.rule = rule
        self.step = float(step)  # magnitude of the coordinated push beyond the honest cloud

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
        u = np.vstack([v.vector() for v in verdicts])
        m = u.shape[1] - 1

        # Honest cloud statistics the attacker can estimate.
        mean = u.mean(axis=0)
        spread = u.std(axis=0) + 1e-9
        s_sign = -1.0 if target == Decision.ALLOW else 1.0

        direction = np.zeros(u.shape[1])
        direction[0] = s_sign
        if m:
            direction[1] = s_sign

        # Place Byzantine verdicts at mean + step * spread * direction. For median rules this
        # is the largest within-cloud push that still pulls the aggregate; for gmed it biases
        # the weighted center.
        target_point = mean + self.step * spread * direction
        idx = rng.choice(n, size=f, replace=False) if f > 0 else np.array([], dtype=int)
        for k in idx:
            score = float(np.clip(target_point[0], 0.0, 1.0))
            emb = target_point[1:] if m else np.zeros(0)
            decision = int(Decision.BLOCK if score >= 0.5 else Decision.ALLOW)
            verdicts[k] = Verdict(
                decision=decision, score=score, embedding=emb, judge_id=int(k), is_byzantine=True
            )
        return verdicts

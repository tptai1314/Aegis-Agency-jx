"""Synthetic judges that emit verdict vectors in the paper's verdict space.

These generators let the mechanism (aggregation, attacks, metrics, theory) be tested and
illustrated without any LLM. They realise the paper's assumptions parametrically:

* honest concentration radius ``r``           -- Assumption 1
* decision margin ``gamma``                   -- Assumption 2
* honest-error correlation ``rho``            -- Proposition 3 (RQ4)

For a payload with ground-truth label y* the honest reference score is
``0.5 + gamma`` if y*=1 (block) or ``0.5 - gamma`` if y*=0 (allow), and honest verdict
vectors are drawn within radius ~``r`` of a class-conditional reference. A shared latent
component of relative weight ``sqrt(rho)`` induces pairwise error correlation ~``rho`` across
judges, so diverse committees (small rho) concentrate while homogeneous committees (large
rho) fail together (Proposition 3).

Everything here is synthetic; outputs are never paper results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aegis_agency.data.schemas import Decision, Payload, Verdict
from aegis_agency.judges.base import JudgeModel
from aegis_agency.utils.validation import check_unit_interval


@dataclass
class SyntheticJudgePopulation:
    """Generates an honest committee's verdicts for a payload (Assumptions 1-2; Prop. 3)."""

    n_judges: int
    margin: float = 0.25          # gamma: |E[score] - 0.5|
    radius: float = 0.08          # r: honest concentration (std of score/embedding noise)
    correlation: float = 0.0      # rho: pairwise correlation of honest errors
    embedding_dim: int = 4        # m; verdict dim d = 1 + m
    threshold: float = 0.5        # tau (reference decision boundary)

    def __post_init__(self) -> None:
        check_unit_interval(self.margin, "margin")
        if self.radius < 0:
            raise ValueError("radius (r) must be >= 0.")
        check_unit_interval(self.correlation, "correlation")
        if self.embedding_dim < 0:
            raise ValueError("embedding_dim (m) must be >= 0.")

    def reference_score(self, label: int) -> float:
        """Class-conditional honest reference score (Assumption 2)."""
        return self.threshold + self.margin if int(label) == Decision.BLOCK else self.threshold - self.margin

    def _class_center(self, label: int) -> np.ndarray:
        """Deterministic class-conditional embedding center in R^m."""
        if self.embedding_dim == 0:
            return np.zeros(0)
        sign = 1.0 if int(label) == Decision.BLOCK else -1.0
        center = np.zeros(self.embedding_dim)
        center[0] = sign  # separate the two classes along the first embedding axis
        return center

    def generate_honest(self, payload: Payload, rng: np.random.Generator) -> list[Verdict]:
        """Generate honest verdicts for the whole committee (shared latent -> correlation)."""
        n, m = self.n_judges, self.embedding_dim
        ref_s = self.reference_score(payload.true_label)
        center = self._class_center(payload.true_label)

        rho = self.correlation
        # Shared vs independent noise weights giving pairwise correlation ~ rho.
        w_shared = np.sqrt(rho)
        w_indep = np.sqrt(max(0.0, 1.0 - rho))

        shared_s = rng.standard_normal()
        shared_e = rng.standard_normal(m) if m else np.zeros(0)

        verdicts: list[Verdict] = []
        for k in range(n):
            indep_s = rng.standard_normal()
            noise_s = self.radius * (w_shared * shared_s + w_indep * indep_s)
            score = float(np.clip(ref_s + noise_s, 0.0, 1.0))
            if m:
                indep_e = rng.standard_normal(m)
                noise_e = self.radius * (w_shared * shared_e + w_indep * indep_e)
                emb = center + noise_e
            else:
                emb = np.zeros(0)
            decision = int(Decision.BLOCK if score >= self.threshold else Decision.ALLOW)
            verdicts.append(
                Verdict(decision=decision, score=score, embedding=emb, judge_id=k, is_byzantine=False)
            )
        return verdicts


class SyntheticHonestJudge(JudgeModel):
    """Single independent honest judge (rho = 0), for the :class:`JudgeModel` interface."""

    def __init__(self, judge_id: int, margin: float = 0.25, radius: float = 0.08, embedding_dim: int = 4):
        super().__init__(judge_id=judge_id, backbone="synthetic")
        self._pop = SyntheticJudgePopulation(
            n_judges=1, margin=margin, radius=radius, correlation=0.0, embedding_dim=embedding_dim
        )

    def verdict(self, payload: Payload, rng: np.random.Generator) -> Verdict:
        v = self._pop.generate_honest(payload, rng)[0]
        # Re-stamp the judge id.
        return Verdict(
            decision=v.decision, score=v.score, embedding=v.embedding,
            judge_id=self.judge_id, is_byzantine=False,
        )

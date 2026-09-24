"""Payload isolation model (Definition 1: epsilon-isolation; Proposition 2).

A judge is epsilon-isolated if an injected variant of the payload changes its verdict law by
at most epsilon in total variation (Def. 1). We model this operationally: with probability
at most ``epsilon`` a second-order injection perturbs an honest judge's verdict; with
probability >= 1 - epsilon the judge is unaffected. epsilon = 0 is perfect isolation.

This is a *model* of the empirical isolation property, used for synthetic stress tests. On
EC2, epsilon is measured from real judges under JudgeDeceiver-style injections (RQ2); see
docs/ec2_experiment_guide.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aegis_agency.utils.validation import check_unit_interval


@dataclass
class IsolationModel:
    """Parametric epsilon-isolation model (Def. 1)."""

    epsilon: float = 0.0

    def __post_init__(self) -> None:
        check_unit_interval(self.epsilon, "epsilon")

    def leaks(self, rng: np.random.Generator) -> bool:
        """Return True with probability ``epsilon`` (the injection perturbs this judge)."""
        if self.epsilon <= 0.0:
            return False
        return bool(rng.random() < self.epsilon)

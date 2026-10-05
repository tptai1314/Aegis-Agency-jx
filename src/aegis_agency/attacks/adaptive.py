"""Adaptive-on-aggregation attacks (Section 5.3 capability iv; Xie et al., 2019; Fang et al., 2020).

Two attackers live here:

``AdaptiveAggregationAttack`` (``attack: adaptive``) -- the original **heuristic surrogate**: it
    places the Byzantine verdicts a fixed step beyond the honest cloud along the target
    direction. Cheap and deterministic, but its magnitude is not optimised and the same tampered
    committee is used for every rule.

``AdaptiveSearchAttack`` (``attack: adaptive_search``) -- a **search-based** attacker that
    optimises against the specific rule: it enumerates candidate Byzantine verdict vectors
    (extreme + within-cloud shifts of several magnitudes + the honest edge) and, for each, every
    placement of the ``f`` Byzantine slots (or a deterministic sample of placements when the
    count is large), evaluates the *actual aggregator* the pipeline uses and keeps the assignment
    that pushes the aggregate furthest past the decision threshold in the attacker's favour. The
    harness instantiates it once per pipeline (``per_rule = True``), so ``aegis_cmed``,
    ``aegis_gmed``, ``aegis_krum``, ``autodefense`` (mean coordinator), ``majority_vote`` and
    ``single_model`` each face an attacker tuned to them.

Neither attack is claimed to be *optimal*: the search is over a finite candidate set, not a
global optimisation, and it is exactly what an attacker with the honest verdicts in hand can do
against a known rule.
"""

from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np

from aegis_agency.attacks.base import Attack
from aegis_agency.data.schemas import Decision, Payload, Verdict
from aegis_agency.methods.aggregators import aggregate
from aegis_agency.utils.logging import get_logger

logger = get_logger(__name__)

#: Score coordinate of the attack target when the attacker wants the gate to ALLOW / BLOCK.
_TARGET_SCORE = {int(Decision.BLOCK): 1.0, int(Decision.ALLOW): 0.0}

#: Rules already reported as un-evaluable, so the warning is emitted once per process and not
#: once per payload (the harness builds a fresh attack instance for every call).
_WARNED_RULES: set[str] = set()


class AdaptiveAggregationAttack(Attack):
    """Directional manipulation tuned to the aggregation rule (heuristic surrogate)."""

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


class AdaptiveSearchAttack(Attack):
    """Search-based attacker that optimises the Byzantine verdicts against a known rule.

    Parameters
    ----------
    rule : str
        Aggregation rule to optimise against (``cmed``/``gmed``/``krum``/``majority``/``mean``/
        ``single``). Passed by the harness per pipeline; the default only applies when the attack
        is used directly.
    threshold : float
        Decision threshold tau the attacker wants to cross (the pipeline's current, calibrated
        threshold when driven by the harness).
    n_grid : int
        Number of within-cloud shift magnitudes tried, evenly spaced on ``[0.25, radius_scale]``
        times the honest per-coordinate spread.
    radius_scale : float
        Largest within-cloud shift, as a multiple of the honest spread.
    max_subsets : int
        Cap on the number of Byzantine-slot placements evaluated per candidate. Placements are
        enumerated exhaustively when ``C(n, f)`` is small and sampled deterministically (from the
        trial rng) otherwise.
    """

    name = "adaptive_search"
    per_rule = True

    def __init__(
        self,
        rule: str = "gmed",
        threshold: float = 0.5,
        n_grid: int = 5,
        radius_scale: float = 1.5,
        max_subsets: int = 200,
    ):
        self.rule = rule
        self.threshold = float(threshold)
        self.n_grid = max(1, int(n_grid))
        self.radius_scale = float(radius_scale)
        self.max_subsets = max(1, int(max_subsets))

    # ------------------------------------------------------------------------ candidate sets
    def _candidates(self, u: np.ndarray, target: int) -> list[np.ndarray]:
        """Candidate Byzantine verdict vectors, all aimed at the attacker's target decision."""
        d = u.shape[1]
        mean = u.mean(axis=0)
        spread = u.std(axis=0) + 1e-9
        direction = np.zeros(d)
        direction[0] = 1.0 if target == Decision.BLOCK else -1.0

        candidates: list[np.ndarray] = []
        extreme = np.zeros(d)
        extreme[0] = _TARGET_SCORE[target]
        candidates.append(extreme)
        for z in np.linspace(0.25, self.radius_scale, self.n_grid):
            point = mean + z * spread * direction
            point[0] = float(np.clip(point[0], 0.0, 1.0))
            candidates.append(point)
        # The honest verdict already closest to the target side (a "be one of them" move).
        edge_index = int(np.argmax(u[:, 0])) if target == Decision.BLOCK else int(np.argmin(u[:, 0]))
        candidates.append(u[edge_index].copy())
        return candidates

    def _placements(self, n: int, f: int, rng: np.random.Generator) -> list[tuple[int, ...]]:
        """Byzantine slot placements: exhaustive when small, else a deterministic sample."""
        total = 1
        for i in range(f):
            total = total * (n - i) // (i + 1)
        if total <= self.max_subsets:
            return list(itertools.combinations(range(n), f))
        seen: set[tuple[int, ...]] = set()
        guard = 0
        while len(seen) < self.max_subsets and guard < 100 * self.max_subsets:
            guard += 1
            seen.add(tuple(sorted(int(x) for x in rng.choice(n, size=f, replace=False))))
        return sorted(seen)

    # ------------------------------------------------------------------------- aggregate score
    def _aggregate_score(self, matrix: np.ndarray, f: int) -> float:
        """Score coordinate of the aggregate under the rule being attacked."""
        if self.rule == "mean":            # AutoDefense's single (non-robust) Coordinator
            return float(matrix[:, 0].mean())
        if self.rule == "single":          # single hardened model: the first judge
            return float(matrix[0, 0])
        if self.rule == "majority":
            votes = (matrix[:, 0] >= 0.5).astype(int)
            return 1.0 if 2 * int(votes.sum()) > matrix.shape[0] else 0.0
        return float(aggregate(matrix, self.rule, f=f)[0])

    # --------------------------------------------------------------------------------- apply
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
        if f == 0 or n == 0:
            return verdicts
        target = self.target_decision(payload)
        u = np.vstack([v.vector() for v in verdicts])
        candidates = self._candidates(u, target)
        placements = self._placements(n, f, rng)

        best_gain = -np.inf
        best_verdicts = verdicts
        for candidate in candidates:
            for idx in placements:
                trial = list(verdicts)
                for k in idx:
                    trial[k] = Verdict(
                        decision=target,
                        score=float(candidate[0]),
                        embedding=candidate[1:].copy(),
                        judge_id=int(k),
                        is_byzantine=True,
                    )
                matrix = np.vstack([v.vector() for v in trial])
                try:
                    score = self._aggregate_score(matrix, f)
                except ValueError as exc:  # e.g. Krum with n - f - 2 < 1
                    if self.rule not in _WARNED_RULES:
                        _WARNED_RULES.add(self.rule)
                        logger.warning(
                            "adaptive_search: rule %r could not be evaluated (%s); leaving the "
                            "committee honest and skipping the search.",
                            self.rule,
                            exc,
                        )
                    return verdicts
                gain = score - self.threshold if target == Decision.BLOCK else self.threshold - score
                if gain > best_gain:
                    best_gain = gain
                    best_verdicts = trial
        return best_verdicts

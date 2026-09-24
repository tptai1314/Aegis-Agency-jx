"""Byzantine-robust verdict aggregation rules (Algorithm 2, Section 6.3).

Each rule maps an (n, d) matrix of verdict vectors u_k = (s_k, e_k) to a single aggregate
verdict vector u_hat in R^d. The final decision is D(u_hat) = 1{ pi_1(u_hat) >= tau }.

Rules
-----
majority_vote      : decision-only baseline, 1{ sum d_k > n/2 }  (Section 6.3)
coordinate_median  : per-coordinate median, tolerant for f < n/2  (Alg. 2; Prop. 1)
geometric_median   : smoothed Weiszfeld iteration, tolerant for f < n/2  (Alg. 2; Lemma 1)
krum               : select verdict minimising sum of sq. dist. to n-f-2 nearest (Alg. 2; Lemma 2)

References map directly to the manuscript; see audits/math_to_code_audit.md for edge-case
and consistency notes.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from aegis_agency.utils.logging import get_logger
from aegis_agency.utils.validation import check_verdict_matrix

logger = get_logger(__name__)


# --------------------------------------------------------------------------- majority vote
def majority_vote(
    decisions: np.ndarray,
    tie_breaking: str = "paper_strict",
    rng: np.random.Generator | None = None,
) -> int:
    """Plain majority vote over binary decisions: 1{ sum d_k > n/2 } (Section 6.3).

    Parameters
    ----------
    decisions : array of {0, 1}, shape (n,)
    tie_breaking : {"paper_strict", "block_on_tie", "random"}
        Behaviour when exactly n/2 judges block (n even). The paper uses strict ``>``,
        which resolves a tie to ALLOW; we default to that ("paper_strict") and expose the
        safer "block_on_tie" and randomised "random" options. See math_to_code_audit.md.
    rng : Generator, required iff tie_breaking == "random".

    Returns
    -------
    int : aggregate decision in {0, 1}.
    """
    d = np.asarray(decisions, dtype=int).ravel()
    if d.size == 0:
        raise ValueError("majority_vote requires at least one decision.")
    if not np.all(np.isin(d, (0, 1))):
        raise ValueError("Decisions must be binary {0, 1}.")
    n = d.size
    blocks = int(d.sum())
    if 2 * blocks > n:
        return 1
    if 2 * blocks < n:
        return 0
    # Tie: exactly n/2 blocks.
    if tie_breaking == "paper_strict":
        return 0  # strict '>' fails -> allow
    if tie_breaking == "block_on_tie":
        return 1
    if tie_breaking == "random":
        if rng is None:
            raise ValueError("tie_breaking='random' requires an rng.")
        return int(rng.integers(0, 2))
    raise ValueError(f"Unknown tie_breaking: {tie_breaking!r}.")


# ---------------------------------------------------------------------- coordinate median
def coordinate_median(u: np.ndarray) -> np.ndarray:
    """Coordinate-wise median of verdict vectors (Alg. 2; Prop. 1).

    Returns the vector whose j-th coordinate is the median of the j-th coordinates. For
    f < n/2 Byzantine rows, each output coordinate lies within the honest range (Prop. 1).
    For even n, ``np.median`` averages the two central values; this still lies in the honest
    range, preserving the guarantee (see math_to_code_audit.md).
    """
    u = check_verdict_matrix(u)
    return np.median(u, axis=0)


# ----------------------------------------------------------------------- geometric median
def geometric_median(
    u: np.ndarray,
    eta: float = 1e-8,
    max_iter: int = 200,
    tol: float = 1e-7,
) -> np.ndarray:
    """Geometric median via smoothed Weiszfeld iteration (Alg. 2; Lemma 1).

    Solves argmin_y sum_k || y - u_k ||_2 using the weights
    ``w_k = 1 / max(|| y - u_k ||, eta)`` exactly as Algorithm 2 specifies. The ``eta`` floor
    prevents division by zero when the iterate coincides with a data point.

    Notes
    -----
    Lemma 1's displacement bound || u_hat - u* || <= C_alpha r is stated for the *exact*
    geometric median; this routine returns an ``eta``-approximation. The theory check in
    :mod:`aegis_agency.metrics.theory` therefore tests the bound with a tolerance rather than
    asserting exact equality (conservative choice).
    """
    u = check_verdict_matrix(u)
    n = u.shape[0]
    if n == 1:
        return u[0].copy()
    y = u.mean(axis=0)  # Alg. 2 initialisation
    for it in range(max_iter):
        dist = np.linalg.norm(u - y, axis=1)
        w = 1.0 / np.maximum(dist, eta)
        y_new = (w[:, None] * u).sum(axis=0) / w.sum()
        shift = float(np.linalg.norm(y_new - y))
        y = y_new
        if shift < tol:
            return y
    logger.warning(
        "geometric_median did not converge within %d iterations (last shift below tol=%g "
        "not reached); returning last iterate.",
        max_iter,
        tol,
    )
    return y


# ------------------------------------------------------------------------------------ Krum
def krum(u: np.ndarray, f: int, return_index: bool = False):
    """Krum selection (Alg. 2; Lemma 2 restated from Blanchard et al., 2017).

    Selects u_{k*} with k* = argmin_k sum_{j in N_k} || u_k - u_j ||^2, where N_k are the
    n - f - 2 nearest neighbours of k (excluding k itself).

    The tolerance guarantee requires 2f + 2 < n; the computation itself only requires
    n - f - 2 >= 1. A warning is logged when 2f + 2 >= n (guarantee not in force).
    """
    u = check_verdict_matrix(u)
    n = u.shape[0]
    n_neighbors = n - f - 2
    if n_neighbors < 1:
        raise ValueError(
            f"Krum requires n - f - 2 >= 1 (n={n}, f={f}); increase n or decrease f."
        )
    if 2 * f + 2 >= n:
        logger.warning(
            "Krum tolerance condition 2f+2 < n is violated (n=%d, f=%d); the selection is "
            "still computed but the Byzantine-tolerance guarantee does not hold.",
            n,
            f,
        )
    # Pairwise squared distances.
    diff = u[:, None, :] - u[None, :, :]
    sq = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(sq, np.inf)  # exclude self
    scores = np.empty(n, dtype=float)
    for k in range(n):
        nearest = np.sort(sq[k])[:n_neighbors]
        scores[k] = float(nearest.sum())
    k_star = int(np.argmin(scores))  # ties -> lowest index (deterministic)
    if return_index:
        return u[k_star].copy(), k_star
    return u[k_star].copy()


# ------------------------------------------------------------------------- dispatch helper
def aggregate(
    u: np.ndarray,
    rule: str,
    f: int = 0,
    tie_breaking: str = "paper_strict",
    rng: np.random.Generator | None = None,
    **kwargs,
) -> np.ndarray:
    """Dispatch to an aggregation rule and return an aggregate verdict vector in R^d.

    For ``rule='majority'`` the decision is taken over the score coordinate thresholded at
    0.5 to form binary decisions (a decision-only rule); the returned vector carries the
    majority decision in its score coordinate so downstream code is uniform.
    """
    u = check_verdict_matrix(u)
    if rule == "gmed":
        return geometric_median(u, **kwargs)
    if rule == "cmed":
        return coordinate_median(u)
    if rule == "krum":
        return krum(u, f=f)
    if rule == "majority":
        decisions = (u[:, 0] >= 0.5).astype(int)
        maj = majority_vote(decisions, tie_breaking=tie_breaking, rng=rng)
        out = coordinate_median(u).copy()
        out[0] = float(maj)  # score coordinate reports the majority decision
        return out
    raise ValueError(f"Unknown aggregation rule: {rule!r}.")


AGGREGATORS: dict[str, Callable[..., np.ndarray]] = {
    "majority": lambda u, **kw: aggregate(u, "majority", **kw),
    "cmed": lambda u, **kw: aggregate(u, "cmed", **kw),
    "gmed": lambda u, **kw: aggregate(u, "gmed", **kw),
    "krum": lambda u, **kw: aggregate(u, "krum", **kw),
}

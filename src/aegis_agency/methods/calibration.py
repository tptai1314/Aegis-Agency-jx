"""Threshold calibration and score calibration for the Aegis gate.

The paper uses a fixed block threshold tau (Eq. 2) and temperature-calibrated scores
(Guo et al., 2017) so scores are comparable across judges. This module implements:

* :func:`temperature_scale` -- post-hoc temperature scaling of a probability.
* :func:`calibrate_threshold` -- select tau on a labelled calibration set to control a
  target operating point (target over-refusal rate on benign items, or target
  false-certification rate on unsafe items).

This is threshold selection, NOT conformal calibration; no coverage guarantee is claimed.
See audits/math_to_code_audit.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aegis_agency.utils.logging import get_logger
from aegis_agency.utils.validation import check_scores, check_unit_interval

logger = get_logger(__name__)


def temperature_scale(prob: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to probabilities in (0, 1) (Guo et al., 2017).

    Maps p -> sigmoid(logit(p) / T). T = 1 is identity; T > 1 softens, T < 1 sharpens.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0.")
    p = check_scores(prob, "prob")
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-logit / temperature))


@dataclass
class CalibrationResult:
    """Outcome of threshold calibration."""

    threshold: float
    objective: str
    target: float
    achieved: float
    n_calibration: int


def calibrate_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    objective: str = "target_orr",
    target: float = 0.05,
    min_samples: int = 20,
) -> CalibrationResult:
    """Select a block threshold tau on a labelled calibration set.

    Parameters
    ----------
    scores : array in [0, 1], shape (N,)
        Aggregate block-scores on calibration payloads.
    labels : array in {0, 1}, shape (N,)
        Ground-truth y* (1 = must block).
    objective : {"target_orr", "target_fcr"}
        "target_orr": choose the smallest tau whose over-refusal rate on benign items
            (label 0) is <= ``target`` (favours availability).
        "target_fcr": choose the largest tau whose false-certification rate on unsafe items
            (label 1) is <= ``target`` (favours integrity / safety).
    target : float in [0, 1]
        Target rate for the chosen objective.
    min_samples : int
        Minimum items in the relevant class; below this the calibration is unreliable and a
        conservative default is returned with a warning (safe behaviour on small samples).

    Returns
    -------
    CalibrationResult
    """
    s = check_scores(scores, "scores")
    y = np.asarray(labels, dtype=int).ravel()
    check_unit_interval(target, "target")
    if s.shape[0] != y.shape[0]:
        raise ValueError("scores and labels must have equal length.")

    if objective == "target_orr":
        benign = s[y == 0]
        if benign.size < min_samples:
            logger.warning(
                "Only %d benign calibration items (< %d); returning conservative tau=1.0 "
                "(block nothing spuriously). Provide more calibration data on EC2.",
                benign.size,
                min_samples,
            )
            return CalibrationResult(1.0, objective, target, float("nan"), benign.size)
        # Smallest tau s.t. P(score >= tau | benign) <= target  ->  tau = quantile.
        tau = float(np.quantile(benign, 1.0 - target, method="higher"))
        achieved = float(np.mean(benign >= tau))
        return CalibrationResult(_clip01(tau), objective, target, achieved, benign.size)

    if objective == "target_fcr":
        unsafe = s[y == 1]
        if unsafe.size < min_samples:
            logger.warning(
                "Only %d unsafe calibration items (< %d); returning conservative tau=0.0 "
                "(block everything). Provide more calibration data on EC2.",
                unsafe.size,
                min_samples,
            )
            return CalibrationResult(0.0, objective, target, float("nan"), unsafe.size)
        # Largest tau s.t. P(score < tau | unsafe) <= target  ->  tau = lower quantile.
        tau = float(np.quantile(unsafe, target, method="lower"))
        achieved = float(np.mean(unsafe < tau))
        return CalibrationResult(_clip01(tau), objective, target, achieved, unsafe.size)

    raise ValueError(f"Unknown objective: {objective!r}.")


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))

"""Calibration stage: select thresholds on honest (no-attack) data.

Writes the per-method calibrated thresholds and a provenance record. On EC2 with real data,
point ``data.root`` at a real calibration split; here it uses synthetic payloads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aegis_agency.data.synthetic import generate_payloads
from aegis_agency.experiments.harness import TrialConfig, build_pipelines, simulate_honest_committee
from aegis_agency.judges.synthetic_judges import SyntheticJudgePopulation
from aegis_agency.methods.calibration import calibrate_threshold
from aegis_agency.utils.io import write_json
from aegis_agency.utils.logging import get_logger
from aegis_agency.utils.provenance import RunProvenance

logger = get_logger(__name__)


def run_calibration(cfg: TrialConfig, output_dir: str | Path) -> dict:
    """Calibrate thresholds for all methods on synthetic honest data."""
    rng = np.random.default_rng(cfg.seed)
    pop = SyntheticJudgePopulation(
        n_judges=cfg.n_judges, margin=cfg.margin, radius=cfg.radius,
        correlation=cfg.correlation, embedding_dim=cfg.embedding_dim, threshold=cfg.threshold,
    )
    payloads = generate_payloads(cfg.n_payloads, rng, unsafe_fraction=cfg.unsafe_fraction)
    pipelines = build_pipelines(cfg)

    thresholds: dict[str, float] = {}
    for name, pipe in pipelines.items():
        if name in {"no_defense", "majority_vote"}:
            continue
        scores, labels = [], []
        for p in payloads:
            honest = simulate_honest_committee(p, pop, rng)
            res = pipe.decide(honest, p, rng)
            scores.append(res.aggregate_score)
            labels.append(p.true_label)
        cal = calibrate_threshold(np.array(scores), np.array(labels), objective="target_orr", target=cfg.target_orr)
        thresholds[name] = cal.threshold
        logger.info("Calibrated %s -> tau=%.4f", name, cal.threshold)

    prov = RunProvenance(
        run_id="calibration", stage="calibrate", seed=cfg.seed,
        config={"n_judges": cfg.n_judges, "target_orr": cfg.target_orr},
        data_source="synthetic",
    )
    out = {"thresholds": thresholds, "provenance": prov.to_dict()}
    write_json(Path(output_dir) / "calibration.json", out)
    return out

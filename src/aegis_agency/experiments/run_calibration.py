"""Calibration stage: select thresholds on honest (no-attack) data.

Writes the per-method calibrated thresholds and a provenance record. With ``data.root`` set
in the config, payloads are loaded from a real benchmark on disk (verdict simulation still
synthetic); otherwise synthetic payloads are generated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aegis_agency.experiments.harness import (
    TrialConfig,
    build_honest_committee,
    build_pipelines,
    load_payloads,
    prefetch_committee_verdicts,
    simulate_honest_committee,
)
from aegis_agency.methods.calibration import calibrate_threshold
from aegis_agency.utils.io import write_json
from aegis_agency.utils.logging import get_logger
from aegis_agency.utils.provenance import RunProvenance

logger = get_logger(__name__)


def run_calibration(cfg: TrialConfig, output_dir: str | Path) -> dict:
    """Calibrate thresholds for all methods on honest data."""
    rng = np.random.default_rng(cfg.seed)
    committee = build_honest_committee(cfg)
    payloads, data_source = load_payloads(cfg, rng)
    pipelines = build_pipelines(cfg)

    # Real-judge backends collect their verdicts up front (one backbone session at a time,
    # engines released afterwards when max_resident_engines == 1); synthetic is untouched.
    new_calls = prefetch_committee_verdicts(committee, payloads, cfg)
    if new_calls:
        logger.info("Prefetched %d honest judge calls for %d payloads.", new_calls, len(payloads))

    thresholds: dict[str, float] = {}
    for name, pipe in pipelines.items():
        if name in {"no_defense", "majority_vote"}:
            continue
        scores, labels = [], []
        for p in payloads:
            honest = simulate_honest_committee(p, committee, rng)
            res = pipe.decide(honest, p, rng)
            scores.append(res.aggregate_score)
            labels.append(p.true_label)
        cal = calibrate_threshold(np.array(scores), np.array(labels), objective="target_orr", target=cfg.target_orr)
        thresholds[name] = cal.threshold
        logger.info("Calibrated %s -> tau=%.4f", name, cal.threshold)

    prov = RunProvenance(
        run_id="calibration", stage="calibrate", seed=cfg.seed,
        config={
            "n_judges": cfg.n_judges, "target_orr": cfg.target_orr,
            "data_root": cfg.data_root, "benchmark": cfg.benchmark, "data_split": cfg.data_split,
            "judge_backend": cfg.judge_backend, "models": list(cfg.models),
            "isolation": cfg.isolation, "max_resident_engines": cfg.max_resident_engines,
            "temperature": cfg.temperature, "max_tokens": cfg.max_tokens,
            "model_revision": cfg.model_revision or "<unpinned>",
        },
        data_source=data_source,
    )
    out = {"thresholds": thresholds, "provenance": prov.to_dict()}
    write_json(Path(output_dir) / "calibration.json", out)
    return out

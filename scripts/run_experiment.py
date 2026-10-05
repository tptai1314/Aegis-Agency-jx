#!/usr/bin/env python3
"""Run an experiment stage (calibrate / evaluate / ablate).

    python scripts/run_experiment.py --config configs/experiment_template.yaml --stage calibrate --output outputs/exp
    python scripts/run_experiment.py --config configs/experiment_template.yaml --stage evaluate --output outputs/exp
    python scripts/run_experiment.py --config configs/experiment_template.yaml --stage ablate   --output outputs/exp

On EC2, point the config's data section at real data (see docs/ec2_experiment_guide.md). By
default this runs on synthetic verdict data and produces synthetic smoke-test outputs.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from aegis_agency.cli import _load_cfg, _parse_seeds
from aegis_agency.experiments.run_ablation import run_ablation
from aegis_agency.experiments.run_calibration import run_calibration
from aegis_agency.experiments.run_evaluation import run_evaluation
from aegis_agency.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description="Aegis-Agency experiment runner.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=["calibrate", "evaluate", "ablate"])
    ap.add_argument("--output", default="outputs/experiment")
    ap.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds for the evaluate stage (e.g. 0,1,2,3,4); default: the "
        "config's single seed. Repeated sweeps yield evaluation_summary.csv (mean +/- std).",
    )
    ap.add_argument(
        "--cache",
        default=None,
        help="Override judges.verdict_cache (JSONL path). Pass a NEW path for a cold-cache run "
        "so measured RQ5 latency/token numbers come from real judge calls, not cache hits.",
    )
    args = ap.parse_args()

    configure_logging()
    cfg = _load_cfg(args.config)
    out = Path(args.output)
    if args.cache:
        cfg = replace(cfg, verdict_cache=args.cache)
        logger.info("Overriding judges.verdict_cache -> %s", args.cache)
    seeds = _parse_seeds(args.seeds)
    if seeds is not None and args.stage != "evaluate":
        logger.warning("--seeds only affects the evaluate stage; ignoring for '%s'.", args.stage)

    if args.stage == "calibrate":
        run_calibration(cfg, out)
    elif args.stage == "evaluate":
        run_evaluation(cfg, out, seeds=seeds)
    elif args.stage == "ablate":
        run_ablation(cfg, out)
    logger.info("Stage '%s' complete; outputs in %s.", args.stage, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

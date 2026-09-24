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
from pathlib import Path

from aegis_agency.cli import _load_cfg
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
    args = ap.parse_args()

    configure_logging()
    cfg = _load_cfg(args.config)
    out = Path(args.output)

    if args.stage == "calibrate":
        run_calibration(cfg, out)
    elif args.stage == "evaluate":
        run_evaluation(cfg, out)
    elif args.stage == "ablate":
        run_ablation(cfg, out)
    logger.info("Stage '%s' complete; outputs in %s.", args.stage, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

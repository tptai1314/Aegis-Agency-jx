#!/usr/bin/env python3
"""Run the synthetic smoke-test demo end to end.

    python scripts/run_synthetic_demo.py --config configs/synthetic_demo.yaml --output outputs/synthetic_demo

Produces calibration.json, evaluation_sweep.csv, and asr_vs_f.png -- all clearly labelled as
synthetic smoke-test output, NOT paper results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aegis_agency.cli import _load_cfg
from aegis_agency.experiments.plot_results import plot_evaluation_sweep
from aegis_agency.experiments.run_calibration import run_calibration
from aegis_agency.experiments.run_evaluation import run_evaluation
from aegis_agency.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description="Aegis-Agency synthetic smoke-test demo.")
    ap.add_argument("--config", default="configs/synthetic_demo.yaml")
    ap.add_argument("--output", default="outputs/synthetic_demo")
    args = ap.parse_args()

    configure_logging()
    cfg = _load_cfg(args.config)
    out = Path(args.output)

    logger.info("Running calibration ...")
    run_calibration(cfg, out)
    logger.info("Running evaluation sweep ...")
    run_evaluation(cfg, out)
    logger.info("Plotting ...")
    plot_evaluation_sweep(out / "evaluation_sweep.csv", out / "asr_vs_f.png", synthetic=True)
    logger.info("Done. Synthetic smoke-test outputs in %s (NOT paper results).", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

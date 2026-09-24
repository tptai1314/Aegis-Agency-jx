#!/usr/bin/env python3
"""Generate plots from result CSVs.

    python scripts/make_plots.py --input outputs/synthetic_demo/evaluation_sweep.csv \
        --output outputs/synthetic_demo/asr_vs_f.png

Plots are always generated from files, never from hard-coded numbers. Synthetic result plots
carry a visible 'synthetic smoke-test' banner.
"""

from __future__ import annotations

import argparse

from aegis_agency.experiments.plot_results import plot_evaluation_sweep
from aegis_agency.utils.logging import configure_logging


def main() -> int:
    ap = argparse.ArgumentParser(description="Aegis-Agency plotting.")
    ap.add_argument("--input", required=True, help="evaluation_sweep.csv path.")
    ap.add_argument("--output", required=True, help="Output image path.")
    ap.add_argument("--real", action="store_true", help="Mark as real (omit synthetic banner). Use ONLY for verified real results.")
    args = ap.parse_args()
    configure_logging()
    plot_evaluation_sweep(args.input, args.output, synthetic=not args.real)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

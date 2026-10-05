#!/usr/bin/env python3
"""Measure the paper's empirical quantities: r, gamma, rho (RQ4) and epsilon (RQ2).

    # homogeneous committee on one backbone (r, gamma, rho + epsilon isolation on/off)
    python scripts/measure_empirical.py --config configs/real_llama.yaml --output outputs/measure_llama

    # diverse committee (RQ4), several seeds for dispersion, capped epsilon sample
    python scripts/measure_empirical.py --config configs/real_diverse.yaml --output outputs/measure_diverse \
        --benchmark formal --seeds 0,1,2,3,4 --max-pairs 200

    # smoke-test the estimator offline (synthetic verdicts, no GPU)
    python scripts/measure_empirical.py --config configs/synthetic_demo.yaml --output outputs/measure_synthetic

This is an **additive** measurement entry point. It does not replace, reorder, or shorten the
``calibrate -> evaluate -> ablate`` protocol; it writes separate artefacts
(``empirical_committee.csv``, ``empirical_gamma.csv``, ``empirical_rho_pairs.csv``,
``empirical_isolation.csv``, ``empirical_provenance.json``).

Measured values are NOT paper results until their provenance is recorded in
``audits/result_integrity_audit.md``; ``is_paper_result`` stays ``False`` here by construction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aegis_agency.cli import _load_cfg, _parse_seeds
from aegis_agency.experiments.measure_empirical import (
    DEFAULT_TV_BINS,
    INJECTION_VARIANTS,
    run_measurement,
)
from aegis_agency.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description="Aegis-Agency empirical r/gamma/rho/epsilon measurement.")
    ap.add_argument("--config", required=True, help="YAML config (judge backend, backbones, data root).")
    ap.add_argument("--output", default="outputs/measure", help="Output directory.")
    ap.add_argument(
        "--benchmark",
        default=None,
        help="Benchmark for r/gamma/rho (default: data.benchmark from the config).",
    )
    ap.add_argument(
        "--pairs-benchmark",
        default="second_order",
        help="Benchmark whose _clean/_attack rows are used for epsilon (default: second_order).",
    )
    ap.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds, e.g. 0,1,2,3,4 (default: the config's single seed).",
    )
    ap.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Cap the number of clean/attack pairs used for epsilon (default: all pairs).",
    )
    ap.add_argument(
        "--variants",
        default=None,
        help="Comma-separated injection realisations used for epsilon, from "
        f"{sorted(INJECTION_VARIANTS)} (default: all of them). Definition 1 asks for a supremum "
        "over injections; the max over the measured family is the closest honest stand-in.",
    )
    ap.add_argument(
        "--tv-bins", type=int, default=DEFAULT_TV_BINS, help="Score bins for the binned-TV epsilon."
    )
    ap.add_argument(
        "--skip-isolation-ablation",
        action="store_true",
        help="Measure epsilon with isolation ON only, skipping the un-isolated arm (halves the calls).",
    )
    args = ap.parse_args()

    configure_logging()
    cfg = _load_cfg(args.config)
    out = Path(args.output)
    variants = (
        [v.strip() for v in args.variants.split(",") if v.strip()]
        if args.variants
        else tuple(INJECTION_VARIANTS)
    )

    result = run_measurement(
        cfg,
        out,
        benchmark=args.benchmark,
        pairs_benchmark=args.pairs_benchmark,
        isolation_modes=[True] if args.skip_isolation_ablation else [True, False],
        variants=variants,
        tv_bins=args.tv_bins,
        seeds=_parse_seeds(args.seeds),
        max_pairs=args.max_pairs,
    )
    logger.info(
        "Measurement complete: %d committee row(s), %d gamma row(s), %d isolation row(s) "
        "(%d variant(s)), %d isolation summary row(s) in %s (is_paper_result=%s).",
        len(result["committee"]),
        len(result["gamma"]),
        len(result["isolation"]),
        len(variants),
        len(result["isolation_summary"]),
        out,
        result["provenance"]["is_paper_result"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

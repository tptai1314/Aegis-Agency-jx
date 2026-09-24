"""Command-line interface for Aegis-Agency.

Examples
--------
    python -m aegis_agency.cli --help
    python -m aegis_agency.cli info
    python -m aegis_agency.cli demo   --config configs/synthetic_demo.yaml --output outputs/synthetic_demo
    python -m aegis_agency.cli evaluate --config configs/experiment_template.yaml --output outputs/eval
    python -m aegis_agency.cli plot   --input outputs/eval/evaluation_sweep.csv --output outputs/eval/asr_vs_f.png

All stages run on synthetic data by default and produce synthetic smoke-test outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from aegis_agency import __version__
from aegis_agency.experiments.harness import TrialConfig
from aegis_agency.experiments.plot_results import plot_evaluation_sweep
from aegis_agency.experiments.run_ablation import run_ablation
from aegis_agency.experiments.run_calibration import run_calibration
from aegis_agency.experiments.run_evaluation import run_evaluation
from aegis_agency.utils.io import load_yaml
from aegis_agency.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _config_from_dict(d: dict[str, Any]) -> TrialConfig:
    """Build a TrialConfig from a (possibly nested) config dict."""
    exp = d.get("experiment", d)
    fields = {f: exp[f] for f in TrialConfig.__dataclass_fields__ if f in exp}
    if "rules" in fields:
        fields["rules"] = tuple(fields["rules"])
    return TrialConfig(**fields)


def _load_cfg(path: str | None) -> TrialConfig:
    if path is None:
        return TrialConfig()
    return _config_from_dict(load_yaml(path))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aegis_agency", description="Byzantine-robust multi-agent LLM defense pipeline (Aegis-Agency).")
    p.add_argument("--version", action="version", version=f"aegis_agency {__version__}")
    p.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="Print package and mechanism information.")

    for name, help_text in [
        ("demo", "Run the synthetic smoke-test demo (calibrate + evaluate + plot)."),
        ("calibrate", "Calibrate thresholds on synthetic honest data."),
        ("evaluate", "Sweep the Byzantine fraction and record metrics."),
        ("ablate", "Run the component ablations."),
    ]:
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--config", default=None, help="YAML config path (defaults to built-in defaults).")
        sp.add_argument("--output", default="outputs/run", help="Output directory.")

    pp = sub.add_parser("plot", help="Plot an evaluation-sweep CSV.")
    pp.add_argument("--input", required=True, help="Path to evaluation_sweep.csv.")
    pp.add_argument("--output", required=True, help="Output image path.")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)

    if args.command == "info":
        print(f"Aegis-Agency v{__version__}")
        print("Byzantine-robust verdict aggregation for multi-agent LLM defense pipelines.")
        print("Aggregation rules: majority, cmed, gmed, krum.")
        print("Baselines: no_defense, single_model, majority_vote, autodefense.")
        print("Attacks: compromise, collusion, injection, adaptive.")
        print("NOTE: outputs generated here are synthetic smoke-test artefacts, not paper results.")
        return 0

    if args.command == "plot":
        plot_evaluation_sweep(args.input, args.output, synthetic=True)
        return 0

    cfg = _load_cfg(args.config)
    out = Path(args.output)

    if args.command == "calibrate":
        run_calibration(cfg, out)
    elif args.command == "evaluate":
        run_evaluation(cfg, out)
    elif args.command == "ablate":
        run_ablation(cfg, out)
    elif args.command == "demo":
        run_calibration(cfg, out)
        run_evaluation(cfg, out)
        plot_evaluation_sweep(out / "evaluation_sweep.csv", out / "asr_vs_f.png", synthetic=True)
        logger.info("Synthetic demo complete. Outputs in %s (synthetic smoke-test, not paper results).", out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

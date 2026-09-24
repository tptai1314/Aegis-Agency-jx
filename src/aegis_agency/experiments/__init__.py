"""Experiment harness and runners (synthetic mechanism validation + EC2-ready protocol)."""

from aegis_agency.experiments.harness import (
    TrialConfig,
    apply_attack,
    build_pipelines,
    run_trial,
    simulate_honest_committee,
    sweep_colluding_fraction,
)

__all__ = [
    "TrialConfig",
    "build_pipelines",
    "simulate_honest_committee",
    "apply_attack",
    "run_trial",
    "sweep_colluding_fraction",
]

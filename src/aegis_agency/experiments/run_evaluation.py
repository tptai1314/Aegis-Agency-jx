"""Evaluation stage: sweep the Byzantine fraction and record per-method metrics (RQ1).

Writes a tidy CSV (one row per (f, method)) plus a provenance JSON. Every artefact is a
synthetic smoke-test output, clearly flagged in provenance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from aegis_agency.experiments.harness import TrialConfig, sweep_colluding_fraction
from aegis_agency.utils.io import write_csv, write_json
from aegis_agency.utils.logging import get_logger
from aegis_agency.utils.provenance import RunProvenance

logger = get_logger(__name__)


def run_evaluation(
    cfg: TrialConfig,
    output_dir: str | Path,
    f_values: Sequence[int] | None = None,
) -> dict:
    """Run the colluding-fraction sweep and write results to ``output_dir``."""
    if f_values is None:
        # Sweep 0 .. floor((n-1)/2), the median-rule tolerance boundary (f < n/2).
        f_values = list(range((cfg.n_judges - 1) // 2 + 1))
    rows = sweep_colluding_fraction(cfg, f_values)
    out_dir = Path(output_dir)
    write_csv(out_dir / "evaluation_sweep.csv", rows)

    prov = RunProvenance(
        run_id="evaluation", stage="evaluate", seed=cfg.seed,
        config={
            "n_judges": cfg.n_judges, "attack": cfg.attack, "rules": list(cfg.rules),
            "margin": cfg.margin, "radius": cfg.radius, "correlation": cfg.correlation,
            "f_values": list(f_values),
        },
        data_source="synthetic",
    )
    write_json(out_dir / "evaluation_provenance.json", prov.to_dict())
    logger.info("Wrote %d rows to %s (synthetic smoke-test output).", len(rows), out_dir / "evaluation_sweep.csv")
    return {"rows": rows, "provenance": prov.to_dict()}

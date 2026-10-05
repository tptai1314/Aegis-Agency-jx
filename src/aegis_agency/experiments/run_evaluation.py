"""Evaluation stage: sweep the Byzantine fraction and record per-method metrics (RQ1).

Writes a tidy CSV (one row per ``(seed, f, method)``) plus:

* ``evaluation_summary.csv`` -- mean / std across seeds per ``(f, method)``;
* ``evaluation_cost.csv``    -- **measured** judge latency / token cost, written only when the
  judge backend records per-call statistics (real vLLM judges on the GPU server);
* ``evaluation_provenance.json`` -- config, seeds, data source.

Every artefact is flagged with its data source in provenance.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

from aegis_agency.experiments.harness import (
    TrialConfig,
    data_source_tag,
    summarize_sweep_rows,
    sweep_colluding_fraction,
)
from aegis_agency.utils.io import write_csv, write_json
from aegis_agency.utils.logging import get_logger
from aegis_agency.utils.provenance import RunProvenance

logger = get_logger(__name__)


def run_evaluation(
    cfg: TrialConfig,
    output_dir: str | Path,
    f_values: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
) -> dict:
    """Run the colluding-fraction sweep and write results to ``output_dir``.

    Parameters
    ----------
    f_values : sequence of int, optional
        Byzantine counts to sweep; defaults to ``0 .. floor((n_judges - 1) / 2)``.
    seeds : sequence of int, optional
        Seeds to repeat each sweep with. Defaults to ``[cfg.seed]`` (single-seed behaviour,
        unchanged from before); pass e.g. ``[0, 1, 2, 3, 4]`` to obtain the mean ± std the
        paper's reproducibility checklist asks for.
    """
    if f_values is None:
        # Sweep 0 .. floor((n-1)/2), the median-rule tolerance boundary (f < n/2).
        f_values = list(range((cfg.n_judges - 1) // 2 + 1))
    seed_list = [int(s) for s in seeds] if seeds else [int(cfg.seed)]

    rows: list[dict] = []
    cost_rows: list[dict] = []
    cost_payload_rows: list[dict] = []
    for seed in seed_list:
        rows.extend(
            sweep_colluding_fraction(
                replace(cfg, seed=seed),
                f_values,
                cost_rows=cost_rows,
                cost_payload_rows=cost_payload_rows,
            )
        )

    out_dir = Path(output_dir)
    write_csv(out_dir / "evaluation_sweep.csv", rows)
    summary = summarize_sweep_rows(rows)
    write_csv(out_dir / "evaluation_summary.csv", summary)
    if cost_rows:
        write_csv(out_dir / "evaluation_cost.csv", cost_rows)
    if cost_payload_rows:
        write_csv(out_dir / "evaluation_cost_payloads.csv", cost_payload_rows)

    prov = RunProvenance(
        run_id="evaluation", stage="evaluate", seed=cfg.seed,
        config={
            "n_judges": cfg.n_judges, "attack": cfg.attack, "rules": list(cfg.rules),
            "margin": cfg.margin, "radius": cfg.radius, "correlation": cfg.correlation,
            "f_values": list(f_values), "seeds": seed_list,
            "data_root": cfg.data_root, "benchmark": cfg.benchmark, "data_split": cfg.data_split,
            "judge_backend": cfg.judge_backend, "models": list(cfg.models),
            "isolation": cfg.isolation, "max_resident_engines": cfg.max_resident_engines,
            "temperature": cfg.temperature, "max_tokens": cfg.max_tokens,
            "model_revision": cfg.model_revision or "<unpinned>",
            "cost_model": {
                "l_in": cfg.l_in, "l_out": cfg.l_out, "analyze_tokens": cfg.analyze_tokens,
                "judge_latency_model": cfg.judge_latency_model,
                "agg_latency_model": cfg.agg_latency_model,
            },
        },
        data_source=data_source_tag(cfg),
    )
    write_json(out_dir / "evaluation_provenance.json", prov.to_dict())
    logger.info(
        "Wrote %d sweep rows and %d summary rows%s%s to %s (%s smoke-test output).",
        len(rows),
        len(summary),
        f" plus {len(cost_rows)} measured-cost rows" if cost_rows else "",
        f" and {len(cost_payload_rows)} per-payload cost rows" if cost_payload_rows else "",
        out_dir,
        data_source_tag(cfg),
    )
    return {
        "rows": rows,
        "summary": summary,
        "cost_rows": cost_rows,
        "cost_payload_rows": cost_payload_rows,
        "provenance": prov.to_dict(),
    }

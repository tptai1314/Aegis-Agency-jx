"""Ablation stage: isolate each Aegis component (Section 10 ablations).

Ablations:
* robust aggregation on/off  -> Aegis (gmed) vs AutoDefense (mean coordinator).
* payload isolation on/off   -> epsilon = 0 vs epsilon > 0 under the injection attack.
* hardened judges on/off     -> small vs large honest radius r (proxy for hardening quality).
* committee size             -> n in {1..7}.
* backbone diversity         -> correlation rho low vs high (RQ4).

Writes a tidy CSV, plus measured cost files when the judge backend records per-call
measurements (real vLLM judges): ``ablation_cost.csv`` (per ablation / committee size) and
``ablation_cost_payloads.csv`` (per payload, with measured parallel/sequential latency).
The committee-size rows (n = 1..7) are the substrate for the RQ5 cost frontier.

Synthetic smoke-test output only unless a real judge backend is configured.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from aegis_agency.experiments.harness import (
    TrialConfig,
    data_source_tag,
    make_ablation_row,
    run_trial,
)
from aegis_agency.utils.io import write_csv, write_json
from aegis_agency.utils.logging import get_logger
from aegis_agency.utils.provenance import RunProvenance

logger = get_logger(__name__)


def run_ablation(cfg: TrialConfig, output_dir: str | Path) -> dict:
    """Run the component ablations and write a tidy CSV."""
    rows: list[dict] = []
    cost_rows: list[dict] = []
    cost_payload_rows: list[dict] = []

    # Ablations isolate *aggregation robustness*, so we hold the threshold fixed (tau = 0.5,
    # which cleanly separates the synthetic class-conditional scores given a positive margin)
    # rather than recalibrating per row -- otherwise small-sample calibration fallbacks would
    # confound the component comparison. Also ensure enough payloads for stable rates.
    cfg = replace(cfg, calibrate=False, n_payloads=max(cfg.n_payloads, 400))

    def record(label: str, trial_cfg: TrialConfig, methods: tuple[str, ...]) -> None:
        out = run_trial(trial_cfg)
        for method in methods:
            if method in out["methods"]:
                rows.append(
                    make_ablation_row(
                        ablation=label,
                        method=method,
                        seed=trial_cfg.seed,
                        metrics=out["methods"][method],
                    )
                )
        measured = out.get("cost_measured")
        if measured:
            cost_rows.append(
                {
                    "ablation": label,
                    "seed": trial_cfg.seed,
                    "n_judges": trial_cfg.n_judges,
                    "f": trial_cfg.f,
                    **measured,
                }
            )
        for payload_id, cost in sorted((out.get("cost_payloads") or {}).items()):
            cost_payload_rows.append(
                {
                    "ablation": label,
                    "seed": trial_cfg.seed,
                    "n_judges": trial_cfg.n_judges,
                    "f": trial_cfg.f,
                    "payload_id": payload_id,
                    **cost,
                }
            )

    # robust aggregation on/off
    record("robust_agg_on", replace(cfg, rules=("gmed",), attack="collusion"), ("aegis_gmed",))
    record("robust_agg_off", replace(cfg, rules=("gmed",), attack="collusion"), ("autodefense",))
    # payload isolation on/off (injection attack)
    record("isolation_on", replace(cfg, rules=("gmed",), attack="injection", attack_kwargs={"epsilon": 0.0}), ("aegis_gmed",))
    record("isolation_off", replace(cfg, rules=("gmed",), attack="injection", attack_kwargs={"epsilon": 0.5}), ("aegis_gmed",))
    # hardened judges on/off (radius proxy)
    record("hardened_on", replace(cfg, rules=("gmed",), radius=0.05), ("aegis_gmed",))
    record("hardened_off", replace(cfg, rules=("gmed",), radius=0.25), ("aegis_gmed",))
    # backbone diversity (rho)
    record("diverse_backbones", replace(cfg, rules=("gmed",), correlation=0.0), ("aegis_gmed",))
    record("homogeneous_backbones", replace(cfg, rules=("gmed",), correlation=0.8), ("aegis_gmed",))
    # committee size
    for n in (1, 3, 5, 7):
        record(f"committee_n{n}", replace(cfg, n_judges=n, rules=("gmed",), f=min(cfg.f, (n - 1) // 2)), ("aegis_gmed",))

    out_dir = Path(output_dir)
    write_csv(out_dir / "ablation.csv", rows)
    if cost_rows:
        write_csv(out_dir / "ablation_cost.csv", cost_rows)
    if cost_payload_rows:
        write_csv(out_dir / "ablation_cost_payloads.csv", cost_payload_rows)
    prov = RunProvenance(
        run_id="ablation", stage="ablation", seed=cfg.seed,
        config={"n_judges": cfg.n_judges, "data_root": cfg.data_root, "benchmark": cfg.benchmark,
             "judge_backend": cfg.judge_backend, "models": list(cfg.models),
             "isolation": cfg.isolation, "max_resident_engines": cfg.max_resident_engines,
             "temperature": cfg.temperature, "max_tokens": cfg.max_tokens,
             "model_revision": cfg.model_revision or "<unpinned>",
             "cost_model": {
                 "l_in": cfg.l_in, "l_out": cfg.l_out, "analyze_tokens": cfg.analyze_tokens,
                 "judge_latency_model": cfg.judge_latency_model,
                 "agg_latency_model": cfg.agg_latency_model,
             }},
        data_source=data_source_tag(cfg),
    )
    write_json(out_dir / "ablation_provenance.json", prov.to_dict())
    logger.info(
        "Wrote %d ablation rows%s to %s (%s).",
        len(rows),
        f" plus {len(cost_rows)} measured-cost rows" if cost_rows else "",
        out_dir / "ablation.csv",
        data_source_tag(cfg),
    )
    return {
        "rows": rows,
        "cost_rows": cost_rows,
        "cost_payload_rows": cost_payload_rows,
        "provenance": prov.to_dict(),
    }

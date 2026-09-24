"""Ablation stage: isolate each Aegis component (Section 10 ablations).

Ablations:
* robust aggregation on/off  -> Aegis (gmed) vs AutoDefense (mean coordinator).
* payload isolation on/off   -> epsilon = 0 vs epsilon > 0 under the injection attack.
* hardened judges on/off     -> small vs large honest radius r (proxy for hardening quality).
* committee size             -> n in {1..7}.
* backbone diversity         -> correlation rho low vs high (RQ4).

Writes a tidy CSV. Synthetic smoke-test output only.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from aegis_agency.experiments.harness import TrialConfig, run_trial
from aegis_agency.utils.io import write_csv, write_json
from aegis_agency.utils.logging import get_logger
from aegis_agency.utils.provenance import RunProvenance

logger = get_logger(__name__)


def run_ablation(cfg: TrialConfig, output_dir: str | Path) -> dict:
    """Run the component ablations and write a tidy CSV."""
    rows: list[dict] = []

    # Ablations isolate *aggregation robustness*, so we hold the threshold fixed (tau = 0.5,
    # which cleanly separates the synthetic class-conditional scores given a positive margin)
    # rather than recalibrating per row -- otherwise small-sample calibration fallbacks would
    # confound the component comparison. Also ensure enough payloads for stable rates.
    cfg = replace(cfg, calibrate=False, n_payloads=max(cfg.n_payloads, 400))

    def record(label: str, trial_cfg: TrialConfig, methods: tuple[str, ...]) -> None:
        out = run_trial(trial_cfg)
        for method in methods:
            if method in out["methods"]:
                m = out["methods"][method]
                rows.append({
                    "ablation": label, "method": method,
                    "asr_uc": m["asr_uc"], "orr": m["orr"],
                    "defense_success_rate": m["defense_success_rate"],
                })

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
    prov = RunProvenance(
        run_id="ablation", stage="ablation", seed=cfg.seed,
        config={"n_judges": cfg.n_judges}, data_source="synthetic",
    )
    write_json(out_dir / "ablation_provenance.json", prov.to_dict())
    logger.info("Wrote %d ablation rows to %s (synthetic).", len(rows), out_dir / "ablation.csv")
    return {"rows": rows, "provenance": prov.to_dict()}

# Aegis-Agency

**Byzantine-robust, injection-hardened multi-agent LLM defense pipelines — reference implementation.**

This repository implements the mechanism of the manuscript *Aegis-Agency: Byzantine-Robust,
Injection-Hardened Multi-Agent Defense Pipelines for Large Language Models*
(`AegisAgency_main.pdf`, in this project folder): a committee of hardened LLM "judge" agents
inspects a candidate output, and their verdicts are combined by a **Byzantine-robust
aggregation rule** (coordinate-wise median, geometric median, or Krum) instead of a single
trusted coordinator, while the untrusted payload is **isolated** from the judges' instruction
channel.

> ⚠️ **This repository does not automatically download datasets or run paper-level
> experiments. Real experiments are intended to be executed on EC2 after the required
> datasets, benchmarks, models, and credentials are manually provisioned.**
>
> ⚠️ **No paper-level experimental results are claimed by this repository unless real
> benchmark outputs are generated and their provenance is recorded.** Everything produced by
> the synthetic demo is a *smoke-test artefact*, not a paper result.

## Why a verdict-space implementation
The paper's theory is stated over **verdict vectors** `u_k = (s_k, e_k)` (Eq. 2), so the core
mechanism — aggregation rules, attacks, metrics, and the theorems — is implemented and tested
directly in that space. This lets the algorithm and math be validated offline with no LLM,
while real hardened LLM judges and jailbreak/injection benchmarks are reached through
documented adapters for EC2. See `docs/implementation_notes.md`.

## Paper-to-code map (summary)
| Paper | Code |
|---|---|
| Verdict / verdict vector (Eq. 1-2) | `src/aegis_agency/data/schemas.py` |
| Aggregation: cmed / gmed / Krum / majority (Alg. 2) | `src/aegis_agency/methods/aggregators.py` |
| Aegis adjudication (Alg. 1) | `src/aegis_agency/methods/gate.py` |
| Attacks: compromise / collusion / injection / adaptive (§5.3) | `src/aegis_agency/attacks/` |
| Metrics: ASR-UC, ORR, tolerance, detection F1/AUROC (Eq. 4-5, §10) | `src/aegis_agency/metrics/metrics.py` |
| Theory: C_alpha, integrity condition, correlated variance (L1, T1, P3) | `src/aegis_agency/metrics/theory.py` |
| Baselines: no-defense / single-model / majority / AutoDefense (§10) | `src/aegis_agency/baselines/` |
Full mapping: `audits/paper_to_code_traceability.csv`.

## Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # runtime + pytest/ruff/mypy
```
Requires Python ≥ 3.11. No GPU, no network needed for tests or the demo.

## Quick start (synthetic smoke-test)
```bash
make demo
# or:
python scripts/run_synthetic_demo.py --config configs/synthetic_demo.yaml --output outputs/synthetic_demo
```
Produces `outputs/synthetic_demo/{calibration.json, evaluation_sweep.csv, asr_vs_f.png}`,
all labelled synthetic. The demo sweeps the Byzantine count `f` and shows the robust rules
holding ASR-under-compromise near zero while the single-model baseline degrades.

CLI:
```bash
python -m aegis_agency.cli --help
python -m aegis_agency.cli info
python -m aegis_agency.cli evaluate --config configs/experiment_template.yaml --output outputs/eval
python -m aegis_agency.cli plot --input outputs/eval/evaluation_sweep.csv --output outputs/eval/asr.png
```

## Real-data usage (EC2)
This repo never downloads data. Place benchmarks on disk (see `docs/data_format.md`), wire the
real LLM-judge and baseline adapters (`docs/baseline_adapters.md`), and follow
`docs/ec2_experiment_guide.md`. Then run `scripts/run_experiment.py` stages against real
verdicts.

## Folder structure
```
AegisAgency/
  README.md  pyproject.toml  requirements.txt  Makefile  .gitignore
  configs/        default.yaml  synthetic_demo.yaml  experiment_template.yaml
  src/aegis_agency/
    cli.py
    data/         schemas.py  synthetic.py  adapters.py
    judges/       base.py  synthetic_judges.py  isolation.py
    methods/      aggregators.py  calibration.py  gate.py
    attacks/      compromise.py  collusion.py  injection.py  adaptive.py
    metrics/      metrics.py  theory.py  confidence_intervals.py  cost.py
    baselines/    no_defense.py  single_model.py  majority_vote.py  autodefense.py  external_wrappers.py
    experiments/  harness.py  run_calibration.py  run_evaluation.py  run_ablation.py  plot_results.py
    utils/        logging.py  seeding.py  io.py  validation.py  provenance.py
  scripts/        run_synthetic_demo.py  run_experiment.py  make_plots.py
  tests/          (47 tests)
  examples/       example_config.yaml  synthetic_data/  README.md
  docs/           implementation_notes.md  data_format.md  baseline_adapters.md  reproducibility.md  ec2_experiment_guide.md
  outputs/        (regenerable synthetic artefacts)
  audits/         paper_implementation_spec.md  paper_to_code_traceability.csv  math_to_code_audit.md
                  implementation_gaps.md  result_integrity_audit.md
  TODO_IMPLEMENTATION.md
```

## Implemented algorithms
Coordinate-wise median, geometric median (smoothed Weiszfeld), Krum selection, plain majority
vote; the Aegis adjudication gate (analyze → isolate → judge → robust-aggregate → allow/block/
escalate); threshold + temperature calibration.

## Metrics
ASR-under-compromise (Eq. 4), over-refusal rate (Eq. 5), defense success rate, Byzantine
tolerance fraction, malicious-verdict detection F1/AUROC, utility retention, group-conditional
ASR; token/latency cost models; Wilson / bootstrap confidence intervals; theory checks
(C_alpha, integrity condition, correlated-failure variance, injection flip bound).

## Baselines
Implemented: always-submit, single hardened model, plain majority vote, AutoDefense
(single-Coordinator mean). External (adapter stubs, EC2): real AutoDefense, SecAlign, StruQ,
JudgeDeceiver. A dummy baseline exists for tests only and is never presented as a real system.

## Configuration guide
YAML configs under `configs/` drive the synthetic simulation: committee size `n_judges`,
aggregation `rules`, honest margin/radius/correlation (`margin`/`radius`/`correlation`),
`attack`, Byzantine count `f`, and calibration target. The `data:` and `judges:` sections in
`experiment_template.yaml` are consumed only by the EC2 real-data adapters.

## Reproducibility
Fixed seeds via `utils/seeding.get_rng`; provenance JSON with every result; plots generated
from files with synthetic banners. See `docs/reproducibility.md`.

## Limitations and known gaps
- The synthetic layer validates the **mechanism and math**, not the paper's empirical claims.
- Real LLM judges, benchmarks, and external baselines are **adapter stubs** (EC2 only).
- Krum's constant is not re-derived (the paper restates it); the aggregator is implemented.
- `epsilon`-isolation and honest concentration are modelled parametrically in synthetic runs;
  their real values must be measured on EC2 (RQ2/RQ4).
Full lists: `audits/implementation_gaps.md`, `TODO_IMPLEMENTATION.md`.

## Tests
```bash
pytest -q          # 47 tests, synthetic only, no network
ruff check .       # lint (passes)
mypy src           # types (passes)
```

## License
MIT (placeholder — confirm before release).

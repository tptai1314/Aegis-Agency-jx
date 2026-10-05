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
> experiments. Real experiments are run on a GPU server after the required datasets,
> benchmarks, models, and credentials are manually provisioned** — see
> `docs/gpu_server_runbook.md` (SSH workflow) and `docs/ec2_experiment_guide.md`.
>
> ⚠️ **No paper-level experimental results are claimed by this repository unless real
> benchmark outputs are generated and their provenance is recorded.** Everything produced by
> the synthetic demo is a *smoke-test artefact*, not a paper result.

## Why a verdict-space implementation
The paper's theory is stated over **verdict vectors** `u_k = (s_k, e_k)` (Eq. 2), so the core
mechanism — aggregation rules, attacks, metrics, and the theorems — is implemented and tested
directly in that space. This lets the algorithm and math be validated offline with no LLM,
while real hardened LLM judges and jailbreak/injection benchmarks are reached through
documented adapters for a GPU server. See `docs/implementation_notes.md`.

## Paper-to-code map (summary)
| Paper | Code |
|---|---|
| Verdict / verdict vector (Eq. 1-2) | `src/aegis_agency/data/schemas.py` |
| Aggregation: cmed / gmed / Krum / majority (Alg. 2) | `src/aegis_agency/methods/aggregators.py` |
| Aegis adjudication (Alg. 1) | `src/aegis_agency/methods/gate.py` |
| Attacks: compromise / collusion / injection / adaptive / adaptive_search (§5.3) | `src/aegis_agency/attacks/` |
| Metrics: ASR-UC, ORR, tolerance, detection F1/AUROC (Eq. 4-5, §10) | `src/aegis_agency/metrics/metrics.py` |
| Theory: C_alpha, integrity condition, correlated variance (L1, T1, P3) | `src/aegis_agency/metrics/theory.py` |
| Baselines: no-defense / single-model / majority / AutoDefense (§10) | `src/aegis_agency/baselines/` |
| Empirical `r`, `gamma`, `rho`, `epsilon` (Asm. 1-2, Prop. 3, Def. 1) | `src/aegis_agency/experiments/measure_empirical.py` |
Full mapping: `audits/paper_to_code_traceability.csv`.

## Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # runtime + pytest/ruff/mypy
```
Requires Python ≥ 3.11. No GPU, no network needed for tests or the demo. Real judges need the
extra: `pip install -e ".[dev,vllm]"` (Linux + CUDA).

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

## Real-data usage (GPU server)
This repo never downloads data. Place benchmarks on disk (see `docs/data_format.md`) and
point a config's `data:` section at them — the harness loads real benchmark payloads via
`CsvBenchmarkAdapter` (see `configs/real_experiment.yaml`, and `configs/real_diverse.yaml` for a
diverse committee). While verdicts are synthetic it produces labelled real-traffic smoke tests
(`data_source="benchmark:<name>"`), not paper numbers. Wire the remaining baseline adapters
(`docs/baseline_adapters.md`), follow `docs/gpu_server_runbook.md`, then run
`scripts/run_experiment.py` stages against real verdicts. `scripts/measure_empirical.py`
measures the paper's empirical quantities (honest radius `r`, margin `gamma`, correlation
`rho`, isolation leakage `epsilon` over a family of injection realisations and with the
isolation ablation) from the same judges.

To run the whole protocol (RQ1-RQ6) in one submission — queue friendly, resumable:

```bash
bash scripts/run_all.sh --dry-run      # preview every command
bash scripts/run_all.sh                # real run on the GPU box
sbatch scripts/slurm_run_all.sbatch    # submit to SLURM (see docs/run_all_commands.md)
```

## Folder structure
```
AegisAgency/
  README.md  pyproject.toml  requirements.txt  Makefile  .gitignore
  configs/        default.yaml  synthetic_demo.yaml  experiment_template.yaml  real_experiment.yaml  real_diverse.yaml
  src/aegis_agency/
    cli.py
    data/         schemas.py  synthetic.py  adapters.py
    judges/       base.py  synthetic_judges.py  isolation.py  llm_judge.py
    methods/      aggregators.py  calibration.py  gate.py
    attacks/      compromise.py  collusion.py  injection.py  adaptive.py
    metrics/      metrics.py  theory.py  confidence_intervals.py  cost.py
    baselines/    no_defense.py  single_model.py  majority_vote.py  autodefense.py
                  external_wrappers.py  llm_baselines.py
    experiments/  harness.py  run_calibration.py  run_evaluation.py  run_ablation.py  plot_results.py
                  measure_empirical.py
    utils/        logging.py  seeding.py  io.py  validation.py  provenance.py  cache.py
  scripts/        run_synthetic_demo.py  run_experiment.py  make_plots.py  measure_empirical.py
                  run_all.sh  slurm_run_all.sbatch
  tests/          (163 tests)
  examples/       example_config.yaml  synthetic_data/  README.md
  docs/           implementation_notes.md  data_format.md  baseline_adapters.md  reproducibility.md  ec2_experiment_guide.md
                  gpu_server_runbook.md  run_all_commands.md  runner_metrics_plan.md
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
ASR; model-based and **measured** token/latency cost; Wilson / bootstrap confidence intervals;
theory checks (C_alpha, integrity condition, correlated-failure variance, injection flip bound).

## Baselines
Implemented: always-submit, single hardened model, plain majority vote, AutoDefense
(single-Coordinator mean). External systems are **re-implemented** on the local judge machinery
(`baselines/llm_baselines.py`: AutoDefense as analyzer → judge(s) → coordinator, plus SecAlign /
StruQ hardened single models) and can run head-to-head via `experiment.external_baselines`. They
are re-implementations from the published designs, not the authors' code, and their checkpoint
ids / prompt templates must be verified before numbers are reported; their rows describe the
system *without* a compromised judge. The JudgeDeceiver optimiser remains a stub (its generated
payload pairs ship in `data/benchmarks/second_order/`). A dummy baseline exists for tests only
and is never presented as a real system.

## Configuration guide
YAML configs under `configs/` drive the synthetic simulation: committee size `n_judges`,
aggregation `rules`, honest margin/radius/correlation (`margin`/`radius`/`correlation`),
`attack`, Byzantine count `f`, and calibration target. The `data:` section points at on-disk
benchmarks and the `judges:` section configures real judges (`backend`, `backbones`,
`isolation`, `max_resident_engines`, `temperature`, `max_tokens`, `model_revision`,
`verdict_cache`); the optional `cost:` section supplies the analytic cost parameters behind the
`*_model` CSV columns.

## Reproducibility
Fixed seeds via `utils/seeding.get_rng`; provenance JSON with every result (including the judge
side controls and decoding parameters); plots generated from files with synthetic banners.
`evaluate` can sweep several seeds (`--seeds 0,1,2,3,4`) and additionally writes
`evaluation_summary.csv` (mean ± std across seeds) plus measured cost files when real judges
report latency/tokens. See `docs/reproducibility.md`.

## Limitations and known gaps
- The synthetic layer validates the **mechanism and math**, not the paper's empirical claims.
- The real LLM-judge path *is* implemented (`judges/llm_judge.py`: vLLM, isolation on/off,
  homogeneous and round-robin diverse committees, verdict cache, measured latency/tokens), but it
  has never been executed here — no GPU run is recorded in this checkout. It currently emits
  **score-only** verdicts (`m = 0`, no rationale embedding), so the verdict vector is `d = 1`
  rather than the paper's `d = 1 + m`; the aggregation, attacks and metrics are unaffected.
  External baselines (real AutoDefense / SecAlign / StruQ / JudgeDeceiver) remain adapter stubs.
- `epsilon`-isolation and honest concentration are modelled parametrically in synthetic runs;
  `scripts/measure_empirical.py` measures them from real judges, but the values must be produced
  on the GPU server (RQ2/RQ4).
- RQ5 latency/token numbers require a **cold-cache** run; cached runs report no measured cost.
- Krum's constant is not re-derived (the paper restates it); the aggregator is implemented.
Full lists: `audits/implementation_gaps.md`, `TODO_IMPLEMENTATION.md`.

## Tests
```bash
pytest -q          # 163 tests, synthetic only, no network
ruff check .       # lint (passes)
mypy src           # types (passes)
```

## License
MIT (placeholder — confirm before release).

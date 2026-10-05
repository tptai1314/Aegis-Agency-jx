# Reproducibility

## Seeds
All stochastic code draws from an explicit `numpy.random.Generator` created via
`aegis_agency.utils.seeding.get_rng(seed)`. Every config carries a `seed`; the same seed and
config reproduce identical trials (see `tests/test_reproducibility.py`). For whole-process
determinism (third-party libraries), call `set_global_seed(seed)`.

## Config capture
Each runner writes a `*provenance.json` (see `utils/provenance.py`) recording:
`run_id`, `stage`, `seed`, `config`, `data_source` (`synthetic`/`real`), `is_paper_result`
(always `False` for synthetic runs), package version, Python version, and platform. The
`config` block also records the judge-side controls that a reader needs to reproduce a run:
`models`, `isolation`, `max_resident_engines`, `temperature`, `max_tokens`, and
`model_revision` (`"<unpinned>"` when no HF revision was pinned — a pinned revision is passed
through to the vLLM engine and gets its own engine-registry entry).

## Result provenance
- Tables/plots are always generated **from result files**, never from hard-coded numbers.
- Synthetic plots carry a visible "synthetic smoke-test — NOT a paper result" banner.
- `tests/test_reproducibility.py::test_no_committed_result_files_claim_paper_results` fails
  the suite if any shipped provenance file claims `is_paper_result: true`.

## Metric export (runner extension — docs/runner_metrics_plan.md)
`run_evaluation` writes result files that are all generated from code, never hard-coded:

| File | Content |
|------|---------|
| `evaluation_sweep.csv` | one row per `(seed, f, method)`. The **first six columns are the historical schema** (`f, n_judges, method, asr_uc, orr, defense_success_rate`) in that exact order with unchanged values. Appended: `utility_retention`, `escalation_rate`, `byzantine_tolerance_fraction`, Wilson 95 % bounds for ASR-UC and ORR, `detection_auroc` / `detection_f1` (NaN when the committee has no mixed Byzantine mask), the **analytic** cost columns `tokens_model` / `latency_parallel_model` / `latency_sequential_model` (NaN unless a `cost:` config section is supplied — never a misleading zero), `group_asr_json` (strict JSON; non-finite rates become `null`) and `seed`. |
| `evaluation_summary.csv` | mean and sample std (ddof = 1; NaN with a single seed) across seeds per `(n_judges, f, method)`, plus `n_seeds` and the seeds used. |
| `evaluation_cost.csv` | **measured** judge cost per `(seed, f)`: `n_calls`, latency mean/p50/p95, prompt/completion tokens. Written only when the judge backend records per-call measurements (real vLLM judges). |
| `evaluation_cost_payloads.csv` | **measured** per-payload cost: `max_latency_s` = Eq. (lat-par) `max_k lat(J_k)` and `sum_latency_s` = Eq. (lat-seq) `sum_k lat(J_k)` over the judge calls actually served, plus token counts. This is the measured RQ5 substrate. |
| `ablation.csv` | the same metric columns, keeping the historical prefix `ablation, method, asr_uc, orr, defense_success_rate`. **Backend caveat:** the `diverse_backbones` / `homogeneous_backbones` rows are driven by the *synthetic* correlation parameter `rho` and `hardened_on/off` by `radius` `r`; with a real judge backend those knobs do nothing, so those rows are the same configuration repeated. RQ4 evidence for a real run comes from comparing the homogeneous and diverse **configs** plus `empirical_rho_pairs.csv`, not from these rows. |
| `ablation_cost.csv`, `ablation_cost_payloads.csv` | as above, keyed by `ablation` + `n_judges`: the measured cost of the ablation trials **as configured**. With a shared cache the later committee-size rows have few or no calls, so do **not** read the RQ5 frontier off this file — produce it with one cold-cache `evaluate` per `n` (see `docs/gpu_server_runbook.md` §7.1). |

Cache hits are not calls, so a fully cached run writes **no** cost file rather than a fabricated
zero. Per-payload rows only cover payloads for which a call was actually made.

### External-system rows (`experiment.external_baselines`)

When a re-implemented external system (AutoDefense / SecAlign / StruQ) is enabled, its row in
`evaluation_sweep.csv` comes from the system's **own** verdicts, not from the committee, so:
its row describes the system **without a compromised judge** (a committee-level Byzantine attack is
not injectable into a system with no verdict interface) — do not read its `asr_uc` as the same
quantity as `asr_uc(f)` of an Aegis committee, and say so in the write-up. `describe()` on the
adapter records `kind: "reimplementation"`, the role/backbone layout, the cache namespaces, the
prompt sources and the standing caveats.

### Empirical quantities (RQ2/RQ4): `scripts/measure_empirical.py`

| File | Content |
|------|---------|
| `empirical_committee.csv` | one row per `(label, seed)`: `r` profile (`r_p50/p90/p95/max/rms`), honest-disagreement `r_within_p95/max`, `gamma_min`, `margin_condition_ok`, and `rho` at the decision / residual / raw-score level with the number of judge pairs whose correlation is defined. |
| `empirical_gamma.csv` | per class: estimated reference score `pi_1(u*)`, `gamma`, whether the reference decision equals ground truth, and `condition_ok`. |
| `empirical_rho_pairs.csv` | per judge pair (`backbone_a`, `backbone_b`): the three correlations — the RQ4 diversity diagnostic. |
| `empirical_isolation.csv` | per `(label, seed, isolation, variant)`: `epsilon_decision_flip` (the per-judge event probability Proposition 2 bounds), `epsilon_tv_score` (binned total-variation estimate of Definition 1), `mean_abs_score_shift`, clean/attack block rates, and per-judge flip rates as strict JSON. |
| `empirical_isolation_summary.csv` | per `(label, seed, isolation)`: the **worst case over injection variants** (`epsilon_decision_flip_max` + which variant attained it, `epsilon_tv_score_max`, `mean_abs_score_shift_max`) and the mean over variants. Definition 1 asks for a supremum over injections, which is unattainable; this max is the closest honest stand-in and must be reported as such, never as the supremum. |
| `empirical_provenance.json` | config, seeds, variants, reference convention, and the **caveats** that must accompany any reported number. |

Definitions and conventions (matching the manuscript):

- `u*` is estimated per ground-truth class as the **class-conditional mean of honest verdicts** —
  it is not known a priori; `r` is therefore reported as a profile rather than one number.
- `gamma` uses `tau` from the config and additionally checks `D(u*) = y*` (Assumption 2).
- `rho` follows Proposition 3 (`Z_k = 1{J_k decides correctly}`). A homogeneous temperature-0
  committee has no between-judge variance, so `rho` is **undefined** there and reported as NaN
  with zero defined pairs.
- `epsilon` is measured over a **family** of injection realisations (default: `judgedeceiver`,
  `naive_override`, `roleplay_takeover`, `delimiter_escape`) rather than a single attack. Each
  variant gets its own payload id, so no variant can read another's cached verdict. Report the
  per-variant rows plus the max over variants; Definition 1 asks for a supremum over injections,
  which is not attainable empirically.
- The un-isolated arm (RQ2 ablation) uses a **separate verdict-cache namespace**
  (`<backbone>+noisolation`), so it can never read isolated verdicts.
- Values produced with the `synthetic` backend are plumbing checks only, never measurements.

Real values are not paper results until `audits/result_integrity_audit.md` records their
provenance; `is_paper_result` stays `false` in these files by construction.


Seed sweep (for the mean ± std the checklist below asks for):

```bash
python scripts/run_experiment.py --config configs/real_llama.yaml --stage evaluate \
    --output outputs/real_vllm_llama --seeds 0,1,2,3,4
# or: python -m aegis_agency.cli evaluate --config <cfg> --output <dir> --seeds 0,1,2,3,4
```

Default behaviour is unchanged: with no `--seeds` the sweep uses the config's single seed.
Per-seed dispersion is reported as Wilson intervals in `evaluation_sweep.csv` and as mean ± std
across seeds in `evaluation_summary.csv`; no significance test is applied.

## Environment
- Python >= 3.11.
- Runtime deps: `numpy`, `pyyaml`, `matplotlib` (see `requirements.txt`).
- Dev deps: `pytest`, `ruff`, `mypy`, `pandas` (`pip install -e ".[dev]"`).
- No GPU, no network access required for tests or the synthetic demo.

## Commands
```bash
pip install -e ".[dev]"
pytest -q                 # 120 tests, synthetic only
ruff check .              # lint
mypy src                  # types
make demo                 # synthetic smoke-test demo -> outputs/synthetic_demo/
```

## Determinism caveats
- Floating-point reductions (e.g. geometric-median Weiszfeld) are deterministic for a fixed
  input but may differ at the last ULP across BLAS builds; comparisons in tests use
  tolerances.
- `matplotlib` is used in `Agg` (headless) mode; the first run builds a font cache.

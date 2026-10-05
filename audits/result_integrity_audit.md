# Result Integrity Audit

## Were any real experiments run?
**No.** No real LLM judge, no real benchmark, and no external baseline was executed during
repository creation. The environment was used only to build the code, run unit tests on
synthetic data, and generate a synthetic smoke-test demo.

## Are the outputs synthetic smoke-test outputs?
**Yes.** Everything under `outputs/` is generated from the synthetic verdict-space simulation
(`aegis_agency.judges.synthetic_judges` + `aegis_agency.experiments.harness`). Each result
carries a provenance record with `data_source: "synthetic"` and `is_paper_result: false`.

## Where did each shipped output come from?
| File | Origin | Real result? |
|------|--------|--------------|
| `outputs/synthetic_demo/calibration.json` | `run_calibration` on synthetic honest data | No — synthetic |
| `outputs/synthetic_demo/evaluation_sweep.csv` | `run_evaluation` colluding-fraction sweep, synthetic | No — synthetic |
| `outputs/synthetic_demo/evaluation_provenance.json` | provenance for the above | metadata |
| `outputs/synthetic_demo/asr_vs_f.png` | `plot_evaluation_sweep` from the CSV above | No — synthetic, watermarked |

## Were any paper placeholder numbers replaced?
**No.** The manuscript's result tables/plots are placeholders and remain so. This repository
does not fill them in; it provides the pipeline that would, once real runs are executed on
EC2 with recorded provenance.

## Are generated plots conceptual, synthetic, or real?
**Synthetic.** `asr_vs_f.png` is drawn from a synthetic result CSV and carries a visible
"Synthetic smoke-test output — NOT a paper result" watermark. The `--real` flag (off by
default) removes the watermark and must be used only for verified real results.

## Are all result tables generated from code?
**Yes.** No table or plot contains hard-coded numbers; all are produced by scripts from result
files. `tests/test_reproducibility.py::test_no_committed_result_files_claim_paper_results`
enforces that no shipped provenance file claims `is_paper_result: true`.

## What the synthetic demo legitimately demonstrates
The synthetic sweep validates the **mechanism and math**, not paper claims: under a
decision-flipping compromise attack, the robust rules (cmed/gmed/krum) and majority vote hold
ASR-under-compromise near zero for `f < n/2`, while the single-model baseline degrades as
`P(judge_0 compromised) = f/n`, and no-defense sits at ASR = 1. These are properties of the
algorithm (Theorem 1), reproduced on controlled synthetic inputs — they are **not** evidence
about real LLM judges or benchmarks.

## Bottom line
No paper-level experimental results are claimed. All artefacts are synthetic and clearly
labelled. Real results require the EC2 protocol in `docs/ec2_experiment_guide.md`, after which
this file must be updated to record their provenance.

## Addendum — runner metric export (2026-09-25)
- The evaluation runner now emits additional metric columns plus `evaluation_summary.csv`
  (mean ± std across seeds) and, for real judges, `evaluation_cost.csv` (measured latency /
  tokens). See `docs/runner_metrics_plan.md` §9 and `docs/reproducibility.md`.
- This changed **no computed value**: re-running the configuration behind
  `outputs/real_formal/` reproduces all six historical columns
  (`f, n_judges, method, asr_uc, orr, defense_success_rate`) byte-for-byte (28/28 rows).
- No new result file is claimed here. `outputs/real_formal/` still pairs real **payloads**
  (`benchmark:formal`) with **synthetic** verdicts and keeps `is_paper_result: false`.
- Repository-state note: `data/benchmarks/` now contains real, hash-recorded benchmark CSVs
  (see `data/benchmarks/PROVENANCE.json`), which is newer than the "no real benchmark" wording
  above. The datasets were only *placed*: no real judge has ever been executed
  (`outputs/cache/` does not exist and no vLLM run is recorded).

## Addendum — RQ2/RQ4/RQ5 measurement tooling (2026-09-25)
- A measurement entry point now exists for the paper's empirical quantities:
  `scripts/measure_empirical.py` writes `empirical_committee.csv` (r profile, gamma,
  rho at decision/residual/score level), `empirical_gamma.csv`, `empirical_rho_pairs.csv` and
  `empirical_isolation.csv` (epsilon with isolation ON and OFF, using the 1000 paired
  clean/injected rows shipped in `data/benchmarks/second_order/`).
- **No measured value is claimed here.** The tooling has been exercised only against the
  synthetic backend (plumbing) and against a stubbed vLLM engine in the unit tests; real numbers
  require a GPU run. Every file carries `is_paper_result: false`, and
  `empirical_provenance.json` records the caveats that must accompany the numbers (u* estimated
  from the judges themselves; epsilon is one attack realisation, not the supremum of Def. 1; rho
  is undefined for a homogeneous temperature-0 committee).
- RQ5 measured-cost plumbing is in place (`evaluation_cost*.csv`, `ablation_cost*.csv`, per-payload
  measured `max_k`/`sum_k` latency) but produces files only when real judge calls occur, i.e. on a
  cold-cache GPU run. A cached run writes nothing rather than a fabricated value.
- RQ4 diverse committees are now supported in a single run (round-robin backbones with one
  backbone session resident at a time, `configs/real_diverse.yaml`); still unexecuted.

## How to update this file after the GPU run
1. Record the exact commands, cache paths, seeds, and `platform` from each
   `*_provenance.json` / `empirical_provenance.json`.
2. State which numbers enter the manuscript and in which table/figure.
3. Only then set `is_paper_result: true` on the relevant artefacts — and keep
   `tests/test_reproducibility.py::test_no_committed_result_files_claim_paper_results` green
   (shipped outputs must stay `false`; the flag belongs to the documented run record).

## Addendum — external baselines re-implemented (2026-09-25)
- The AutoDefense / SecAlign / StruQ stubs are now **re-implementations** on the local judge path
  (`baselines/llm_baselines.py`), runnable head-to-head via `experiment.external_baselines`. No
  result is claimed: they have never been executed, and their `describe()` marks them
  `kind: "reimplementation"` with the caveat that checkpoint ids and prompt templates must be
  verified against the authors' repositories before any number is reported.
- A defect found during integration review is fixed: a role answering with the *published*
  AutoDefense contract (`Judgment: VALID/INVALID`) without a score used to raise and abort the
  sweep; it is now honoured at the decision level and recorded as `score_source: "validity_text"`,
  while a completion with neither contract still raises (no invented verdicts).
- Rows produced by an external system describe that system **without** a compromised judge and are
  not comparable to `asr_uc(f)` of a verdict-space committee; `docs/reproducibility.md` says so.

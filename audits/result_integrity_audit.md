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

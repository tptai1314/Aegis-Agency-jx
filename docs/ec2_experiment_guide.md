# EC2 Experiment Guide

This repository is **EC2-ready** but does not download data or run paper-level experiments
during generation. Real experiments are run here, after you provision data, models, and
credentials manually.

> **Nothing in this repo auto-downloads datasets, weights, or benchmarks.** All of that is a
> manual, deliberate step you perform on the server.

## 1. Provision the environment
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q            # confirm the mechanism works on synthetic data
```
For real LLM judges you will additionally need the backbone runtimes (e.g. `transformers`,
`vllm`, or API SDKs) and any GPUs those require — these are **not** listed as dependencies
because the synthetic mechanism does not need them.

## 2. Place datasets (manual)
Prepare each benchmark as a CSV in the format of `docs/data_format.md`:
```
/data/harmbench/test.csv
/data/injecagent/test.csv
/data/benign/test.csv
```
Sources requiring manual download / license / registration (see `audits/implementation_gaps.md`):
- jailbreak: AdvBench/GCG, PAIR, TAP, GPTFuzzer, in-the-wild DAN, HarmBench.
- injection: formal injection benchmark, InjecAgent, universal injection.
- second-order: JudgeDeceiver optimiser (external repo) to generate injected payloads.

## 3. Wire real judges and baselines
The real LLM judge path is **already implemented** (`judges/llm_judge.py`, local vLLM) — nothing
to write for the judge itself; you select it in the config. What still needs real code is the
external *baselines*:

- `data.adapters.LLMJudgeAdapter` — implemented by `judges.llm_judge.VllmLLMJudge`
  (vLLM, temperature 0, payload isolation with an ON/OFF switch, verdict cache, measured
  latency/tokens). It is reached with `judges.backend: vllm` and has never been executed without
  a GPU — validate it on the server first.
- `baselines.external_wrappers.{AutoDefenseAdapter, SecAlignAdapter, StruQAdapter}` — **no longer
  stubs**: they delegate to re-implementations in `baselines/llm_baselines.py`
  (`AutoDefenseStyleBaseline` = analyzer → judge → coordinator; `HardenedSingleModelAdapter` for
  SecAlign/StruQ checkpoints). They are *re-implementations from the published designs*, not the
  authors' code; verify prompts and checkpoint ids against the authors' repositories before
  reporting numbers. See `docs/baseline_adapters.md`.
- `baselines.external_wrappers.JudgeDeceiverAdapter` — still a stub (external optimiser).
  For RQ2 the *paired* payloads it would generate are already shipped in
  `data/benchmarks/second_order/` (1000 clean/injected pairs), so `scripts/measure_empirical.py`
  can measure epsilon without the external optimiser.

Each returns/consumes the schema documented in `docs/baseline_adapters.md`. Provide checkpoint
paths / API endpoints via `ExternalBaselineConfig.model_path_or_endpoint`; **never commit
secrets** — read them from environment variables or a mounted secrets file.

## 4. Run the protocol (RQ1-RQ6)
Payload loading from on-disk benchmarks is wired: set `data.root` / `data.benchmark` / `data.split`
in the config and the harness reads `<root>/<benchmark>/test.csv` via `CsvBenchmarkAdapter`. With
`judges.backend: vllm` the verdicts are real judge verdicts; with the default backend they are
synthetic. Then:
```bash
python scripts/run_experiment.py --config configs/real_experiment.yaml --stage calibrate --output outputs/real
python scripts/run_experiment.py --config configs/real_experiment.yaml --stage evaluate --output outputs/real --seeds 0,1,2,3,4
python scripts/run_experiment.py --config configs/real_experiment.yaml --stage ablate   --output outputs/real
python scripts/make_plots.py --input outputs/real/evaluation_sweep.csv --output outputs/real/asr_vs_f.png --real
```
Add a cold-cache run for the measured RQ5 cost (`--cache <new path>`) and run
`scripts/measure_empirical.py` for the empirical `r`, `gamma`, `rho`, `epsilon`. See
`docs/gpu_server_runbook.md` (SSH/GPU-server workflow) and `docs/reproducibility.md` for every
output file. Only pass `--real` to plots when the inputs are genuinely real, verified results; set
`is_paper_result` in provenance only with documented provenance.

## 5. Record provenance
Keep the `*provenance.json` files with every result, and update
`audits/result_integrity_audit.md` to state exactly which outputs are real, where the data
came from, and which numbers (if any) enter the paper.

## What EC2 does not change
- The mechanism, math, and metrics are identical to the synthetic runs; only the *source of
  verdicts* changes (synthetic generator → real LLM judges).
- The paper's tables/plots remain placeholders until these real runs are executed and their
  provenance recorded.

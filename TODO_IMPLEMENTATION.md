# TODO — Implementation (EC2 / real experiments)

The repository is complete and runnable on synthetic data (129+ tests pass; synthetic demo
runs). The items below are what remains to run the **paper-level** experiments; none block the
synthetic mechanism.

## Data (manual provisioning — never auto-downloaded)
- [x] Place the jailbreak/benign/injection benchmarks that are actually available as CSVs per
      `docs/data_format.md` under `data.root`: harmbench, advbench, dan, injecagent, formal,
      second_order, benign, benign_xstest (see `data/benchmarks/PROVENANCE.json`).
- [ ] **Out of scope unless the paper needs them**: PAIR, TAP, GPTFuzzer and the universal-injection
      set are *attack generators* / trained artefacts (payloads depend on the victim model), not
      downloadable payload files — see `docs/gpu_server_runbook.md` §8.4. Report coverage honestly.
- [ ] Generate second-order injections with a real JudgeDeceiver optimiser (the paired clean/injected
      payloads it produced already ship in `data/benchmarks/second_order/`).

## Real models / baselines (weights, APIs, GPUs)
- [x] Real judge implementing `LLMJudgeAdapter.judge()` — `judges/llm_judge.py` (local vLLM, isolation
      on/off, homogeneous + diverse committees, verdict cache, measured latency/tokens). **Never
      executed yet**; also score-only (`m = 0`, no rationale embedding).
- [x] `AutoDefenseAdapter.predict()` / `SecAlignAdapter.predict()` / `StruQAdapter.predict()` —
      re-implementations from the published designs on the local engine
      (`baselines/llm_baselines.py`), wired into the harness for a head-to-head via
      `experiment.external_baselines` + `external_baseline_kwargs` (off by default).
      **Verify prompts/checkpoints against the authors' repositories before reporting any number**;
      `JudgeDeceiverAdapter.inject()` remains a stub.
- [ ] Provide checkpoint paths / endpoints via `ExternalBaselineConfig`; read secrets from env,
      never commit them.

## Empirical quantities to measure (replace synthetic parameters)
- [x] Measurement tooling for honest radius `r`, margin `gamma`, correlation `rho` and isolation
      leakage `epsilon` — `scripts/measure_empirical.py`. `epsilon` is measured over a **family** of
      injection realisations (4 variants incl. a delimiter-escape attack on the isolation channel)
      with the max-over-variants reported as the closest stand-in for the supremum of Definition 1.
      **Run it on the GPU** to obtain real values.
- [ ] Run it on real judges and record the measured `r`, `gamma`, `rho` (homogeneous vs diverse)
      and `epsilon` (isolation on/off, all variants) with their provenance and caveats.
- [x] RQ4 diverse committee in a single run (round-robin backbones, one backbone session resident
      at a time): `configs/real_diverse.yaml`.
- [ ] Execute the diverse committee on the GPU and compare `rho` against the homogeneous runs.
- [x] Adaptive-on-aggregation attack that genuinely optimises against each pipeline's rule and
      calibrated threshold: `attack: adaptive_search` (`per_rule = True`). The older `adaptive`
      heuristic is kept and documented as a surrogate.

## Runs and reporting
- [x] Wire the extra metrics into the runners (RQ2 detection F1/AUROC, RQ5 model-cost columns and
      measured cost, RQ6 utility retention, Wilson CIs, group-conditional ASR, seed sweep) —
      see `docs/runner_metrics_plan.md` §9–§11. **Measured** RQ5 numbers still need a GPU run with
      a cold verdict cache (`--cache <new path>`).
- [ ] Execute calibrate → evaluate → ablate on real verdicts (`docs/ec2_experiment_guide.md`,
      `docs/gpu_server_runbook.md`), including a cold-cache run for the RQ5 cost frontier.
- [ ] Fill the manuscript's placeholder Tables 5-6 and Figure 6 with measured numbers +
      dispersion + significance tests.
- [ ] Update `audits/result_integrity_audit.md` with the provenance of every real result.
- [ ] Use `make_plots.py --real` only for verified real results.

## Optional theory strengthening
- [ ] Re-derive Krum's constant in the verdict space, or keep the cited restatement (currently
      `intentionally_out_of_scope`).
- [ ] Add a Weiszfeld approximation-error term to the Lemma 1 displacement check.

## Author / release
- [ ] Confirm the MIT license placeholder and add author/citation metadata.
- [ ] Confirm the majority-vote tie policy with the paper's authors (documented ambiguity).

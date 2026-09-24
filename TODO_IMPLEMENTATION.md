# TODO — Implementation (EC2 / real experiments)

The repository is complete and runnable on synthetic data (47 tests pass; synthetic demo
runs). The items below are what remains to run the **paper-level** experiments; none block the
synthetic mechanism.

## Data (manual provisioning — never auto-downloaded)
- [ ] Place jailbreak benchmarks (AdvBench/GCG, PAIR, TAP, GPTFuzzer, in-the-wild DAN,
      HarmBench) as CSVs per `docs/data_format.md` under `data.root`.
- [ ] Place injection benchmarks (formal injection benchmark, InjecAgent, universal injection).
- [ ] Place held-out benign traffic for over-refusal / utility.
- [ ] Generate second-order injections with a real JudgeDeceiver optimiser.

## Real models / baselines (weights, APIs, GPUs)
- [ ] Implement `data.adapters.LLMJudgeAdapter.judge()` for real hardened judges on
      Llama-3 / Qwen2.5 / Mistral / GPT-4o / Claude-3.5 with SecAlign/StruQ hardening +
      payload isolation.
- [ ] Implement `AutoDefenseAdapter.predict()` (real AutoDefense system).
- [ ] Implement `SecAlignAdapter.predict()` and `StruQAdapter.predict()` (checkpoints).
- [ ] Implement `JudgeDeceiverAdapter.inject()` (real second-order injection).
- [ ] Provide checkpoint paths / endpoints via `ExternalBaselineConfig`; read secrets from env,
      never commit them.

## Empirical quantities to measure (replace synthetic parameters)
- [ ] Honest concentration radius `r` and margin `gamma` from real judges (Assumptions 1-2).
- [ ] Error correlation `rho` for homogeneous vs diverse committees (RQ4).
- [ ] Isolation leakage `epsilon` under JudgeDeceiver, isolation on/off (RQ2).

## Runs and reporting
- [ ] Execute calibrate → evaluate → ablate on real verdicts (`docs/ec2_experiment_guide.md`).
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

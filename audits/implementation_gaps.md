# Implementation Gaps

Honest inventory of everything that is *not* fully realised locally, why, and what is needed
to close it. Nothing here is hidden in the code.

## 1. Real datasets / benchmarks (require manual provisioning on EC2)
| Benchmark | Why not local | What is needed |
|-----------|---------------|----------------|
| AdvBench / GCG, PAIR, TAP, GPTFuzzer, in-the-wild DAN, HarmBench | large, license/registration, not auto-downloaded by policy | place as CSV per `docs/data_format.md`; or subclass `BenchmarkAdapter` |
| Formal injection benchmark, InjecAgent, universal injection | external repos / licenses | same as above |
| Benign traffic (held-out instruction/QA) | must be disjoint from attack sets | curate and place as CSV |
| JudgeDeceiver second-order injections | external optimiser | implement `JudgeDeceiverAdapter.inject()` |

Status: `not_implemented_requires_manual_data`. A `CsvBenchmarkAdapter` with a documented
schema and an example CSV (`examples/synthetic_data/example_benchmark.csv`) are provided so
real data drops in without code changes.

## 2. Real LLM judges (require weights / APIs + GPUs)
- Hardened judges on Llama-3 / Qwen2.5 / Mistral / GPT-4o / Claude-3.5 with SecAlign/StruQ
  hardening and payload isolation.
- Status: `adapter_stub_waiting_for_external_dependency` (`data.adapters.LLMJudgeAdapter`).
- Needed: backbone runtime, checkpoints/endpoints, and a `judge()` implementation returning a
  `Verdict`.

## 3. External baselines (real systems)
- Real AutoDefense system, SecAlign checkpoint, StruQ checkpoint.
- Status: `adapter_stub_waiting_for_external_dependency`
  (`baselines.external_wrappers.{AutoDefenseAdapter, SecAlignAdapter, StruQAdapter}`).
- The *structural* AutoDefense baseline (non-robust mean coordinator) IS implemented for the
  mechanism-level comparison; the stub is for a faithful head-to-head.

## 4. Theory: Krum constant
- The paper restates Blanchard et al.'s Krum guarantee rather than re-deriving it. The Krum
  aggregator is implemented; its theoretical constant is **not** re-derived here.
- Status: `intentionally_out_of_scope`. See `audits/math_to_code_audit.md`.

## 5. Parametric stand-ins for empirical quantities
- Honest concentration radius `r`, margin `gamma`, error correlation `rho`, and isolation
  leakage `epsilon` are **parameters** of the synthetic judge model. Their real values are
  empirical (RQ2/RQ4) and must be measured from real judges on EC2.
- Status: `implemented_with_synthetic_demo`.

## 6. Geometric-median approximation
- Lemma 1's displacement bound is for the *exact* geometric median; the code uses a Weiszfeld
  approximation. The theory check tests the bound with a tolerance and does not assert the
  exact-median inequality on the iterate.
- Status: implemented with a documented conservative tolerance.

## 7. Paper result tables/plots
- The manuscript's result tables and Figure 6 are **placeholders** (no measured numbers). This
  repo ships only synthetic smoke-test outputs. Real numbers require the EC2 runs above.
- Status: `implemented_with_synthetic_demo` for the pipeline; results themselves pending.

## Known ambiguity (documented, not silently resolved)
- **Majority-vote ties** (n even, exactly n/2 blocks): the paper's strict `>` resolves ties to
  ALLOW. We follow the paper by default and expose `block_on_tie` / `random` alternatives.
  Flagged `paper_inconsistency_detected` (minor) in the traceability CSV and
  `math_to_code_audit.md`.

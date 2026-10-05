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

Status: partially closed. Eight families **are** present on disk under `data/benchmarks/`
(harmbench, advbench, dan, injecagent, benign, benign_xstest, formal, second_order) with sources
and SHA-256 recorded in `data/benchmarks/PROVENANCE.json`; `data/` is git-ignored, so a fresh
clone starts empty and the CSVs must be re-provisioned. Still **not** present: PAIR, TAP,
GPTFuzzer, and the universal-injection set. A `CsvBenchmarkAdapter` with a documented schema and
an example CSV (`examples/synthetic_data/example_benchmark.csv`) is provided so real data drops in
without code changes. Status: `data_placed_provenance_recorded` (partial coverage).

## 2. Real LLM judges (require weights / APIs + GPUs)
- Hardened judges on Llama-3 / Qwen2.5 / Mistral / GPT-4o / Claude-3.5 with SecAlign/StruQ
  hardening and payload isolation.
- Status: **implemented** in `judges/llm_judge.py` (local vLLM, temperature 0, `<|data|>`
  isolation channel with an ON/OFF switch for the RQ2 ablation, verdict cache keyed by
  `(model, payload_id, judge_id)` with a separate namespace for the un-isolated arm, homogeneous
  and round-robin diverse committees, engine residency cap, measured latency/token stats).
  **Never executed**: this checkout has no GPU run recorded (`vllm` is not installed and
  `outputs/cache/` does not exist), so the class is unvalidated against real weights.
- Still missing for full coverage: SecAlign/StruQ-hardened **checkpoints** (the code applies
  prompt-level isolation and hardening discipline, not a preference-optimised checkpoint),
  proprietary backends (GPT-4o / Claude-3.5) — the implemented path is local vLLM only — and
  **rationale embeddings**: the judge returns `embedding = zeros(0)`, so the verdict vector is
  score-only (`d = 1`) instead of the paper's `d = 1 + m` with a sentence encoder on the
  rationale (`experiments.tex:54`; `docs/data_format.md` specifies `e_k` as part of the
  interface). Aggregation, attacks, metrics and theory are all dimension-agnostic, so results
  remain valid for `d = 1`, but the paper's `m > 0` configuration is not reproduced.
- Status: `implemented_local_vllm_not_yet_executed`.

## 3. External baselines (real systems)
- Real AutoDefense system, SecAlign checkpoint, StruQ checkpoint.
- Status: **re-implemented** (not the authors' code) in `baselines/llm_baselines.py` and wired into
  the harness through `experiment.external_baselines`: `AutoDefenseStyleBaseline`
  (analyzer → judge(s) → coordinator; prompts quoted verbatim from `XHMY/AutoDefense`
  `data/prompt/defense_prompts.json` block `explicit_3_agent`, with two documented adaptations) and
  `HardenedSingleModelAdapter` (`SecAlignHardenedAdapter` / `StruQHardenedAdapter`).
  **Standing caveats**: the SecAlign/StruQ checkpoint ids are third-party HF mirrors and their
  prompt templates are re-implementation defaults that must be verified against the authors'
  repositories before any number is reported; the AutoDefense runs use one local backbone per role,
  not the authors' model configuration; rows for these systems are *uncompromised-system* rows.
  `describe()` carries all of this. The re-implementations have never been executed on a GPU.
- The *structural* AutoDefense baseline (non-robust mean coordinator) IS implemented for the
  mechanism-level comparison; the re-implementation is for the agent-level head-to-head.
- Remaining stub: `JudgeDeceiverAdapter.inject()` (external optimiser).

## 4. Theory: Krum constant
- The paper restates Blanchard et al.'s Krum guarantee rather than re-deriving it. The Krum
  aggregator is implemented; its theoretical constant is **not** re-derived here.
- Status: `intentionally_out_of_scope`. See `audits/math_to_code_audit.md`.

## 5. Parametric stand-ins for empirical quantities
- Honest concentration radius `r`, margin `gamma`, error correlation `rho`, and isolation
  leakage `epsilon` are **parameters** of the synthetic judge model in synthetic runs. Their real
  values are empirical (RQ2/RQ4).
- Status: **measurement tooling implemented** (`scripts/measure_empirical.py` →
  `empirical_committee.csv` for the `r` profile / `gamma` / `rho`, `empirical_gamma.csv`,
  `empirical_rho_pairs.csv`, `empirical_isolation.csv` + `empirical_isolation_summary.csv` for
  `epsilon`). `epsilon` is now measured over a **family** of injection realisations
  (`judgedeceiver`, `naive_override`, `roleplay_takeover`, `delimiter_escape` — the last one
  closing the isolation channel, i.e. attacking Definition 1's assumption directly), each with
  its own verdict-cache identity, using the 1000 paired clean/injected rows of `second_order`.
  The summary file reports the max over variants, which is the closest attainable stand-in for
  the supremum in Definition 1. **No real value is claimed yet**: the tooling has only been
  exercised against the synthetic backend and a stubbed engine in tests.
- Standing limitations to report with any number: `u*` is estimated from the judges themselves
  (class-conditional mean); `epsilon` is a max over a measured family, not the supremum;
  `rho` is undefined for a homogeneous temperature-0 committee (no between-judge variance).
- Status: `measurement_tooling_implemented_values_pending`.

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

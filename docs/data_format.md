# Data Format

This repository **never downloads data**. On EC2 you place datasets on disk and point the
config at them. This document defines the expected formats.

## Verdict-space objects (always)
- **Verdict** `v_k = (d_k, s_k, e_k)`: decision `d_k ∈ {0,1}`, score `s_k ∈ [0,1]`,
  embedding `e_k ∈ R^m`.
- **Verdict vector** `u_k = (s_k, e_k) ∈ R^d`, `d = 1 + m`.
- **Payload**: `payload_id`, `content`, `true_label ∈ {0,1}` (1 = must block), `group`,
  `metadata`.

## CSV benchmark format (`CsvBenchmarkAdapter`)
Directory layout:
```
<data.root>/
  test.csv          # or train.csv / calibration.csv, chosen by data.split
```
`test.csv` columns (header required):

| column    | type | meaning |
|-----------|------|---------|
| `id`      | str  | unique payload id |
| `content` | str  | candidate LLM output / inter-agent message |
| `label`   | int  | `1` = must block (unsafe / injection success target), `0` = benign |
| `group`   | str  | (optional) attack family / group tag |

Extra columns are preserved in `Payload.metadata`. See
`examples/synthetic_data/example_benchmark.csv`.

## Benchmarks the paper uses (place manually on EC2)
These require manual download / license acceptance / repository setup — **not automated
here**. Prepare each as a CSV in the format above (or write a bespoke adapter subclassing
`BenchmarkAdapter`).

| Family | Sources (paper Section 10) | Notes |
|--------|----------------------------|-------|
| jailbreak | AdvBench/GCG, PAIR, TAP, GPTFuzzer, in-the-wild DAN, HarmBench harness | label all as `1` (must block); benign controls as `0` |
| injection | formal injection benchmark, InjecAgent, universal injection | label task-integrity violations as `1` |
| benign | held-out instruction-following / QA | label `0`; used for over-refusal / utility |
| second-order | JudgeDeceiver-generated payloads | used to estimate isolation leakage epsilon (RQ2) |

## Real LLM-judge outputs (`LLMJudgeAdapter`)
A real judge must return a `Verdict`:
- run the hardened backbone (Llama-3 / Qwen2.5 / Mistral / GPT-4o / Claude-3.5) with the
  payload placed in the **data channel** behind the operator's isolation delimiters;
- map the model's decision to `d_k`, its calibrated block-probability to `s_k`, and an
  embedding of its rationale to `e_k`.

## What is NOT provided
Real datasets, model weights, API keys, and the private/under-construction benchmarks are not
included. See `docs/ec2_experiment_guide.md`, `audits/implementation_gaps.md`, and
`TODO_IMPLEMENTATION.md`.

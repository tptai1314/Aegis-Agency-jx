# Baseline Adapters

The paper compares against three baselines and references external systems. This document
separates what is **implemented**, what is an **adapter stub**, and what is a **dummy for
tests only** — these are never conflated.

## Implemented baselines (from the paper, fully specified)
| Baseline | Class | What it does |
|----------|-------|--------------|
| Always-submit (no defense) | `baselines.no_defense.NoDefensePipeline` | releases everything (upper bound on ASR) |
| Single hardened model | `baselines.single_model.SingleModelPipeline` | decision from one judge's score |
| Plain majority vote | `baselines.majority_vote.MajorityVotePipeline` | decision-only committee vote |
| AutoDefense (single Coordinator) | `baselines.autodefense.AutoDefensePipeline` | non-robust **mean** aggregation (models the single trust point) |

These run on synthetic verdicts and on real verdicts alike.

## Adapter stubs (require external code / weights on EC2)
Defined in `baselines/external_wrappers.py` and `data/adapters.py`. Each raises
`NotImplementedError` with setup instructions rather than fabricating a result.

| System | Adapter | Required to activate |
|--------|---------|----------------------|
| Real AutoDefense system | `AutoDefenseAdapter` | clone the AutoDefense repo; configure Coordinator/analyzer/judge LLMs; implement `predict()` |
| SecAlign-hardened judge | `SecAlignAdapter` | provide the SecAlign checkpoint path; implement `predict()` |
| StruQ-hardened judge | `StruQAdapter` | provide the StruQ checkpoint; implement `predict()` |
| JudgeDeceiver injector | `JudgeDeceiverAdapter` | provide the JudgeDeceiver optimiser; implement `inject()` |
| Real LLM judge | `data.adapters.LLMJudgeAdapter` | load a backbone; return a `Verdict` |

Expected input/output schema for every judge adapter:
- **Input:** `Payload(payload_id, content, group, metadata)`.
- **Output:** `Verdict(decision ∈ {0,1}, score ∈ [0,1], embedding ∈ R^m)`.

## Dummy baseline (tests ONLY)
`baselines.external_wrappers.DummyExternalBaseline` returns a fixed verdict from the payload
label. It exists solely to test pipeline plumbing and is **never** a stand-in for a real
external baseline. Do not report its outputs as any system's results.

## Why AutoDefense appears twice
- `AutoDefensePipeline` (implemented) is the *structural* baseline: a single non-robust
  Coordinator (mean) over verdicts, which is all that is needed to demonstrate the lack of
  Byzantine tolerance in the verdict space.
- `AutoDefenseAdapter` (stub) is the *real* AutoDefense system with its LLM agents, for a
  faithful head-to-head on EC2. Use the stub when you need the genuine system; use the
  implemented pipeline for the mechanism-level comparison.

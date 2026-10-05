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

## Re-implementations of the external systems (`baselines/llm_baselines.py`)
The published systems cannot be shipped here (external repos, checkpoints, licences), so the
adapters now carry **re-implementations built on this repo's own local vLLM judge path**:

| System | Class | What it runs |
|--------|-------|--------------|
| AutoDefense (Zeng et al. 2024) | `AutoDefenseStyleBaseline` | the published **analyzer → judge → coordinator** agent design over local backbones: a single Coordinator aggregates the judges' votes (no Byzantine tolerance), exactly the trust structure the paper attacks |
| SecAlign (Chen et al. 2025) | `HardenedSingleModelAdapter` | one released hardened checkpoint, with its own prompt/query format and its own verdict-cache namespace |
| StruQ (Chen et al. 2024) | `HardenedSingleModelAdapter` | same, with the StruQ structured-query format and checkpoint |

Both return a `Verdict` (`predict(payload, rng)`) and expose `describe()` for provenance, and both
can be run head-to-head against the Aegis pipelines:

```yaml
experiment:
  external_baselines: [autodefense, secalign, struq]
  external_baseline_kwargs:
    secalign: {checkpoint: "<verified HF id>"}   # required: nothing is guessed
    struq:    {checkpoint: "<verified HF id>"}
```

Accepted configuration keys per class, the run cost, and the two documented AutoDefense
adaptations are listed in `docs/gpu_server_runbook.md` §8.8.

**Answer parsing.** A role's completion is read as a JSON score first; if that fails, the
*published* free-text contract (`Judgment: VALID/INVALID`) is honoured and the verdict is taken at
the decision level (score 0/1) with `last_trace[...]["score_source"] == "validity_text"`. A
completion carrying neither contract raises a clear error — a verdict is never invented. The
coordinator's own fallback (`vote_share`) is unchanged.

> ⚠️ **These are re-implementations, not the authors' code.** The AutoDefense prompts come from the
> published repository where they could be retrieved (URL is recorded in `describe()["sources"]`);
> the SecAlign/StruQ prompt templates and checkpoint ids are **defaults that must be verified
> against the authors' repositories before any number is reported**. `describe()["kind"]` is
> `"reimplementation"` and `describe()["caveats"]` lists the standing caveats, so no result can be
> mistaken for a head-to-head against the original systems.

| System | Adapter | Remaining work for full fidelity |
|--------|---------|----------------------------------|
| JudgeDeceiver injector | `JudgeDeceiverAdapter` | still a stub: the optimiser is external. Its *effect* is covered for RQ2 because the paired clean/injected payloads it generated already ship in `data/benchmarks/second_order/` (see `scripts/measure_empirical.py`) |
| Real LLM judge | `judges.llm_judge.VllmLLMJudge` | **implemented** — set `judges.backend: vllm` + `backbones`; needs a GPU with vLLM. Never executed in this checkout |

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
- `AutoDefenseStyleBaseline` / `AutoDefenseAdapter` is the *agent-level* system with its LLM
  roles, for a faithful head-to-head on the GPU server — implemented here as a re-implementation
  from the published design, with the caveats above.

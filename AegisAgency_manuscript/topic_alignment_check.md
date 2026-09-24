# Topic Alignment Check — Aegis-Agency

Comparison of the locked idea (`idea.pdf`, see `idea_fidelity_report.md` and
`topic_signature.json`) against the final manuscript. Each row must PASS.

## Idea topic vs. final title
- **Idea:** "Aegis-Agency: Byzantine-Robust, Injection-Hardened Multi-Agent Defense
  Pipelines."
- **Manuscript title:** "Aegis-Agency: Byzantine-Robust, Injection-Hardened Multi-Agent
  Defense Pipelines for Large Language Models."
- **PASS.** Adds only the scope qualifier "for Large Language Models"; keeps every core term
  (Aegis-Agency, Byzantine-robust, injection-hardened, multi-agent defense pipelines).

## Idea keywords vs. manuscript keywords
- **Idea keywords:** multi-agent defense pipeline, Byzantine-robust verdict aggregation
  (Krum/median), 2f+2<n, insider/colluding defenders, second-order injection, payload
  isolation (StruQ), hardened judges (SecAlign/StruQ), jailbreak vs. injection/task-integrity,
  adaptive attacks, over-refusal, correlated failure, backbone diversity, security-vs-cost.
- **Manuscript keywords (frontmatter + abstract):** LLM security, multi-agent systems,
  Byzantine-robust aggregation, prompt injection, jailbreak defense, LLM-as-a-judge, insider
  threat, graceful degradation; abstract additionally names payload isolation, hardened
  judges, second-order injection, task-integrity, correlated failure, backbone diversity,
  security-versus-cost.
- **PASS.** Full coverage; no foreign keyword dominates.

## Idea theorem directions vs. manuscript theorem directions
- **Idea:** "under 2f+2<n, up to f Byzantine judges cannot force an adversarial verdict — to
  be proven, not assumed"; correlated-failure analysis; latency linear in agents.
- **Manuscript:** Theorem 1 (decision integrity: up to f Byzantine judges cannot flip the
  aggregate decision, under 2f+2<n / f<n/2 + concentration + margin) — PROVEN in Appendix A;
  Proposition 3 (correlated-failure floor) — PROVEN; Section 9 (latency/token linear in n).
- **PASS.** The exact "to be proven" property is Theorem 1, proven under stated assumptions;
  correlated failure and cost both present. Condition $2f+2<n$ appears explicitly.

## Idea algorithm vs. manuscript algorithm
- **Idea:** Krum/geometric-median selection over {v_k} in verdict/embedding space replacing
  the Coordinator; verdict = allow/block + calibrated score + rationale embedding; payload in
  isolated data channel.
- **Manuscript:** Algorithm 1 (Aegis adjudication) + Algorithm 2 (cmed/gmed/Krum) over
  $\bm u_k=(s_k,\bm e_k)$; isolation step; verdict exactly as idea specifies.
- **PASS.** Algorithm name (Aegis-Agency) and aggregation family (Krum/cmed/gmed) preserved.

## Idea experiment plan vs. manuscript experiment plan
- **Idea:** backbones Llama-3/Qwen2.5/Mistral/GPT-4o/Claude-3.5; 1–7 judges; compromised /
  colluding / second-order / adaptive-on-aggregation attacks; baselines AutoDefense / single
  SecAlign / majority vote; metrics ASR-under-compromise, over-refusal, tolerance fraction,
  detection F1/AUROC, latency, token cost, utility; ablations (agg on/off, isolation on/off,
  hardened on/off, n, colluding fraction, homogeneous vs diverse).
- **Manuscript:** Section 10 reproduces all of the above verbatim in structure (RQ1–RQ6,
  datasets, models, baselines, attacks, metrics, ablations, reproducibility) — with results
  as clearly-labelled placeholders.
- **PASS.** No datasets/attacks/metrics/baselines from a different topic substituted.

## Idea seed references vs. manuscript related work
- **Idea seeds:** AutoDefense (2403.04783), Krum (Blanchard NeurIPS 2017), SecAlign
  (2410.05451), StruQ (2402.06363), Llama-Guard (named).
- **Manuscript:** all five resolved with correct metadata and cited centrally in Intro /
  Related Work / System Model / Experiments (`zeng2024autodefense`, `blanchard2017krum`,
  `chen2025secalign`, `chen2024struq`, `inan2023llamaguard`).
- **PASS.**

## Forbidden-drift scan (must be absent from title/abstract/contributions/theorems/experiments)
| Forbidden drift | Present as topic? | Where mentioned (allowed context only) |
|---|---|---|
| federated learning as object | NO | Related Work only, as the *source* of imported robust aggregation, explicitly contrasted |
| over-the-air / wireless / analog | NO | absent |
| differential privacy | NO | absent (privacy is not a goal) |
| mechanism design / incentives | NO | absent |
| blockchain / BFT protocol design | NO | Lamport/PBFT cited as *context*, marked "not our mechanism" |
| watermarking / multimodal / RL | NO | absent |
| new jailbreak attack as contribution | NO | attacks used only to evaluate the defense |
| single-model hardening as contribution | NO | it is a component + baseline, stated |
| "reduced ASR = secure" | NO | explicitly denied throughout |

## Canonical failure checks (from idea_fidelity_report.md)
1. Title has Aegis-Agency + Byzantine-robust + injection — ✓
2. Threat model includes compromised/colluding defenders AND second-order injection — ✓
3. Aggregation from Krum/median/geometric-median family — ✓
4. Condition 2f+2<n appears in theory — ✓ (and f<n/2 for median)
5. Baselines include AutoDefense, single SecAlign, plain majority vote — ✓
6. Evaluation has 1–7 judge sweep and the four attack classes — ✓
7. No forbidden topic in title/abstract/contributions — ✓

## RESULT: ALL PASS
The manuscript is consistent with the locked paper specification. No topic-mismatch revision
required.

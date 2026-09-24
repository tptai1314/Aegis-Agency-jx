# Idea Fidelity Report — Idea 05: Aegis-Agency

Source of truth: `idea.pdf` (4 pages) and its LaTeX source `idea.tex` in
`/Users/sonvatuananh/Documents/RMIT/SonHa/SecureAgent/SecureMAS_Research/Idea_05_AegisAgency/`.
The PDF and the `.tex` source are textually identical; both were read in full before this
report was written. Every item below is extracted from that document, with no additions.

## 1. Exact title / working title

**"Aegis-Agency: Byzantine-Robust, Injection-Hardened Multi-Agent Defense Pipelines"**

The system name is written `Aegis-Agency` (typewriter font `\texttt{Aegis-Agency}` in the
contribution list). The manuscript must keep this name.

## 2. Core research field

Security of LLM-based multi-agent systems; specifically, the security of *multi-agent
defense pipelines* that filter LLM outputs (AutoDefense-style response-filtering
agencies). Target venues named in the idea: ACM CCS (primary), USENIX Security,
IEEE S&P, NDSS, and the journal IEEE TDSC (dependability/fault-tolerance scope).
This is an applied-security + fault-tolerance paper, not a federated-learning,
mechanism-design, or wireless paper.

## 3. Core technical keywords

- multi-agent defense pipeline / defense "agency"
- LLM output filtering / response filtering
- Byzantine-robust verdict aggregation (Krum, coordinate-wise median, geometric median)
- Krum tolerance condition 2f + 2 < n
- compromised / colluding internal defenders (insider compromise)
- second-order prompt injection (injection against the judges via the inspected payload)
- payload isolation (StruQ-style delimiting / structured input; data channel vs. instruction channel)
- hardened judge agents (SecAlign / StruQ fine-tuning)
- jailbreak vs. prompt injection / task-integrity violations
- instruction / privilege hierarchy
- adaptive attacks on the pipeline and on the aggregation rule
- graceful degradation under insider compromise
- ASR-under-compromise; over-refusal / false positives
- security-vs-cost frontier (latency, token/API cost vs. Byzantine tolerance)
- backbone diversity; correlated failure of judges sharing a base model

## 4. System model entities

- **Input agent** — receives the inspected payload.
- **Analyzer agents** (one or more) — e.g., intention analysis, original-prompt analysis.
- **Judge agents** — a committee of 1–7 judges, each emitting a verdict.
- **Output agent** — enforces the final decision.
- **Byzantine-robust aggregation rule** — replaces AutoDefense's single Coordinator.
- **Inspected payload** — a candidate LLM response and/or an inter-agent message; untrusted.
- **Attacker** — black-box to gray-box, adaptive, may compromise/collude with internal
  defenders and simultaneously inject the inspected content.

## 5. Main mathematical objects

- Verdict of judge k: v_k = (allow/block decision + calibrated score + rationale embedding).
- Verdict/embedding space over which Krum / geometric-median selection operates.
- Byzantine tolerance parameters n (number of judges), f (Byzantine judges),
  with the Krum condition 2f + 2 < n.
- Honest-majority condition: at least n − f judges honest.
- Colluding fraction (swept up to the 2f + 2 < n boundary).
- Correlation of judge failures (residual correlation when judges share a backbone).
- Cost objects: latency (linear in the number of agents), token/API cost.
- Metrics: defense success rate, ASR-under-compromise, targeted-ASR, false-positive /
  over-refusal rate, Byzantine tolerance fraction, detection F1/AUROC of malicious
  verdicts, utility retention on benign traffic.

## 6. Proposed algorithm name

The architecture is named **Aegis-Agency**. No separate algorithm name is given; the
aggregation step is described as "a Krum/geometric-median selection over {v_k} in
verdict/embedding space". The manuscript may therefore name the aggregation procedure
descriptively (e.g., "robust verdict aggregation") but must keep Aegis-Agency as the
architecture name and must keep Krum / coordinate-wise median / geometric median as
the aggregation family.

## 7. Theorem names and theorem directions

No theorem is named. One explicit theorem direction is stated:

> "under the Krum condition 2f + 2 < n, up to f Byzantine judges cannot force an
> adversarial verdict — a target property to establish for the semantic verdict space,
> **to be proven, not assumed**."

So the theoretical goal is: prove, for the semantic verdict space (decision + score +
embedding), that Krum-style / robust-median aggregation guarantees the aggregate verdict
cannot be flipped by up to f Byzantine judges, under explicit assumptions. Secondary
analytical directions stated in the idea:

- Diversity/correlated-failure analysis: "Krum tolerance is vacuous if all judges fail
  identically" — analyze majority/robust-vote failure under correlated judge errors
  (flagged *(hypothesis)*).
- Cost analysis: latency linear in the number of agents ("quantified"); token/API cost.
- Explicit non-claim: "lower ASR-under-compromise is evidence of robustness, not a proof
  of security."

## 8. Experiment plan

- **LLM families:** open (Llama-3, Qwen2.5, Mistral) and proprietary (GPT-4o, Claude-3.5)
  backbones; homogeneous and diverse committees.
- **Number of defenders:** 1–7 judges.
- **Adversary sweep:** malicious/colluding fraction from independent single compromise to
  colluding minorities up to the 2f + 2 < n boundary; static vs. adaptive attacks on the
  pipeline and the aggregation rule.
- **Attacks:** compromised judge; colluding defenders; second-order injection of the
  inspected payload; adaptive attack on aggregation.
- **Metrics:** defense success rate; ASR-under-compromise and targeted-ASR;
  false-positive/over-refusal rate; Byzantine tolerance fraction; detection F1/AUROC of
  malicious verdicts; latency; token/API cost; utility retention on benign traffic.
- **Ablations:** robust aggregation on/off; payload isolation on/off; hardened judges
  on/off; number of defenders 1–7; colluding fraction; homogeneous vs. diverse backbones
  (correlated-failure test).

## 9. Baseline methods

1. **AutoDefense** (single Coordinator) — arXiv:2403.04783.
2. **A single hardened model (SecAlign)** — arXiv:2410.05451.
3. **Plain majority vote** over judges.

## 10. Seed references (as given in the idea PDF)

| Seed | Identifier in idea PDF | Status |
|---|---|---|
| AutoDefense | arXiv:2403.04783 | to resolve full metadata |
| Krum | Blanchard et al., NeurIPS 2017 | to resolve full metadata |
| SecAlign | arXiv:2410.05451 | to resolve full metadata |
| StruQ | arXiv:2402.06363 | to resolve full metadata |
| Llama-Guard | named in AutoDefense description ("optional Llama-Guard fourth agent") | to resolve full metadata |

## 11. Explicit risks / assumptions stated in the idea

- **Cost/latency:** many hardened judges plus aggregation raise latency and token/API
  cost; the frontier may be unattractive at small n.
- **Small-committee tolerance:** with 3–5 judges, 2f + 2 < n tolerates very few faults
  (f = 1 needs n ≥ 4); whether this is meaningful robustness is an *open empirical
  question, not assumed*.
- **Correlated failure:** judges sharing a base model may fail identically, voiding
  Byzantine tolerance; required diversity may be hard to achieve *(hypothesis)*.
- **Utility/interpretation:** robust aggregation biased toward blocking could raise
  false positives; over-refusal measured explicitly; reduced ASR-under-compromise is
  not "secure"; residual failures reported.
- **Trusted:** the aggregation substrate executes correctly; at least n − f judges honest;
  inspected payload confined to a data channel.
- **Out of scope:** all judges compromised; identical correlated failure across all judges
  (studied as a limiting case, not assumed defended).

## 12. What the paper is NOT about

- NOT federated learning, over-the-air aggregation, wireless channels, or gradient
  aggregation as the object of study (Krum is imported *from* that literature; the
  gradients-under-a-trusted-master setting is explicitly contrasted, not adopted).
- NOT mechanism design, incentives, auctions, or payments.
- NOT differential privacy (privacy is not a stated goal of the idea).
- NOT a new jailbreak *attack* paper: attacks are built only to evaluate the defense.
- NOT a single-model hardening paper: SecAlign/StruQ hardening is a *component and a
  baseline*, not the contribution.
- NOT blockchain/consensus-protocol design: the aggregation substrate is trusted;
  no BFT state-machine replication protocol is proposed.
- NOT a claim of provable end-to-end security: the idea explicitly says reduced ASR is
  evidence, not proof, and residual failures must be reported.
- The idea mentions "Contagion" only as a *sibling idea in the same repository* used for
  a confidence comparison; it is NOT a dependency and must not be imported as content.

---

# LOCKED PAPER SPECIFICATION

- **Locked topic:** securing the multi-agent defense pipeline itself — an
  AutoDefense-style LLM response-filtering agency — against insider (Byzantine/colluding)
  defenders and second-order prompt injection, via Byzantine-robust verdict aggregation,
  payload isolation, and hardened judges, generalized from jailbreak filtering to
  prompt-injection/task-integrity filtering.

- **Locked problem statement:** existing multi-agent LLM defenses assume all internal
  defenders are honest, aggregate through a single trusted Coordinator, expose their
  judges to the untrusted content they inspect, are evaluated only against static
  attacks, and target jailbreak rather than injection/task-integrity. Design and analyze
  a defense agency that removes the honest-insider and trusted-coordinator assumptions
  and closes the second-order injection surface, and characterize what robustness it
  does and does not provide.

- **Locked method family:** committee of n hardened judge agents (SecAlign/StruQ-style
  fine-tuning), StruQ-style structural isolation of the untrusted payload into a data
  channel, and Byzantine-robust aggregation of judge verdicts (Krum-style selection,
  coordinate-wise median, geometric median) replacing the single Coordinator; verdict =
  (decision, calibrated score, rationale embedding).

- **Locked theoretical goals:** (i) prove, under explicit assumptions on honest-verdict
  concentration in the verdict/embedding space, that with 2f + 2 < n (Krum-style
  selection) or f < n/2 (median-style rules) up to f Byzantine judges cannot flip the
  aggregate decision (integrity: cannot force pass; availability: cannot force block);
  (ii) quantify degradation of these guarantees under correlated judge failures;
  (iii) latency/token-cost complexity in n. All stated with honest assumptions; no
  claim that this equals end-to-end security.

- **Locked experiment family:** jailbreak and prompt-injection benchmarks; judges on
  Llama-3/Qwen2.5/Mistral/GPT-4o/Claude-3.5 backbones; 1–7 judges; compromised-judge,
  colluding-minority, second-order-injection, and adaptive-aggregation attacks; ablations
  exactly as listed in §8 above. Results are PLACEHOLDER/protocol-only unless real data
  exists (none exists in PROJECT_PATH).

- **Locked baseline family:** AutoDefense (single Coordinator), single hardened model
  (SecAlign), plain majority vote. (Secondary baselines may be added only if verified
  literature supplies them and they do not displace these three.)

- **Terms that MUST appear in title/abstract/introduction:** Aegis-Agency;
  Byzantine-robust (verdict aggregation); prompt injection; multi-agent defense
  pipeline(s); compromised/colluding defenders (insider); second-order injection;
  jailbreak; task-integrity; hardened judges.

- **Terms that must NOT dominate the paper** (allowed only as brief, cited context):
  federated learning; gradient aggregation; differential privacy; blockchain/BFT
  consensus protocols; mechanism design/incentives; wireless/over-the-air; watermarking;
  multimodal agents; reinforcement learning. None of these may appear in the title,
  abstract, contribution list, theorem statements, or experiment design as a topic.

## Canonical failure checks

A draft is INVALID and must be discarded if any of the following holds:
1. Title lacks "Aegis-Agency" or lacks Byzantine-robust(ness) or injection terms.
2. The threat model does not include compromised/colluding internal defenders AND
   second-order injection.
3. The aggregation rule is not from the Krum/median/geometric-median family.
4. The condition 2f + 2 < n does not appear in the theory.
5. Baselines omit AutoDefense, single SecAlign model, or plain majority vote.
6. The evaluation omits the 1–7 judge sweep or the four attack classes.
7. A forbidden topic (FL, DP, mechanism design, blockchain, wireless) appears in the
   title, abstract, or contributions.

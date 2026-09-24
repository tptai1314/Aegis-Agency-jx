# Paper Implementation Specification — Aegis-Agency

Source: `AegisAgency_main.pdf` (the manuscript in this project folder). This spec was
extracted directly from the manuscript sources (title, abstract, system model, problem
formulation, method, algorithm, theory, complexity, experiments).

## Paper title
*Aegis-Agency: Byzantine-Robust, Injection-Hardened Multi-Agent Defense Pipelines for
Large Language Models.*

## Central task
Secure the multi-agent LLM **defense pipeline itself**. A committee of $n$ LLM "judge"
agents inspects a candidate output (the *payload*); their verdicts are combined into an
allow/block decision. The paper hardens this pipeline against (a) compromised/colluding
internal judges (Byzantine minority) and (b) second-order prompt injection carried by the
payload the judges read.

## Method name
**Aegis-Agency** — four components: (1) Byzantine-robust verdict aggregation, (2) payload
isolation, (3) hardened judges, (4) jailbreak→task-integrity generalisation.

## Algorithms (from the paper)
- **Algorithm 1 — Aegis adjudication.** analyze → isolate payload → query $n$ hardened
  judges in parallel → robust-aggregate verdict vectors → decide allow/block/escalate
  (escalate when the aggregate score is within $\delta$ of the threshold).
- **Algorithm 2 — Robust aggregation subroutines.** `CoordinateMedian`, `GeometricMedian`
  (Weiszfeld iteration), `Krum` (select the verdict minimising the sum of squared distances
  to its $n-f-2$ nearest neighbours).
- Baseline rule: **plain majority vote** over decisions (sign-vote special case).

## Equations that must be implemented
| Eq | Object | Form |
|----|--------|------|
| (1) | verdict | $v_k=(d_k, s_k, \bm e_k)$, $d_k\in\{0,1\}$, $s_k\in[0,1]$, $\bm e_k\in\mathbb R^m$ |
| (2) | verdict vector | $\bm u_k=(s_k,\bm e_k)\in\mathbb R^d$, $d=1+m$; $\pi_1(\bm u_k)=s_k$; decision $\mathsf D(\bm u)=\mathbf 1\{\pi_1(\bm u)\ge\tau\}$ |
| (3) | $\varepsilon$-isolation | $\sup_c \mathrm{TV}(P(\cdot\mid c,x),P(\cdot\mid c,x'))\le\varepsilon$ |
| (4) | ASR-under-compromise | $\textsc{asr-uc}(f)=\Pr[\hat d=0\mid y^\star=1]$ |
| (5) | over-refusal | $\textsc{orr}(f)=\Pr[\hat d=1\mid y^\star=0]$ |
| (6) | gmed displacement (Lemma 1) | $\lVert\hat{\bm u}-\bm u^\star\rVert\le C_\alpha r$, $C_\alpha=\frac{2(n-f)}{n-2f}=\frac{2(1-\alpha)}{1-2\alpha}$ |
| (7) | integrity condition (Thm 1) | $\gamma > C_\alpha r \Rightarrow \mathsf D(\hat{\bm u})=\mathsf D(\bm u^\star)$ |
| (8) | correlated variance (Prop 3) | $\mathrm{Var}(\bar Z)=\mu(1-\mu)[\frac{1-\rho}{n-f}+\rho]$ |
| — | token cost | $\text{Tokens}(n)=n(\ell_{in}+\ell_{out})+\text{Tokens}_{analyze}$ |
| — | latency | $\text{Lat}_\parallel(n)=\max_k \text{lat}(J_k)+\text{lat}(\mathsf{Agg})+\text{lat}_{analyze}$; sequential $=\sum_k$ |

## Data structures
- `Verdict(d, s, e)`; `VerdictVector` = concatenation $(s, \bm e)$.
- `Payload` (candidate output + ground-truth label $y^\star$ + group/metadata).
- `CommitteeConfig` ($n$, aggregation rule, assumed $f$, backbones, homogeneous/diverse).
- `DecisionResult` (decision ∈ {allow, block, escalate}, aggregate score, threshold, mode,
  group, per-judge outlier scores).

## Inputs and outputs
- Input: a payload (in evaluation, an LLM output/benchmark item); a committee of judges;
  an aggregation rule; a calibrated threshold $\tau$ (+ escalation band $\delta$).
- Output: allow/block/escalate decision with metadata; per-run metrics.

## Metrics (Section 10 / Problem Formulation)
Defense success rate; ASR-under-compromise (4); targeted-ASR; over-refusal rate (5);
Byzantine tolerance fraction; detection F1/AUROC of malicious verdicts (aggregation as an
outlier detector); latency; token/API cost; utility retention. Also measured: honest radius
$r$, margin $\gamma$, correlation $\rho$, isolation leakage $\varepsilon$.

## Baselines
1. **AutoDefense** — single-Coordinator committee (no Byzantine tolerance). *Implementable*
   as a non-robust aggregator (single point / mean); the *real* AutoDefense system needs an
   external repo → adapter stub.
2. **Single hardened model (SecAlign)** — one judge. *Implementable* structurally; the *real*
   SecAlign checkpoint needs external weights → adapter stub.
3. **Plain majority vote** — decision-only aggregation. *Fully implementable*.

## Attacks (adaptive by construction)
Compromised judge; colluding defenders ("a-little-is-enough", stay within honest radius $r$);
second-order payload injection (JudgeDeceiver-style; modelled by isolation leakage
$\varepsilon$); adaptive-on-aggregation (inner-product manipulation / local model poisoning).

## Datasets / simulation settings
- **Real (EC2, adapters only):** jailbreak sets — DAN in-the-wild, GCG/AdvBench, PAIR, TAP,
  GPTFuzzer, HarmBench harness; injection — formal injection benchmark, InjecAgent, universal
  injection; benign traffic; JudgeDeceiver second-order injections. Backbones: Llama-3,
  Qwen2.5, Mistral, GPT-4o, Claude-3.5.
- **Synthetic (local, this repo):** verdict-space simulation of honest/Byzantine judges with
  controllable margin $\gamma$, radius $r$, correlation $\rho$, leakage $\varepsilon$ — used
  to test the mechanism and empirically illustrate Lemma 1 / Theorem 1 / Proposition 3.

## Experimental protocol
RQ1 integrity vs colluding fraction; RQ2 second-order injection; RQ3 adaptivity; RQ4
correlated failure (homogeneous vs diverse); RQ5 cost frontier ($n=1..7$); RQ6 utility.
Ablations: robust aggregation on/off, isolation on/off, hardened judges on/off, $n$,
colluding fraction, backbone diversity.

## Assumptions
A1 honest concentration ($\lVert\bm u_k-\bm u^\star\rVert\le r$); A2 decision margin
($|\pi_1(\bm u^\star)-\tau|\ge\gamma$, $\mathsf D(\bm u^\star)=y^\star$); A3 tolerable
fraction ($f<n/2$ for median rules, $2f+2<n$ for Krum); Def.1 $\varepsilon$-isolation.

## Limitations / conceptual-only / placeholder components
- **All paper result tables/plots are placeholders** — no measured numbers exist. This repo
  therefore ships **synthetic smoke-test outputs only**, never paper results.
- Real LLM judges, benchmarks, and external baselines (AutoDefense/SecAlign/StruQ/
  JudgeDeceiver) are **adapter stubs** requiring EC2 + real data/weights.
- Lemma 2 (Krum) is a cited restatement in the paper; the Krum aggregator is implemented, but
  its theoretical constant is not re-derived here.
- $\varepsilon$-isolation is an empirical property; modelled parametrically in synthetic runs.

## What the paper is NOT
Not federated learning, over-the-air, differential privacy, mechanism design, blockchain/BFT
protocol design, a new jailbreak attack, or a single-model hardening paper. The implementation
must not drift into those.

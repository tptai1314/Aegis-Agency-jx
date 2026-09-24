# Notation Consistency Check — Aegis-Agency

Cross-checked against `sections/preliminaries.tex` (Table 1) and every equation/theorem.

## Symbol inventory (no conflicting reuse)
| Symbol | Meaning | Defined in | Reused elsewhere consistently? |
|---|---|---|---|
| $n$ | committee size | Prelim, Table 1 | yes (system model, complexity, experiments) |
| $f,\ \alpha=f/n$ | Byzantine count / fraction | Prelim, Table 1 | yes (theory, complexity) |
| $\mathcal H,\ \mathcal B$ | honest / Byzantine index sets | Prelim | yes (proofs) |
| $x$ | inspected payload | Prelim | yes (algorithm, threat model) |
| $v_k=(d_k,s_k,e_k)$ | full verdict | Eq. (1) | yes |
| $\bm u_k=(s_k,\bm e_k)\in\mathbb R^d$ | verdict vector; $d=1+m$ | Eq. (2) | yes (theory, proofs) |
| $\pi_1(\bm u)=s$ | score projection | Prelim | yes (Thm 1, proof) |
| $\tau$ | block threshold | Prelim | yes; $\delta$ (escalation band) distinct |
| $\mathsf D(\cdot)$ | decision function | Prelim | yes |
| $\mathsf{Agg},\ \hat{\bm u}$ | aggregation rule / aggregate | Prelim | yes |
| $\mathrm{cmed},\mathrm{gmed},\textsc{Krum}$ | the three rules | Prelim | yes |
| $\bm u^\star$ | honest reference verdict | Table 1 / Assumption 1 | yes (Lemma 1, Thm 1, proofs) |
| $r$ | honest concentration radius | Assumption 1 | yes |
| $\gamma$ | decision margin | Assumption 2 | yes |
| $C_\alpha$ | displacement constant | Lemma 1 | yes (Thm 1, Prop 1 with $C_\alpha=1$) |
| $\rho$ | honest-error correlation | Prop 3 | yes |
| $\varepsilon$ | isolation leakage | Def 1 | yes (Prop 2) |
| $m,\ d=1+m$ | embedding dim / verdict dim | Eq. (1)-(2) | yes |
| $\mu$ | honest per-judge correctness prob. | Prop 3 | local to Prop 3 only |
| $N=n-f$ | honest count (proof shorthand) | Appendix A.5 | local, defined at use |
| $T$ | Weiszfeld iteration count | Complexity / Alg 2 | local, defined at use |
| $\delta$ | escalation band half-width | Algorithm 1 | local, defined at use |

## Potential collisions checked and cleared
- $d$ is **verdict-vector dimension** ($d=1+m$); the per-judge **decision** is $d_k$
  (subscripted). No clash: $d$ never appears without meaning fixed by context, and $d_k$
  always carries a subscript. ✓ (Documented here to pre-empt reviewer confusion; a sentence
  could be added to Prelim if desired — noted in TODO.)
- $s$ is **score**; $\bm e$ is **embedding**; $e$ is not used for Euler's number anywhere. ✓
- $\alpha=f/n$ (Byzantine fraction) is not reused for significance level or step size. ✓
- $C_\alpha$ vs. Corollary label "Cor." — different namespaces (math symbol vs. cref). ✓

## Equation labels — all referenced or intentionally standalone
- (1) verdict, (2) verdict vector — referenced. (3) isolation TV — referenced (Prop 2).
- (4) asr-uc, (5) orr — referenced. (6) gmed bound — referenced (Thm 1, proof).
- (7) integrity condition — referenced. (8) corrvar — referenced (Prop 3, proof).
- token-cost / latency eqs — referenced in complexity + experiments.
- No dangling `\eqref` and no equation labelled but never used (checked via compile: zero
  "reference undefined").

## Theorem-variable ↔ system-model-variable agreement
- Theorem 1 uses exactly $n,f,\alpha,r,\gamma,\bm u^\star,\hat{\bm u},\tau$ — all system-model
  symbols. No undefined constant appears in any final theorem statement. ✓
- $C_\alpha$ is the only derived constant in a theorem statement; fully defined in Lemma 1. ✓

## Result: PASS
No symbol carries two conflicting meanings; every symbol in every displayed equation is
defined before use; theorem variables match the system model. One optional clarity note
($d$ vs $d_k$) is logged in TODO_before_submission.md.

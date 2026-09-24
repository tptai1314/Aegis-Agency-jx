# Implementation Notes

How the Aegis-Agency paper (`AegisAgency_main.pdf`) was translated into code.

## Guiding idea: implement in the verdict space
The paper's theory operates on **verdict vectors** `u_k = (s_k, e_k) ∈ R^d` (Eq. 2), not on
raw LLM text. Every theorem (Lemma 1, Theorem 1, Proposition 3) is a statement about how an
aggregation rule maps a set of verdict vectors to an aggregate. This makes the *mechanism*
fully implementable and testable without any LLM: we simulate honest and Byzantine judges as
distributions over verdict vectors, and the aggregation / metrics / theory code is exactly
what would run on real verdicts. Real LLM judges and benchmarks are reached through adapters
(`data/adapters.py`, `baselines/external_wrappers.py`) on EC2.

This separation is deliberate and honest: the synthetic layer validates the *algorithm and
math*; it does not and cannot produce the paper's *empirical* results, which require real
hardened LLM judges on real jailbreak/injection benchmarks.

## Module map
| Paper object | Module | Notes |
|---|---|---|
| Verdict / verdict vector (Eq. 1-2) | `data/schemas.py` | dataclasses with validation |
| Coordinate median / gmed / Krum / majority (Alg. 2) | `methods/aggregators.py` | exact rule definitions |
| Aegis adjudication (Alg. 1) | `methods/gate.py` | analyze→isolate→judge→aggregate→decide |
| Threshold + score calibration | `methods/calibration.py` | threshold selection (not conformal) |
| Honest committee (Assumptions 1-2, rho) | `judges/synthetic_judges.py` | parametric verdict generator |
| epsilon-isolation (Def. 1) | `judges/isolation.py`, `attacks/injection.py` | leakage model |
| Attacks (Section 5.3) | `attacks/` | compromise / collusion / injection / adaptive |
| Metrics (Eq. 4-5, Section 10) | `metrics/metrics.py` | guarded rates + detection AUROC/F1 |
| Theory checks (Lemma 1, Thm 1, Prop 3) | `metrics/theory.py` | C_alpha, integrity, variance |
| Cost models (Section 9) | `metrics/cost.py` | token/latency |
| Baselines (Section 10) | `baselines/` | no_defense, single_model, majority, autodefense |
| External systems | `baselines/external_wrappers.py`, `data/adapters.py` | stubs w/ schemas |
| Experiment loop (RQ1-RQ6) | `experiments/` | harness + runners |

## Design decisions and their justification
- **Krum design-`f` vs attack-`f`.** Krum's neighbour count uses a *design* fault budget
  (how many faults the operator provisions for), which must satisfy `n - f - 2 >= 1`. The
  *attack* may inject a different number of Byzantine judges. The harness sets Krum's design
  `f` to the largest value its committee supports and lets the attack vary independently, so
  Krum can be observed degrading when the attack exceeds its design budget.
- **Majority vote is not threshold-calibrated.** Its aggregate "score" is already a binary
  decision; calibrating a continuous threshold on `{0,1}` is degenerate, so it keeps `tau=0.5`.
- **Escalation defaults to the safe side.** Algorithm 1's escalate branch collapses to
  `block` for metric purposes (configurable), so escalation is never scored as an evasion.
- **Detection metric.** The paper frames robust aggregation as an outlier detector over
  verdicts; we score it with AUROC/F1 using each verdict's distance to the aggregate as the
  suspicion score (implemented with a dependency-free rank-sum AUROC).

See `audits/math_to_code_audit.md` for equation-level edge cases and the one documented
ambiguity (majority-vote ties).

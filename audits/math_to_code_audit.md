# Math-to-Code Audit — Aegis-Agency

For each mathematical object implemented, its paper location, code interpretation, edge
cases, internal-consistency check, and any correction made for a sound implementation.

---

## Verdict space and decision (Eq. 1–2)
- **Paper location:** Preliminaries, Eq. (1)–(2).
- **Object:** $v_k=(d_k,s_k,\bm e_k)$; verdict vector $\bm u_k=(s_k,\bm e_k)\in\mathbb R^d$;
  $\mathsf D(\bm u)=\mathbf 1\{\pi_1(\bm u)\ge\tau\}$.
- **Code:** `data/schemas.py` (`Verdict`, `verdict_vector`), `methods/gate.py` (`decision`).
- **Edge cases:** score clipped to $[0,1]$; embedding may be empty ($m=0$, so $d=1$, decision
  from score alone) — validated. Decision uses `>=` at the threshold (matches Eq. 2).
- **Consistent?** Yes. `>=` vs `>` at $\tau$: paper uses $\ge$; implemented as `>=`.

## Coordinate-wise median (Alg. 2; Prop. 1)
- **Object:** per-coordinate median; lies in the honest range for $f<n/2$.
- **Code:** `methods/aggregators.py::coordinate_median` = `np.median(U, axis=0)`.
- **Edge cases:** even $n$ → `np.median` averages the two middle values. The bracketing proof
  (Prop. 1) uses a single middle order statistic; the average of the two central honest-
  bracketed values still lies in $[a,b]$, so the guarantee is preserved. Documented.
- **Consistent?** Yes.

## Geometric median / Weiszfeld (Alg. 2; Lemma 1)
- **Object:** $\hat{\bm u}=\arg\min_y\sum_k\lVert y-\bm u_k\rVert$; Alg. 2 weights
  $w_k=1/\max(\lVert y-\bm u_k\rVert,\eta)$.
- **Code:** `methods/aggregators.py::geometric_median` (smoothed Weiszfeld, `eta`, `max_iter`,
  `tol`).
- **Edge cases:** (i) a query point coincident with a data point → division by zero, avoided
  by the $\max(\cdot,\eta)$ floor exactly as Alg. 2 specifies; (ii) non-convergence → capped
  at `max_iter` with a logged warning; (iii) $n=1$ → returns that point.
- **Consistent?** Yes. **Note:** Lemma 1's bound $C_\alpha r$ is for the *exact* geometric
  median; the Weiszfeld output is an $\eta$-approximation. The theory-check
  (`metrics/theory.py`) therefore tests the bound with a tolerance and flags the
  approximation — it does NOT assert the exact-median inequality on the iterate. This is the
  mathematically conservative choice.

## Krum (Alg. 2; Lemma 2 restated)
- **Object:** select $\bm u_{k^\star}$, $k^\star=\arg\min_k\sum_{j\in\mathcal N_k}
  \lVert\bm u_k-\bm u_j\rVert^2$, $\mathcal N_k$ = $n-f-2$ nearest neighbours.
- **Code:** `methods/aggregators.py::krum`.
- **Edge cases:** requires $n-f-2\ge 1$ i.e. $n\ge f+3$; the paper's guarantee needs
  $2f+2<n$ (i.e. $n\ge 2f+3$). Code raises `ValueError` if $n-f-2<1$ and logs a warning if
  $2f+2\ge n$ (guarantee not in force but computation still defined). Ties in the score →
  lowest index (deterministic).
- **Consistent?** Paper is internally consistent. Lemma 2 is a *restatement* (not reproven);
  the code implements the selector but the theory module does not claim Krum's constant.

## Majority vote (baseline rule)
- **Object:** $\hat d=\mathbf 1\{\sum_k d_k>n/2\}$.
- **Code:** `baselines/majority_vote.py`, `methods/aggregators.py::majority_vote`.
- **Edge cases — TIE ($n$ even, $\sum d_k=n/2$):** the paper's strict `>` makes a tie resolve
  to **allow** ($\hat d=0$). For a *defense*, allow-on-tie is the less safe choice. We
  therefore (a) implement the paper's strict `>` as the default to match the text, and (b)
  expose `tie_breaking ∈ {"paper_strict","block_on_tie","random"}`; `random` requires a seed
  (randomised tie-breaking, relevant to the paper's remark on ties). **Documented as a
  potential paper ambiguity**; default follows the paper.
- **Consistent?** The paper specifies `>`; we follow it and warn.

## $\varepsilon$-isolation (Def. 1; Prop. 2)
- **Object:** TV distance between verdict laws with/without injection $\le\varepsilon$.
- **Code:** modelled in `attacks/injection.py` and `judges/isolation.py`: with probability
  $\le\varepsilon$ an honest judge's verdict is perturbed by the injection. Prop. 2's count
  bound $\binom{n-f}{t+1}\varepsilon^{t+1}$ is implemented in `metrics/theory.py`
  (`injection_flip_bound`) under the independence assumption the proposition states.
- **Edge cases:** $\varepsilon=0$ → no honest verdict changes (perfect isolation). $\varepsilon
  \in[0,1]$ validated.
- **Consistent?** Yes; independence assumption surfaced in a docstring and a warning.

## ASR-under-compromise / ORR (Eq. 4–5)
- **Object:** $\Pr[\hat d=0\mid y^\star=1]$ and $\Pr[\hat d=1\mid y^\star=0]$.
- **Code:** `metrics/metrics.py::asr_under_compromise`, `over_refusal_rate`.
- **Edge cases:** empty conditioning set (no unsafe / no benign items) → returns `nan` with a
  logged warning, never a divide-by-zero. Escalate decisions are counted per an explicit
  policy flag (`escalate_as ∈ {"block","allow","exclude"}`, default `"block"` for asr-uc and
  `"exclude"`-aware handling documented). Defaults documented in the function.
- **Consistent?** Yes.

## $C_\alpha$ displacement and integrity condition (Eq. 6–7; Lemma 1, Thm 1)
- **Object:** $C_\alpha=\frac{2(n-f)}{n-2f}$; integrity iff $\gamma>C_\alpha r$.
- **Code:** `metrics/theory.py::c_alpha`, `integrity_condition`, `displacement_holds`.
- **Edge cases:** requires $n-2f>0$; `c_alpha` raises `ValueError` for $f\ge n/2$ (constant
  undefined/infinite), matching the theorem's domain. $C_0=2$, $C_{1/4}=3$, $C_{1/3}=4$ unit-
  tested.
- **Consistent?** Yes; the code enforces the theorem's domain rather than returning a bogus
  value.

## Correlated-failure variance (Eq. 8; Prop. 3)
- **Object:** $\mathrm{Var}(\bar Z)=\mu(1-\mu)[\frac{1-\rho}{N}+\rho]$, $N=n-f$.
- **Code:** `metrics/theory.py::correlated_variance`; asymptotic floor
  $\mu(1-\mu)\rho$ as `correlated_variance_floor`.
- **Edge cases:** $\rho\in[0,1]$, $\mu\in[0,1]$, $N\ge1$ validated. For validity of an
  exchangeable-Bernoulli correlation, $\rho\ge-1/(N-1)$; we restrict to $\rho\ge0$ (the
  paper's regime) and document.
- **Consistent?** Variance identity verified symbolically in the unit test against a
  Monte-Carlo estimate from an exchangeable-Bernoulli sampler.

## Cost models
- **Object:** token cost linear in $n$; parallel/sequential latency.
- **Code:** `metrics/cost.py`. Pure arithmetic from the paper; no external claim.
- **Consistent?** Yes.

---

## Summary of internal-consistency findings
- **No hard contradictions** between definitions, theorems, and algorithms were found.
- **One ambiguity (majority-vote ties):** paper uses strict `>`; we follow it as default and
  expose safer/randomised options. Logged as `paper_inconsistency_detected` (minor) in the
  traceability CSV.
- **One approximation gap (gmed exact vs Weiszfeld):** the theory bound is stated for the
  exact geometric median; the code tests it with tolerance and never asserts the exact
  inequality on the iterate. Conservative and documented.
- All divide-by-zero / empty-set / domain issues are guarded and logged, never silently
  defaulted.

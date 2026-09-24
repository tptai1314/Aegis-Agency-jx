# Math Audit — Aegis-Agency

Audited objects: Lemma 1 (gmed displacement), Theorem 1 (decision integrity),
Corollary 1 (availability), Proposition 1 (coordinate median), Lemma 2 (Krum, restated),
Proposition 2 (isolation containment), Proposition 3 (correlated failure). Proofs are in
`appendix/proofs.tex` (§Appendix A) except Lemma 2 which is restated from a cited source.

---

## Lemma 1 — Geometric-median displacement
- **Exact statement.** For verdicts $\bm u_1,\dots,\bm u_n\in\mathbb R^d$ with an honest
  subset $\mathcal H$, $|\mathcal H|=n-f$, obeying $\lVert\bm u_k-\bm u^\star\rVert\le r$,
  and $f<n/2$, the geometric median $\hat{\bm u}$ satisfies
  $\lVert\hat{\bm u}-\bm u^\star\rVert\le C_\alpha r$, $C_\alpha=\frac{2(n-f)}{n-2f}
  =\frac{2(1-\alpha)}{1-2\alpha}$, $\alpha=f/n$.
- **Required assumptions.** Honest concentration (Assumption 1); $f<n/2$ (Assumption 3,
  median branch). No distributional assumption; deterministic worst case.
- **Proof status.** COMPLETE. Uses only optimality of the geometric median (F(û)≤F(u*)) and
  the triangle / reverse-triangle inequalities. Reproduced in full in Appendix A.1.
- **External theorem used.** None invoked as a black box; the result is consistent with the
  1/2 breakdown point of the geometric median (Lopuhaä–Rousseeuw 1991; Minsker 2015), cited
  for context only, not used as a proof step.
- **Dimensional consistency.** Both sides are lengths in $\mathbb R^d$; $C_\alpha$ is
  dimensionless. ✓
- **Constant/asymptotic dependency.** $C_\alpha$ increasing on $[0,1/2)$, $C_\alpha\to\infty$
  as $\alpha\to1/2$; requires $n-2f>0$, i.e. $f<n/2$, used at the division step. ✓
- **Is the claim too strong?** No optimality/matching claim is made; explicitly stated as a
  *sufficient* displacement bound (Remark after the lemma and in Appendix A.1).
- **Possible gap / required revision.** None. Note: the bound assumes the *exact* geometric
  median; the Weiszfeld iteration (Algorithm 2) computes an approximation, so a deployment
  should use an approximation-error term (flagged in TODO_before_submission.md).

## Theorem 1 — Decision integrity under a Byzantine minority
- **Exact statement.** Under Assumptions 1–3 (median rule) and margin condition
  $\gamma>C_\alpha r$, $\mathsf D(\hat{\bm u})=\mathsf D(\bm u^\star)=y^\star(x)$ for every
  Byzantine choice; up to $f$ colluding judges cannot force a pass or a block.
- **Required assumptions.** Assumptions 1 (concentration $r$), 2 (margin $\gamma$, and
  $\mathsf D(\bm u^\star)=y^\star$), 3 ($f<n/2$); condition $\gamma>C_\alpha r$.
- **Proof status.** COMPLETE (Appendix A.2). Chains Lemma 1 with 1-Lipschitzness of the
  score projection $\pi_1$ and the margin.
- **External theorem used.** Lemma 1 (internal).
- **Dimensional consistency.** $|\pi_1(\hat u)-\pi_1(u^\star)|\le\lVert\hat u-u^\star\rVert$
  (score is a coordinate, 1-Lipschitz). ✓
- **Constant/asymptotic dependency.** Guarantee holds iff $\gamma>C_\alpha r$; both measured
  empirically (Section 10 / Appendix B). ✓
- **Is the claim too strong?** The theorem is explicitly CONDITIONAL. The paper repeatedly
  states it does not equal end-to-end security. No "optimal/first/tight" wording attached.
- **Possible gap / required revision.** The dependence on the *reference* $\bm u^\star$: the
  theorem certifies the decision equals $\mathsf D(\bm u^\star)$, and Assumption 2 ties
  $\mathsf D(\bm u^\star)=y^\star$. If honest judges are collectively wrong, $\bm u^\star$ may
  not equal ground truth — this is exactly what Proposition 3 handles and what the text flags.
  No revision needed; the coupling is stated.

## Corollary 1 — Availability
- **Status.** COMPLETE (Appendix A.2). Immediate from the symmetric second implication in
  Theorem 1's proof. ✓ No overreach.

## Proposition 1 — Coordinate-wise median integrity
- **Statement.** With $f<n/2$, each coordinate of $\mathrm{cmed}$ lies in the honest range;
  score-coordinate displacement $\le r$ (i.e. $C_\alpha\to1$), so $\gamma>r$ suffices.
- **Proof status.** SKETCH in main text; full 1-D order-statistic argument in Appendix A.3
  (complete). Consistent with Yin et al. 2018 (coordinate median tolerance).
- **Dimensional / constant checks.** ✓ ($C_\alpha=1$ for cmed on the score coordinate).
- **Too strong?** No. Bracketing argument is standard and correct.

## Lemma 2 — Krum selection (restated)
- **Statement.** Under $2f+2<n$, Krum selects a vector whose (expected) squared distance to
  the honest barycenter is bounded by a constant times honest variance.
- **Proof status.** RESTATED from Blanchard et al. 2017 (NeurIPS); NOT reproven. Labelled as
  such in the text and in Table 3.
- **External theorem used.** Blanchard et al. 2017, Proposition/analysis of Krum. Cited.
- **Caveats transplanted.** (i) bounds distance to barycenter, feeding Theorem 1 only via
  Assumption 1 with a possibly larger effective $r$ — stated; (ii) high-dimension
  degradation (El Mhamdi et al. 2018 / Bulyan) — stated, and mitigated by low-dimensional
  verdict space and by offering cmed/gmed. 
- **Too strong?** No; explicitly a restatement, and its limits are disclosed.
- **Required revision.** Before submission, either (a) reproduce the precise constant from
  Blanchard et al. adapted to the verdict space, or (b) keep as a cited restatement and do
  not rely on it beyond "cannot be forced arbitrarily far." Currently (b). Flagged in
  TODO_before_submission.md and citation_todo.md.

## Proposition 2 — Isolation containment
- **Statement.** If every honest judge is $\varepsilon$-isolated, an injection changes a
  fixed honest decision w.p. $\le\varepsilon$; $>t$ flips w.p. $\le\binom{n-f}{t+1}
  \varepsilon^{t+1}$ (independent draws); $\varepsilon=0$ leaves honest verdicts unchanged.
- **Proof status.** SKETCH in text; completed in Appendix A.4. Per-judge bound is exact from
  the TV definition; the count bound assumes independence of honest responses (stated), which
  is the complement of the Proposition 3 regime.
- **Too strong?** No; explicitly conditional on the measured $\varepsilon$ and on
  independence. The $\varepsilon=0$ idealisation is labelled as such.
- **Possible gap.** The union-bound constant $\binom{n-f}{t+1}$ is loose but valid; not
  claimed tight.

## Proposition 3 — Correlated honest failure
- **Statement.** For exchangeable Bernoulli honest-correctness with mean $\mu$, pairwise
  correlation $\rho$: $\mathrm{Var}(\bar Z)=\mu(1-\mu)[\frac{1-\rho}{n-f}+\rho]\to
  \mu(1-\mu)\rho$; for $\mu<1/2$, Chebyshev gives an $n$-independent floor on majority
  failure.
- **Proof status.** COMPLETE (Appendix A.5). Exact variance computation + Chebyshev.
- **Dimensional / algebraic check.** Variance formula verified:
  $\frac1{N^2}[N\mu(1-\mu)+N(N-1)\rho\mu(1-\mu)]=\mu(1-\mu)[\frac{1-\rho}{N}+\rho]$. ✓
- **Too strong?** No. Presented as a limiting-case impossibility for the *honest majority*,
  explicitly "not a defense." One-sided, asymptotic — stated.
- **Possible gap.** Exchangeability with a single $\rho\ge0$ is a modelling choice (requires
  $\rho\ge-1/(N-1)$ for validity; we take $\rho\in[0,1]$). Stated as a model, not a law of
  nature.

---

## Cross-cutting checks
- No proof takes the contrapositive of a one-sided upper bound to assert achievability. ✓
- No classical theorem (Krum, coordinate-median, geometric-median breakdown, Chebyshev) is
  invoked without its assumptions being present in the local model. ✓
- Upper bounds are never used to claim a matching attack exists. The colluding-attack
  *feasibility* is treated empirically (Section 10, Appendix B), not derived from the bounds. ✓
- High-probability vs. expectation: Lemma 1, Theorem 1, Prop 1, Prop 2 (per-judge), Prop 3
  variance are deterministic/exact; Lemma 2 (expectation, cited) and Prop 3 majority-floor
  (probabilistic) are labelled accordingly. ✓

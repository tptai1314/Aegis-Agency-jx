# Citation TODO — Aegis-Agency

Claims that still need a citation, or citation tasks to finish before submission. There are
**no `\textcolor{red}{[Citation needed]}` markers left in the manuscript** — every asserted
claim is either self-contained (our own definitions/theorems), attributed to a verified
reference, or explicitly flagged as a placeholder/expected trend.

## Style / metadata tasks (not missing citations)
1. Expand `and others` in long author lists (`xi2023agentsurvey`, `bai2022constitutional`,
   `touvron2023llama2`) to the full list if the target journal requires complete authors.
2. Add page numbers / DOIs for arXiv-only entries at camera-ready (bibtex "empty pages"
   warnings are cosmetic).
3. Re-confirm canonical venue tags (HarmBench→ICML 2024, TAP→NeurIPS 2024, MetaGPT→ICLR 2024,
   PsySafe→ACL 2024, He et al.→Findings ACL 2025, ASB→ICLR 2025) against the published
   proceedings for the final reference list.

## Optional strengthening (would improve, not required)
4. **Lemma 2 (Krum) constant.** Currently a cited restatement of Blanchard et al. 2017. If a
   reviewer wants a self-contained bound in the verdict space, either reproduce Blanchard's
   constant adapted to $\bm u_k$ or add a citation to a follow-up that states the constant
   explicitly. Tracked in `math_audit.md`.
5. **Geometric-median approximation error.** The exact-gmed bound (Lemma 1) vs. the Weiszfeld
   approximation (Algorithm 2) gap could be supported by a convergence-rate citation for
   smoothed Weiszfeld (already have `pillutla2022rfa`; add its explicit rate statement when
   the approximation term is added to the theorem).
6. **Calibration across heterogeneous judges.** `guo2017calibration` supports single-model
   temperature scaling; if the camera-ready claims cross-judge score comparability more
   strongly, add a multi-model calibration reference.

## Claims deliberately left uncited (correct as-is)
- All statements of our own Definitions, Assumptions, Lemmas, Theorems, Propositions,
  Corollaries — original, proven in Appendix A or labelled restated/sketch.
- The security-vs-cost frontier numbers in Table 4 — arithmetic from $f<n/2$, not an external
  claim.
- Placeholder result cells/curves — labelled non-results, need data not citations.

# TODO Before Submission — Aegis-Agency

Ordered by importance. Nothing here blocks compilation; all items are for turning this
rigorous design+analysis draft into a submittable paper.

## 1. Placeholder results (MUST replace)
- [ ] Run the full evaluation of Section 10 / Appendix B and replace every "--" in
      **Table 5** (main comparison) and **Table 6** (ablation).
- [ ] Replace the conceptual curves in **Figure 6** (`pgfplots_placeholder_results.tex`) with
      measured data; remove the "Conceptual / Placeholder" wording once real.
- [ ] Report mean ± std over seeds and an explicit significance test for every comparison.
- [ ] Measure and report $r$, $\gamma$, $\rho$, and $\varepsilon$ (the quantities Thm 1,
      Prop 2, Prop 3 depend on) — these are the crux of whether the guarantees bind.

## 2. Unverified / incomplete proofs
- [ ] **Lemma 2 (Krum):** currently a cited restatement. Either reproduce Blanchard et al.'s
      constant adapted to the verdict space, or keep as restatement and ensure no result
      relies on more than "cannot be forced arbitrarily far."
- [ ] **Lemma 1 vs. Algorithm 2:** add a Weiszfeld approximation-error term to the exact-gmed
      bound, or state that the deployment uses exact gmed to tolerance $\eta$.
- [ ] Optionally promote Proposition 1 (coordinate median) sketch to a full inline proof.

## 3. Missing experiments / analysis the idea flags
- [ ] Small-committee tolerance study (is $f=1$ meaningful?) — RQ5.
- [ ] Correlated-failure test: homogeneous vs. diverse backbones, with measured $\rho$ — RQ4.
- [ ] Security-vs-cost frontier in measured latency and token cost — RQ5.
- [ ] Second-order injection with isolation on/off to estimate $\varepsilon$ — RQ2.

## 4. Author / venue metadata
- [ ] Fill real author names, affiliations, emails, ORCIDs (currently placeholders in
      `main.tex`).
- [ ] Choose the target venue and switch class/style accordingly:
      - Journal (IEEE TDSC / Computers & Security): keep `elsarticle` or switch to IEEEtran.
      - Conference (CCS/USENIX/S&P/NDSS): switch to `acmart` / `usenix` / IEEE `sig-alternate`
        and re-fit page limits (this draft is ~47 pp, longer than most conference limits).
- [ ] Add acknowledgements, funding, ethics/broamder-impact, and artifact-availability
      statements per venue.

## 5. Citation / bibliography finishing
- [ ] Expand `and others` author lists (`xi2023agentsurvey`, `bai2022constitutional`,
      `touvron2023llama2`) to full lists if required.
- [ ] Add page numbers/DOIs for arXiv-only entries; re-confirm canonical venue tags at
      camera-ready (see `citation_todo.md`).

## 6. LaTeX warnings (cosmetic)
- [ ] Reduce the 85 overfull hboxes (max ~75 pt) — mostly wide tables and long inline math;
      tighten column specs or add `\small`/`\resizebox` where needed.
- [ ] Silence the `OT1/cmr/m/scit` (small-caps-italic) font substitution by avoiding
      `\textsc` inside italic contexts, or load a font that provides the shape.
- [ ] Optional clarity note: add one sentence in Preliminaries distinguishing $d$ (verdict
      dimension) from $d_k$ (per-judge decision) — see `notation_consistency_check.md`.

## 7. Assumptions requiring empirical validation (state in camera-ready)
- [ ] Honest concentration (Assumption 1): do real hardened LLM judges concentrate? Measure.
- [ ] Decision margin (Assumption 2): measure $\gamma$ near policy boundaries.
- [ ] $\varepsilon$-isolation (Definition 1): measure residual leakage of hardened+isolated
      judges under JudgeDeceiver.
- [ ] Embedding separability: verify the sentence encoder separates honest vs. manipulated
      rationales; treat encoder robustness as an attack surface.

## 8. Figures (polish, optional)
- [ ] Optional: make the Byzantine judge the top committee row in Figure 1 so the "compromise"
      arrow drops vertically (current diagonal is acceptable but could be cleaner).

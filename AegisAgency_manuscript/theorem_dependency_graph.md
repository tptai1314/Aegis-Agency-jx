# Theorem Dependency Graph — Aegis-Agency

Textual companion to Figure 4 (`figures/tikz_theory_diagram.tex`).

## Chain

```
Assumptions
  A1  honest concentration (radius r)
  A2  decision margin (gamma; D(u*) = y*)
  A3  tolerable Byzantine fraction (f < n/2  |  2f+2 < n)
  Def.1  epsilon-isolation

        A1, A3(median) ─────────────► Lemma 1  (gmed displacement,  ||û−u*|| ≤ C_alpha r)
                                           │
        A2 ────────────────────────────────┼──► Theorem 1  (decision integrity)
                                           │        │
                                           │        └──► Corollary 1  (availability, symmetric)
        A1, A3(median) ──► Prop. 1 (coordinate-median integrity; C_alpha → 1)  ──► Theorem 1 (score coord.)
        A3(Krum) ───────► Lemma 2 (Krum selection, restated [Blanchard17]) ──► Theorem 1 (via A1, larger eff. r)
        Def.1 ──────────► Prop. 2 (second-order injection containment)

        A1 (violated as rho→1) ──► Prop. 3 (correlated-failure floor)  ⟂ voids A1
```

## Reading

- **Assumptions → Lemmas.** A1+A3 give Lemma 1 (geometric median) and, via the cited
  result, Lemma 2 (Krum). Def.1 underlies Prop. 2.
- **Lemmas → Theorems.** Lemma 1 (or Prop. 1 for cmed, or Lemma 2 for Krum) plus A2 gives
  Theorem 1. Corollary 1 is the availability direction of the same proof.
- **Theorems → Design/Experiments.** Theorem 1's condition `gamma > C_alpha r` is what
  Section 10 / Appendix B measure (r, gamma, rho). Prop. 3 is the limiting-case counterweight
  that *voids* A1 when honest errors correlate; it feeds RQ4 (homogeneous-vs-diverse test)
  and the diversity requirement, not a defense.
- **Goals mapping.**
  - G1 (integrity)   ⟸ Lemma 1, Theorem 1, Prop. 1, Lemma 2.
  - G1 (availability) ⟸ Corollary 1.
  - G2 (injection containment) ⟸ Def. 1, Prop. 2.
  - G3 (graceful degradation) ⟸ C_alpha growth in Lemma 1 + Prop. 3.
  - G4 (cost) ⟸ Section 9 (complexity), independent of the theorem chain.

## No circular dependencies
Each arrow points from lower to higher abstraction (Assumption → Lemma → Theorem →
Corollary → Experiment). Prop. 3 does not feed any theorem; it bounds the validity region of
A1 and is therefore drawn as a side constraint, not an input.

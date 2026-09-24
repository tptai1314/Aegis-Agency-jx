# Claim-Strength Audit — Aegis-Agency

Every strong-claim word was searched in the manuscript source. Below: each occurrence, its
context, and the verdict (justified / weakened / removed / not used).

## "first"
- **Intro contribution 1 & Related Work positioning:** "the first threat model, to our
  knowledge within the verified literature, that drops AutoDefense's honest-agent and
  single-coordinator assumptions..." and "To the best of our knowledge, within the verified
  literature, Aegis-Agency is the first defense that...".
  - **Verdict: WEAKENED and retained.** Always hedged with "to the best of our knowledge,
    within the verified literature." Never a bare "first." Justified by the related-work
    survey (AutoDefense, SecAlign, StruQ, Krum, MAS-security papers) none of which model an
    insider minority inside a defense pipeline with robust verdict aggregation.

## "optimal"
- Searched: appears only as "order-optimal" when *describing cited work* (Yin et al. 2018,
  coordinate median). Never claimed for Aegis-Agency results.
  - **Verdict: not used for our claims.** ✓

## "sharp" / "tight" / "matching"
- **Not used anywhere for our results.** Lemma 1's constant is explicitly called a
  *sufficient* bound with "We make no optimality or matching claim" (Appendix A.1 remark and
  theory remark). No "matching lower bound," no "matching impossibility."
  - **Verdict: not used / explicitly disclaimed.** ✓

## "state-of-the-art" / "superior" / "outperforms"
- **Not used.** All empirical numbers are placeholders; the paper states "No superiority
  claim is made on their basis" (Section 10) and every result table is labelled placeholder.
  - **Verdict: not used.** ✓

## "guarantee" / "prove" / "secure"
- "guarantee" and "prove" are used for Lemma 1, Theorem 1, Corollary 1, Prop 3 — all with
  complete proofs in Appendix A, and all stated CONDITIONALLY (assumptions named).
- "secure": used only to *deny* the claim — "reduced ASR under compromise is not 'secure'",
  "not a proof of end-to-end security", Section 6.3 "What 'secure' does and does not mean."
  - **Verdict: justified (proofs) and carefully bounded (non-claim stated repeatedly).** ✓

## "no efficiency loss" / "universal" / "exact"
- **Not used** as claims about Aegis-Agency. "universal" appears only in cited titles
  (universal adversarial attacks / universal prompt injection). "exact" appears only in the
  technical sense "exact assumptions/statement."
  - **Verdict: not used for our claims.** ✓

## "graceful degradation"
- Central claim, used throughout.
  - **Verdict: justified.** Backed by Lemma 1's $C_\alpha$ growth (bounded, increasing
    displacement, not a cliff) and Prop 3 (smooth variance floor). Explicitly contrasted with
    "not a proof of security."

## Novelty phrasing audit
- All novelty sentences use safe framing: "To the best of our knowledge, within the verified
  literature, ...", "Under Assumptions X–Y, we show ...", "The analysis characterises ...",
  "The result suggests ... and will be tested empirically." ✓

## Summary
| Word | Occurrences for our claims | Action |
|---|---|---|
| first | 2 (both hedged) | weakened, retained |
| optimal | 0 | — |
| sharp/tight/matching | 0 | explicitly disclaimed |
| state-of-the-art/superior | 0 | — |
| secure (as achieved) | 0 | used only as denied non-claim |
| universal/exact/no-loss | 0 | — |
| guarantee/prove | many | justified by complete conditional proofs |

**No unjustified strong claim remains.** The one residual empirical-strength risk is that the
conditional guarantees could be *read* as unconditional; this is mitigated by the explicit
non-claim in the abstract, Section 6.3, Discussion, and Limitations, and is listed in
reviewer_audit.md as the top objection to pre-empt.

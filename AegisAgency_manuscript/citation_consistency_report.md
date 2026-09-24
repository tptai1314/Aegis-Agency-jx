# Citation Consistency Report — Aegis-Agency

Automated + manual cross-check between `references_verified.bib`, the `\cite` keys in the
manuscript, and `citation_claim_map.csv`.

## Cite-key ↔ bib coverage
- **`.bib` entries:** 66.
- **Distinct `\cite` keys in manuscript:** 66.
- **`bibcite` entries resolved in `main.aux`:** 66/66.
- **Every `\cite{...}` key exists in the `.bib`:** YES — final compile reports **no undefined
  citations** (verified: `grep "There were undefined" main.log` → none on the final pass).
- **Every `.bib` entry is cited:** YES — the three initially-uncited entries
  (`mcmahan2017fedavg`, `qian2023chatdev`, `yi2024jailbreaksurvey`) were wired into
  Related Work / Introduction; `comm -23 defined cited` now returns empty.

## Metadata correctness (spot + systematic check against `citation_claim_map.csv`)
- **Author order:** taken verbatim from fetched arXiv/proceedings/Crossref pages (see
  `citation_search_log.md`). Multi-author entries with `others` (`xi2023agentsurvey`,
  `bai2022constitutional`, `touvron2023llama2`) use `and others` deliberately to avoid
  transcription error on very long lists — flagged in TODO for full expansion if the venue
  requires it.
- **Titles:** exact, including capitalisation protected with braces (e.g. `{AutoDefense}`,
  `{Byzantine}`, `{StruQ}`).
- **arXiv IDs match titles:** verified per-entry (each `eprint` was the fetched page).
- **DOIs match titles:** verified via Crossref for the 7 DOI entries (`chen2017byzantine`,
  `pillutla2022rfa`, `minsker2015geometric`, `lopuhaa1991breakdown`, `lamport1982byzantine`,
  `lai2016agnostic`, `kuncheva2003diversity`).

## Claim-support check (no citation used for an unsupported claim)
- Every `\cite` was placed against the sentence it supports, per `citation_claim_map.csv`
  (`exact_claim_supported`, `source_location`).
- **No empirical paper cited as proving a theorem.** Byzantine guarantees are attributed only
  to Blanchard 2017 (Krum), Yin 2018 (coordinate median), Chen 2017 (geometric median),
  Minsker 2015 / Lopuhaä–Rousseeuw 1991 (breakdown/concentration) — all theory sources.
- **No centralized method cited as decentralized, no digital-agg cited as analog-agg, no DP
  cited as Byzantine robustness** — these confusions are structurally impossible here (topic
  is verdict aggregation), and the FL sources are explicitly framed as trusted-server.
- **`castro1999pbft` and `lai2016agnostic`** are marked `metadata-only, not used for
  substantive claims` and are cited only for context (BFT framing / high-dim robustness
  pointer), never as a proof source. Consistent with their use in the text.

## Topic-dominance check
- Directly-on-topic references (seed + injection/jailbreak/judge/MAS-security + Byzantine
  aggregation) constitute > 70% of the 66 entries; peripheral entries (FL, alignment,
  embeddings, calibration, BFT classics) are each tied to a specific supporting sentence and
  do not dominate. No neighbouring-topic cluster displaces the locked topic.

## Duplicates / versions
- Llama Guard appeared as both a seed and a moderation candidate → merged to a single entry
  (`inan2023llamaguard`); logged in `references_rejected.md`.
- No conference/journal/arXiv triple-entries; where a paper has multiple versions the most
  appropriate single verified version is cited (e.g. SecAlign → CCS 2025 with arXiv eprint).

## Residual items (also in citation_todo.md / TODO_before_submission.md)
- Long author lists using `and others` should be fully expanded to match the target journal's
  reference style before submission.
- Page numbers are absent for arXiv-only entries (bibtex "empty pages" warnings) — cosmetic;
  add on final typesetting if the venue requires.
- Venue tags for a few entries reflect canonical publication where the arXiv page listed only
  subject classes (HarmBench→ICML'24, TAP→NeurIPS'24, etc.); re-confirm against the published
  proceedings at camera-ready.

## RESULT: CONSISTENT
No undefined citations, no uncited entries, no unsupported-claim attributions, no
topic-foreign dominance. Outstanding items are cosmetic/style and tracked.

# Reviewer-Style Self-Audit — Aegis-Agency

## Locked topic
Securing the multi-agent LLM defense pipeline itself against (a) compromised/colluding
internal defenders and (b) second-order prompt injection, via Byzantine-robust verdict
aggregation + payload isolation + hardened judges, generalised from jailbreak to
prompt-injection/task-integrity. (See `idea_fidelity_report.md`.)

## Does the manuscript match idea.pdf?
YES. Verified line-by-line in `topic_alignment_check.md` (title, keywords, theorem
directions, algorithm, experiment plan, seed references all PASS; forbidden-drift scan clean).

## Final contribution list
1. Threat model dropping AutoDefense's honest-agent + single-coordinator assumptions, adding
   second-order injection and adaptivity (Sec. 4–5).
2. Aegis-Agency architecture: Byzantine-robust verdict aggregation + payload isolation +
   hardened judges + task-integrity generalisation (Sec. 6–7).
3. Theory: decision-integrity theorem (Thm 1), correlated-failure floor (Prop 3),
   security-vs-cost complexity (Sec. 9).
4. Adaptive evaluation protocol vs. AutoDefense / single SecAlign / majority vote (Sec. 10),
   with placeholder results.

## Main novelty claims and supporting evidence
| Novelty claim | Evidence in paper | Verified citation boundary |
|---|---|---|
| Insider (Byzantine/colluding) threat inside a defense pipeline | Threat model Sec. 5.3; Thm 1 | AutoDefense assumes honest agents `zeng2024autodefense`; MAS-security papers attack task teams, not defensive ones `lee2024promptinfection,he2025redteaming,zhang2024psysafe` |
| Robust verdict aggregation replacing the single coordinator | Sec. 6.3, Alg. 2, Thm 1 | Krum/median are trusted-server gradient rules `blanchard2017krum,yin2018byzantine`; transplanted to verdicts |
| Closing the second-order injection surface | Def. 1, Prop. 2 | JudgeDeceiver shows judges are injectable `shi2024judgeinjection`; StruQ/spotlighting isolate `chen2024struq,hines2024spotlighting` |
| Generalisation jailbreak→task-integrity | Sec. 6.4 | instruction hierarchy `wallace2024instruction`; injection benchmarks `liu2024formalizing` |

## Missing citations
None blocking (see `citation_todo.md`); only style/metadata and optional-strengthening items.

## Theorem/proof risks
- **Thm 1 is conditional** (concentration + margin). Risk: read as unconditional. Mitigation:
  explicit non-claim in abstract, Sec. 6.3, Discussion, Limitations. **Top objection to
  pre-empt.**
- **Lemma 2 (Krum) is restated, not reproven.** Risk: reviewer wants a verdict-space
  constant. Mitigation: cmed/gmed carry complete proofs; Krum offered as one of three rules.
- **Prop 3 uses an exchangeable-correlation model.** Risk: reviewer questions realism.
  Mitigation: presented as a limiting-case impossibility + measured $\rho$ in evaluation.
- **Lemma 1 uses exact gmed; Alg. 2 approximates.** Risk: approximation gap. Mitigation:
  logged in TODO; add error term at camera-ready.

## Experiment gaps
- **All results are placeholders.** The empirical contribution is pending. This is disclosed
  everywhere (abstract, Sec. 10 header, every table/figure caption, Limitations). A reviewer
  expecting measured results will see the paper is at the design+analysis+protocol stage.
- Feasibility risks the idea itself flags (small-committee tolerance; achievable diversity)
  are directly targeted by RQ4/RQ5 ablations but not yet answered.

## Possible reviewer objections (and where addressed)
1. "Conditional guarantee ≠ security." → Sec. 6.3, Discussion, Limitations; framed as
   graceful degradation.
2. "Small committees tolerate only f=1." → Table 4, Limitations, RQ5 — treated as an open
   empirical question, not assumed away.
3. "Adaptive attackers break robust aggregation (Fang/Baruch/Xie)." → Related Work +
   evaluation is adaptive by construction; Thm 1 stated as conditional.
4. "Correlated judges void tolerance." → Prop 3 proves it; diversity made a requirement.
5. "Aggregating embeddings assumes a good encoder." → Discussion + Limitations disclose the
   encoder assumption and treat it as an attack surface.
6. "Why not agent debate among judges?" → Discussion: star topology deliberately removes the
   inter-agent channel; debate left as future work.

## Overclaiming risks
Audited in `claim_strength_audit.md`: no bare "first/optimal/sharp/matching/SOTA"; all novelty
hedged "to the best of our knowledge, within the verified literature"; "secure" used only as a
denied non-claim.

## TikZ / figure readability status
6/6 figures PASS (`figure_quality_audit.md`); rendered to PNG and inspected; no overlaps;
placeholder plots labelled as non-results.

## LaTeX compilation status
Compiles with `latexmk -pdf main.tex`; 47 pages; **no undefined references or citations**;
66/66 citations resolved; remaining warnings are cosmetic (85 overfull hboxes, max ~75pt,
mostly in wide tables/inline math; and small-caps-italic font substitution). See
`TODO_before_submission.md`.

## Citation verification status
66 references, all verified against a primary source (arXiv/proceedings/Crossref); see
`citation_search_log.md`, `citation_claim_map.csv`, `citation_consistency_report.md`. 0
fabricated. 5 seed references resolved with correct metadata.

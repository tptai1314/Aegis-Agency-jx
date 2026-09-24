# Figure Quality Audit — Aegis-Agency

All figures rendered to PNG at 130 DPI under `figure_previews/` and visually inspected.
Colour grammar (blue=honest, red=attack, green=defense, purple=theory, gray=infra) is shared
via `figures/tikz_styles.tex` and reinforced by shape/dash for grayscale readability. Icons
are pure TikZ (no external images).

| Item | Value |
|---|---|
| Fig. 1 filename | `figures/tikz_architecture.tex` |
| Label | `fig:architecture` |
| Page after compile | 11 |
| Preview | `figure_previews/page_11-11.png` |
| Icons used | yes (server, model, lock, judge/gavel, shield, doc, attacker) |
| Arrows routed cleanly | yes (blue data lanes, red dashed adversarial, gray control) |
| Any arrow overlaps text | no — labels on white chips (`elabel`) |
| Any text overlaps text | no |
| Arrow labels readable | yes ("verdicts", "compromise", "second-order injection", "data ch.") |
| Colours consistent | yes |
| Grayscale-readable | yes (Byzantine judge dashed; lanes differ by dash) |
| Caption informative | yes |
| Rendered & inspected | yes |
| Required fixes | minor: the red "compromise" diagonal to Judge k passes near neighbouring judge boxes but does not overlap them; acceptable. Optional future tweak: make the Byzantine judge the top committee row for a fully vertical drop. |
| Final | PASS |

| Item | Value |
|---|---|
| Fig. 2 filename | `figures/tikz_threat_privacy_robustness.tex` |
| Label | `fig:threat` | Page | 13 | Preview | `page_13-13.png` |
| Icons | yes (attacker, lock, shield, doc, server) |
| Routing / overlap | three clean lanes (red top / blue spine / green bottom); no overlaps |
| Labels | white chips ("shift ε", "f Byz.", "bias û", "contains", "resists", "bounds") |
| Legend | yes (attack lever / defense / honest pipeline) |
| Grayscale | yes (Byzantine node dashed; lanes by position) |
| Caption / inspected | yes / yes |
| Final | PASS |

| Item | Value |
|---|---|
| Fig. 3 filename | `figures/tikz_method_pipeline.tex` |
| Label | `fig:pipeline` | Page | 15 | Preview | `page_15-15.png` |
| Icons | yes | Routing | single left-to-right numbered flow; escalate branch gray dashed |
| Overlap | none; short equations on chips beneath stages |
| Caption / inspected | yes / yes | Final | PASS |

| Item | Value |
|---|---|
| Fig. 4 filename | `figures/tikz_theory_diagram.tex` |
| Label | `fig:theory` | Page | 21 | Preview | `page_21-21.png` |
| Node classes | assumptions (purple) / lemmas (green) / theorems (blue) / corollary (gray) / correlated-failure (red) |
| Routing | orthogonal routed connectors; grouping box around assumptions; one annotated edge ("ρ↑ voids r") |
| Overlap | none material; the A1→Prop.3 edge is routed below Prop.1 without crossing text |
| Caption / inspected | yes / yes | Final | PASS |

| Item | Value |
|---|---|
| Fig. 5 filename | `figures/tikz_experimental_setup.tex` |
| Label | `fig:expsetup` | Page | 28 | Preview | `page_28-28.png` |
| Icons | yes (doc, attacker, shield, check) |
| Routing | datasets→SUT→metrics→outputs; attacks (red) in; baselines (blue) compare |
| Overlap | none; "payloads"/"compare" on chips | Final | PASS |

| Item | Value |
|---|---|
| Fig. 6 filename | `figures/pgfplots_placeholder_results.tex` |
| Label | `fig:results` (subfigs `fig:results-a`, `fig:results-b`) | Page | 31 | Preview | `page_31-31.png` |
| Type | PGFPlots, two subfigures |
| Placeholder labelled | YES — caption bolded "Conceptual expected trends; not experimental results. Placeholder curves; replace with actual experimental data before submission." |
| Legends / axes | yes, both panels; axis labels present |
| Colours consistent | yes (green=Aegis, amber=majority, gray=single model, red=cost) |
| Grayscale | distinct markers (circle/square/triangle) + dashes |
| Caption / inspected | yes / yes | Final | PASS |

## Summary
6/6 figures PASS the overlap/readability audit. No arrow crosses text, node, icon, or
equation; every figure has caption, label, and (where lanes/colours differ) a legend or
in-caption legend; all rendered to PNG and visually inspected. Placeholder plots are
unambiguously marked as non-results.

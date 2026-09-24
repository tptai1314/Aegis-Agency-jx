# Citation Search Log — Aegis-Agency

## Objective
Build a verified bibliography (~50–80 references) for a manuscript that faithfully
expands `idea.pdf` (Idea 05: Aegis-Agency — Byzantine-robust, injection-hardened
multi-agent LLM defense pipelines). Final count: **66 verified references**.

## Scholarly databases / sources searched
- arXiv official abstract pages (`arxiv.org/abs/…`) — primary existence + metadata source.
- ACL Anthology (`aclanthology.org`) — for EMNLP/ACL/Findings versions.
- PMLR proceedings (`proceedings.mlr.press`) — ICML/AISTATS/UAI.
- NeurIPS proceedings (`papers.neurips.cc` / `papers.nips.cc`).
- OpenReview (`openreview.net`) — ICLR.
- USENIX presentation pages (`usenix.org/conference/...`).
- Crossref DOI registry (`api.crossref.org/works/DOI`) — for journal / ACM / IEEE / statistics
  papers whose publisher pages block automated fetches (ACM DL, Project Euclid, IEEE Xplore).
- DBLP (`dblp.org`) — for OSDI/older proceedings metadata.

## Inclusion criteria
1. Paper existence confirmed by fetching a real arXiv/proceedings/DOI page.
2. Exact title, ordered author list, year, and venue captured verbatim from that page.
3. Topical relevance to the locked specification (see `idea_fidelity_report.md`):
   prompt injection, jailbreak, LLM-as-judge, multi-agent LLM systems and their security,
   Byzantine-robust aggregation, robust statistics, BFT, and the specific backbones/benchmarks/
   alignment methods the idea's method and evaluation depend on.
4. At least 60% of the corpus directly on the locked topic (satisfied: the seed +
   competing-methods + jailbreak/injection/judge/MAS-security + Byzantine-aggregation clusters
   are all directly on-topic; peripheral entries are limited to per-sentence support).

## Exclusion criteria
- Could not verify existence / metadata from a primary source → rejected.
- Off-topic for the locked specification (e.g. generic NLP unrelated to safety/security).
- Would be cited only for keyword overlap rather than a specific supported sentence.
- Predatory or unverifiable venue; blog posts; unverifiable GitHub repos as sole source.
- Duplicate of a version already included, unless both versions are intentionally discussed.

## Seed references from idea.pdf and resolved metadata
| Idea PDF label | Resolved title | Authors | Venue/Year | Verified via |
|---|---|---|---|---|
| AutoDefense (arXiv:2403.04783) | AutoDefense: Multi-Agent LLM Defense against Jailbreak Attacks | Zeng, Wu, Zhang, Wang, Wu | arXiv 2024 (rev. Nov 2024) | arxiv.org/abs/2403.04783 |
| Krum (Blanchard et al., NeurIPS 2017) | Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent | Blanchard, El Mhamdi, Guerraoui, Stainer | NeurIPS 2017 | papers.nips.cc (2017 hash f4b9ec30…) |
| SecAlign (arXiv:2410.05451) | SecAlign: Defending Against Prompt Injection with Preference Optimization | Chen, Zharmagambetov, Mahloujifar, Chaudhuri, Wagner, Guo | ACM CCS 2025 | arxiv.org/abs/2410.05451 |
| StruQ (arXiv:2402.06363) | StruQ: Defending Against Prompt Injection with Structured Queries | Chen, Piet, Sitawarin, Wagner | USENIX Security 2025 | arxiv.org/abs/2402.06363 |
| Llama-Guard (named inside AutoDefense) | Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations | Inan et al. (11 authors) | arXiv 2023 | arxiv.org/abs/2312.06674 |

All five seed references had complete, resolvable metadata; no "Anonymous"/incomplete seed
entries required arXiv-ID cross-resolution beyond the IDs already given in the idea PDF.

## Major literature clusters (final corpus)
1. **Core seed** (5): AutoDefense, Krum, SecAlign, StruQ, Llama Guard.
2. **Closest competing / directly related** (16): indirect & universal prompt injection
   (Greshake, Liu-formalizing, Liu-universal, InjecAgent), instruction hierarchy (Wallace),
   spotlighting (Hines), LLM-as-judge & its failures (Zheng, Wang-notfair, JudgeDeceiver),
   MAS security (Prompt Infection, Evil Geniuses, Flooding, CORBA, Agent-in-the-Middle/He,
   PsySafe, Agent Security Bench).
3. **Foundational theory** (11): Krum(seed), coordinate-median/trimmed-mean (Yin),
   geometric-median GD (Chen-2017), Bulyan (El Mhamdi), RFA (Pillutla), signSGD majority
   vote (Bernstein), geometric-median concentration (Minsker), breakdown points
   (Lopuhaä–Rousseeuw), Byzantine Generals (Lamport), PBFT (Castro–Liskov), high-dim robust
   estimation (Diakonikolas-2016, Lai, Diakonikolas-survey).
4. **System / robustness background** (moderation & guardrails, MAS frameworks):
   AutoGen, MetaGPT, CAMEL, ChatDev, multi-agent debate, agent survey, NeMo Guardrails,
   LlamaFirewall, ShieldGemma, OpenAI moderation.
5. **Experimental baselines / attacks / datasets / backbones** (jailbreak & defense):
   GCG/AdvBench (Zou), HarmBench, PAIR, TAP, DAN, GPTFuzzer, jailbreak survey, SmoothLLM,
   baseline defenses (Jain), certified erase-and-check (Kumar), Llama 2 backbone.
6. **Peripheral, per-sentence** (FL & poisoning & diversity & alignment & embeddings &
   calibration): FedAvg, backdoor FL, local model poisoning (Fang), "A Little Is Enough"
   (Baruch), inner-product manipulation (Xie), ensemble diversity (Kuncheva), InstructGPT,
   Constitutional AI, DPO, Sentence-BERT, SimCSE, calibration (Guo).

## Rejected candidate references
See `references_rejected.md`. Summary of rejection reasons encountered: (a) book-length
references not cleanly verifiable via arXiv/proceedings/DOI (Huber, *Robust Statistics*);
(b) duplicate versions (Llama Guard appeared as both a seed and a cluster-1 candidate — merged
into a single entry `inan2023llamaguard`); (c) venue-tag uncertainty resolved rather than
rejected (arXiv HTML intermittently dropped headers for InstructGPT, DPO, Lai et al., PsySafe,
Agent-in-the-Middle — cross-verified via Crossref/ACL Anthology).

## Verification-source notes
- Two entries were confirmed through ACL Anthology when the arXiv HTML summariser dropped the
  header: `zhang2024psysafe` (2401.11880 → ACL 2024) and `he2025redteaming` (2502.14847 →
  Findings ACL 2025).
- ACM/IEEE/Project-Euclid journal entries (`chen2017byzantine`, `pillutla2022rfa`,
  `minsker2015geometric`, `lopuhaa1991breakdown`, `lamport1982byzantine`, `lai2016agnostic`,
  `kuncheva2003diversity`) were verified through Crossref DOI records because the publisher
  pages block automated fetches.
- Venue labels marked as conference proceedings for papers whose arXiv page lists only subject
  classes (e.g. HarmBench→ICML 2024, TAP→NeurIPS 2024) reflect their canonical publication; the
  arXiv identifiers themselves are all confirmed. Any residual venue-only uncertainty is logged
  in `citation_claim_map.csv` (verification_level column) and `TODO_before_submission.md`.

## Final reference count
- Verified and included in `references_verified.bib`: **66**.
- Directly-on-topic fraction: > 70% (clusters 1–5), comfortably above the 60% requirement.

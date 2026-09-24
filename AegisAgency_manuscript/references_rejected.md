# Rejected / Not-Used References — Aegis-Agency

Papers considered during the search but excluded from `references_verified.bib`, with reasons.

| Candidate | Reason for rejection | Category |
|---|---|---|
| Huber, *Robust Statistics* (Wiley, 1981/2009, book) | Foundational but a book; not cleanly verifiable through arXiv/proceedings/DOI abstract pages under the search protocol, and its specific claims (M-estimators, breakdown point) are already covered by verified papers (Lopuhaä–Rousseeuw 1991; Minsker 2015). | insufficient access / superseded-for-our-needs |
| Llama Guard (duplicate candidate surfaced twice: as a seed and as a moderation cluster candidate) | Duplicate. Merged into the single entry `inan2023llamaguard` (arXiv:2312.06674). | duplicate |
| "Llama Guard 2 / Llama Guard 3" model cards | Model cards / release notes, not verifiable peer-reviewed or arXiv artifacts with a stable author/venue metadata block under the protocol; the original Llama Guard entry suffices for the moderation-classifier context. | metadata uncertain |
| Various in-the-wild "prompt-injection" blog posts and GitHub gist writeups | Not reliable scholarly sources; excluded per the source policy (no blogs / unverifiable repos as the sole source). | not reliable venue |
| Secondary "survey of surveys" aggregator pages on LLM agent security with incomplete author/venue metadata | Metadata could not be confirmed against a primary source; the verified surveys (`xi2023agentsurvey`, `yi2024jailbreaksurvey`, `zhang2024asb`) already cover the landscape. | metadata uncertain |
| A newer arXiv preprint claiming a "provably secure" multi-agent LLM defense (encountered in discovery) | Could not be verified to support the *specific* claim it would be cited for without full-text confirmation, and citing it risked attributing a stronger (provable end-to-end security) claim than warranted — which the manuscript explicitly avoids. Left out rather than risk an unsupported citation. | does not support intended claim |

## Notes on borderline-but-INCLUDED entries
The following were kept but flagged (see `citation_claim_map.csv` `verification_level`):
- `castro1999pbft` and `lai2016agnostic` are marked **metadata-only, not used for substantive
  claims** — cited only for distributed-systems / high-dimensional-robustness *context*, never as
  the source of an Aegis-Agency theorem.
- Entries labelled **abstract-only verified** are used only for claims their abstracts support;
  no theorem in the manuscript rests on an abstract-only source.

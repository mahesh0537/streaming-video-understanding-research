---
zotero_key: null
authors: (anonymous / camera-ready pending — ICLR 2026)
year: 2026
arxiv: null
openreview: 738HjJEbml
pdf: https://openreview.net/pdf?id=738HjJEbml
tier: compact
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---

# QueryStream: Query-Aware Pruning and Proactive Response

**Lineage role:** frontier (ICLR 2026) bridge between the query-aware token-pruning thread and the proactive-trigger thread — rejects the query-agnostic "visual change = important" heuristic used by change-based triggers like [[timechat-online]].

**Compact summary** (no arXiv PDF; the OpenReview PDF sits behind a browser-verification wall, so this stays compact-tier from abstract/secondary sources):

- **Problem:** streaming pipelines prune tokens and schedule responses by *visual* dynamics, conflating "something changed on screen" with "something relevant to the query happened."
- **Key idea:** inject query-awareness into both halves: **QDP** (Query-Aware Differential Pruning) filters tokens by query relevance + temporal novelty against a smoothed history; **RTAR** (Relevance-Triggered Active Response) fires the proactive response when accumulated query-relevant evidence crosses threshold. Training-free, lightweight.
- **Claimed results:** SOTA on [[streamingbench]] and [[ovo-bench]] while pruning >70% of visual tokens (n/r — not verifiable against the PDF from here).
- **Why it matters:** a distinct trigger family — *relevance*-conditioned rather than change-conditioned — and evidence that the pruning (memory) and triggering (proactive) threads are converging; cf. [[streamov]]'s evidence-guided triggering.

**Upgrade path:** if the paper lands on arXiv, promote to deep (full method + figures) and set `arxiv:`.

*Verification: abstract-level only; numbers marked n/r.*

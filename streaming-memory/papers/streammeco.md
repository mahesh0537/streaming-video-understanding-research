---
zotero_key: null
authors: Junxi Wang, Te Sun, Jiayi Zhu et al. (Shanghai Jiao Tong University / Fudan / Shanghai AI Lab / HKUST / MemTensor)
year: 2026
arxiv: 2604.09000
pdf: https://arxiv.org/pdf/2604.09000
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---

# StreamMeCo: Long-Term Agent Memory Compression for Efficient Streaming Video Understanding

**Lineage role:** A memory-*graph* compression layer bolted onto an existing agent-memory streaming-video system (M3-Agent): it shrinks the accumulating text-node population of the memory graph with two connectivity-aware pruners (edge-free minmax sampling for isolated nodes, edge-aware weighted pruning for connected nodes) and adds a time-decay retrieval rule — trading storage/latency for retained (even slightly improved) accuracy. ACL 2026 Findings.

## Problem — what was limited before this paper (short)
Agent-memory approaches to streaming video (notably M3-Agent) build a persistent, growing **memory graph**: as the stream runs, they keep accumulating *text nodes* (episodic/semantic memory items) alongside entity nodes (faces, voices). The graph never stops growing, so both storage and — more painfully — **retrieval latency** blow up as the number of text nodes climbs into the thousands (their benchmarks average ~1,000–2,500 text nodes per video). Naive fixes (random dropping, plain clustering) throw away accuracy. Nothing exploited the *structure* of the memory graph to decide what to keep.

## Key idea — the core insight, 2-4 sentences
Text nodes in the memory graph split cleanly into two structural classes: **isolated** nodes (no edges to entity nodes) and **connected** nodes (linked to face/voice entities via edges). These deserve different compression policies. For isolated nodes there is no edge signal, so pick a maximally-diverse representative subset via spherical-KMeans clustering plus a farthest-point ("minmax") sampler; for connected nodes, score each by a fusion of its entity-importance (edge weight) and its embedding redundancy, and keep the top-K. Finally, at query time, weight retrieval toward **recent** segments with an exponential time-decay allocation (TMR), mimicking human recency bias.

![[streammeco.png]]

> **Crux (Figure 3).** The StreamMeCo overview: the memory graph is split into isolated text nodes (top, EMsampling: spherical KMeans then minmax farthest-point selection) and connected text nodes (bottom, EWpruning: fuse an entity-importance weight matrix $W$ with an embedding-similarity matrix $S$, then Top-K), and the compressed graph is queried through the TMR mechanism that pulls more memories from recent time segments. *Wang et al. (2026), arXiv:2604.09000. Embedded for personal research reference.*

## Method + math — the mechanism, then the central objective/equations in full

StreamMeCo is a **training-free** post-hoc layer over M3-Agent's already-built memory graph. M3-Agent (the base system) ingests a video stream, extracts entity nodes (faces, voices) and text nodes (episodic events + semantic facts), links them with edges, and answers questions by retrieving from this graph. StreamMeCo does not touch the base VLM; it compresses the text-node set and changes the retrieval allocation.

### 1. Text-node partition
Given the memory graph, text nodes are partitioned by connectivity:
- **Isolated text nodes** ($N_1$ of them): no edge to any entity node → handled by **EMsampling**.
- **Connected text nodes** ($N_2$ of them): edge-linked to $M$ entity (face/voice) nodes → handled by **EWpruning**.

### 2. Edge-Free Minmax Sampling (EMsampling) — isolated nodes
No edge signal exists, so the goal is a **diverse, representative** retained subset. First cluster the isolated nodes by their content embeddings $e_i$ using Spherical KMeans (Dhillon & Modha, 2001) into $N_1 \times a$ clusters, where $a$ is a **cluster ratio** hyperparameter:

$$\{C_j\} = \mathrm{KMeans}(\{e_i\},\, a), \qquad i = 1,\dots,N_1$$

Assume an overall retention ratio $\alpha$ for isolated nodes, distributed evenly across clusters, so cluster $C_j$ retains $|C_j|\times\alpha$ nodes. Within each cluster, seed the selected set $S_j$ with the node closest to the cluster centre, then **greedily add the farthest node** (max–min / farthest-point sampling). The distance from an unselected node $u_i \in U_j$ to the selected set is its minimum distance to any already-selected node:

$$D(u_i, S_j) = \min_{s_k \in S_j} \lVert e_{u_i} - e_{s_k}\rVert$$

and the next node added is the one maximizing that:

$$u_i^{*} = \arg\max_{u_i \in U_j} D(u_i, S_j)$$

Repeat until $|S_j|$ reaches $|C_j|\times\alpha$. This "minmax" rule spreads the retained nodes to cover the semantic space rather than clumping near cluster centres.

### 3. Edge-Aware Weighting Pruning (EWpruning) — connected nodes
For the $N_2$ connected text nodes (linked to $M$ entity nodes), build two matrices:
- an **entity-importance weight edge matrix** $W \in \mathbb{R}^{M \times N_2}$ (edge weights from entities to text nodes); row-wise summed and normalized to a per-text-node importance vector $W'$ (higher = attached to more/heavier entities).
- an **embedding-similarity matrix** $S \in \mathbb{R}^{N_2 \times N_2}$ among the connected text nodes; a redundancy score is derived and **inverted** ($1 - s'$) into $S'$ so that nodes with *less* similarity to others (more unique) score higher.

Fuse the two into a per-node retention score with a balance coefficient $b$:

$$r_j = b\, W'_j + (1-b)\, S'_j$$

then keep the **Top-K** connected nodes by $r_j$. So a node survives if it is either attached to important entities or carries non-redundant content.

### 4. Time-decay Memory Retrieval (TMR) — the retrieval rule
Compression alone can't guarantee that the *recent* segments (usually most relevant to a live query) are well-represented at retrieval time. TMR reallocates the retrieval budget across the $t$ temporal segments with an **exponential decay** favoring recent ones. For segment $j$ with base relevance $\bar E_j$, the decayed relevance is:

$$E'_j = \bar E_j \cdot e^{-\lambda (t - j)}$$

where $\lambda$ is the decay coefficient and $t$ is the current (most recent) segment index — so $t-j$ is the age. The number of nodes retrieved from segment $j$ out of a total budget $k$ is allocated proportionally:

$$\mathrm{Num}_j = \left( \frac{E'_j}{\sum_{i=1}^{t} E'_i} \right) \times k$$

This runs on top of the base system's character-based and semantic-based retrieval; the net effect is "more memories are retrieved from recent video segments."

## Explicit design choices — concrete decisions (raw material for new systems)
- **Training-free / plug-in**: no fine-tuning of the underlying VLM; operates purely on the constructed memory graph, so it drops onto M3-Agent (and, in the appendix, onto Mem0's graph memory) without retraining.
- **Structure-conditioned dual policy**: the *edge topology* of the graph (isolated vs connected) selects the compression algorithm — the paper's central design bet.
- **Isolated → coverage, connected → importance+uniqueness**: farthest-point diversity sampling for the edgeless nodes; entity-weight × redundancy fusion for the edged nodes.
- **Retention knobs**: cluster ratio $a$ (best 0.05), fusion balance $b$ (best 0.1), decay $\lambda$ (default 0.1), and a global compression rate reported at 30% / 50% / 70%.
- **Exponential over linear/piecewise decay** for TMR recency weighting (exponential wins in the ablation).
- **Underlying stack**: memory-graph construction uses Gemini-2.5-Pro + OpenAI `text-embedding-3-large` (an expensive dependency the authors flag as a cost limit).
- **Compression + retrieval are separable**: "+StreamMeCo" (compression only) and "+StreamMeCo+TMR" (compression + recency retrieval) are reported as distinct configs; TMR is what turns a small accuracy *loss* into an accuracy *gain*.

## Key results / what to remember — exact headline numbers with setting
Benchmarks (accuracy, %): **M3-Bench-robot** (100 videos, 1,276 QA, ~2,040s avg), **M3-Bench-web** (920 videos, 3,214 QA, ~1,631s avg), **Video-MME-Long** (300 videos, 900 QA, ~2,467s avg). Base system = **M3-Agent**.

Main table (Table 1), accuracy on robot / web / Video-MME-Long / **Average**:
- M3-Agent baseline: 30.3 / 47.9 / 56.0 / **44.7**
- +StreamMeCo (compression only) ↓30%: 30.7 / 47.0 / 54.8 / **44.2**
- +StreamMeCo ↓50%: 30.6 / 44.7 / 54.4 / **43.2**
- +StreamMeCo ↓70%: 28.4 / 41.9 / 53.3 / **41.2**
- +StreamMeCo+TMR ↓30%: 34.6 / 50.7 / 55.2 / **46.8**
- +StreamMeCo+TMR ↓50%: 33.4 / 48.9 / 56.6 / **46.3**
- +StreamMeCo+TMR ↓70%: 34.2 / 47.9 / 54.9 / **45.7**

Headline claim: **at 70% compression, +TMR reaches 45.7% average vs 44.7% baseline = +1.0% accuracy, with a 1.87× memory-retrieval speedup** (averaged across the three datasets). So it removes 70% of text nodes and *still improves* accuracy while nearly doubling retrieval speed — the recency retrieval (TMR) is doing the accuracy work; compression alone at 70% drops to 41.2%.

Baselines at ↓30% (robot / web): Random 28.5 / 44.6; Clustering 29.6 / 45.3 — both below +StreamMeCo's 30.7 / 47.0, confirming the structure-aware selection beats naive dropping.

Ablation (Table 2, 30% compression, robot / web, % of baseline retained): Baseline 30.3 / 47.9; Random 28.5 (94.1%) / 44.6 (93.1%); EMsampling only 30.3 (100%) / 45.7 (95.4%); +Entity Importance 30.1 / 46.1; +Embedding Similarity 30.2 / 46.1; **All components 30.7 (101.3%) / 47.0 (98.1%)** — each fusion term adds a bit, full model best.

Decay method (Table 3, M3-Bench-robot): exponential $\lambda{=}0.1$ = **34.6%** (best), beats linear at every $\lambda$ and beats a piecewise strategy (33.8%). Exponential degrades as $\lambda$ grows (34.6 → 32.0 at $\lambda{=}2.0$).

Also generalizes to **Mem0 (graph) memory** on Office/Meeting benchmarks (Table 6, appendix) — evidence the compression is not M3-Agent-specific.

No Zotero highlights present.

Takeaways: (1) Split memory-graph text nodes by edge connectivity and compress each class with its own rule. (2) A recency-weighted retrieval allocation (TMR) recovers/exceeds baseline accuracy after heavy compression. (3) 70% of memory can be discarded for ~1.87× retrieval speedup *without* accuracy loss — the win is latency, not accuracy per se.

## How it connects (evolution)
- [[streammem]], [[streamkv]], [[savemem]], [[omnimem]] — sibling long-term memory-compression schemes for streaming video; StreamMeCo is distinguished by operating on a *graph* memory (edges) rather than a KV cache or flat store.
- [[visual-agentic-memory]], [[eventmemagent]] — agent-memory framing (persistent structured memory an agent queries), the family M3-Agent belongs to.
- [[fluxmem]], [[hermes-kv]], [[streamingtom]] — memory-eviction / token-merging counterparts operating at the KV/token level instead of the graph-node level.
- [[streamforest]], [[flash-vstream]] — streaming-video memory baselines it compares against in Table 1.
- [[streaming-memory]] — the sub-topic hub tying these together.

## Open questions / limitations
- **Narrow eval surface**: because M3-Agent only ships memory graphs for M3-Bench-robot and M3-Bench-web, the third benchmark (Video-MME-Long) is the only extra they could build, limited by API budget/time — so cross-dataset generality of the graph-compression claim rests on few points.
- **Expensive graph construction**: the memory graph itself needs Gemini-2.5-Pro + `text-embedding-3-large` calls per stream — StreamMeCo speeds *retrieval*, not the (dominant) construction cost.
- **Accuracy margins are thin and noisy**: many deltas are ~0.3–1% and non-monotone in compression rate (e.g. robot 34.6→33.4→34.2 across 30/50/70%), so the "+1.0% at 70%" headline should be read as "no loss," not a robust gain.
- **Hyperparameter fragility**: strong sensitivity to $a$, $b$, $\lambda$ (best values 0.05 / 0.1 / 0.1) with no adaptive tuning — unclear how to set them for an unseen deployment.

*Verification: equations (1)–(6) transcribed from the arXiv HTML method section (Sec. 4.2) and cross-checked against the rendered PDF page 4 (Figure 3 fusion formula $r_j = bW'_j + (1-b)S'_j$); all accuracy/speedup numbers taken from Tables 1–3 and the abstract of arXiv:2604.09000v2. No Zotero access (offline).*

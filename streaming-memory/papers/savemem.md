---
zotero_key: null
authors: Hang Wu et al. (UC Merced; US Bank; Univ. of Queensland)
year: 2026
arxiv: 2605.07897
pdf: https://arxiv.org/pdf/2605.07897
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# Semantic-Aware Adaptive Visual Memory for Streaming Video Understanding

**Lineage role:** A training-free, two-stage streaming-memory wrapper for an off-the-shelf VLM (Qwen2.5-VL) that injects a *semantic* prior (a fixed pseudo-question bank) into three-tier token compression and adds an *anchor-conditioned recency gate* so retrieval scope adapts per query — lifting OVO-Bench 52.27 → 62.69 at ~half the GPU memory.

## Problem — what was limited before this paper (short)
Online streaming video understanding forces a VLM to ingest an unbounded frame stream under a fixed memory budget and answer at unpredictable moments, so *what to retain* is the core decision. Prior work handled this in two mostly-separate ways: (1) compress visual tokens before they enter the LLM using **visual-similarity heuristics** (e.g. FluxMem's Otsu-thresholded hierarchical memory), which drop redundancy but ignore whether a token is *semantically* informative; and (2) add **KV-cache-level retrieval** at inference (ReKV, LiveVLM, WeaveTime), which is bolted on *after* compression is finalized and often needs fine-tuning. The two consequences: compression rarely uses semantic signals, and retrieval and retention are hard to coordinate as one pipeline. SimpleStream even showed that just feeding the last-K frames is already competitive, hinting at an unexploited present-vs-past query trade-off.

## Key idea — the core insight, 2-4 sentences
Decouple the pipeline *at query-arrival time* into a **query-agnostic** compression stage and a **query-aware** retrieval stage that share one memory. Stage 1 scores every visual token once against a small **fixed pseudo-question bank** (generic probes for objects, counting, actions, scene change, spatial layout) so long-term retention is shaped by *semantic salience* rather than visual similarity alone, all under a constant token budget maintained across a three-tier (short / mid / long) memory. Stage 2 reads that memory: an **anchor-conditioned recency gate** decides whether the recent short-term anchor already suffices (present-focused query) or whether to reach into aged mid/long tiers (historical query), and ColBERT-style late interaction picks the candidate frames to feed the VLM.

![[savemem.png]]
> **Crux (Figure 2).** The two-stage framework: Stage 1 (query-agnostic) streams frames through Recency Anchor → Temporal Semantic Pruning → Spatial Semantic Selection → Selective Forgetting, guided by a pseudo-question semantic score; Stage 2 (query-aware) runs an anchor-guided recency gate then late-interaction scoring to retrieve candidate frames for the MLLM. *Wu et al. (2026), arXiv:2605.07897. Embedded for personal research reference.*

## Method + math — the mechanism, then the objective/equations in full

**Setup.** At time $C$ a causal constraint restricts the model to frames in $[0, C]$ — no bidirectional attention or global pooling. Newly encoded visual tokens must be triaged online.

**Query-agnostic semantic prior (Stage 1).** A fixed pseudo-question bank $Q$ of five generic probes is tokenized once at model load and shared across all videos/queries. Each visual token $v$ gets a scalar semantic salience via late-interaction **MaxSim** over the bank:
$$s(v) \;=\; \max_{q \in Q} \cos(v, q).$$
The max across probes captures whether a token aligns with at least one generic semantic axis (object presence, counting, action/event, scene change, spatial layout). Crucially $s(v)$ is computed **once per token at encoding time** and reused across every later tier transition, so the prior adds only one MaxSim per frame to streaming cost.

**Three-tier memory with selective forgetting.** Tokens flow through three tiers, each with its own compression operator:
- **$\mathcal{M}_{\text{short}}$ (Recency Anchor):** a FIFO buffer that keeps recent frames at *full fidelity* (keep all tokens). Anchors the most recent context.
- **$\mathcal{M}_{\text{mid}}$ (Temporal Semantic Pruning):** when short-term fills, the oldest frame is evicted here; tokens well-represented by their **temporal neighbours** are removed (drop cross-frame redundancy), but tokens with high $s(v)$ or at scene boundaries are spared. Semantic salience thus *augments* the similarity baseline rather than overriding it.
- **$\mathcal{M}_{\text{long}}$ (Spatial Semantic Selection):** when mid-term overflows, frames pass here; the operator enforces **spatial dispersion as a hard constraint** while ranking tokens by $s(v)$ (drop intra-frame noise).
- **Selective Forgetting:** once total tokens exceed the global budget $B$, the lowest-scoring tokens in $\mathcal{M}_{\text{long}}$ are discarded, giving an $\mathcal{O}(1)$ footprint independent of video length.

**Anchor-conditioned recency gate (Stage 2).** On query arrival, the gate first tests whether the recent anchor already contains enough query-relevant signal, comparing the query's affinity to short-term memory against an adaptively calibrated threshold:
$$\text{MaxSim}(\mathcal{M}_{\text{short}}, q) \;\geq\; \rho \cdot \overline{\sigma}^{\,\text{ema}}_{\text{short}},$$
where $\overline{\sigma}^{\,\text{ema}}_{\text{short}}$ is an **exponential moving average** of past short-term affinities (dynamic calibration), and $\rho$ adapts to query type — $\rho = 0.1$ for present-focused queries, $\rho = 2.0$ for historical queries. If satisfied, $\mathcal{M}_{\text{short}}$ is returned directly and aged-tier retrieval is bypassed.

**Query-aware late interaction.** Otherwise, for each candidate frame $g \in \mathcal{M}_{\text{mid}} \cup \mathcal{M}_{\text{long}}$, relevance is a ColBERT-style mean-of-max:
$$\sigma(g) \;=\; \frac{1}{|g|} \sum_{i=1}^{|g|} \max_{q_j \in q} \cos(g_i, q_j).$$
Each retained visual token $g_i$ contributes its max cosine similarity to the query tokens; the frame score is the mean of those per-token maxima.

**Adaptive top-K.** Frames are ranked by $\sigma(\cdot)$ and retrieval size adapts to score dispersion: tightly-clustered scores (diffuse evidence) → keep more frames; well-separated scores → aggressive filtering. The recency anchor is *unconditionally appended*, yielding the final set $\mathcal{M}^{*} = \mathcal{M}_{\text{short}} \cup \mathcal{R}$, which is passed to the MLLM to answer. The overall procedure is Algorithm 1 (two-stage streaming visual memory).

## Explicit design choices — concrete decisions (raw material for new systems)
- **Training-free wrapper:** no fine-tuning; applies directly to Qwen2.5-VL-7B and -3B backbones.
- **Fixed pseudo-question bank of 5 generic probes** (Appendix B): "What objects are visible…", "How many items/people…", "What actions/events…", "What has changed…", "Describe the spatial arrangement…"; instantiated once at model load, shared across all videos.
- **Semantic score computed once per token at encode time**, reused across all tier transitions — adds only one MaxSim/frame.
- **Three concrete tier sizes / operators:** short-term ≈ 4 frames FIFO full-fidelity; mid-term ≈ 16 frames temporal-pruned; long-term heavily pruned; global budget $B$ ≈ 2048 tokens.
- **Semantic salience augments (never overrides) the visual-similarity redundancy criterion**; scene boundaries explicitly spared.
- **Spatial dispersion is a hard constraint** in long-term selection (keep coverage, then rank by $s(v)$).
- **Recency gate uses an EMA-calibrated, per-query-type threshold** ($\rho=0.1$ present / $\rho=2.0$ historical) rather than a fixed cutoff.
- **Late interaction is ColBERT mean-of-max**, cheap and applied only to aged tiers when the gate fails.
- **Anchor always appended** to the retrieved set regardless of gate outcome.

## Key results / what to remember — headline numbers WITH setting (verified against the paper's tables)
- **OVO-Bench overall (Qwen2.5-VL-7B):** 52.27 → **62.69** (+10.41). Real-Time avg 59.90 → 74.93 (+15.03); Backward-Tracing avg 44.65 → 50.44 (+5.79). Biggest per-task gains: OCR 67.79→91.95 (+24.16), STU 42.13→65.73 (+23.60), HLD 23.66→37.63 (+13.97). EPM slightly regresses (51.52→50.84, −0.68).
- **OVO-Bench overall (Qwen2.5-VL-3B):** 52.23 → **57.34** (+5.12); Real-Time avg 61.27→70.15 (+8.88), BT avg 43.18→44.53 (+1.35).
- **Training-free competitors on OVO-Bench (7B):** FluxMem 57.22, HERMES 59.20, **SAVEMem 62.69 (best)**.
- **StreamingBench Real-Time (7B):** 73.9 → **76.0** (+2.1); (3B) 68.9 → 70.7 (+1.8).
- **ODV-Bench (7B):** Static 48.3 → **57.0** (+8.7), Dynamic 57.5 → **60.7** (+3.2); (3B) Static 46.0→47.9, Dynamic 55.9→56.6.
- **Efficiency (Fig. 3):** token count sub-linear — ~1.8k @ 8 frames → ~3.7k @ 128 frames; peak GPU memory 18.5 GB @128 frames vs baseline 35.8 GB = **48% reduction**.
- **Ablation (a) two-stage (7B):** baseline 52.27; Stage-1-only 57.79; Stage-2-only 59.95; both **62.69** — stages are complementary.
- **Ablation (b) semantic prior:** random vectors 56.05; single prompt 59.73; full question bank **62.69** — the semantic bank matters, and diverse probes beat a single prompt.
- **Ablation (c) recency gate:** No gate 59.06; Always gate 59.40 (BT collapses to 45.18); **EMA gate 62.69** — adaptive gating, not always-retrieve, is what preserves backward-tracing.

No Zotero highlights present.

Takeaways: (1) a *cheap, fixed* semantic prior injected into compression recovers most of the gap left by pure visual-similarity pruning; (2) treating present-vs-past queries with an adaptive gate (rather than always retrieving) is essential — the "Always gate" variant tanks backward-tracing; (3) decoupling compression (query-agnostic) from retrieval (query-aware) over a *shared* memory is what lets both improve without fine-tuning.

## How it connects (evolution)
- [[fluxmem]] — the strong recent visual-similarity / Otsu-threshold hierarchical-memory baseline SAVEMem is positioned against and beats (57.22 vs 62.69).
- [[rekv]] — KV-cache retrieval line (retrieve query-relevant KV entries at inference); SAVEMem instead retrieves at the *token/frame* level over an explicitly compressed memory.
- [[livevlm]] — KV-cache retrieval contemporary cited as the other bolt-on-retrieval approach SAVEMem contrasts with.
- [[weavetime]] — coarse-to-fine recall triggered by prediction uncertainty; a different adaptive-retrieval trigger than SAVEMem's recency gate.
- [[hermes-kv]] — training-free KV/memory competitor on OVO-Bench (59.20) that SAVEMem outperforms.
- [[ovo-bench]] — the primary benchmark (online-video, real-time + backward-tracing task split) all headline numbers report on.

## Open questions / limitations
- The pseudo-question bank is a *fixed* 5-probe set chosen by hand; whether it generalizes beyond these five semantic axes (or should be learned/expanded per domain) is untested.
- The recency-gate uses hard, hand-picked $\rho$ values (0.1 vs 2.0) keyed to a present/historical query classification — the paper does not detail how query type is inferred at runtime, and EPM already regresses slightly.
- Gains shrink notably on the 3B backbone and on StreamingBench/ODV-dynamic (+0.7 to +2.1), so the large OVO-Bench jump may be partly benchmark/backbone-specific.
- Evaluated only on Qwen2.5-VL; portability of the semantic-score-at-encode-time mechanism to other VLM tokenizers is unverified.

*Verification: numbers and equations checked against the paper's own Tables 1–4, Fig. 3 efficiency text, and Eq. (2) MaxSim / recency-gate / late-interaction formulas as extracted from arXiv:2605.07897v1 (arXiv HTML + PDF stream text); code repo https://github.com/wuhang03/savemem cited in the PDF but not fetched.*

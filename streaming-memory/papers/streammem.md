---
zotero_key: null
authors: Yanlai Yang et al. (Meta AI / New York University)
year: 2025
arxiv: 2508.15717
pdf: https://arxiv.org/pdf/2508.15717
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# StreamMem: Query-Agnostic KV Cache Memory for Streaming Video Understanding

**Lineage role:** Training-free, *query-agnostic* KV-cache compression for streaming video LLMs — it prunes/merges the KV cache online using chat-template tokens as a stand-in "generic query," so it needs neither the future question nor the video length, and holds a *fixed* memory budget.

## Problem — what was limited before this paper (short)
Feeding a long video to an MLLM produces a huge number of visual tokens: they overflow the context window, and storing/attending to their KV cache during decoding is expensive in both memory and compute. Prior streaming-memory approaches each had a hole. ReKV keeps *all* KV by offloading to CPU/disk and retrieving per query — memory grows unbounded, not scalable to ultra-long streams. LiveVLM uses a *fixed compression ratio* plus a FIFO eviction policy, so it forgets informative early content. Query-*dependent* compression (pick tokens relevant to the question) needs the question up front — impractical for open-ended long-video and multi-turn chat where questions arrive later or repeatedly. Naive frame/token pooling loses information (e.g. an action spans frames and no single frame captures it).

## Key idea — the core insight, 2-4 sentences
Salient-token selection normally needs the user query to score which visual tokens matter; StreamMem removes that dependence by appending the model's own **chat-template tokens** (`<|im_end|><|im_start|>assistant\n`) after each clip's visual tokens and using *their* attention over the visual tokens as a proxy saliency signal. Because captioning data dominates MLLM pretraining, those template tokens implicitly elicit a generic "describe this video" query, so the attention they emit highlights broadly-useful content without any real question. StreamMem then keeps a fixed-size memory by, per incoming clip, **pruning** low-attention KV and **merging** each frame's tokens into a compact weighted prototype. The result is a bounded KV budget that works online, query-free, and is model-agnostic / training-free.

![[streammem.png]]
> **Crux (Figure 2).** (a) The streaming workflow — each incoming clip is redundancy-filtered, encoded, appended with query-agnostic chat-template tokens, run through the MLLM, then the KV cache (old memory + new frames) is compressed and written back. (b) The compression module: attention-based *pruning* of both previous-memory and new-frame KVs (scored against the chat-template proxy query, red X = evicted) plus frame-wise *weighted merge* into per-frame prototypes (darker squares) → the fixed-size compressed KV memory. *Yang et al. (2025), arXiv:2508.15717. Embedded for personal research reference.*

## Method + math — the mechanism, then the central objective/equations in full

**Pipeline (Algorithm 1, per clip).** The video arrives as a stream; StreamMem processes it in clips of **8 frames**. For each new clip: (1) *input filtering* removes temporally-redundant frames; (2) the surviving frames are encoded and the visual tokens are appended with the chat-template proxy tokens and run through the MLLM, producing per-layer keys/values plus attention scores; (3) if the total cache exceeds the budget $M$, it is **pruned** to the top-scoring tokens; (4) each new frame's KV is additionally **merged** into a single prototype and inserted. Old memory and new-frame KV are compressed *together* every step, so the footprint never grows.

**1. Input frame filtering.** Compute cosine similarity between consecutive frame embeddings; when it exceeds a threshold $\delta$, merge the redundant frames by averaging. Default $\delta = 0.95$. This cuts redundancy in the *input* space before it ever enters the cache.

**2. Attention-based pruning (query-agnostic saliency).** Let $Q \in \mathbb{R}^{q \times d}$ be the query representation of the appended chat-template tokens and $K^i_t \in \mathbb{R}^{n \times d}$ the key matrix of the $n$ visual tokens of the current clip at layer $i$. Cross-attention of the proxy tokens onto the visual tokens is

$$
A^i_t = \mathrm{Softmax}\!\left( \frac{Q\,(K^i_t)^{\top}}{\sqrt{d}} \right), \qquad A^i_t \in \mathbb{R}^{q \times n}.
$$

Aggregating $A^i_t$ over the $q$ proxy-token rows gives a per-visual-token importance score; the top-$k$ tokens (per layer) are retained and the rest evicted. This is applied to *both* the previous-memory KVs and the new-frame KVs, so old-but-still-salient content can survive while stale content is dropped (unlike FIFO).

**3. Frame-wise KV merging (prototypes).** In addition to keeping top-$k$ raw tokens, each new frame's $n$ tokens are condensed into one prototype key/value via an attention-weighted sum, using the normalized importance weights $\alpha^i_j$:

$$
\bar{K}^i_t = \sum_{j=1}^{n} \alpha^i_j\, K^i_{t,j}, \qquad
\bar{V}^i_t = \sum_{j=1}^{n} \alpha^i_j\, V^i_{t,j}.
$$

Weighted merging preserves the spatial information a hard prune would discard, in a single compact slot per frame. (Ablation: weighted merge > average merge > no merge.)

**4. Fixed memory budget.** A hard cap enforces the fixed footprint across the whole stream, distributed evenly over the $L$ transformer layers:

$$
\sum_{i=1}^{L} \big\| K^{i\,\prime}_t \big\|_0 \le M,
$$

where $K^{i\,\prime}_t$ is the compressed key cache at layer $i$ and $M$ the total token budget.

**5. Positional embedding via YaRN.** To let a fixed-context MLLM span a long stream without corrupting spatial/temporal structure, StreamMem applies **YaRN** context-window extension rather than re-assigning position IDs (which would throw away original spatio-temporal order). The scaling factor $\lambda$ is model-specific: $\lambda=8$ for LLaVA-OneVision, $\lambda=2$ for Qwen2-VL, $\lambda=1$ for Qwen2.5-VL.

Everything above is **training-free** — no fine-tuning of the base MLLM; it is a drop-in inference-time cache controller.

## Explicit design choices
- **Proxy query = chat-template tokens** `<|im_end|><|im_start|>assistant\n` appended after visual tokens; leans on captioning-heavy pretraining to act as an implicit generic query (no real question needed).
- **Attention-based token scoring** against that proxy (Eq. 1), aggregated over proxy rows → per-token saliency; keep top-$k$ per layer.
- **Prune old memory + new frames jointly** every clip → salient early content can be retained; contrast with LiveVLM's FIFO forgetting.
- **Frame-wise weighted merge** (Eq. 2) into one prototype per frame, in addition to raw top-$k$ — preserves spatial detail cheaply.
- **Input-space filtering** by consecutive-frame cosine similarity, threshold $\delta = 0.95$ (sweet spot in ablation).
- **Fixed KV budget $M$**, split evenly across layers (Eq. 3); evaluated at 6K / 12K / 24K vs a ~50K full cache.
- **Clip granularity:** 8 frames per streaming chunk.
- **YaRN** visual-context extension with per-model $\lambda$ (8 / 2 / 1); large gains vs no scaling.
- **Model-agnostic / training-free:** demonstrated on LLaVA-OneVision-7B, Qwen2-VL-7B, Qwen2.5-VL-3B.

## Key results / what to remember
Numbers verified against the paper's own tables (HTML full text).

*Offline long-video understanding (Table 1), all at a 6K KV budget:*
- **LLaVA-OneVision-7B:** StreamMem MLVU-Medium **66.9** / MLVU-Long **63.0** / EgoSchema **56.6** / VideoMME **59.4** — beats the 6K baseline (64.7 / 60.1 / 54.7 / 56.9) and matches/exceeds ReKV (which needs ~353K/hour of stored KV) and LiveVLM.
- **Qwen2-VL-7B:** StreamMem MLVU-Long **67.2**, EgoSchema **62.4** at 6K, vs InfiniPot-V (65.6 / 60.8) and the 50K full-cache baseline (65.2).
- **Qwen2.5-VL-3B:** StreamMem 6K MLVU-Medium 62.3 / EgoSchema 60.1, roughly matching InfiniPot-V and the 50K baseline.
- Headline claim: with a **24K** budget (<½ the full cache), StreamMem *surpasses the full-KV* setting.

*Variable budget (Table 3, MLVU / Qwen2-VL-7B), "All":* StreamMem 6K **65.9**, 12K **66.0**, 24K **66.3** — vs Full KV (50K) 65.9 and InfiniPot-V 65.8 / 66.0 / 65.7. (Table 8, Qwen2.5-VL-3B, "All": 6K 62.3 → 12K 63.1 → 24K 64.3 vs 63.3 full.)

*Streaming VideoQA (Table 2, LLaVA-OneVision-7B, GPT-3.5 judge, acc / 1-5 score):* StreamMem RVS-Ego **57.6 / 3.8**, RVS-Movie **52.7 / 3.4** — competitive with Flash-VStream (57.0 / 4.0; 53.1 / 3.3) and InfiniPot-V (57.9 / 3.5; 51.4 / 3.5); full ReKV-with-offloading is higher (63.7 / 54.4) but stores unbounded KV.

*Ablations (MLVU, LLaVA-OneVision, "All"):* proxy query — chat-template **66.9** ≈ generic-text 66.7 < true-query oracle 68.1 (Table 4); merging — weighted **66.9** > average 66.3 > none 65.6 (Table 5); filtering — $\delta{=}0.95$ **66.9** > 0.90/0.97 > none 65.4 (Table 6); YaRN — $\lambda{=}8$ **66.9** vs $\lambda{=}1$ 61.5 (Table 7).

No Zotero highlights present.

Takeaways: (1) chat-template tokens are a shockingly good *free* proxy query — within ~1.2 pts of the oracle true query; (2) prune-and-merge together, at a fixed budget, matches or beats an unbounded cache; (3) YaRN context extension is not optional — it is worth ~5 pts on MLVU here; (4) the method is a training-free, model-agnostic inference-time wrapper.

## How it connects (evolution)
- [[rekv]] — the query-*dependent*, store-everything-and-retrieve baseline StreamMem argues against (unbounded memory); direct comparison in Table 2.
- [[livevlm]] — fixed-ratio + FIFO KV compression whose early-content forgetting StreamMem targets with joint prune-of-old-and-new.
- [[infinipot-v]] — concurrent fixed-budget streaming KV-compression method; StreamMem's main head-to-head baseline across Tables 1/3/8.
- [[flash-vstream]] — external-memory streaming VideoQA baseline compared on RVS-Ego/RVS-Movie (Table 2).
- [[hermes-kv]] — sibling KV-cache-memory approach in this sub-topic (query-agnostic KV compression lineage).
- [[streamkv]] — adjacent streaming KV-memory method for cross-referencing the design axis.

## Open questions / limitations
- The chat-template proxy assumes captioning-heavy pretraining elicits a useful *generic* query; for tasks needing narrow, rare details a true query still wins (oracle 68.1 vs 66.9) — StreamMem may prune content no generic caption would attend to.
- YaRN scaling factor is a hand-tuned per-model hyperparameter ($\lambda$ 8/2/1); no principled rule, and mis-set $\lambda$ costs several points.
- Streaming-VideoQA (Table 2) still trails full offloading ReKV — the fixed budget genuinely loses information on some ultra-long, detail-heavy streams.
- Evaluated on ≤7B MLLMs and specific benchmarks; behavior at larger scales, very high fps, or truly unbounded (hours-long) live streams is not stress-tested.

*Verification: equations (Eqs. 1-3), the pruning/merging mechanism, and all headline numbers checked against the paper's own tables (Tables 1-8) via the arXiv HTML full text (arxiv.org/html/2508.15717); crux figure cropped from the PDF (arxiv.org/pdf/2508.15717, Figure 2, page 2).*

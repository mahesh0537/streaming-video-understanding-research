---
zotero_key: null
authors: Ruanjun Li, Yuedong Tan, Yuanming Shi, Jiawei Shao (TeleAI, China Telecom · ShanghaiTech · Xidian)
year: 2025
arxiv: 2503.09387
pdf: https://arxiv.org/pdf/2503.09387
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# VideoScan: Enabling Efficient Streaming Video Understanding via Frame-level Semantic Carriers

**Lineage role:** pushes streaming-video memory to the extreme of **one token per frame** — a "semantic carrier" whose average-pooled embedding *and* its cached KV pairs jointly carry a frame's meaning forward, giving a fixed-size, query-agnostic, reusable streaming memory.

## Problem — what was limited before this paper (short)
Streaming video LLMs must ingest a frame-by-frame stream in real time, but each frame yields hundreds of visual tokens (e.g. ~210 for LLaVA-Video), so the KV cache and compute grow without bound as a video lengthens — incompatible with robotics / AR latency budgets. Prior token reduction splits into two unsatisfying camps: (a) pooling / sparse sampling only cuts 30–50% of tokens, not enough for hundred-frame videos; (b) instruction-guided compression reaches 1–2 tokens/frame but ties the compressed representation to a *specific query*, so the memory is neither reusable across questions nor temporally coherent — a poor fit for open-ended streaming interaction.

## Key idea — the core insight, 2-4 sentences
The authors observe a "semantic flow" property of decoder transformers: through cascaded self-attention, later token positions accumulate a weighted summary of preceding tokens, so pruning up to ~90% of low-attention vision tokens after a shallow layer barely hurts accuracy — the model reconstructs what it needs from the surviving high-attention tokens. VideoScan exploits this by representing **each frame with a single semantic carrier token** placed at the end of that frame's vision tokens. The carrier has two coupled parts: an **input embedding** = average pool of the frame's visual features (a stable, query-independent summary), and a **semantic flow** = the carrier's **KV pairs** cached at every layer, which inherit the frame's in-context information. After prefilling, all frame-level visual tokens are discarded and decoding runs purely off the carriers, kept in a fixed-capacity memory bank.

![[videoscan.png]]
> **Crux (Figure 3).** The VideoScan inference framework: in the prefilling phase a frame's vision-encoder embeddings $\mathbf{E}_t$ are average-pooled into one semantic-carrier embedding $\mathbf{C}_t$ (retrieved/merged with a memory of past carriers, projected, and fed with the text query through the LLM), and the carrier's per-layer KV pairs $\mathbf{P}_{t,l}$ are cached; in the decoding phase the frame tokens are gone and the LLM answers from the carrier embeddings + KV memory alone — so cost is $O(1)$ visual tokens per frame and GPU memory is duration-independent. *Li et al. (2025), arXiv:2503.09387. Embedded for personal research reference.*

## Method + math — the mechanism, then the central objective/equations in full

**Why one token can work — semantic flow.** For input embeddings $\mathbf{X}=[\mathbf{x}_1,\dots,\mathbf{x}_s]^\top\in\mathbb{R}^{s\times d}$, a self-attention layer computes
$$\mathbf{Y}=\mathrm{Softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V},\qquad \mathbf{Q}=\mathbf{X}\mathbf{W}_Q,\ \mathbf{K}=\mathbf{X}\mathbf{W}_K,\ \mathbf{V}=\mathbf{X}\mathbf{W}_V .$$
The output at the $(s{+}1)$-th position is a weighted aggregation of $\mathbf{V}$, i.e. a summary of the preceding $s$ tokens; stacked over $L$ layers this summarization is refined and propagated. Empirically, pruning low-attention vision tokens after a shallow layer costs almost nothing, and the degradation *shrinks* as the pruning layer index $l$ grows — LLMs "derive knowledge from tokens following the high-attention ones" rather than from the exact positions where visual content first appears. This licenses collapsing a frame into a single trailing carrier.

**Semantic carrier construction.** For frame $V^t\in\mathbb{R}^{H\times W\times 3}$ at time $t$, the vision encoder gives $\mathbf{E}^t\in\mathbb{R}^{N\times d}$ ($N$ = tokens/frame). The carrier's input embedding is the mean:
$$\mathbf{C}^t := \frac{1}{N}\sum_{i=1}^{N}\mathbf{E}^{t,i},\qquad \mathbf{C}^t\in\mathbb{R}^{1\times d}.$$
Placed at the end of the frame's token block, $\mathbf{C}^t$ also aggregates the frame's context into its layer-wise **KV pairs** $\mathbf{P}_{t,l}$ (the "semantic flow"), which are the second, complementary channel of the carrier.

**Two-phase inference.**
- *Prefilling:* encode $V^t\to\mathbf{E}^t$, pool to $\mathbf{C}^t$, retrieve up to $M$ historical carriers from the memory bank $\mathcal{M}$, project, concatenate with the system instruction $\mathbf{X}_t$ and text query $\mathbf{Q}_t$, and run one LLM forward pass (generating the first token and caching $\mathbf{P}_{t,l}$ for all layers $l\le L$).
- *Decoding:* frame-level visual tokens are dropped; every subsequent token is produced using only the cached carrier embeddings and KV pairs $\mathbf{P}_{t,l}$ — no re-attention over raw visual tokens.

**Memory mechanism (fixed-capacity, query-agnostic).** $\mathcal{M}$ holds at most $M$ carriers, each as (embedding, KV). On overflow, a **feature-duplication eviction** compares adjacent carriers by cosine similarity and drops the *older* member of the most-similar pair:
$$\text{evict } \arg\max_{t}\ \cos\!\big(\mathbf{C}^{t},\mathbf{C}^{t+1}\big),$$
discarding temporally redundant frames while keeping distinct events — this is what makes GPU memory (~18 GB) independent of video length.

**Two-stage training** (Fig. 4). *Stage 1* — joint LoRA fine-tune of **both** vision encoder and LLM on LLaVA-Video-178K, so the encoder learns to emit a good average-poolable representation and the LLM learns to reason from one token per frame. *Stage 2* — LLM-only fine-tune under a **semantic-aware causal mask** that forces unidirectional semantic flow: each frame's visual content is visible only to its own carrier, and later positions may attend to prior *carriers* but not to prior frames' raw visual tokens — guaranteeing that at decode time nothing is lost when the raw tokens are dropped. Stage 2 prioritizes clips with rich temporal dynamics.

## Explicit design choices
- **Backbone:** LLaVA-Video-7B (the base whose accuracy VideoScan retains ~85% of).
- **One token per frame** = average-pooled embedding **+** its cached KV, not either alone (ablation shows both are needed).
- Carrier is placed **at the end** of the frame's vision-token block so semantic flow accumulates into it before the block is discarded.
- **Query-agnostic** compression (mean pool), unlike instruction-guided methods — the memory is reusable across arbitrary later questions.
- **Fixed memory bank** of $M$ carriers (paper reports $M{=}64$ and $M{=}128$); overflow handled by cosine-similarity duplicate eviction, not FIFO — keeps distinct events, drops redundant ones.
- Frame-level visual tokens exist **only during prefilling**, then are dropped; decoding is carrier-only.
- **Semantic-aware causal mask** in Stage 2 enforces that all cross-frame information must route through carriers (no leakage from dropped raw tokens).
- Sampled at **1 fps** for streaming (vs fixed 64-frame sampling in the memory-free baseline).

## Key results / what to remember
Base = LLaVA-Video-7B (64 frames, 210 tokens/frame, avg 62.7). VideoScan uses **1 fps, 1 token/frame**.

**Offline VideoQA (Table 1, VideoMME reported without subtitles):**
- VideoScan ($M{=}128$): MVBench **48.9**, MLVU **61.3**, LongVideoBench **49.5**, VideoMME-Overall **55.1** (Short 64.2 / Medium 54.3 / Long 46.7), **Avg 53.7**.
- VideoScan ($M{=}64$): MVBench 48.9, MLVU 59.7, LongVideoBench 47.1, VideoMME 54.0, Avg 52.4.
- vs base LLaVA-Video-7B: MLVU 70.8, VideoMME 63.3, Avg 62.7 → VideoScan keeps **~85%** of base accuracy (53.7/62.7) while cutting **>99%** of vision tokens (210→1).

**vs token-efficient methods on MLVU (Table 2, avg over 7 subtasks):** VideoScan ($M{=}128$, 1 token) **61.3** vs LLaVA-Mini (1 token) **42.8** and LLaMA-VID (2 tokens) 33.2 — a large margin at equal token budget, and without instruction-coupled leakage. Per-subtask VideoScan: TR 81.8, AR 64.0, NQA 67.9, ER 62.5, PQA 64.9, AO 44.0, AC 31.6.

**Online streaming VStream-QA (Table 3, A100, single query-to-answer):**
- VideoScan ($M{=}128$): RVS-Ego **60.9** / score 4.0, RVS-Movie **54.1** / 3.5, latency **2.1 s**, VRAM **18 GB**.
- vs ReKV (LLaVA-OV-7B, offloaded KV): Ego 63.7 / Movie 54.4, latency 2.7 s, VRAM 36 GB → VideoScan is **1.29× faster** (2.1 vs 2.7 s) with **50% less VRAM** (18 vs 36 GB) at competitive accuracy.
- vs Flash-VStream-7B: Ego 57.3 / Movie 53.1, VRAM 19 GB → VideoScan beats it on both while using slightly less memory.
- Throughput: **6 FPS** serving, **5× faster** than the original LLaVA-Video (per Conclusion).

**Ablation — semantic-carrier components (Table 4, VideoMME w/o subtitles):** full 54.0 (Short 62.1 / Med 53.2 / Long 46.6); w/o input embedding 44.5; w/o KV 42.6 → dropping either channel costs ~9–11 points; **both are essential**.

**Ablation — memory mechanism (Table 5, VideoMME):** w/ memory $M{=}128$ **55.1** and $M{=}64$ 54.0 (1 fps) beat the memory-free evenly-sampled baselines (64 frames 52.8; 128 frames 53.8), i.e. the cosine-eviction memory adds ~1–2 points and, crucially, keeps memory bounded for arbitrarily long streams.

No Zotero highlights present.

Takeaways: (1) a frame can be compressed to **one query-agnostic token** if you keep **both** its pooled embedding and its cached KV — the KV is where "semantic flow" lives; (2) redundancy-aware (cosine) eviction, not FIFO, is what turns a token budget into a good long-video memory; (3) query-agnostic compression is the design lever that makes the memory reusable across questions, the property streaming interaction needs.

## How it connects (evolution)
- [[flash-vstream]] — a direct streaming baseline it beats on VStream-QA at lower VRAM.
- [[rekv]] — KV-retrieval streaming memory it matches in accuracy but at 1.29× speed and half the VRAM; VideoScan compresses instead of offloading/retrieving raw KV.
- [[streamkv]] / [[hermes-kv]] / [[streamforest]] — sibling KV-compression / memory approaches for streaming video, contrast the eviction and carrier design.
- [[infinipot-v]] — another bounded-memory long-video KV method; compare capacity-management policies.
- [[streamchat-mem]] — streaming memory for dialogue; shares the reusable-memory-vs-query-coupling tension VideoScan resolves via mean pooling.
- [[streaming-memory]] — the sub-topic hub.

## Open questions / limitations
- **Encoder-only architectures unsupported:** the method relies on autoregressive decoding, attention-sink behavior, and the causal-mask training, so it does not transfer to Transformer *encoder* VLMs.
- **Fine-grained loss:** one token per frame sacrifices precise spatial-temporal detail, capping performance on tasks needing exact visual reasoning (visible in the still-large gap to the 210-token base on MLVU/VideoMME).
- **Hybrid-memory trade-off:** the authors note that keeping a few information-rich tokens or adding instruction-guided retrieval could recover detail, but risks breaking the fixed-memory / efficiency guarantee — an open design tension.
- **Accuracy vs the best retrieval memory:** on RVS-Ego it still trails ReKV (60.9 vs 63.7); the win is efficiency, so whether carrier compression can close the accuracy gap is unresolved.

*Verification: all headline numbers cross-checked against the rendered PDF pages of Tables 1–5 (arXiv:2503.09387, pages 7–8) and the Fig. 3 caption; equations transcribed from the paper's Sec. 3.1–3.3 text. Where the fast HTML extractor mislabeled fields (e.g. VideoMME vs Avg column, the Table 2 margin, and the Table 5 M=64/M=128 rows), I used the values read directly from the table images.*

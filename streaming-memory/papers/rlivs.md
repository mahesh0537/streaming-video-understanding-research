---
zotero_key: null
authors: Vaggelis Dorovatas, Soroush Seifi, Gunshi Gupta, Rahaf Aljundi (Toyota Motor Europe · Archimedes RU, Athena RC · University of Oxford)
year: 2025
arxiv: 2510.17364
pdf: https://arxiv.org/pdf/2510.17364
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# Recurrent Attention-based Token Selection for Efficient Streaming Video-LLMs

**Lineage role:** Oct 2025 (NeurIPS 2025) training-free, backbone-agnostic streaming memory: consolidate each short clip into a tiny LLM-attention-selected visual-token set carried recurrently, and answer queries by retrieving stored *captions* rather than reprocessing frames.

## Problem — what was limited before this paper (short)
Standard Video-LLMs assume full random access to the whole video at query time, so their visual-token count grows linearly with length and blows past the context window on hour-long streams. Streaming methods each pay a cost: training-based summarizers (VideoStreaming, Flash-VStream) need bespoke fine-tuning and extrapolate poorly to arbitrary lengths; training-free KV-cache methods (ReKV) store large decoder key-value caches that are memory-heavy and redundant; caption-only clip methods (Goldfish) are cheap but process clips independently, losing entity/temporal continuity across clips. The gap: a training-free, model-agnostic way to keep a compact persistent memory *and* preserve cross-clip continuity.

## Key idea — the core insight, 2-4 sentences
Because the backbone Video-LLM already computes attention while captioning a clip, its attention *is* a free relevance signal over visual tokens. rLiVS keeps only the top ~6% of visual tokens that the caption attended to (a "compressed memory trace" of the clip), prepends this FIFO history to the next clip so past context steers current attention (recurrent processing), and — critically — answers questions not from visual tokens but from the *stored captions*, retrieved by MMR against the query. It is training-free and plugs into any short-clip Video-LLM (LLaVA-OneVision, Qwen2.5-VL) with no external modules.

![[rlivs.png]]
> **Crux (Figure 1).** Left: each short clip is prepended with attention-selected visual tokens from past clips (history $S^1$), captioned, and a new sparse token subset is selected from the current clip's attention scores to grow the FIFO memory. Right: at query time, accumulated per-clip text descriptions are matched to the query embedding and the most similar captions are fed to the LLM for answering. *Dorovatas et al. (2025), arXiv:2510.17364. Embedded for personal research reference.*

## Method + math — the mechanism, then the central objective/equations in full
The backbone is a frozen short-clip Video-LLM = visual encoder VE + projector P + LLM. For a $T$-frame clip $\mathbf{V}$ and an instruction $\mathbf{X}_I$ ("Describe what is happening in the video."), it emits projected visual tokens and a caption $C$:
$$\mathbf{X}_V = \mathrm{P}(\mathrm{VE}(\mathbf{V})), \qquad C = \mathrm{LLM}(\mathbf{X}_V, \mathbf{X}_I) \tag{1}$$
with $\mathbf{X}_V \in \mathbb{R}^{T N_V \times D}$, $N_V$ spatial tokens per frame. The goal is a sparse memory $S \in \mathbb{R}^{T \times N_S \times D}$ with $N_S \ll N_V$.

**3.1 Attention-based token selection.** By the last generated token, form the concatenation $\mathbf{X}^l=[\mathbf{X}_V^l,\mathbf{X}_I^l,\mathbf{X}_C^l]$ (visual, instruction, and generated-caption tokens, $N_C$ caption tokens). Per layer $l$ and head $h$, standard scaled dot-product attention:
$$\mathbf{Q}^{l,h}=\mathbf{X}^l\mathbf{W}_Q^{l,h},\ \ \mathbf{K}^{l,h}=\mathbf{X}^l\mathbf{W}_K^{l,h},\qquad \mathbf{A}^{l,h}=\mathrm{Softmax}\!\left(\frac{\mathbf{Q}^{l,h}(\mathbf{K}^{l,h})^\top}{\sqrt{d_k}}\right) \tag{2,3}$$
Slice out the cross-attention block from caption tokens onto visual tokens, $\mathbf{A}_V^{l,h}=\mathbf{A}^{l,h}[TN_V{+}N_I : TN_V{+}N_I{+}N_C,\ 0:TN_V]$ (Eq. 4). The global importance of visual token $j$ averages this over caption tokens, heads, and a subset of layers:
$$a_j=\frac{1}{L}\sum_{l=1}^{L}\frac{1}{H}\sum_{h=1}^{H}\left(\frac{1}{N_C}\sum_{i=1}^{N_C}\mathbf{A}_{V\,ij}^{l,h}\right) \tag{5}$$
Select the top $N_S$ tokens (kept in original temporal order):
$$S=\mathbf{X}_V[\pi(1),\pi(2),\dots,\pi(N_S),:],\qquad \pi=\mathrm{argsort}(\mathbf{a})\ \text{(descending)} \tag{6}$$
Note the values $\mathbf{V}^{l,h}$ are never needed — only attention weights — so their projection is skipped.

**3.2 Recurrent processing.** For clip $t{+}1$, prepend the FIFO queue of past selections $[\,S^{(0)},S^{(1)},\dots,S^{(t)}\,]$ to the current raw clip tokens $\mathbf{X}_V^{(t+1)}$ as LLM context, up to context window $W$ or a compute budget; oldest entries pop when the memory is full. This lets past experience shape current attention/captioning across clips.

**3.3 Efficient QA.** Store each generated caption $\{\mathbf{X}_C^{(t)}\}$. At query time, embed query $q$ and select the top-$K$ captions by **Maximal Marginal Relevance (MMR)** — a score balancing query relevance (cosine of caption vs. query) against diversity (cosine of a candidate vs. already-selected captions), reducing redundancy from recurrent caption generation. The retrieved captions $C'$ plus query are fed to the LLM to generate the answer. Empirically, feeding *captions only* beats feeding visual tokens (or both). Algorithm 1 sketch: streaming loop buffers `CLIP_SIZE=16` frames, runs `Attn_Selection`, appends $S$ to a `MAX_MEM=16`-clip FIFO memory $M_s$ and the caption to a log $M_l$; answering does `embed(query) → Retrieve_TopK(Q, M_l) → LLM_Generate_Answer`.

## Explicit design choices
- Training-free, backbone-agnostic: freeze a short-clip Video-LLM; add no learnable modules and no external embedding encoder (caption IDs are stored, not CLIP features).
- Selection signal = the backbone's own caption→visual cross-attention, averaged over caption tokens, heads, and only **4 of 28** layers (compute cap); average aggregation (max pooling noted as a straightforward alternative).
- Aggressive compression: retain **196 of 3,136** visual tokens per clip = **6.25%** (~95% discarded).
- Recurrent FIFO visual memory of `MAX_MEM=16` clips; per clip 32 frames budgeted as **16 current + 16 recurrent-memory** frames.
- Answer from retrieved **captions**, not visual tokens; retrieval by **MMR** (relevance + diversity), `K` captions, within a **10K-token** context window (no external memory offloading).
- Frame rates follow prior work: RVS-Ego/Movie and CG-Bench / offline VS-Stream at 0.5 FPS, MovieChat at 1 FPS.
- Backbones tested: LLaVA-OneVision 7B (default) and 0.5B; also Qwen2.5-VL-7B to show plug-and-play.

## Key results / what to remember
No Zotero highlights present.

- **Streaming (Realtime VStream-QA, Table 3), LLaVA-OV-7B:** rLiVS **65.3** Acc / 4.0 Sco (RVS-Ego) and **57.7** / 3.6 (RVS-Movie) vs ReKV 63.7 / 54.4 — new SOTA, +1.6 / +3.3 Acc; latency **1.9s vs 2.7s** and peak VRAM **25GB vs 36GB** (11GB less, and no per-hour KV-cache growth vs ReKV's 18.8 GB/h).
- **Streaming, LLaVA-OV-0.5B:** rLiVS 57.6 / 51.3 vs ReKV 54.7 / 44.6; 1.5s, 11GB VRAM. The 0.5B rLiVS even beats the 7B ReKV by +2.9 (Ego) / +6.7 (Movie) — caption-based answering scales down well.
- **Streaming, Qwen2.5-VL-7B backbone:** rLiVS **68.1** Acc on RVS-Ego (best reported), 56.1 on RVS-Movie — model-agnostic.
- **Offline long video (Table 1), LLaVA-OV-7B:** VS-Ego **61.0** Acc / 3.9, VS-Movie **59.3** / 3.6, MovieChat **78.0** / 4.0, CG-Bench **33.1** Acc — beats prior best (Flash-VStream-7B 59.0 / 56.1) by +2 / +3 on VS-Ego / VS-Movie.
- **Selection ablation (Table 2, NextQA-valset):** attention selection **77.0%** @6% and **78.4%** @12% vs full model 78.6%; beats uniform (75.5 / 76.7), mean-pool (70.7 / 75.5), K-Means (76.8 / 78.1) — near-full accuracy at 12%, only ~1.5% drop at 6%.
- **Recurrency ablation (Table 4):** removing the FIFO history drops RVS-Ego 65.3→62.5, RVS-Movie 57.7→53.7, MovieChat 78.0→74.1 (≈3–4 pts) — recurrence matters.
- **Modality (Table 5, per text/appendix):** answering from stored captions beats answering from retrieved visual tokens (and beats using both) — retrieval over visual tokens performs near random.

## How it connects (evolution)
- [[rekv]] — the training-free KV-cache streaming-QA method rLiVS directly beats; rLiVS replaces the heavy KV cache with tiny attention-selected tokens + caption retrieval.
- [[flash-vstream]] — trained FIFO/weighted-k-means hierarchical memory baseline for offline long video; rLiVS surpasses it training-free.
- [[videostreaming]] — trainable summarization-token compression (ref [24]); rLiVS is the training-free, attention-selected counterpart.
- [[streamkv]] — KV-cache compression for streaming video, a sibling on the memory-compression axis.
- [[infinipot-v]] — training-free streaming visual-token/KV compression under a fixed budget; same "keep only what matters" spirit.
- [[streaming-memory]] — sub-topic hub (persistent memory for streaming video LLMs).

## Open questions / limitations
- Attention as a relevance oracle is only as good as the backbone's captioning attention; the caption-driven selection risks discarding tokens irrelevant to the caption but relevant to future queries (selection is task/instruction-conditioned, not query-conditioned).
- Answering from captions alone can lose fine visual detail (counting, precise spatial layout) that text summaries omit; the paper shows captions beat visual retrieval on these benchmarks, but that trade-off is dataset-dependent.
- Fixed FIFO memory (`MAX_MEM=16` clips) simply drops the oldest context — no importance-aware eviction, so very long streams can forget salient early events.
- Layer subset (4 of 28) and averaging are chosen for compute, not learned; sensitivity to which layers/heads carry the useful attention is only partially explored (appendix ablations).

*Verification: equations (1)–(6) and Algorithm 1 read from the arXiv PDF (arXiv:2510.17364v1, pp. 4–5); all numbers cross-checked against the paper's own Tables 1–4 (pp. 7–8) and Table 5 / implementation text (pp. 6, 8). Title/authors/affiliations confirmed on the arXiv abs page and PDF title page (NeurIPS 2025).*

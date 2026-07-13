---
zotero_key: null
authors: Shrenik Patel, Daivik Patel (Rutgers University; equal contribution)
year: 2025
arxiv: 2511.13644
pdf: https://arxiv.org/pdf/2511.13644
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# CacheFlow: Compressive Streaming Memory for Efficient Long-Form Video Understanding

**Lineage role:** A training-free compressive streaming memory that pairs online token pruning (temporal-redundancy dropping) with a GRU-summarized, CPU-offloaded KV cache and consensus retrieval — pushing [[rekv]]-style KV retrieval toward much higher compression at fixed accuracy.

## Problem — what was limited before this paper (short)
Long-form video QA breaks current VLMs because attention cost and the key–value (KV) cache both grow with runtime: a long stream either overflows GPU memory or forces expensive full-context inference. The two standard escapes each lose something. Sliding-window / recurrent-attention methods keep only recent tokens, so they are near-sighted — distant context is discarded the moment it exits the window. Retrieval-augmented and persistent-memory systems (e.g. [[rekv]], Flash-VStream) reach back further but accumulate redundant frame-level features and store overlapping block caches, so their memory is bloated and their temporal indexing is coarse. Neither achieves compact memory *and* precise long-range temporal reasoning simultaneously, and most require retraining.

## Key idea — the core insight
CacheFlow exploits two complementary facts. First, natural video is temporally redundant: consecutive frames differ only in small spatial patches, so most per-patch tokens can be dropped online before they ever enter the cache. Second, you do not need the full KV tensor resident to *decide* whether a chunk is relevant — a lightweight, frozen recurrent summarizer can compress each block's keys into a single retrieval vector while the full KV pairs are offloaded to CPU and rehydrated on demand. Combining aggressive online token dropping, fixed-size block packing, GRU-based compressive indexing, and consensus-first retrieval yields long-range reasoning under tight memory/latency budgets — entirely training-free (all learned weights, including the GRU, stay frozen).

![[cacheflow.png]]
> **Crux (Figure 1).** CacheFlow's two-phase pipeline: *Encoding* prunes redundant per-patch tokens by inter-frame similarity, packs survivors into fixed 196-token blocks, offloads each block's full KV to CPU, and summarizes its keys into one GRU vector $g_j$; *Retrieval* scores the question's query vector against those summaries at first and last layers, uses consensus + Top-K=64 to pick blocks, rehydrates only those from CPU, and attends over them for the answer. *Patel & Patel (2025), arXiv:2511.13644. Embedded for personal research reference.*

## Method + math — the mechanism, then the central equations in full

CacheFlow is a plug-in cache manager wrapped around a frozen VLM backbone (LLaVA-OneVision). It has an online **encoding** path (per incoming frame) and a query-time **retrieval** path.

### 1. Dynamic Token Dropping (DTD)
For each patch $p$ at frame $t$ with visual feature $x_{t,p}$, compute cosine similarity to the *same patch position* in the previous frame:
$$s_{t,p} = \cos\!\left(x_{t,p},\, x_{t-1,p}\right) = \frac{x_{t,p}\cdot x_{t-1,p}}{\lVert x_{t,p}\rVert\,\lVert x_{t-1,p}\rVert}.$$
Keep a patch only if it has changed enough — i.e. its similarity is *below* a threshold $\tau_{\text{feat}}$:
$$m_{t,p} = \mathbb{1}\!\left[s_{t,p} < \tau_{\text{feat}}\right], \qquad X'_t = \{x_{t,p} : m_{t,p}=1\}.$$
Two guards keep the stream well-formed: the **first frame** ($t=1$) is always kept in full (no prior to compare against), and if a frame masks out *all* its tokens, the single least-similar token is force-kept so no frame vanishes entirely. Redundant static background is thus never written to the cache.

### 2. Dynamic token-to-block packing
The pruned, variable-length survivor sets $X'_t$ from successive frames are concatenated into one temporal token stream and re-chunked into **fixed-size blocks of $B=196$ tokens**. Because a heavily-static frame contributes few tokens and a dynamic frame contributes many, a block may span several frames or a fraction of one — the uniform block size is what makes eviction and retrieval bookkeeping stable and lets the system run in a streaming (bounded-state) fashion.

### 3. GRU-based compressive memory (the index)
When a completed block $B_j$ is offloaded to CPU, its full Key tensor at layer $\ell$, $K^\ell_j$, is reshaped into a sequence of $B$ key vectors $k'_{j,1},\dots,k'_{j,B}$ and fed through a **single-layer, frozen GRU** that acts as a deterministic recurrent summarizer:
$$
\begin{aligned}
r_i &= \sigma\!\left(W_r k'_{j,i} + U_r h_{i-1} + b_r\right),\\
z_i &= \sigma\!\left(W_z k'_{j,i} + U_z h_{i-1} + b_z\right),\\
\tilde h_i &= \tanh\!\left(W_h k'_{j,i} + U_h\,(r_i \odot h_{i-1}) + b_h\right),\\
h_i &= (1-z_i)\odot h_{i-1} + z_i\odot \tilde h_i.
\end{aligned}
$$
The final hidden state $g^\ell_j = h_B \in \mathbb{R}^{D_g}$ is the block's compressed **retrieval key**. Only these small summary vectors stay resident (plus the recent local window); the full $(K,V)$ tensors live on CPU. The GRU weights are never trained — it is used purely as a fixed summarizer, which is why the whole method is training-free.

### 4. Consensus-first retrieval
At query time the question is tokenized into a query vector $q^\ell$ per layer, and cosine similarity is computed against every stored block summary:
$$s^\ell_j = \frac{q^\ell \cdot g^\ell_j}{\lVert q^\ell\rVert\,\lVert g^\ell_j\rVert}.$$
Rather than trust a single layer, CacheFlow ranks blocks at the **first layer** ($R_0$, ordered by $s^0_j$) and the **last layer** ($R_L$, ordered by $s^L_j$) and merges them consensus-first to fill $K$ slots:
1. **Consensus** — take blocks in $R_0 \cap R_L$ first.
2. **Last-layer fills** — add highest-scoring blocks from $R_L \setminus R_0$.
3. **First-layer fills** — if slots remain, add from $R_0 \setminus R_L$.

The Top-$K$ ($K=64$ blocks) selected this way are the only ones rehydrated.

### 5. Position-aware rehydration and attention
The selected blocks' full KV tensors are pulled back from CPU and **prepended to the local window**. Query tokens are assigned absolute positions $[\,n_{\text{mem}}+1,\dots,n_{\text{mem}}+n_q\,]$ so that RoPE rotational phases stay continuous across the rehydrated memory and the fresh tokens (an offset, not re-encoding). The answer is then produced by ordinary attention over the assembled cache:
$$\mathrm{Attn}(Q^\ell, K^\ell_{\text{cache}}, V^\ell_{\text{cache}}) = \mathrm{softmax}\!\left(\frac{Q^\ell (K^\ell_{\text{cache}})^{\top}}{\sqrt{d_k}}\right) V^\ell_{\text{cache}}.$$
Because only local window + $KB$ tokens are ever attended, cost drops from $O(T^2)$ in total stream length $T$ to $O\!\left((n_{\text{local}} + KB)^2\right)$, independent of how long the video runs.

## Explicit design choices
- **Training-free / plug-in.** Backbone (LLaVA-OneVision 0.5B and 7B) is frozen; even the GRU summarizer weights are frozen — no fine-tuning of any component.
- **Redundancy signal = per-patch cosine to previous frame**, thresholded at $\tau_{\text{feat}}=0.5$ (the robust operating point giving ~80–87% token drop with minimal accuracy loss).
- **Fixed block size $B=196$ tokens** — decouples cache/retrieval bookkeeping from variable per-frame survivor counts.
- **Frozen single-layer GRU as the compressor**: summarizes a whole block's keys into one $\mathbb{R}^{D_g}$ vector; chosen over mean-pooling because recurrence preserves order/temporal structure.
- **Full KV offloaded to CPU**, only summary vectors + local window kept on GPU; blocks rehydrated on demand.
- **Consensus retrieval across first + last transformer layers**, not a single layer — combines low-level and semantic matching.
- **Top-$K=64$ blocks** rehydrated per query.
- **RoPE position offset** on rehydrated blocks + queries to keep rotary phases continuous (no positional discontinuity between memory and new tokens).
- **First frame always fully kept; force-keep least-similar token** if a frame would otherwise be fully pruned.

## Key results / what to remember
All numbers are CacheFlow with a frozen LLaVA-OneVision (LLaVA-OV) backbone; "drop" = token-drop %. Verified against the paper's Tables 1–5 (as reported via the arXiv HTML; a few could not be independently re-derived and are marked).

**Offline long-video QA (Table 1).**
- *LLaVA-OV-0.5B:* QAEgo4D 50.8% (81.8% drop) vs ReKV 48.2%; MLVU 54.6% (79.7% drop) vs ReKV 54.2%; EgoSchema 27.9% (82.6% drop) vs ReKV 25.6%; ActivityNet-QA 50.6% / 3.47 score (80.8% drop) vs ReKV 49.9% / 3.01.
- *LLaVA-OV-7B:* QAEgo4D 55.4% (70.9% drop) vs ReKV 54.8%; MLVU 66.9% (67.1% drop) vs ReKV 66.5%; EgoSchema 60.2% (72.0% drop) vs ReKV 60.7%; ActivityNet-QA 59.6% / 3.64 (70.9% drop) vs ReKV 60.2% / 3.52.
- Context baselines: GPT-4V 57.0% and Gemini 2.5 Pro 66.7% on ActivityNet-QA; LongVA-7B 56.3% on MLVU; Flash V-Stream-7B 38.1% on EgoSchema.

**Streaming video QA — RVS benchmarks (Table 2).**
- *0.5B:* RVS-Ego 54.3% / 3.87 (86.8% drop, 1.38 s latency, 2.9 GB GPU) vs ReKV 51.9% / 3.80 (1.60 s, 19 GB); RVS-Movie 42.6% / 3.34 (70.7% drop).
- *7B:* RVS-Ego 61.6% / 3.94 (78.5% drop, 2.03 s latency, 26.4 GB GPU) vs ReKV 59.7% / 3.90 (3.30 s, 38 GB); RVS-Movie 50.5% / 3.44 (58.0% drop); vs Flash V-Stream-7B 56.3% (RVS-Ego) / 53.3% (RVS-Movie).
- Headline efficiency: matches/beats ReKV while dropping up to **~87% of tokens**, cutting GPU memory (e.g. 19 GB → 2.9 GB at 0.5B) and latency.

**Ablations.**
- *DTD threshold (Table 3, MLVU / RVS-Ego):* $\tau=0.25$ → 52.1% / 52.8% at 98.7% / 99.3% drop (too aggressive); $\tau=0.50$ → 54.6% / 54.3% at 79.7% / 86.8% drop (chosen); $\tau=0.75$ → 53.4% / 55.4% at only 35.9% / 45.8% drop.
- *GRU vs mean pooling (Table 4):* GRU beats mean-pool — QAEgo4D 48.2% vs 47.6%; RVS-Ego 52.9% / 3.85 vs 51.9% / 3.80.
- *Consensus retrieval (Table 5):* first+last-layer consensus > last-layer-only — QAEgo4D 50.8% vs 50.2%; EgoSchema 27.9% vs 27.6%.

No Zotero highlights present.

Takeaways: (1) Most of the compute win comes from *online* token dropping (temporal redundancy), not from smarter retrieval — but retrieval + GRU indexing is what keeps accuracy from collapsing at 80%+ drop. (2) $\tau_{\text{feat}}\approx0.5$ is the knee: below it accuracy drops with the tokens; above it you stop compressing. (3) The whole thing is training-free and backbone-agnostic, so it reads as a drop-in efficiency layer over any KV-cached VLM, and it beats [[rekv]] on both accuracy and memory/latency simultaneously.

## How it connects (evolution)
- [[rekv]] — the direct predecessor and primary baseline: retrieval over stored KV caches for streaming video; CacheFlow adds online token dropping + GRU compression to cut its cache redundancy.
- [[flash-vstream]] — hierarchical flash memory (synopsis + detail) for real-time streaming; a contrasting memory design CacheFlow compares against (throughput-first vs compression-first).
- [[infinipot-v]] — training-free KV-cache compression for long video on VLMs; sibling in the "compress the KV cache without retraining" line.
- [[streamingvlm]] — streaming-oriented VLM cache/attention management; adjacent efficiency approach for unbounded streams.
- [[hermes-kv]] — KV-cache-centric memory for streaming video; same design axis (what to keep/evict in the cache).
- [[streamkv]] — streaming KV retrieval memory; closely related retrieval-over-KV formulation.

## Open questions / limitations
- **Per-patch, position-aligned cosine dropping** assumes a roughly static camera / aligned patch grid; under strong camera motion or scene cuts, "changed" patches proliferate and DTD's compression should degrade toward zero (as the $\tau=0.75$ row already hints).
- **The GRU is frozen and never trained** — it is a fixed summarizer, so its retrieval keys are only as good as an untrained recurrence over raw key vectors; a learned compressor might index far better, at the cost of the training-free property.
- **Consensus uses only first + last layers**; whether intermediate-layer agreement or a learned gating would help is untested, and the gains over last-layer-only are small (≤0.6 pts in the reported ablation).
- **CPU↔GPU rehydration cost** is folded into the reported latency but its scaling with a very large offloaded store (millions of blocks over hours-long video) is not stress-tested; retrieval quality at extreme memory sizes is unverified.

*Verification: equations (DTD mask, GRU gates, cosine retrieval, offset RoPE attention, O((n_local+KB)^2) complexity) and all numbers cross-checked against the arXiv HTML rendering of the paper's Tables 1–5 and Section 3 (arXiv:2511.13644v1); Figure 1 cropped directly from the arXiv PDF. Table values were not independently re-computed, so any single-cell transcription is only as reliable as the source table.*

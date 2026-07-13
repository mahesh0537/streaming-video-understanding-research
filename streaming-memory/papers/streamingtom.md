---
zotero_key: null
authors: Xueyi Chen, Keda Tao, Kele Shao, Huan Wang et al. (Westlake University; CUHK; Zhejiang University; SII)
year: 2025
arxiv: 2510.18269
pdf: https://arxiv.org/pdf/2510.18269
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# StreamingTOM: Streaming Token Compression for Efficient Video Understanding

**Lineage role:** A training-free, two-stage token-compression scheme for streaming video LLMs that attacks *both* pre-LLM prefill cost (via causal temporal token reduction) and post-LLM KV-cache memory (via online 4-bit quantized memory with query-guided retrieval) — pushing the [[streammem]] / [[rekv]] / [[livevlm]] / [[infinipot-v]] line of streaming-memory work upstream of the LLM as well.

## Problem — what was limited before this paper (short)
Streaming video VLMs must run under strict **causality** (no future frames; each frame finalized in a single pass) and suffer from unbounded **token accumulation** over time. This creates a dual bottleneck: (1) **prefill compute** grows as $O(tNLd^2)$ because every accumulated visual token traverses the full transformer stack, and (2) the **KV-cache memory** grows linearly with stream length (~18.8 GB for one hour of video on LLaVA-OV-7B, exceeding typical GPU capacity). Prior training-free methods (KV eviction / retrieval like [[rekv]], [[streammem]]) only regulate the *post-LLM* KV-cache; they leave the costly *pre-LLM* prefill untouched, so per-frame compute stays $O(NLd^2)$.

## Key idea — the core insight, 2-4 sentences
The two bottlenecks demand a *coordinated* solution with a fixed order: reduce tokens **before** the LLM (compute) and quantize the cache **after** the LLM (memory) — neither alone suffices, since pre-LLM compression cannot stop cache growth and post-LLM compression cannot reclaim compute already spent. StreamingTOM composes **Causal Temporal Reduction (CTR)**, which condenses each frame's $N$ tokens to a fixed quota of $G$ tokens using a 2-frame causal window, with **Online Quantized Memory (OQM)**, which stores those $G$-token groups at 4-bit precision and, on a query, retrieves and dequantizes only the top-$k$ relevant groups. A fixed-size **group** (one group = one frame = $G$ tokens) is the interface decoupling reduction from storage and keeping per-frame latency predictable.

![[streamingtom.png]]
> **Crux (Figure 2).** The two coordinated pipelines: the *vision pipeline* encodes each frame, applies CTR to squeeze $N$ tokens into a compact $G$-token group, prefills it, and writes it to memory; the *query pipeline* uses the question embedding to retrieve the top-$k$ groups from a 4-bit quantized memory, dequantizes only those, and joint-prefills them with the question for decoding. *Chen, Tao, Shao, Wang (2025), arXiv:2510.18269. Embedded for personal research reference.*

## Method + math — the mechanism, then the objective in full

### Setup and the compression objective
A stream $\mathcal{V}=\{v_1,v_2,\dots\}$ is processed causally: at time $t$ only $\mathcal{V}_{\le t}$ is available and each frame is finalized immediately (no retrospective reprocessing). Each frame is encoded once into visual tokens $\mathbf{H}_t = \mathcal{E}_v(v_t)\in\mathbb{R}^{N\times d}$ and, across $L$ layers, mapped to key/value pairs; the cache accumulates by concatenation
$$\mathcal{K}_t^{(l)}=[\mathcal{K}_{t-1}^{(l)};K_t^{(l)}],\qquad \mathcal{V}_t^{(l)}=[\mathcal{V}_{t-1}^{(l)};V_t^{(l)}].$$
Per-frame compute stays $O(NLd^2)$ and the cache grows without bound:
$$\frac{d\,\text{memory}}{dt}=2\cdot L\cdot N\cdot d\cdot \text{sizeof(dtype)}\cdot \text{fps}.$$
Under LLaVA-OV-7B settings ($L{=}28$, $N{=}196$, $H_{kv}{=}4$, $d_h{=}128$, FP16, 0.5 fps → 1800 frames/hour) this integrates to $\approx 18.8$ GB/hour. The compression objective, with query distribution $p$ and memory budget $M$, is
$$\min_{(\hat{\mathcal{K}},\hat{\mathcal{V}})\in\mathcal{F}_M}\ \mathbb{E}_{q\sim p}\,\mathcal{L}\big(q,\hat{\mathcal{K}},\hat{\mathcal{V}}\big),\qquad \mathcal{F}_M=\{(\hat{\mathcal{K}},\hat{\mathcal{V}}):|\hat{\mathcal{K}}|+|\hat{\mathcal{V}}|\le M\},$$
i.e. compress query-agnostically and causally while preserving task accuracy. The full framework is the fixed composition
$$\text{StreamingTOM}=\text{OQM}_{16\to4}\circ \text{CTR}_{N\to G}.$$

### Stage 1 — Causal Temporal Reduction (CTR)
Goal: cut prefill compute from $O(TNLd^2)$ to $O(TGLd^2)$ by emitting a **fixed budget** of $G$ tokens per frame. Using only frames $t{-}1$ and $t$ (a 2-frame causal window), token-wise temporal similarity is the cosine
$$s_t^{(i)}=\frac{F_t^{(i)}\cdot F_{t-1}^{(i)}}{\lVert F_t^{(i)}\rVert\,\lVert F_{t-1}^{(i)}\rVert},$$
and tokens are partitioned by a threshold $\tau_c$ (default $0.9$) into **static** $S_t$ (high similarity) and **dynamic** $D_t$ (low similarity). Spatial saliency $\alpha_t^{(i)}$ comes from the vision encoder. The budget $G$ is split adaptively between the two paths:
$$k_s=\Big\lfloor G\cdot\frac{|S_t|}{|S_t|+|D_t|}\Big\rfloor,\qquad k_d=G-k_s.$$
Dual-path processing: **dynamic** tokens keep the top-$k_d$ by saliency; **static** tokens are merged (density-based clustering) down to $k_s$ representatives. Output is exactly $G$ tokens/frame. State memory is $O(Nd)$ (independent of stream length) and per-frame cost is $O(N+G^2)$.

### Stage 2 — Online Quantized Memory (OQM)
Each $G$-token group's KV is stored with **per-group, per-channel 4-bit** quantization:
$$Q_4(X_t)=\big(\text{uint4}(\hat X_t),\, s_t,\, m_t,\, \bar k_t\big),\quad s_t=\frac{\max(X_t)-\min(X_t)}{15},\quad m_t=\min(X_t),$$
where $\bar k_t$ is a **representative key** (mean of the group's keys, kept in full precision for retrieval). Dequantization is $Q_4^{-1}(\text{uint4}(\hat X);s,m)=\text{depack}(\hat X)\odot s+m$. On a query, only the most relevant groups are pulled back to FP16:
$$R=\text{TopK}\{\,\text{sim}(q,\bar k_i)\,\}_{i=1}^{t},\qquad \text{KV}_{\text{active}}=\bigcup_{i\in R} Q_4^{-1}(G_i),$$
with a layer-wise group count $k=\min(\lceil B/G\rceil,\ T^{(l)}/G)$ ($B$ = per-layer token budget). Complete history is preserved at $O(T\,G\,d)/4$ storage, while only $O(k\,G\,d)$ ($k\ll T$) is active during decoding.

### Combined ratio
Storage FP16/4-bit $\approx (N/G)\cdot(16/4)=4N/G$; with $N{=}196,\ G{=}50$ this is $\approx 15.7\times$. For the one-hour stream, KV-cache drops from 18.8 GB to ~1.2 GB.

## Explicit design choices
- **Fixed order, two stages:** CTR (pre-LLM, compute) then OQM (post-LLM, memory) — the paper argues neither ordering is interchangeable nor either stage sufficient alone.
- **Fixed per-frame token quota $G=50$** (out of $N=196$) → predictable, bounded prefill latency per frame regardless of scene complexity.
- **2-frame causal window** for temporal redundancy (only $t{-}1$, $t$) — keeps state memory $O(Nd)$ and honors strict causality (no future frames).
- **Static/dynamic split at $\tau_c=0.9$**, adaptive $k_s/k_d$ budget allocation, saliency-based top-k for dynamic + density clustering (merge) for static.
- **Group abstraction = one frame = $G$ tokens** as the interface between CTR and OQM; frame-aligned so retrieved content stays temporally coherent (whole frames).
- **Per-group, per-channel 4-bit (uint4) quantization** with scale/offset; representative key kept FP16 for retrieval fidelity.
- **Question-guided top-$k$ retrieval + selective dequantization** — only retrieved groups reconstructed to FP16 (active KV), rest stay compressed.
- **OQM global budget 12k tokens**; sampling 0.5 fps for clips <30 min, 0.2 fps for longer.
- **Training-free**, drop-in on frozen VLMs (LLaVA-OV-7B; also Qwen2.5-VL-7B). Greedy decoding; single A6000 (48 GB), FP16.

## Key results / what to remember
No Zotero highlights present.

- **Offline VideoMME (LLaVA-OV-7B, 0.5/0.2 fps), Overall = 59.9** (Short 50.6 / Medium 57.8 / Long 71.3), beating the 32-frame offline reference (58.4) and training-free peers +StreamMem (59.4) and +LiveVLM (57.3). MLVU **67.9**, EgoSchema **63.7**, three-benchmark Avg **63.8** vs StreamMem 63.1. (Table 1.)
- **Online RVS streaming:** RVS-Ego Acc **58.3** / Score 3.9, RVS-Movie Acc **53.2** / Score 3.5, Avg Acc **55.8** / Score 3.7 — best average among training-free streaming methods (StreamMem 55.2, InfiniPot-V 54.6, Flash-VStream 55.0). (Table 2.)
- **Efficiency (batch 8):** one-hour KV-cache **1.2 GB** vs 18.8 GB baseline; at 256 frames ~1.2× lower peak memory and ~2× faster TTFT vs LiveVLM. GPU memory stays ~16–18 GB from 16→512 frames; TTFT 0.17→0.30 s; throughput ~20–37 tok/s. (Table 3 / Fig. 6.)
- **Ablation (VideoMME Overall):** default 50 tokens + 4-bit (6.4% retention) = **59.9**, best; 40+4-bit = 58.9; 60+4-bit = 59.3; going to 2-bit drops all (~58.1–58.7). (Table 4.)
- **Generalization to Qwen2.5-VL-7B:** at 0.5 fps VideoMME Avg **63.0** = 102.8% retention vs the 32-frame baseline (61.3); at matched 32 frames 58.9 (96.1%). LongVideoBench: **56.3** at 0.5 fps vs 56.4 baseline. (Tables 5–6.)
- Takeaway: attacking prefill *and* KV-cache together lets a training-free streaming method match or slightly beat offline references and post-LLM-only compressors while cutting memory ~15× and TTFT ~2×.

## How it connects (evolution)
- [[streammem]] — the closest training-free streaming-memory baseline it beats; StreamingTOM adds the pre-LLM CTR stage StreamMem lacks.
- [[rekv]] — retrieval-based KV-cache streaming memory; StreamingTOM's OQM is a quantized, group-retrieval successor idea.
- [[livevlm]] — training-free streaming VLM it compares TTFT/memory against (2× faster, 1.2× lower peak).
- [[infinipot-v]] — bounded-memory KV compression for streaming video, a direct efficiency peer in Table 2.
- [[flash-vstream]] — memory-based online streaming baseline on the RVS benchmarks.
- [[streamkv]] — sibling KV-compression-for-streaming approach; same design axis (what to keep in the cache).

## Open questions / limitations
- 2-bit quantization noticeably hurts accuracy (Table 4), so the memory ratio is effectively floored at 4-bit — long-horizon streams still accumulate ~1.2 GB/hour of compressed cache.
- CTR's fixed $G=50$ quota is uniform per frame; scenes with dense simultaneous motion may exceed the dynamic budget, and the 2-frame window can miss slow drifts spanning many frames.
- Retrieval fidelity hinges on a single mean "representative key" per group; groups whose relevant content is a minority of tokens may be mis-scored and dropped.
- Reported gains are on 7B VLMs (LLaVA-OV, Qwen2.5-VL) and MC/open-ended VideoQA benchmarks; robustness on dense proactive/duplex streaming settings (e.g. [[proactivevideoqa]], [[omnimmi]]) is untested here.

*Verification: equations (Eqs. 1–2, CTR similarity/budget, OQM quantization/retrieval, $4N/G\approx15.7\times$) and all numbers (Tables 1–6, one-hour 18.8 GB→1.2 GB, TTFT/throughput) cross-checked against the arXiv:2510.18269v2 HTML full text and the rendered PDF (Figure 2, captions).*

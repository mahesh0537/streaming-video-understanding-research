---
zotero_key: null
authors: Haowei Zhang, Shudong Yang, Jinlan Fu et al. (Fudan University / OpenMOSS, Shanghai Innovation Institute, NUS)
year: 2026
arxiv: 2601.14724
pdf: https://arxiv.org/pdf/2601.14724
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---

# HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding

**Lineage role:** Reframes the decoder's own KV cache as a layer-stratified hierarchical memory (sensory / working / long-term), giving a training-free, retrieval-free streaming memory manager where the cache *is* the memory — no external store, no query-time recomputation.

## Problem — what was limited before this paper (short)
Streaming video MLLMs must simultaneously deliver (1) stable long-video understanding, (2) real-time low-latency responses, and (3) bounded GPU memory — and existing methods trade one for another. Offline compression methods assume the whole video (and query) is known, so they fail on unpredictable streaming inputs. Prior streaming memory approaches split into *external memory* (store captions / raw patches in a CPU/disk database, then do ad-hoc retrieval + multimodal prefill at query time — high latency, no end-to-end cohesion, often needs training) and *internal memory* in the KV cache. The KV-cache-based training-free methods that do exist ([[rekv]], [[livevlm]]) still offload prior segments to CPU/disk and run an extra retrieval when a query arrives, incurring latency; [[streammem]] uses chat-template tokens to guide compression but manages the cache coarsely (uniform, FIFO-like eviction across all layers) with no mechanistic grounding.

## Key idea — the core insight, 2-4 sentences
A mechanistic study of layer-wise attention in a video MLLM (LLaVA-OV-7B) shows decoder layers specialize by memory granularity: **shallow layers** have intense recency bias (sensory memory), **deep layers** attend sparsely to rhythmic per-frame "anchor" tokens spaced exactly N=196 apart — the frame-level long-term memory — and **middle layers** interpolate the two (working memory). HERMES therefore evicts KV tokens with a *layer-dependent* importance score matched to each layer's role, smooths the retained set across layers, re-indexes positions, and reuses the resulting compact cache directly. Because the compact cache already encodes the video when a user query arrives, HERMES needs **no auxiliary computation or external retrieval at query time**, guaranteeing real-time response.

![[hermes-kv.png]]
> **Crux (Figure 3).** The HERMES pipeline: video chunks are encoded and prefilled into a KV cache treated as hierarchical memory — deep layers keep anchor tokens via Attention+Aggregation, middle layers via Attention+Recency, shallow layers via Recency Bias — with Cross-Layer Smoothing and Position Re-Indexing, so a user question is answered by direct cache reuse ("No extra retrieval", real-time response). *Zhang, Yang, Fu et al. (2026), arXiv:2601.14724. Embedded for personal research reference.*

## Method + math — the mechanism, then the central objective/equations IN FULL

**Setup / sliding window.** The video stream is encoded chunk-by-chunk (16 frames per chunk in the main experiments; 8 in the attention study) and prefilled into the backbone LLM. Each KV cache layer holds a fixed budget of $|M|$ video tokens; when the cache exceeds $|M|$, token eviction (compression) is triggered. Let $i$ be a video token's physical KV cache index at layer $l$, and $T$ the current total number of video tokens in the cache. HERMES computes a per-token importance score $S^l_i$ that governs retention, and the *form of that score is different for each layer band*.

**(1) Shallow layers — sensory memory (recency / forgetting curve).** Modeled on Ebbinghaus exponential forgetting over temporal distance:
$$
S^l_i = \alpha^l_i \cdot e^{-k\,\Delta t_i}, \qquad \Delta t_i = T - 1 - i,
$$
where $k > 0$ is the forgetting rate and $\alpha^l_i$ is a normalization factor. Recent tokens (large $i$, small $\Delta t_i$) score highest.

**(2) Deep layers — long-term memory (attention to anchor tokens).** Deep-layer attention is sparse and stable, and anchor tokens consistently draw high attention, so attention magnitude is a reliable long-term importance signal. Since the real query is unknown mid-stream, a fixed **generic guidance prompt** acts as a pseudo-query to elicit attention weights:
$$
S^l_i = \alpha^l_i \cdot W^l_i,
$$
where $W^l_i$ is the attention weight the $i$-th token receives at layer $l$.

**(3) Middle layers — working memory (interpolate recency ↔ attention).** A layer-dependent mixing weight decays linearly across the middle band:
$$
\omega_l = \omega_0 - \gamma \cdot \frac{l - l_{\text{short}}}{l_{\text{long}} - l_{\text{short}}}, \qquad \omega_0 = 0.75,\ \gamma = 0.6,
$$
with $l_{\text{short}}, l_{\text{long}}$ the band-boundary layer indices. The score blends normalized attention $A^l_i$ and recency $R^l_i$ (from Eqs. 1–2):
$$
S^l_i = (1 - \omega_l)\, A^l_i + \omega_l\, R^l_i.
$$
So shallow-most middle layers are recency-dominated ($\omega \to 0.75$) and deep-most middle layers are attention-dominated.

**Cross-Layer Memory Smoothing.** Evicting each layer independently at the same cache index causes cross-layer inconsistency (misaligned visual memory), which hurts because effective LLM memory relies on cross-layer interaction. Importance signals are propagated from deeper to shallower layers:
$$
\tilde S^l_i = (1 - \lambda_l)\, S^l_i + \lambda_l\, S^{l+1}_i, \qquad \lambda \in [0,1],
$$
where $\lambda_l$ is a (layer-dependent) smoothing strength. Retention then applies Top-K on the smoothed scores to hold the budget $|M|$ per layer:
$$
\mathcal{I}^l = \mathrm{TopK}(\tilde S^l, |M|), \quad K^l = K^l[\mathcal{I}^l], \quad V^l = V^l[\mathcal{I}^l].
$$
To avoid losing long-horizon content, the evicted tokens are **aggregated into a per-layer summary token** that compactly encodes long-term memory and is kept in the cache (Summary Token Aggregation, Algorithm 1 / App. F).

**Position Re-Indexing.** Continuous streaming pushes positional indices past the model's max range, degrading generation, so retained tokens are remapped to a contiguous range $[0, |M|)$. Two variants:
- **Lazy re-indexing** — remap only when indices approach the model limit; preserves recent tokens' original indices, low overhead, less positional drift → preferred for streaming.
- **Eager re-indexing** — remap at every compression step, strictly contiguous RoPE indices; stabilizes long-range semantics but costs more → better for offline.
Implementations are given for 1D RoPE (LLaVA-OV) and 3D M-RoPE (Qwen2.5-VL) (App. E.1 / E.2).

**At query time** the compact hierarchical cache is reused directly for multimodal prefilling + decoding — no retrieval, no re-encoding — which is the source of the latency win.

## Explicit design choices — concrete decisions
- **Training-free, plug-and-play**: no fine-tuning; drops into existing MLLMs (validated on LLaVA-OV 0.5B/7B, Qwen2.5-VL 7B/32B, Qwen3-VL 4B/8B).
- **Layer partition** from the attention study: **10% shallow / 60% middle / 30% deep** layers.
- **Per-layer fixed budget $|M|$**; main results at **4K** and **6K** video tokens (vs 64-frame uniform sampling baselines) — the 4K setting is up to ~68% fewer tokens.
- **Generic guidance pseudo-query** to compute deep/middle-layer attention importance without knowing the real user query (ablated: generic prompt vs alternatives).
- **Summary token per layer** to preserve evicted long-term info instead of hard-dropping it.
- **Lazy re-indexing for streaming, eager for offline**; RoPE-aware remap for both 1D and 3D M-RoPE backbones.
- **Layer-dependent smoothing $\lambda_l$** (configs in App. C) rather than a single global value.
- **Chunk-wise streaming prefill**: 16 frames/chunk at inference; compression fires when budget exceeded.
- Hyperparameters: $\omega_0 = 0.75$, $\gamma = 0.6$; FP16, greedy decoding; efficiency on a single A800 (80 GB).

## Key results / what to remember
Exact headline numbers (verified against the paper's tables; StreamingBench/OVO-Bench "Avg." = mean of real-time visual perception + backward-tracing accuracy):

- **StreamingBench / OVO-Bench (Table 1)** — Qwen2.5-VL-7B + HERMES (4K tokens): **79.44%** StreamingBench, **59.21%** OVO-Bench Avg. — **+6.13** and **+6.93** over the Qwen2.5-VL-7B base (73.31 / 52.28), beating all 7B open-source online+offline models.
- LLaVA-OV-7B + HERMES (4K): **73.23%** StreamingBench, **58.27%** OVO-Bench Avg. — up from base 71.34 / 53.35, and above +ReKV (69.22 / 50.75) and +LiveVLM (72.92 StreamingBench).
- Qwen2.5-VL-32B + HERMES (6K): **80.20%** StreamingBench, **64.82%** OVO-Bench Avg. (base 74.27 / 57.37). Qwen3-VL-8B + HERMES: up to **81.32%** StreamingBench.
- **RVS-Ego / RVS-Movie (Table 2, GPT-3.5-turbo-0125 judge, acc + 1–5 score)** — LLaVA-OV-7B + HERMES (6K): RVS-Ego **60.3** acc / 4.0 score, RVS-Movie **54.4** / 3.6, matching the ReKV upper bound (which caches all frames) on RVS-Movie and beating other training-free methods (StreamMem 57.6, StreamingTOM 58.3 on RVS-Ego). Paper claims up to **+11.4%** accuracy over the base model on streaming (open-ended) tasks.
- **Offline (Table 4)** — LLaVA-OV-7B + HERMES (4K): Egoschema **60.29%**, VideoMME **58.85%** (both above base 59.93 / 57.67), MVBench **56.92%** (≈ base 57.02%) — competitive with far fewer tokens.
- **Efficiency (Fig. 4 / Table 3)** — **10× faster TTFT** than prior SOTA StreamingTOM (256 frames), and 1.04× lower peak memory than LiveVLM. TTFT stays **< 30 ms** and GPU memory **constant (~16.5–17.7 GB)** from 16→512 frames (Table 3) — no OOM, no growth with video length.

No Zotero highlights present.

Takeaways: (i) the KV cache can serve as the *entire* streaming memory if you respect that different depths store different granularities — no external DB and no query-time retrieval needed; (ii) matching the eviction rule to each layer's attention role (recency-decay shallow, attention-driven deep, interpolated middle) plus cross-layer smoothing is what lets an aggressive ~68% token cut stay accurate; (iii) removing query-time computation is the lever for the 10× TTFT and flat-memory story.

## How it connects (evolution)
- [[rekv]] — the KV-cache retrieval baseline HERMES most directly displaces: ReKV caches all prior frames (an upper bound) and retrieves at query time; HERMES gets comparable/better accuracy with a bounded, retrieval-free cache.
- [[livevlm]] — cache-based streaming method HERMES compares memory/latency against; HERMES removes its external offload + query-time cost.
- [[streammem]] — closest prior training-free KV-compression idea (template-token-guided), which HERMES critiques as coarse and non-mechanistic, then improves with layer-stratified management.
- [[infinipot-v]] — another training-free streaming KV/token-budget compression method compared on RVS.
- [[timechat-online]] — motivates the redundancy premise (many streaming video tokens are redundant) that HERMES exploits via compression.
- [[flash-vstream]] — memory-based streaming baseline in the RVS comparison; part of the same lineage of compact streaming video memory.

## Open questions / limitations
- The layer-band structure (10/60/30) and the anchor-token spacing (N=196) are derived on specific backbones (LLaVA-OV, Qwen-VL); it's unclear how the partition transfers to architectures with different visual-token-per-frame counts or attention geometries without re-profiling.
- The **generic guidance pseudo-query** stands in for the real user query when scoring deep/middle importance — genuinely novel or adversarial queries about long-evicted content could still be under-served, since eviction happened before the query was known.
- Long-term information is compressed into a single **summary token per layer**; the fidelity ceiling of that aggregation for very long videos / fine-grained recall is not fully characterized.
- Several hyperparameters ($k$, $\lambda_l$, $\omega_0$, $\gamma$, budget $|M|$) are tuned; robustness of the fixed choices across domains and fps regimes beyond the tested benchmarks is only partially ablated.

*Verification: equations (1)–(6), the 10/60/30 partition, $\omega_0=0.75$, $\gamma=0.6$, and all headline numbers checked against the paper's own text and Tables 1–4 / Table 3 / Fig. 4 in the arXiv PDF (2601.14724v4, pages 4–9); homepage hermes-streaming.github.io and repo github.com/haowei-freesky/HERMES noted from page 1 but not separately fetched.*

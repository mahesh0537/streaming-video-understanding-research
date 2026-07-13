---
zotero_key: null
authors: Zhanzhong Pang, Dibyadip Chatterjee, Fadime Sener, Angela Yao (NUS / Meta Reality Labs)
year: 2026
arxiv: 2605.01858
pdf: https://arxiv.org/pdf/2605.01858
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# Decouple and Cache: KV Cache Construction for Streaming Video Understanding

**Lineage role:** Decouples the streaming KV cache into a *cumulative past* store (built incrementally, condition-degraded) and an *instant* cache (rebuilt on-demand from a small recent buffer), so recency-biased offline VLMs stay accurate on unbounded streams — training-free, via position-agnostic caching.

## Problem — what was limited before this paper (short)
Streaming video understanding needs to answer queries over an unbounded frame stream under fixed memory/compute, and to do it with VLMs that were pretrained on *short* clips. Two coupled failures: (1) the KV cache must be continuously grown and evicted for a stream that never ends; (2) the model must generalize from the short training context to arbitrarily long inference contexts. The prevailing fix — a single "uniform" KV cache that appends every frame's KV and evicts FIFO — has a subtle flaw the authors name the **cumulative effect**: because pretrained LLMs are strongly recency-biased, when a *recent* frame is encoded, its KV is contaminated by conditioning on stale residual context from earlier (already-evicted-in-spirit) frames, degrading the encoding of exactly the tokens that matter most for a just-arrived query.

## Key idea — the core insight, 2-4 sentences
Separate the two jobs a streaming cache does. Keep a **cumulative past cache** $\mathcal{U}$ that is built incrementally and cheaply (each evicted frame is encoded once, conditioned on the accumulated history) to preserve long-range context. But **do not** trust it for the most recent frames: when a query arrives, throw away the recent portion of the cumulative cache and rebuild an **instant cache** $\mathcal{I}$ from a small raw feature buffer $\mathcal{B}$, so recent frames are encoded cleanly (short, in-distribution context). To make this partition/recombine legal, store all KV *before* positional encoding (**position-agnostic encoding**) and re-inject positions only at use time.

![[decouple-and-cache.png]]
> **Crux (Figure 2).** The two operating modes: *Streaming Update (no query)* keeps a FIFO feature buffer $\mathcal{B}_t$ of raw recent frames and, on eviction, folds the oldest frame into the cumulative KV cache $\mathcal{U}_t$ (frozen LLM prefilling, itself FIFO-evicted with attention sinks $A$); *Query-Triggered* builds the instant cache $\mathcal{I}_{t+1}$ from the buffer alone and decodes the answer from $[\mathcal{C}_a, \mathcal{U}_{t+1}, \mathcal{I}_{t+1}]$. *Pang et al. (2026), arXiv:2605.01858. Embedded for personal research reference.*

![[decouple-and-cache-overview.png]]
> **Overview (Figure 1).** Why decoupling helps: offline `uniform sample` / `sliding window` recompute the whole window ($O(TW^2)$, cold-start, memory-limited); a streaming `uniform cache` is cheap ($O(TW)$) but its recent tokens are poorly encoded (76.92); DSCache keeps the cumulative past cache (orange) and rebuilds only the recent instant cache (green) on demand at $O(T(W+C^2))$, reaching 79.12. *Pang et al. (2026), arXiv:2605.01858. Embedded for personal research reference.*

## Method + math — the mechanism, then the objective in full

**Setup.** A frozen VLM's decoder is a stack of $N$ layers. Write the KV-construction operator $\varphi$ that, given new input features $X$ and a conditioning cache $\mathcal{C}$, prefills the model to emit per-layer KV pairs:
$$
\varphi(X,\mathcal{C}) = \Big\{\big(W_k^{(r)}Z^{(r)},\; W_v^{(r)}Z^{(r)}\big)\Big\}_{r=0}^{N-1},\qquad Z^{(0)}=X,\;\; Z^{(r+1)}=g_r\big(Z^{(r)},\mathcal{C}^{(r)}\big),
$$
where $g_r$ is layer $r$'s attention+FFN block reading conditioning cache $\mathcal{C}^{(r)}$. $\mathcal{C}_a$ denotes the **attention sinks** (a few always-kept prefix tokens).

**(1) Position-agnostic encoding.** KV is stored *before* rotary position encoding, and positions are re-assigned at use time.
- *Proposition 4.2:* for RoPE-based LLMs, applying position IDs that are *shifted but preserve relative token distances* yields outputs identical to standard positional encoding — because RoPE attention depends only on relative offsets.
- *Corollary 4.3:* therefore cached entries may be freely **partitioned, recombined, or selectively reused** by simply reassigning contiguous position IDs (this is what lets the recent slice be dropped and the buffer re-encoded without position discontinuities after eviction).

**(2) Feature buffer $\mathcal{B}_t$ (FIFO, size $l_i$).** Raw pre-cache frame features, most recent $l_i$:
$$
\mathcal{B}_t =
\begin{cases}
[\,\mathcal{B}_{t-1},\{X_t\}\,], & |\mathcal{B}_{t-1}| < l_i,\\[4pt]
[\,\mathcal{B}_{t-1}\setminus\{X_{t-l_i}\},\{X_t\}\,], & |\mathcal{B}_{t-1}| = l_i.
\end{cases}
$$
When full, the oldest feature $X_{t-l_i}$ is evicted and promoted into the cumulative cache.

**(3) Cumulative past cache $\mathcal{U}_t$ (incremental build + FIFO evict, budget $l_u$).** The evicted feature is encoded *once*, conditioned on sinks plus a recent slice of the cumulative cache selected by subsampler $\zeta$ (keeps the most recent $l_l\le l_u$ entries):
$$
\mathcal{U}_t = \big[\,\mathcal{U}_{t-1},\; \varphi\big(X_{t-l_i},\,[\mathcal{C}_a,\,\zeta(\mathcal{U}_{t-1})]\big)\,\big].
$$
Length is capped per layer by FIFO eviction when it exceeds $l_u$:
$$
\mathcal{U}_t^{(r)} =
\begin{cases}
\mathcal{U}_t^{(r)}, & |\mathcal{U}_t^{(r)}|\le l_u,\\[4pt]
\big\{(K_i^{(r)},V_i^{(r)})\big\}_{i=|\mathcal{U}_t^{(r)}|-l_u}^{\,|\mathcal{U}_t^{(r)}|-1}, & |\mathcal{U}_t^{(r)}| > l_u.
\end{cases}
$$

**(4) Instant cache $\mathcal{I}_{t+1}$ (rebuilt on query).** Encode the *whole* feature buffer freshly, conditioned only on the sinks — deliberately **independent of $\mathcal{U}$**, so recent frames get a clean, short, in-distribution context:
$$
\mathcal{I}_{t+1} = \varphi\big(\mathcal{B}_{t+1},\,\mathcal{C}_a\big).
$$

**(5) Inference cache.** Concatenate sinks + cumulative past + instant, then decode the answer:
$$
\mathcal{C}_{t+1} = [\,\mathcal{C}_a,\;\mathcal{U}_{t+1},\;\mathcal{I}_{t+1}\,].
$$

The effective context window is $l_W = l_u + l_i$. Cost is $O(T(W+C^2))$ vs $O(TW)$ for the uniform cache and $O(TW^2)$ for offline recompute, where $C\!=\!l_i$ (buffer) $\ll W\!=\!l_W$ — the quadratic term is confined to the tiny instant window, so the extra cost is small.

## Explicit design choices
- **Frozen backbone, training-free.** Bolts onto pretrained offline VLMs (LLaVA-OneVision-7B/0.5B, Qwen2.5-VL-7B) with no finetuning.
- **Decouple recency from history.** Long context lives in the incrementally-built cumulative cache; recent fidelity comes from re-encoding a small raw buffer — the recent slice of the cumulative cache is *discarded* at query time, not reused.
- **Store KV pre-position (position-agnostic), re-index at use.** Enables partition/recombine and clean eviction without position gaps; justified for RoPE by Prop. 4.2.
- **Attention sinks $\mathcal{C}_a$** are always kept and used as the base conditioning for both $\varphi$ calls.
- **Buffer/cumulative sizes:** feature buffer $l_i\!=\!4$ frames, cumulative budget $l_u\!=\!28$ → context window $l_W\!=\!32$ frames (StreamingBench, OVO-Bench); $l_W\!=\!256$ for the longer RVS datasets. Subsampler $\zeta$ conditions each new cumulative entry on only the recent slice.
- **Higher spatial resolution helps under DSCache.** Because tokens are cached and reused, they spend the saved compute on ×1.5–×2.0 spatial resolution for extra gains (ablation Table 9).
- **Composable with cache-compression methods** (ReKV, InfiniPot-V) — DSCache addresses *construction*, they address *compression*, so combining stacks.

## Key results / what to remember
Numbers below are verified against the paper's HTML tables (see verification note — the arXiv PDF endpoint mis-served an unrelated PDF, so figures/text came from the HTML render).

- **StreamingBench (real-time visual understanding, accuracy).** LLaVA-OV-7B **+ DSCache = 79.12** vs + Uniform cache 76.92 (**+2.2**), vs sliding-window 77.40, vs offline uniform-sample 71.12; beats online SOTA StreamBridge 77.04 and TimeChat-Online 75.36. On Qwen2.5-VL-7B: **DSCache 82.32** vs Uniform cache 78.56 (**+3.8**). Largest per-subtask gains vs uniform cache: Text-Rich +6, Spatial/Sequential and Action-type subtasks +5–7.
- **OVO-Bench (overall).** LLaVA-OV-7B **DSCache 57.5** vs Uniform cache 55.6; real-time perception **71.5** vs 68.4, backward-tracing 47.8 vs 46.0, forward-response 53.1 vs 52.9. Qwen2.5-VL-7B **DSCache 55.4** vs Uniform 52.2.
- **RVS (multi-turn streaming QA).** RVS-Ego: LLaVA-OV-7B **DSCache 59.5 acc / 3.97 score** vs Uniform 58.9 / 3.93; RVS-Movie **49.4 / 3.50** vs 48.1 / 3.46. Holds at 0.5B scale too.
- **vs cache-compression methods (LLaVA-OV-7B).** DSCache 79.12 (StreamingBench) > StreamingVLM 76.92 > ReKV 71.06; on Qwen, DSCache 82.32 > StreamingVLM 78.56 > InfiniPot-V 76.40. **Combining** DSCache + ReKV → 79.41, DSCache + InfiniPot-V → 79.60 (further OVO gains too).
- **Runtime (LLaVA-OV-7B, H200, 1 FPS).** DSCache overall latency **0.30 s** (prefill 0.16 + decode 0.14), **16 GB** — vs Uniform cache 0.21 s / 16 GB and Offline 2.85 s / 19 GB. Extra latency is the instant-cache recompute over the tiny buffer; decode cost is identical, memory matches the uniform baseline.
- **Ablation — the instant cache is the load-bearing part.** Removing it (cumulative-only) collapses StreamingBench 79.12 → 74.27; removing the cumulative cache (instant-only) barely moves it (79.12 → 79.06), confirming recent-frame fidelity, not long history, drives the gain on these benchmarks.

No Zotero highlights present.

Takeaways: (i) the "cumulative effect" is a real, nameable failure mode of naive streaming KV caches — recency bias means appended-then-conditioned recent tokens are *worse* encoded than freshly re-encoded ones; (ii) decoupling construction (cheap incremental history vs on-demand clean recent) is a cleaner axis than compression; (iii) position-agnostic (pre-RoPE) caching is the enabling trick that makes partition/recombine sound; (iv) it is orthogonal to and stacks with KV-compression memory methods.

## How it connects (evolution)
- [[streamingvlm]] — the direct streaming-KV baseline it out-scores (76.92 vs 79.12 on LLaVA-OV-7B); DSCache reframes its uniform-cache as the "cumulative effect" problem.
- [[rekv]] — retrieval-based KV compression; DSCache both beats it and *combines* with it (79.12 → 79.41).
- [[infinipot-v]] — training-free KV compression under a fixed budget; a compression sibling that DSCache stacks on (→ 79.60).
- [[streambridge]] — online/streaming VLM SOTA baseline it surpasses on StreamingBench.
- [[streamingbench]] — the primary evaluation benchmark (real-time visual understanding).
- [[ovo-bench]] — the second core benchmark (real-time / backward / forward streaming skills).

## Open questions / limitations
- The ablation shows history (cumulative cache) contributes little on StreamingBench/OVO — so on these benchmarks DSCache is close to "just re-encode the recent window well." Its advantage on genuinely long-horizon, history-dependent queries is less demonstrated (RVS gains are small, ~+0.6–1.3).
- Position-agnostic equivalence (Prop. 4.2) is proved for RoPE; models with learned/ALiBi-style positions or non-standard attention would need re-derivation.
- Buffer/cumulative sizes ($l_i\!=\!4$, $l_u\!=\!28$) are hand-set per benchmark (32 vs 256); no adaptive policy for stream content, and the instant-cache recompute adds ~50% prefill latency that grows with $l_i^2$.
- Gains are largest paired with higher spatial resolution — some of the reported headline lift is compute reinvested, not purely the caching mechanism.

*Verification: equations and all headline numbers cross-checked against the arXiv:2605.01858v1 HTML tables (StreamingBench, OVO-Bench, RVS, runtime, ablation, cache-method comparison); abstract/authors confirmed on the arXiv abs page; figures pulled directly from the HTML assets (framework_new.jpg, intro1.jpg). Note: arXiv's /pdf/2605.01858 endpoint served an unrelated PDF (SVBench, 2502.10810) at fetch time, so no PDF-level re-read was possible — numbers rely on the HTML render and small-model extraction; treat exact per-subtask decimals as HTML-sourced.*

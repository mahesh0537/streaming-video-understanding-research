---
zotero_key: null
authors: Yilong Chen, Xiang Bai, Zhibin Wang et al. (Huazhong Univ. of Science & Technology; South China Univ. of Technology; Peking Univ.)
year: 2025
arxiv: 2511.07278
pdf: https://arxiv.org/pdf/2511.07278
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# StreamKV: Streaming Video Question-Answering with Segment-based KV Cache Retrieval and Compression

**Lineage role:** AAAI-2026 successor to [[rekv]] — replaces uniform KV chunking + store-everything memory with *semantic* segmentation, guidance-prompt compression, and a unified layer-adaptive KV selection module that serves both compression and retrieval.

## Problem — what was limited before this paper (short)
Training-free streaming Video-LLMs that keep the full attention KV cache (notably [[rekv]]) run into two coupled problems as videos grow long. (1) They cut the stream into *uniform* fixed-length chunks, which slices across natural event boundaries and disrupts the continuous semantic structure of the video. (2) They retain the *entire* historical visual KV cache, so GPU memory grows unbounded with video length and latency rises. StreamKV targets accuracy under a hard memory budget: keep only a fraction of the KV cache while still being able to answer arbitrary later questions.

## Key idea — the core insight, 2-4 sentences
Partition the incoming stream into variable-length *semantic* segments (boundaries where adjacent frame embeddings drop in cosine similarity) instead of uniform chunks, and attach a cheap averaged "summary vector" to each segment. Compress each segment's KV cache at encoding time using a fixed, question-agnostic **guidance prompt** as the selection criterion (so compression happens once, before any query, enabling multi-turn dialogue without recomputation), while explicitly *preserving* the summary-vector KV blocks. When a question arrives, retrieve the query-relevant compressed KV blocks. Both compression and retrieval are the *same* operation — a unified **layer-adaptive KV selection** module that allocates the token budget non-uniformly across transformer layers via a binary-searched global threshold.

![[streamkv.png]]
> **Crux (Figure 3).** The StreamKV workflow: a blue *video-processing pipeline* (dynamic semantic partitioning → per-segment summary vector → VLM encoding → guidance-prompt-driven KV compression → KV Bank) and a green *question-answering pipeline* (question-driven KV retrieval → VLM → answer), both driven by the shared layer-wise adaptive selection module (left inset: cosine-similarity scoring → softmax normalization/sort → binary-search budget allocation K_l per layer → top-K_l block selection). *Chen et al. (2025), arXiv:2511.07278. Embedded for personal research reference.*

## Method + math — the mechanism, then the objective in full

**1. Semantic segment partitioning.** Frames are sampled at 0.5 FPS and embedded by the vision encoder. A boundary is placed where the cosine similarity between adjacent frame features drops below the partitioning threshold $\theta$:
$$s_t = \frac{f_{t-1}\cdot f_t}{\lVert f_{t-1}\rVert\,\lVert f_t\rVert}, \qquad \text{new segment when } s_t < \theta.$$
Segment length is clamped between a minimum (4 frames, to avoid degenerate tiny segments) and a maximum (64 frames, to bound temporal context); over-long runs are split and very short ones merged. For segment $i$ with $T_i$ frames a **summary vector** is the mean frame feature:
$$f_s^{i} = \frac{1}{T_i}\sum_{t=1}^{T_i} f_t^{i}.$$
Segments are encoded sequentially with a sliding local window (≈15K tokens); RoPE is applied only *within* the local window during encoding.

**2. Unified layer-adaptive KV selection (`SelectKV`).** For each layer $l$, given candidate representative key vectors $\{r_j^l\}$ (one per KV block, indexed by $\mathrm{idx}(R_l)$) and a *selection criterion* vector $c^l$, score each candidate by cosine similarity $\mathrm{Sim}_l(j)=\cos(r_j^l, c^l)$, then softmax-normalize *within the layer*:
$$\widetilde{\mathrm{Sim}}_l(j) = \frac{\exp(\mathrm{Sim}_l(j))}{\sum_k \exp(\mathrm{Sim}_l(k))}.$$
Select the top-$K_l$ candidates per layer:
$$\mathcal{I}_l = \operatorname{Top}K_{\mathrm{idx}}\big([\widetilde{\mathrm{Sim}}_l(j)]_{j\in\mathrm{idx}(R_l)},\,K_l\big),\qquad \sum_{l=1}^{L} K_l = N.$$
Rather than splitting the budget uniformly ($K_l=N/L$), the per-layer count is set by a **global cumulative threshold** $p$: sort each layer's normalized scores descending ($s_l(j)$ = index of $j$-th largest), and let
$$K_l^{p} = \min\Big\{k \;\Big|\; \sum_{j=1}^{k}\widetilde{\mathrm{Sim}}_l(s_l(j)) \ge p\Big\}.$$
$p$ is chosen (Algorithm 1: binary search over $p\in[0,1]$) so the total budget constraint holds, $\sum_l K_l^{p}=N$. Layers whose similarity mass is concentrated get *more* blocks; diffuse layers get fewer, raising total retained information under a fixed $N$. The whole procedure is written compactly as
$$\{\mathcal{I}_l\}_{l=1}^{L} = \text{SelectKV}\big(\{R_l, c^l\}_{l=1}^{L},\, N\big).$$

**3. KV compression via guidance prompt (at encoding time, per segment $i$).** The selection criterion is a *fixed* guidance prompt $g$ (averaged over its tokens, per layer) that asks the model to keep salient entities, key events/actions, temporal & causal relations, contextual cues and numerical details — deliberately **question-agnostic** so compression is done once and reused across all future turns:
$$g^l = \frac{1}{N_g}\sum_{k=1}^{N_g} g_k^l, \qquad \{I_l^i\}_{l=1}^{L} = \text{SelectKV}\big(\{R_l^i, g^l\}_{l=1}^{L},\, N\big),$$
with per-segment budget $N=\lceil(1-\theta_c)\,T_i\rceil\times L$ under compression ratio $\theta_c$. The summary KV blocks $b_s^{i,l}$ are **never compressed away** — they are appended so each segment always retains a coarse global representation. Compressed blocks + summary blocks are written to the **KV Bank**.

**4. KV retrieval (at question time).** The selection criterion switches to the averaged question tokens:
$$q^l = \frac{1}{N_q}\sum_{k=1}^{N_q} q_k^l,\qquad \{I_l\}_{l=1}^{L} = \text{SelectKV}\big(\{R_l, q^l\}_{l=1}^{L},\, N\big),\quad N = N_r\times L.$$
Retrieved blocks are gathered $P_l=[B_l[j]\mid j\in I_l]$ and fed with the question to the VLM. During QA the retrieved (originally non-contiguous) tokens are treated as *consecutive*, using relative position-based RoPE — decoupling the encoding-phase and QA-phase positional encoding.

## Explicit design choices
- **Base model:** LLaVA-OneVision-Qwen2-7B (training-free; no fine-tuning of the VLM).
- **Sampling:** 0.5 FPS frame sampling.
- **Segmentation:** cosine-similarity boundaries, threshold $\theta=0.99$; min segment 4 frames, max 64 frames; merge/split to keep within bounds.
- **Summary vector:** per-segment mean frame feature; its KV blocks are *exempt* from compression and always kept.
- **Compression is question-agnostic** (guidance prompt), done once at encoding → supports multi-turn without recompression; retrieval is question-specific.
- **Unified selection module** shared by both compression and retrieval; both use cosine-similarity scoring + intra-layer softmax + **layer-adaptive** budget via binary-searched global threshold $p$ (Algorithm 1).
- **Retrieved frames** $N_r$ default = 8 per layer (peak accuracy; more frames *hurt*, unlike ReKV).
- **Positional encoding decoupled** between encode (local-window RoPE) and QA (retrieved tokens made contiguous).
- **Local encoding window** ≈ 15K tokens; hardware NVIDIA H20 (96 GB), FP16.

## Key results / what to remember
Verified against the paper's own tables (StreamingBench; ↓X% = fraction of KV cache discarded).

- **StreamingBench overall (Table 1):** StreamKV-7B **58.9** at ↓60% compression, **57.4** at ↓80%, **56.7** at ↓90% — all above ReKV-7B **53.5** and Dispider-7B **53.1** at the same 0.5 FPS. That is **+5.4** over ReKV even while *discarding 60%* of the cache. Base LLaVA-OV-7B (32 frames) = 56.4; Qwen2-VL-7B = 54.1.
- **Proprietary reference points (Table 1):** Gemini 1.5 Pro 67.1, GPT-4o 60.2, Claude 3.5 Sonnet 57.7 — StreamKV at ↓60% (58.9) beats Claude 3.5 Sonnet.
- **Task-level (↓60%, vs ReKV):** Clips Summarization 87.7 vs 78.6 (+9.1); Anomaly Context Understanding 45.6 vs 31.2 (+14.4); Omni-Source Understanding 51.4 vs 37.4 (+14.0); Real-Time Visual Understanding overall 71.0 vs 69.1.
- **Ablation — semantic vs uniform partitioning (Table 2):** semantic wins at every ratio, gap *widens* under aggressive compression: +1.75 (↓50%), +2.56 (↓60%) … **+5.31 (↓90%)** (56.72 vs 51.41).
- **Ablation — summary vector (Table 3):** with-summary vs without: +1.65 → +2.87 across ↓50%…↓90% (e.g. ↓90%: 56.72 vs 53.85).
- **Ablation — layer-adaptive selection (Table 4):** adaptive-compression + adaptive-retrieval is best at every ratio (59.07 / 58.89 / 58.11 / 57.43 / 56.72 for ↓50→90%), above all uniform/mixed variants.
- **Retrieved-frame count (Figure 4):** StreamKV peaks at ~8 frames then *declines* with more; ReKV *improves* with more frames — evidence StreamKV's retrieval is more precise per frame. (Exact per-point values n/r beyond ~59.5 / ~60 / ~59 for 4 / 8 / 16 frames.)

No Zotero highlights present.

Takeaways: (1) semantic segmentation + a preserved summary vector is what makes aggressive KV compression survivable — the benefit over uniform chunking grows precisely where you compress hardest. (2) Decoupling compression (question-agnostic, once) from retrieval (question-specific) is the design move that makes multi-turn streaming QA cheap. (3) A single layer-adaptive `SelectKV` primitive serves both, and per-layer non-uniform budgeting reliably beats uniform allocation.

## How it connects (evolution)
- [[rekv]] — the direct predecessor: also a training-free KV-cache retrieval streaming Video-LLM, but with uniform chunking + full-cache storage. StreamKV is its semantic + compressed refinement and its main baseline.
- [[infinipot-v]] — likewise attacks streaming KV-cache *memory* via query-agnostic compression under a budget; strong sibling on the compression axis.
- [[hermes-kv]] — KV-cache management/retrieval for streaming video; parallel take on the same bottleneck.
- [[streamingvlm]] — streaming inference with bounded KV/attention state; related efficiency lineage.
- [[dispider]] — streaming Video-LLM baseline compared in Table 1; different (perceive-decide-react) architecture.
- [[streamingbench]] — the primary evaluation benchmark used for all headline and ablation numbers.

## Open questions / limitations
- The guidance prompt is a *fixed*, hand-specified string; how sensitive results are to its wording, and whether a learned criterion would help, is untested (no prompt-ablation reported).
- All results use one 7B base (LLaVA-OneVision) on StreamingBench; generalization to other VLMs/benchmarks (e.g. OVO-Bench, SVBench) is not shown.
- Memory/latency advantages are asserted with qualitative framing; the note lacks a hard wall-clock/GB table here (n/r beyond the ↓X% budget knob) to quantify the efficiency claim precisely.
- Segmentation quality hinges on the 0.99 threshold and vision-encoder frame embeddings; robustness on low-motion or highly repetitive footage (few boundaries → long segments hitting the 64-frame cap) is unexamined.

*Verification: equations and design values cross-checked against arXiv:2511.07278v1 (abstract/method Eqs. 1,5-11, Algorithm 1) and the rendered PDF Figure 3; all reported numbers checked against the paper's Tables 1-4 and Figure 4. Code: github.com/sou1p0wer/StreamKV.*

---
zotero_key: null
authors: Minsoo Kim, Kyuhong Shim, Jungwook Choi, Simyung Chang (Hanyang Univ. / Qualcomm AI Research)
year: 2025
arxiv: 2506.15745
pdf: https://arxiv.org/pdf/2506.15745
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# InfiniPot-V: Memory-Constrained KV Cache Compression for Streaming Video Understanding

**Lineage role:** The fixed-budget, query-agnostic KV-cache compressor for streaming video LLMs (NeurIPS 2025) — it holds peak memory *constant* regardless of stream length by continually compressing the KV cache with two training-free spatiotemporal metrics (temporal-redundancy + value-norm), where prior offline compressors let memory grow linearly and prior streaming methods (ReKV) offloaded to CPU.

## Problem — what was limited before this paper (short)
Long-video MLLMs turn a video into an enormous token sequence, so the KV cache dominates GPU memory and decoding latency. Existing offline compression strategies — frame sampling, input-vision compression (IVC, e.g. LongVU/DyCoke), and post-prefill KV-cache compression (KVC, e.g. SnapKV) — all assume (i) unconstrained memory during compression and (ii) a known query. Neither holds for **streaming video understanding (SVU)**: peak memory grows near-linearly with stream length (KVC must materialize the full cache before compressing, and re-prefill on every new query because its eviction is query-dependent), and the query is not known in advance. ReKV sidesteps memory growth by offloading the KV cache to CPU, but that adds heavy transfer overhead (+18.8 GB/h CPU, ~4x per-frame latency) and fails on shared-memory edge devices where CPU RAM is unavailable.

## Key idea — the core insight, 2-4 sentences
Process the stream in blocks and **continually compress the KV cache whenever a fixed memory budget $|M|$ is hit**, shrinking it back to a small target size $|C|$ — so peak memory never grows with stream length (Continual KV-cache Compression, CKV). Choose which tokens to keep with two cheap, **query-agnostic** spatiotemporal signals computed directly on the cached keys/values: **Temporal-axis Redundancy (TaR)**, which drops patches that are near-static across time (high Key cosine-similarity to recent frames), and **Value Norm (VaN)**, which keeps spatially salient tokens (high $\ell_2$ norm of the Value embedding correlates with high vocabulary-projection entropy = semantic richness). Recent frames are always retained in full; TaR and VaN split the remaining budget.

![[infinipot-v.png]]
> **Crux (Figure 2).** The two eviction criteria and the 3D-reshaped cache they operate on: (a) TaR evicts static "background" patches whose Keys stay similar across frames while retaining moving patches; (b) Key embeddings (dark) carry temporal redundancy far more reliably than Value embeddings (teal) across all layers; (c) InfiniPot-V reshapes the constrained cache into a time×height×width volume and selects TaR (temporal) plus VaN (spatial) tokens, always keeping the $r$ recent frames. *Kim et al. (2025), arXiv:2506.15745. Embedded for personal research reference.*

## Method + math — the mechanism and central objective in full

**Setup.** A decoder-only MLLM caches keys/values $C=(K,V)$ from projections $W_q,W_k,W_v\in\mathbb{R}^{D\times D}$, updated incrementally at decode time as $C_{t+1}=(\{K,k_{t+1}\},\{V,v_{t+1}\})$. Streaming makes this cache unbounded, hence the need for CKV.

**Continual KV-cache Compression (CKV) — the outer loop (Algorithm 1).** Given a memory budget $|M|$, target cache size $|C|$ (with $|M|\gg|C|$), TaR ratio $\alpha$, and recent-frame count $r$:
1. **Process:** append new frame tokens to $K,V$.
2. When $\text{len}(K)\ge|M|$ (budget exceeded): extract the $r$ most recent frames $K_{\text{recent}},V_{\text{recent}}$ and always keep them.
3. **TaR:** score past-frame patches, keep the top $\alpha|C|-|K_{\text{recent}}|$.
4. **VaN:** score all tokens, keep the top $(1-\alpha)|C|$.
5. **Combine:** union the TaR, VaN, and recent-frame indices; compress $K,V$ down to size $|C|$. The freed space $|M|-|C|$ absorbs the next incoming frames.
6. On a user query, generate the answer from the *current* compressed cache — no re-prefill.

Compression adds only ~0.5% overhead relative to frame-processing time.

**TaR — temporal-axis redundancy (Sec. 3.1).** Reshape the current Keys $K\in\mathbb{R}^{H\times(f\times p)\times D}$ (heads $H$, $f$ frames of $p=|M|/f$ patches each, dim $D$) into a 3D volume, split into $K_{\text{recent}}\in\mathbb{R}^{H\times r\times p\times D}$ and $K_{\text{past}}\in\mathbb{R}^{H\times(f-r)\times p\times D}$. For each past patch at frame $t$ and spatial coordinate $(i,j)$, score its distinctiveness as the (negated, averaged) cosine similarity to the same patch coordinate across the $r$ recent frames:
$$
s^{\text{TaR}}(t,i,j) = -\frac{1}{r}\sum_{t'=1}^{r}\cos\!\Big(K^{(t,i,j)}_{\text{past}},\, K^{(t',i,j)}_{\text{recent}}\Big).
$$
The negative sign makes **higher score = less redundant** (more distinctive vs. recent frames). Select the least-redundant past tokens:
$$
\mathcal{I}_{\text{TaR}} = \text{TopK}\big(s^{\text{TaR}},\, |C|-|K_{\text{recent}}|\big),\qquad |K_{\text{recent}}|=rp,
$$
$$
\tilde K_{\text{TaR}}=\text{Concat}\big(K[:,\mathcal{I}_{\text{TaR}},:],\,K_{\text{recent}}\big),\quad
\tilde V_{\text{TaR}}=\text{Concat}\big(V[:,\mathcal{I}_{\text{TaR}},:],\,V_{\text{recent}}\big).
$$
Key insight (Fig. 2b): **Keys** capture static-patch temporal redundancy (high cosine similarity across adjacent frames) far better than Values, so TaR scores on Keys.

**VaN — spatial semantic importance (Sec. 3.2).** Score each token by the $\ell_2$ norm of its **Value** embedding:
$$
s^{\text{VaN}} = \big\|V^{(t,i,j)}\big\|_2 .
$$
Justification: projecting each layer's vision-token representation into vocabulary space and measuring the word-distribution **entropy**, high-VaN tokens have consistently higher entropy (= more informative); retaining high-VaN tokens beats retaining low-VaN ("VaN Reverse") tokens on VideoMME across compression ratios.

**Layer-wise adaptive pooling for VaN.** VaN distributions show strong spatial locality in early/middle layers that fades in deeper layers (measured by center-distance in a $3\times3$ window and coefficient of variation, CV). So an adaptive average-pooling kernel is applied before Top-K, sized inversely to each layer's CV:
$$
\text{PoolSize}(CV_l) = g(CV_l),\qquad g:\mathbb{R}^+\to\{1,3,5,7\},
$$
large kernels (7) for low-CV early layers (high locality → pool aggressively), no pooling (kernel 1) for high-CV deep layers (preserve fine detail). Selection:
$$
\mathcal{I}_{\text{VaN}}=\text{TopK}(\text{VaN}_{\text{pool}},\,|C|),\qquad
\tilde K_{\text{VaN}}=K[:,\mathcal{I}_{\text{VaN}},:],\;\tilde V_{\text{VaN}}=V[:,\mathcal{I}_{\text{VaN}},:].
$$

**Combining TaR + VaN (Sec. 3.3).** Two-stage: first allocate $\alpha|C|$ tokens by TaR (temporal), then fill the remaining $(1-\alpha)|C|$ by VaN (spatial), plus the always-kept recent frames. $\alpha$ and $r$ are swept in the appendix.

## Explicit design choices
- **Fixed memory budget, not a fixed ratio:** compress on hitting $|M|$, always back to target $|C|$ ($|M|\gg|C|$), so peak memory is constant over an arbitrarily long stream. The reported budgets are tiny: 6K (offline) and 4K (streaming) vs. 25K–50K full cache.
- **Fully training-free, plug-in:** no fine-tuning, no architecture change; drops into off-the-shelf MLLMs (Qwen-2-VL, Qwen-2.5-VL, LLaVA-OV, LLaVA-Next-Video).
- **Query-agnostic selection:** eviction depends only on the cached video KV, never on the (unknown, future) user query — the opposite of attention/query-driven KVC like SnapKV. This is what lets it answer multi-turn queries without re-prefill.
- **Keys for temporal, Values for semantic:** TaR uses Key cosine-similarity (Keys encode redundancy), VaN uses Value norm (Values encode semantics) — an empirically motivated split.
- **Always retain the $r$ most recent frames in full** to preserve rapidly-changing / newly-introduced content.
- **3D reshape of the flat cache** into time×height×width so patch coordinates align across frames for the patch-wise TaR comparison.
- **Layer-adaptive VaN pooling** by inverse-CV kernel sizing (1/3/5/7).
- **Patch-wise (not frame-level) similarity** for TaR (ablation: 64.5 vs 62.9 on MLVU).
- **All-on-GPU** — no CPU offloading, deliberately targeting shared-memory edge devices (e.g. 32 GB Jetson AGX Orin).

## Key results / what to remember
No Zotero highlights present.

- **Peak GPU memory cut by up to 94%** and near-constant with stream length; compression overhead ~0.5% (abstract / Fig. 1f).
- **Offline video understanding (Table 1)**, Qwen-2-VL-7B, budget **6K vs. 50K full** (≈12.5%): EgoSchema 65.6 vs 65.2, MLVU 65.8 vs 65.8, VideoMME 62.8 vs 63.9, LongVideoBench 58.4 vs 58.8 — essentially full-cache accuracy at 1/8 the memory. LLaVA-Next-7B at **6K vs 25K**: EgoSchema 65.8, MLVU 65.2, VideoMME 61.1, LVB 60.9 (vs full 67.6 / 68.7 / 62.8 / 63.5). Qwen-2.5-VL-3B at **6K vs 50K**: 61.8 / 62.1 / 59.3 / 56.5.
- **vs. IVC under memory budget (Table 2)**, Qwen-2-VL-7B VideoMME+MLVU avg: **InfiniPot-V 64.3 at 6K** and **63.1 at 3K** — beating TTC/DyCoke (58.4 at 6K) and STC/LongVU (57.9 at 6K), and matching FullKV (64.2 at 50K). At 6K = an ~88% "lossless" compression rate.
- **vs. KVC under the CKV framework (Fig. 4)**: InfiniPot-V beats Uniform Select, SnapKV, and InfiniPot across all four ratios (1/16–1/2) on VideoMME / MLVU / LVB for both LLaVA-Next-7B and Qwen-2-VL-7B; query-dependent SnapKV collapses at high compression while InfiniPot-V holds up even at 1/16.
- **Streaming, offloading-free (Table 3)**, LLaVA-OV-7B, 1-hr video @0.5 fps: InfiniPot-V RVS-Ego 57.9 acc / 3.5 score, RVS-Movie 51.4 / 3.5, at **27.8 GB GPU, 0 CPU offload, 76.3 ms/frame**. ReKV *with* CPU offloading is slightly higher (60.1 / 3.9 Ego) but costs 37.5 GB GPU + 18.8 GB/h CPU and 285.7 ms/frame (~3.7x slower); ReKV *without* offloading drops to 55.8 acc — InfiniPot-V beats it while using no CPU memory.
- **Streaming multiple-choice (Table 4)**, Qwen-2.5-VL-7B, 4K budget: OVO-Bench BW 44.5→47.6, Real 61.1→65.9, FW 47.8→47.9, Avg 51.7→53.6; StreamingBench 75.2→76.4 — consistent gains over Uniform Select.
- **Ablation (Table 5, MLVU):** patch-wise TaR (64.5) > frame-level (62.9); reversed TaR/VaN both hurt, confirming the sign/direction of each metric.
- **Edge (Table 6):** runs streaming video processing on NVIDIA Jetson AGX Orin (32 GB), demonstrating on-device feasibility.

## How it connects (evolution)
- [[rekv]] — the offloading-based streaming KVC baseline InfiniPot-V is measured against (Table 3); InfiniPot-V's pitch is *no CPU offload*.
- [[streaming-memory]] — this note's sub-topic hub: fixed-budget KV compression is a core memory strategy.
- [[flash-vstream]] — another memory-bounded streaming video approach (learned memory) contrasted with this training-free cache-compression route.
- [[ovo-bench]] — streaming benchmark (backward/realtime/forward) used to evaluate InfiniPot-V (Table 4).
- [[streamingbench]] — real-time streaming benchmark used in Table 4.
- [[dispider]] — streaming video LLM in the same problem space (perception under a growing stream) this method could plug into.

## Open questions / limitations
- **Query-agnostic retention is a double-edged sword:** it enables multi-turn without re-prefill, but a token discarded before a query arrives is gone — recall of fine details needed by an unforeseeable later question is bounded by TaR/VaN's guesses (OVO-BW gains are real but modest, 44.5→47.6).
- **Hyperparameters $\alpha$, $r$, and the CV→kernel map $g$** are tuned per model/benchmark; how sensitive real-world streaming deployment is to them is only shown via appendix sweeps.
- **Audio / multi-modal streams and true real-time proactive responding** aren't the focus — evaluation is on QA-style benchmarks, not open-ended proactive dialogue latency budgets end-to-end.
- **VaN as a semantic proxy** rests on the entropy/vocabulary-projection correlation; whether it holds across very different vision encoders or newer native-resolution MLLMs is untested here.

*Verification: equations (TaR Eq. 4–6, VaN norm + adaptive pooling Eq. 7, CKV Algorithm 1) transcribed from the arXiv PDF pages 3–7; all headline numbers cross-checked against the paper's Tables 1–6 and Fig. 1/4 as printed (arXiv HTML was 404, so figures/text taken from the PDF via PyMuPDF text extraction and figtool page renders). No GitHub/project page consulted.*

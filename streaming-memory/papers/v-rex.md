---
zotero_key: null
authors: Donghyuk Kim, Sejeong Yang, Wonjin Shin, Joo-Young Kim (KAIST)
year: 2025
arxiv: 2512.12284
pdf: https://arxiv.org/pdf/2512.12284
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# V-Rex: Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval

**Lineage role:** A *training-free dynamic KV-cache retrieval* algorithm (ReSV) — hash-bit key clustering + a data-adaptive "weighted cumulative-sum" threshold that replaces fixed top-k selection — co-designed with an edge accelerator (HPCA 2026). For this streaming-memory subtopic the load-bearing contribution is the **algorithm**: how to decide *which* cached KV tokens to fetch each frame during the iterative prefill of a streaming video LLM, without retraining. (Hardware internals — the DRE engine, memory hierarchy — are noted only at the surface.)

## Problem — what was limited before this paper (short)
Streaming video LLMs ingest frames continuously, so their KV cache grows roughly as $O(N^2 T)$ (spatial tokens per frame $\times$ temporal duration) and quickly exceeds edge memory. The dominant cost is not text decoding but the **iterative prefill** of each new frame against the whole history. Prior KV-retrieval methods (InfiniGen, ReKV, FlexGen offloading) were built for *text generation* and share two mismatches: (1) they optimize the single-query decode step, whereas streaming is prefill-heavy — the authors measure KV-cache retrieval at ~85% of per-frame latency at a 40K cache (≈40% prediction compute, ≈39% cache fetch from CPU); (2) they use **fixed top-k** selection, but the important-token set shifts widely across layers and heads, so a global $k$ either over-fetches (wasting scarce edge memory/energy) or drops relevant tokens.

## Key idea — the core insight, 2-4 sentences
ReSV exploits the strong **spatial-temporal redundancy of video**: adjacent frames produce near-duplicate key vectors, so keys can be clustered cheaply and a query only needs to score cluster *representatives* rather than every token. On top of that, instead of a fixed top-k it uses **WiCSum (Weighted Cumulative-Sum) thresholding** — sort clusters by relevance-to-query, accumulate their (score × token-count) mass, and stop once a fraction of the total attention mass is captured — which makes the number of fetched tokens adapt per layer and per head. Both steps reduce to bitwise (hash/XOR) and prefix-sum operations, so they are lightweight enough to run on-the-fly and to hide behind execution via prefetch.

![[v-rex.png]]
> **Crux (Fig. 6).** The ReSV pipeline: right after Q/K/V generation the "KV Cache Prediction" for the *next* layer runs — hash-bit key clustering groups the key cache, the query scores cluster representatives ($Key_{cluster}$), and Weighted Cumulative Sum Thresholding picks the surviving clusters; the selected key/value tokens are prefetched so the execution stage does only a light attention over them. *Kim et al. (2025), arXiv:2512.12284. Embedded for personal research reference.*

## Method + math — the mechanism, then the central objective/equations in full

ReSV has **two stages that pipeline across layers** (Fig. 6): while layer $\ell$ runs its *execution* (light attention over already-selected tokens), the *KV-prediction* for layer $\ell{+}1$ runs, and its selected tokens are **prefetched** from CPU/storage so the fetch latency is hidden. KV prediction itself is two steps: hash-bit key clustering, then WiCSum thresholding.

**Step 1 — Hash-bit key clustering (locality-sensitive hashing on keys).**
Each key vector $k$ (dimension $d$) is projected onto $N_{hp}$ random hyperplanes and binarized to an ultra-low-dimensional bit code ($\leq 0.5\%$ of the original dimension):
$$
h(k)_m = \mathbb{1}\!\left[\, w_m^\top k > 0 \,\right], \qquad m = 1,\dots,N_{hp}.
$$
Two keys are placed in the same cluster when their **Hamming distance** (bitwise XOR popcount) is below a threshold $T_{hp}$:
$$
\mathrm{dist}(k_a,k_b) = \sum_{m=1}^{N_{hp}} \big( h(k_a)_m \oplus h(k_b)_m \big) < T_{hp}.
$$
Each cluster $j$ stores a **representative key** $Key_{cluster,j}$ (mean of member keys), its bit code, and a token count $TC_j$. A query then scores only the $|clusters|$ representatives instead of all $N$ tokens:
$$
Score_{i,j} = q_i^\top Key_{cluster,j}.
$$

**Step 2 — WiCSum (Weighted Cumulative-Sum) thresholding.**
For query/head $i$, weight each cluster's relevance by how many tokens it stands for and form the total weighted mass:
$$
Sum_i = \sum_{j} Score_{i,j}\cdot TC_j. \tag{1}
$$
Set a **relative** threshold as a fixed fraction $Th_{r\text{-}wics}$ (a single hyperparameter) of that mass:
$$
Th_{wics,i} = Sum_i \cdot Th_{r\text{-}wics}. \tag{2}
$$
Sort clusters by descending score, let $\sigma(\cdot)$ be that order, and accumulate until the captured mass crosses the threshold:
$$
Acc_i(t) = \sum_{u=1}^{t} Score_{i,\sigma(u)}\cdot TC_{\sigma(u)}, \qquad
\text{select clusters } \sigma(1..t^\*) \text{ where } t^\* = \min\{t : Acc_i(t) > Th_{wics,i}\}. \tag{3}
$$
The tokens inside the selected clusters are the retrieved KV set. Because $t^\*$ depends on the actual score distribution, **each layer and head fetches a different number of tokens** — the adaptivity that fixed top-k lacks. The sort supports **early-exit**: it stops as soon as the running sum exceeds $Th_{wics,i}$, so the full ordering is never needed.

**Execution stage.** With the selected clusters, the model does *light attention* — attention restricted to the retrieved keys/values only — cutting both the memory fetched over PCIe and the attention FLOPs.

*(Hardware, summarized only: a Dynamic Retrieval Engine adds a Hash-bit Cluster Unit — parallel XOR/popcount accumulators — and a WiCSum Threshold Unit with early-exit bucket sorters, plus a KV management unit that reorders tokens by cluster for contiguous PCIe transfers. It occupies ~2% area / ~2.4% power of the accelerator. Details out of scope for this note.)*

## Explicit design choices
- **Retrieval, not eviction:** the full KV cache is offloaded to CPU/storage; ReSV selects and fetches the relevant subset each frame rather than permanently discarding tokens — nothing is lost, so accuracy stays high while resident memory is bounded.
- **Cluster on keys via LSH bit codes** (random-hyperplane hashing, binarized), so grouping is XOR/Hamming — no distance matrix, code length $\leq0.5\%$ of $d$.
- **Score cluster representatives, not tokens:** one dot product per cluster instead of per token; representative = mean key of the cluster.
- **Weight by token count $TC_j$** so a query's attention mass, not just cluster count, drives selection.
- **Relative (fraction-of-mass) threshold** $Th_{r\text{-}wics}$ instead of absolute $k$ → single hyperparameter, automatically per-layer/per-head adaptive.
- **Early-exit sorted accumulation** — stop at the first cluster that crosses the threshold.
- **Layer-pipelined prefetch:** predict layer $\ell{+}1$'s tokens during layer $\ell$'s execution to hide CPU→accelerator fetch latency.
- **Training-free:** operates on a frozen model (Llama-3-8B backbone, SigLIP-ViT-L-384 vision encoder in their setup); no fine-tuning.

## Key results / what to remember
Setting: streaming video LLM = Llama-3-8B + SigLIP-ViT-L-384, evaluated on the **COIN** benchmark (avg 26 frames, 25 question / 39 answer tokens); accelerator RTL at 14nm, 0.8V/800MHz; DRAMSim3 + MQSim; baselines AGX Orin and A100 with FlexGen/ReKV/InfiniGen/Oaken. Verified against the paper's tables:
- **Accuracy:** ReSV loses only **−0.8%** avg accuracy on COIN, vs **−2.0%** for ReKV and **−3.4%** for InfiniGen-prefill, while retrieving far fewer tokens.
- **Retrieval ratio (adaptivity):** ReSV fetches ≈**25–36%** of tokens in the frame/prefill stage and ≈**1.4–2.9%** in generation, vs ~50%/~50% for fixed top-k baselines — about **3.0× fewer tokens** than fixed top-k ReKV at matched accuracy; per-layer/head retrieval spans **~4.2% to ~44%** (Fig. 20).
- **Latency:** end-to-end up to **5.4×** lower at ≥40K caches. Per-frame at 40K cache / batch 1: V-Rex8 (edge) ≈254 ms → 3.9 FPS, **2.2–7.3×** over AGX Orin + FlexGen; V-Rex48 (server) **2.6–7.3×** over A100 baseline (up to **19.7×** at batch 8). Text-gen TPOT: V-Rex8 89–97 ms (1.9–15.1×), V-Rex48 14–15 ms (2.8–16.8×).
- **Energy:** frame processing **5.5–10.2×** (V-Rex8 vs AGX, 1K–40K) and **9.0–29.7×** (V-Rex48 vs A100); text gen up to **18.5×** / **70.6×**.
- **Ablation (40K, batch 1):** ReSV without clustering 1.6× speedup (−0.3% acc); with clustering 9.4× speedup (−0.8% acc); +KVPU hardware 6.0× speedup / 9.2× energy; full V-Rex8 8.1× speedup / 10.2× energy.
- **Roofline (40K, batch 4):** V-Rex8 reaches **71.5%** of peak throughput vs 15% (AGX+ReKV) / 6.6% (AGX+FlexGen). Where Oaken 4-bit KV OOMs beyond ~20K tokens, V-Rex sustains ~7 FPS.

No Zotero highlights present.

**Takeaways:** (1) In streaming video LLMs the bottleneck is *prefill-side KV retrieval*, not decode — algorithms tuned for text decode transfer poorly. (2) Video's spatial-temporal redundancy makes cheap LSH clustering of keys a good proxy for attention importance. (3) Replacing fixed top-k with a *fraction-of-attention-mass* threshold gives free per-layer/per-head adaptivity and the paper's headline token savings — a portable idea even without the custom silicon.

## How it connects (evolution)
- [[rekv]] — the training-free KV-retrieval-for-streaming baseline ReSV directly improves on (fixed top-k → adaptive WiCSum; 3× fewer tokens at matched accuracy).
- [[infinipot-v]] — training-free KV compression/eviction for streaming video under a fixed memory budget; contrast eviction vs ReSV's retrieve-don't-discard.
- [[hermes-kv]] — KV-cache management for streaming/long video; sibling in the streaming-memory KV line.
- [[streamkv]] — KV-cache-centric streaming approach; adjacent method for which tokens to keep/fetch.
- [[flash-vstream]] — memory-architecture design for real-time long video streaming (compressed memory), a different resident-state strategy.
- [[streaming-memory]] — the subtopic hub tying these KV/memory methods together.

## Open questions / limitations
- Results are simulator-based (DRAMSim3/MQSim + RTL synthesis), not silicon; real-system fetch/PCIe behavior may differ from the modeled ideal-prefetch pipeline.
- Evaluated mainly on COIN with one backbone (Llama-3-8B + SigLIP); generality to longer streams, other benchmarks (e.g. StreamingBench/OVO-Bench-style proactive tasks), and other model families is unshown.
- The relative threshold $Th_{r\text{-}wics}$ and hash params ($N_{hp}$, $T_{hp}$) are global hyperparameters; sensitivity and any per-stream tuning cost aren't fully characterized.
- LSH clustering assumes high frame-to-frame key similarity — behavior under scene cuts / fast motion (low redundancy) is a plausible failure mode not deeply stress-tested.

*Verification: equations (1)–(3), the hash-bit/WiCSum mechanism, accuracy (−0.8% vs −2.0%/−3.4%), retrieval ratios (25–36% / 1.4–2.9%, 3.0× fewer than top-k), speedups (per-frame 2.2–7.3×, end-to-end up to 5.4×), energy, and roofline (71.5%) numbers cross-checked against the arXiv HTML (2512.12284v3) method text and results/ablation tables; crux figure is the rendered Fig. 6 from the PDF.*

---
zotero_key: null
authors: Zhenyu Ning, Guangda Liu, Qihao Jin, Chengwei Li, Wenchao Ding, Minyi Guo, Jieru Zhao (Shanghai Jiao Tong University; Fudan University; Guizhou University) — DAC '26
year: 2026
arxiv: 2505.15269
pdf: https://arxiv.org/pdf/2505.15269
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---

# LiveVLM: Efficient Online Video Understanding via Streaming-Oriented KV Cache and Retrieval

**Lineage role:** A training-free, query-agnostic KV-cache framework for streaming video LLMs that pairs an attention-driven **compression** step (Vision Sink Bucketing) with a positional-decoupled **retrieval** step (Position-agnostic KV Retrieval) — sitting at the compression-plus-retrieval end of the streaming-memory design axis, next to [[rekv]], [[streammem]] and [[infinipot-v]].

## Problem — what was limited before this paper (short)
In online/streaming video LLMs the KV cache grows **linearly with video duration**, so memory and per-query latency blow up during hour-long streams. Prior fixes each pay a price: **query-dependent** compression ([[streammem]]-style) keeps only tokens relevant to the *current* query, so it prematurely discards vision tokens future queries will need; **query-agnostic** pruning avoids that but still runs an $O(n^2)$ prefill over the whole retained cache at response time, forcing overly aggressive compression; **CPU offloading** ([[rekv]]) keeps the full cache but pays severe transfer latency, impractical for real-time. LiveVLM targets all three of quality, memory, and speed at once, training-free.

## Key idea — the core insight, 2-4 sentences
Vision-to-vision attention (not text-to-vision) is what reveals **sink vision tokens** — tokens that keep attracting high attention from later tokens — and preserving those under a fixed budget keeps model behavior intact during streaming compression. LiveVLM compresses on the fly during the **encoding phase** with **Vision Sink Bucketing (VSB)**, which balances high-attention tokens against even contextual coverage inside a fixed-size cache. Then at the **response phase**, instead of attending over the whole retained cache, **Position-agnostic KV Retrieval (PaR)** removes positional embeddings before scoring so page-level mean-key tensors become meaningful, retrieves only the query-relevant pages (~40%), and re-inserts positions before the LLM computes — cutting response cost from ~$O(n^2)$ toward ~$O(n)$.

![[livevlm.png]]
> **Crux (Figure 1).** The streaming pipeline: sampled video clips are encoded/projected and streamed into the Video LLM under a FIFO-updated **KV Cache** split into short-term memory (recent) and long-term memory, with **VSB** compressing incoming KVs and **PaR** retrieving query-relevant KVs at response time; insets show it beating [[rekv]]/[[streammem]] on VideoMME/LongVideoBench accuracy while keeping latency (TTFT) and GPU memory nearly flat as frame count grows. *Ning et al. (2026), arXiv:2505.15269. Embedded for personal research reference.*

## Method + math — the mechanism, then the central objective/equations IN FULL

**Setup.** The base model is LLaVA-OneVision-Qwen2-7B (frozen; training-free). Frames are sampled into clips, encoded by the visual encoder, projected to vision tokens, and appended to the Video LLM's KV cache. A **frame filter** and **FIFO update** move KVs between a **short-term memory** (recent frames) and **long-term memory** (compressed history). Two phases: *encoding* (ingest stream, compress) and *response* (a user query arrives, retrieve + answer).

**Vision Sink Bucketing (VSB) — compression during encoding.**
Because the model uses FlashAttention, full attention scores aren't materialized, so VSB approximates importance from only the **last $r$ vision tokens** as an observation window ($r \ll L$, where $L$ is the context length). Let $\mathbf{Q}_i$ be the query tensor of the $i$-th vision token and $\mathbf{K}\in\mathbb{R}^{L\times d}$ all cached key tensors. The importance score is a two-step pooled attention:

$$
\begin{aligned}
\mathbf{W} &= \mathrm{softmax}\!\left(\{\mathbf{Q}_i\}_{i=L-r}^{L}\cdot \mathbf{K}\right), \qquad \mathbf{W}_i\in\mathbb{R}^{r\times L}\\
\mathbf{S} &= \mathrm{MeanPool}\!\left(\{\mathbf{W}_i\}_{i=1}^{r}\right), \qquad \mathbf{S}\in\mathbb{R}^{L}
\end{aligned}
$$

$\mathbf{S}$ gives one importance value per cached vision token; the extra compute/memory is **<1%** of full attention. Crucially the scores are **vision-to-vision** (queries are vision tokens), because text tokens fail to surface sink tokens and would corrupt subsequent attention (Fig. 2–3).

Naive top-$M$ selection on $\mathbf{S}$ over-retains *local* sink tokens (high attention from immediate neighbors but not from later frames), wasting budget and blocking new critical tokens. VSB instead **balances score-mass against context coverage** by bucketing. Partition the context into $N$ position-ordered buckets of equal capacity $B$, with $N\times B = M$ (the cache budget). A token at position $p$ in a length-$L$ context maps to bucket $\lceil p/(L/N)\rceil$ (e.g. token 256 of 1000 with $N{=}10 \to$ bucket 3). Populate in two phases:
1. **Phase 1 (score):** greedily assign the **top-$R$** tokens (highest $\mathbf{S}$, $R$ = retention ratio) to their *positional* buckets — guarantees the most informative tokens survive.
2. **Phase 2 (coverage):** traverse remaining tokens in descending $\mathbf{S}$; keep a token **iff its target bucket still has capacity**. Stops once $M$ tokens are selected.

Concatenate the retained tokens (with their KVs) per bucket into the compressed cache. Retention-ratio analysis (Fig. 4) shows VSB keeps a **higher fraction of answer-relevant tokens than TopK across all layers, especially shallow ones**.

**Position-agnostic KV Retrieval (PaR) — retrieval during response.**
Query-agnostic caches still hold much query-irrelevant content, so PaR retrieves only relevant KVs before the full attention. But token-discarding compression leaves **discontinuous positions** among retained tokens, so their RoPE-augmented keys are dissimilar within a page — making page-level mean-key retrieval fail (Fig. 6; Table 1). Fix: **decouple retrieval from positional embeddings.**
1. Encode the specific question with the foundation model to get query tensors.
2. **Remove positional embeddings** from cached key tensors → position-agnostic keys (high mutual similarity within a page).
3. Partition into pages of size $C$; take the **mean key tensor** per page as its representative.
4. Concatenate mean keys and compute **simplified (pooled) attention** vs. the query tensors → per-page relevance score.
5. **Retrieve the top pages** (about 40% of the cache) by pooled score.
6. **Re-insert positional embeddings** into the retrieved keys before the Video LLM's full attention, restoring the positional info the model reasons over.

Retrieval therefore depends on **intrinsic semantic similarity**, not on (broken) positions, and the LLM's response-phase full-attention runs over a much smaller retrieved cache — the source of the ~$O(n^2)\to O(n)$ response saving.

## Explicit design choices
- **Training-free / query-agnostic:** no fine-tuning; drops onto an existing Video LLM. Base = **LLaVA-OneVision-Qwen2-7B**, **FP16** mixed precision.
- **Cache budget $M = 12\text{k}$ tokens**; **bucket capacity $B = 1$** (i.e. $N = M$ single-token positional buckets, maximal even spread).
- **Observation window $r \ll L$** (only the last $r$ vision tokens' queries score importance; <1% overhead).
- **Importance metric = vision-to-vision attention**, not text-to-vision (the paper's core empirical claim, Fig. 2–3).
- **VSB two-phase fill:** Phase-1 top-$R$ by score into positional buckets, Phase-2 descending-score fill of remaining capacity — trades pure top-$M$ for score+coverage balance.
- **PaR page size $C = 16$**; retrieve **40% of the cache** (Table 6 optimum); strip RoPE before scoring, restore RoPE after selection.
- **KV cache split** into FIFO-updated short-term (recent) + long-term (compressed) memory.
- **Hardware target:** single NVIDIA 4090D, **24 GB**, no CPU offloading (deliberately, for real-time fairness vs. offloading baselines).

## Key results / what to remember
No Zotero highlights present.

- **StreamingBench (Table 4, accuracy %):** LiveVLM-7B **overall 63.10**, beating **GPT-4o 62.50** and **Claude 3.5 Sonnet 59.08**, and the open-source online baselines **ReKV-7B 57.20** and **Dispider-7B 55.65**; +**4.25 pp** over its own foundation LLaVA-OneVision-7B (58.85 overall). Real-Time Visual Understanding subscores e.g. OP 79.84, CR 79.69, CS 84.86.
- **Offline benchmarks (Table 2, accuracy):** on LLaVA-OneVision-7B, LiveVLM gives **VideoMME(all) 59.6** (vs ReKV 58.3, StreamMem 59.4), **VideoMME-Long 51.3** (best; StreamMem 50.1), **VideoMME-Medium 57.0** (best), **LongVideoBench 56.1** (best among online), **MLVU 68.1** (ReKV 68.2 is marginally higher). Reported as **+3.4 pp on MLVU** and **+5.1 pp on VideoMME-Long** over the foundation model.
- **Online VideoQA — RVS-Ego / RVS-Movie (Table 3), 24 GB, no offloading:** LiveVLM **avg acc 55.6 / score 3.8** (RVS-Ego 57.8/3.9, RVS-Movie 53.4/3.6), best among no-offload methods (StreamMem 55.2/3.6, InfiniPot-V 54.6/3.5, Flash-VStream 55.0/3.6, ReKV-w/o-offload 53.3/3.4). ReKV **with** CPU offloading scores higher (59.0/3.8) but its offloading latency is impractical for real-time.
- **Efficiency (Fig. 8), 256 frames:** peak GPU memory **3.02× lower than Dispider** and **1.19× lower than ReKV**; response latency (TTFT) **1.73× faster than ReKV**. LiveVLM's memory and latency stay roughly flat as frames grow to $10^3$, while Dispider's surge.
- **Ablation (Table 5, MLVU-all):** baseline 64.7 → +VSB 66.2 (+1.5) → +VSB+PaR **68.1** (+3.4 total); per-subtask PaR gains +2.6/+4.1/+1.7.
- **Retrieval ratio (Table 6, MLVU-all):** 0.2→66.4, **0.4→68.1 (optimum)**, 0.6→66.7, 0.8→66.2, 1.0→66.2 — so retrieving the *full* cache is worse than a selective 40%, confirming heavy redundancy.
- **Positional-decoupling check (Table 1, MLVU-all):** retrieval **without** positions 68.1 vs **with** positions 64.8 — a 3.3 pp gap that justifies PaR's design.

## How it connects (evolution)
- [[rekv]] — the CPU-offload + in-context retrieval baseline LiveVLM most directly argues against (offloading latency); PaR is the on-GPU retrieval alternative.
- [[streammem]] — query-dependent memory compression LiveVLM contrasts with (premature discarding vs. query-agnostic VSB).
- [[infinipot-v]] — constrained-budget query-agnostic compression (temporal-axis redundancy, value-norm selection); same design family, different compression rule.
- [[flash-vstream]] — memory-augmented streaming architecture used as a baseline on RVS-Ego/Movie.
- [[dispider]] — disentangled streaming perception/reasoning; LiveVLM's efficiency baseline (memory/latency surge comparison).
- [[streamingbench]] — the real-time streaming benchmark where LiveVLM tops GPT-4o/Claude 3.5.
- [[streamkv]] — sibling streaming KV-cache/retrieval approach in this sub-topic.

## Open questions / limitations
- **Tuned to LLaVA-OneVision-7B**: budget $M{=}12$k, page $C{=}16$, ratio 0.4 are chosen for one 7B model on 24 GB; generality to other Video LLMs / larger contexts is asserted (training-free) but not swept.
- **Fixed 40% retrieval ratio** is a global constant; per-query adaptivity (some questions need little history, some need lots) is unexplored — the 1.0-ratio degradation hints a smarter budget could help.
- **Importance from only the last $r$ tokens** could miss a sink that stops attracting attention within the window but matters much later; robustness across very long streams (beyond $10^3$ frames) isn't stress-tested for accuracy, only for memory/latency.
- **Compression is still lossy and query-agnostic**: VSB fixes what to keep before the question is known, so a query needing an evicted detail is unrecoverable — PaR only re-ranks what VSB already retained.

*Verification: equations (VSB $\mathbf{W},\mathbf{S}$), design constants ($M{=}12$k, $B{=}1$, $C{=}16$, 40% retrieval), and all headline numbers checked against the rendered arXiv v2 PDF pages — Fig. 1 (p.1), method Secs. 3–4 (pp.3–4), Tables 1–6 and Fig. 8 (pp.4–6); cross-read with the arXiv HTML abstract/method. GitHub: github.com/sjtu-zhao-lab/LiveVLM (not fetched).*

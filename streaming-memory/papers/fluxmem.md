---
zotero_key: null
authors: Yiweng Xie, Bo He, Junke Wang et al. (Fudan University / Shanghai Innovation Institute / UMD College Park)
year: 2026
arxiv: 2603.02096
pdf: https://arxiv.org/pdf/2603.02096
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding

**Lineage role:** A *training-free, plug-and-play* streaming memory that cascades visual tokens through short→mid→long tiers, compressing on overflow with two data-driven (Otsu-thresholded) reducers — temporally-variant selection then spatial merging — while reusing the same statistics as a zero-cost proactive-response trigger (CVPR 2026).

## Problem — what was limited before this paper (short)
A streaming video LLM must ingest frames forever without the KV/token budget exploding, but naive per-frame token retention grows linearly and blows up latency and GPU memory. Prior compression schemes either use fixed, manually-tuned drop ratios / similarity thresholds that fail across the wildly variable motion of streaming video (over-drop in static scenes, under-drop in busy ones), or require training/fine-tuning to learn a memory. There was no cheap, causal, single-pass memory that adapts its retention strength to the scene *and* comes for free on top of an off-the-shelf VLM.

## Key idea — the core insight, 2-4 sentences
FluxMem maintains three cascaded memory tiers and only compresses *on overflow*: when short-term memory fills, **Temporal Adjacency Selection (TAS)** keeps the temporally *variant* tokens (those that changed vs neighbors) and passes them to mid-term; when mid-term fills, **Spatial Domain Consolidation (SDC)** merges spatially redundant regions into mean "anchors" for long-term. Every threshold is set automatically per-frame by **Otsu's method** on the token-similarity distribution, so retention self-calibrates to scene dynamics with no tuning. Crucially, the backward-similarity score TAS already computes on token entry doubles as a **zero-overhead scene-change trigger** for proactive LLM output.

![[fluxmem.png]]
> **Crux (Figure 2).** The short→mid→long cascade: each encoded frame's tokens $v_{t,h,w}$ enter short-term memory; on overflow TAS retains temporally-variant tokens into mid-term, on further overflow SDC merges spatially-redundant tokens into long-term anchors, and the same TAS statistics fire a Trigger for active LLM responses. *Xie et al. (2026), arXiv:2603.02096. Embedded for personal research reference.*

## Method + math — the mechanism, then the central objective/equations IN FULL

**Streaming hierarchical memory.** Each incoming frame $F_t$ is encoded to a grid of visual tokens $v_{t,h,w}$ ($H\times W$ spatial positions) and written into a **short-term** buffer. The buffer holds recent full-resolution frames; when it exceeds capacity the oldest frame is *evicted through* TAS into **mid-term** memory; when mid-term exceeds its capacity $c_m$ its content is consolidated through SDC into **long-term** memory. At query time the LLM attends over the concatenation of all three tiers (long ⊗ mid ⊗ short in the figure). The whole procedure is strictly **causal, single-pass, and $\mathcal{O}(HW)$ per overflow event** — no extra heads, no backward pass.

**Temporal Adjacency Selection (TAS) — keep what changed.** For a token at position $(h,w)$ in frame $t$, compute its minimum feature distance to a local $3\times3$ window in the *previous* and *next* frames (local windows absorb small motion so only genuine change survives):

$$
\begin{aligned}
s_{t,h,w}^{-} &= \min_{(i,j)\in \mathcal{N}_{3\times3}(h,w)} d\!\left(v_{t,h,w},\, v_{t-1,i,j}\right) \\
s_{t,h,w}^{+} &= \min_{(i,j)\in \mathcal{N}_{3\times3}(h,w)} d\!\left(v_{t,h,w},\, v_{t+1,i,j}\right)
\end{aligned}
$$

A token is **retained** (written to mid-term) iff it is temporally variant on *either* side — high min-distance means "not explained by a neighbor," i.e. new content:

$$
\text{retain}(t,h,w) \iff \left(s_{t,h,w}^{-} > \Theta_t^{-}\right) \ \lor\ \left(s_{t,h,w}^{+} > \Theta_t^{+}\right)
$$

Static background (low distance both sides) is dropped; moving/appearing content is kept.

**Spatial Domain Consolidation (SDC) — merge what is redundant.** Operating *only* on the TAS-retained token set (so the graph is already sparse), for each retained token examine the other retained tokens in its original $3\times3$ spatial neighborhood and connect them if their pairwise distance $\le \Theta_t$, forming a sparse **8-connected graph**. A **union-find** pass (near-linear time) yields connected components $\{\mathcal{C}_{t,k}\}_k$; each locally-coherent component is replaced by its single **mean anchor**:

$$
a_{t,k} = \frac{1}{|\mathcal{C}_{t,k}|} \sum_{(i,j)\in \mathcal{C}_{t,k}} v_{t,i,j}
$$

The anchor set $\mathcal{A}_t=\{a_{t,k}\}_k$ is appended to long-term memory, removing spatial redundancy while preserving one representative per region.

**Proactive Response Triggering — a free trigger.** Reusing the backward scores $s_{t,h,w}^{-}$ that TAS already computed on token entry, define the fraction of the frame that changed:

$$
r_t^{-} = \frac{1}{HW}\sum_{h,w} \mathbf{1}\!\left[s_{t,h,w}^{-} > \Theta_t^{-}\right]
$$

A **scene switch** (and hence a moment worth responding) is declared when $r_t^{-} > \gamma$, with $\gamma\in[0,1]$ a single tunable sensitivity. This adds no new computation — it is a byproduct of the memory write.

**Adaptive thresholding (Otsu).** Every $\Theta$ above ($\Theta_t^{-},\Theta_t^{+}$ for TAS, $\Theta_t$ for SDC) is set automatically per-frame by **Otsu's method** on the empirical distribution of that frame's similarity/distance scores — the classic non-parametric threshold that splits scores into keep/drop by maximizing inter-class variance:

$$
\Theta_t = \arg\max_{\theta} \Big[\, \omega_1(\theta)\,\omega_2(\theta)\,\big(\mu_1(\theta)-\mu_2(\theta)\big)^2 \,\Big]
$$

where $\omega_1,\omega_2$ are the mass and $\mu_1,\mu_2$ the means of the two groups below/above $\theta$. This makes retention strength data-driven: static scenes drop aggressively, high-motion scenes retain more, with no manual drop-ratio.

## Explicit design choices — concrete decisions (raw material for new systems)
- **Three cascaded tiers, compress-on-overflow only** — short (recent, full-res) → mid (TAS-filtered) → long (SDC anchors); no work done until a tier is full.
- **Two orthogonal reducers**: TAS removes *temporal* redundancy (drop static tokens between frames), SDC removes *spatial* redundancy (merge coherent regions within a frame). Temporal filter runs first, so SDC's graph is over a pre-filtered, sparse set.
- **Local $3\times3$ windows** for both temporal matching and spatial linking — tolerates small motion / registration error instead of strict pixel-aligned comparison.
- **Union-find over an 8-connected graph** to get connected components in near-linear time; each replaced by a **mean anchor** (barycenter), not a learned token.
- **Otsu, not a hyperparameter**: every threshold is recomputed per frame from the score histogram — the one design move that makes it robust across low/high motion without tuning.
- **Trigger = reused statistic**: proactive-response detection piggybacks on TAS's backward scores → genuinely zero-overhead active output; single knob $\gamma$.
- **Training-free & plug-and-play** on a frozen VLM (**Qwen2.5-VL-7B**), 1 fps; an *optional* SFT variant squeezes out more (see Table 5).
- Strictly **causal, single-pass, $\mathcal{O}(HW)$** per overflow — suitable for unbounded streams.

## Key results / what to remember — exact headline numbers WITH setting
Base VLM: **Qwen2.5-VL-7B @ 1 fps**; "†" baseline = same model without FluxMem.

- **StreamingBench (real-time subtasks): 76.4** for FluxMem vs **73.9** baseline (**+2.5**).
- **OVO-Bench (real-time subtasks): 67.2** vs **63.3** baseline (**+3.9**).
- **OVO-Bench (overall): 53.3** (**+3.5** vs baseline).
- **MLVU: 73.1** (**+5.2**) — the largest offline-benchmark gain.
- **Video-MME (w/o subtitles): 65.3** (**+2.0**).
- **LongVideoBench: 61.1** (**+0.4**) — smallest gain.
- **Efficiency (Table 3)**: on OVO-Bench, latency **812 ms (↓69.9%)** and peak memory **23.5 GB (↓34.5%)** while *improving* accuracy +3.5; on MLVU, latency **2014 ms (↓44.3%)**, peak memory **28.4 GB (↓31.2%)**.
- **Memory-tier ablation (Table 4)**: S-only 67.8 / M-only 69.9 / L-only 70.9 / **S+M+L full 73.1** on MLVU, with the full config at **64.3% token reduction** — the full cascade wins on accuracy while keeping heavy compression (L-only drops 85.1% of tokens but loses accuracy).
- **Training-free vs SFT (Table 5)**: training-free FluxMem = 53.3 OVO-overall / 76.4 StreamingBench; adding **SFT → 61.4 (+11.6)** OVO-overall and **76.7 (+2.8)** StreamingBench.
- **Robustness to drop ratio (Fig 3a)**: sustains **73.1** on MLVU at 64% drop and still **~70.1** even at ~85% aggressive compression.

No Zotero highlights present.

Takeaways: (1) redundancy in streaming video is both temporal *and* spatial — attack them with two cheap, separate passes rather than one learned compressor; (2) the win is making thresholds *data-driven per frame* (Otsu) instead of a global drop ratio; (3) you can get proactive triggering for free by reusing memory-write statistics; (4) the whole thing is a training-free wrapper, so gains transfer to any grid-token VLM, and light SFT compounds them.

## How it connects (evolution)
- [[streammem]], [[streamkv]], [[hermes-kv]], [[streamforest]] — other streaming *memory / KV-compression* schemes for online video LLMs; FluxMem's tiered, Otsu-adaptive, training-free variant sits alongside these.
- [[flash-vstream]], [[rekv]], [[infinipot-v]] — hierarchical / long-context memory for unbounded video; same "compress-on-overflow across tiers" lineage.
- [[dispider]], [[mmduet]] — proactive / active-response streaming models; FluxMem contributes a zero-cost trigger reusing its own TAS statistics.
- [[streamingbench]], [[ovo-bench]] — the two real-time benchmarks FluxMem is primarily evaluated on.
- [[streaming-memory]] — the sub-topic hub.

## Open questions / limitations
- Mean-anchor SDC replaces a region with its barycenter — lossy for scenes where fine within-region spatial detail (small text, faces in a crowd) matters; OCR-heavy long-term recall could degrade.
- Otsu assumes a **bimodal** keep/drop score distribution; in scenes with uniformly high or uniformly low motion the two-class split may be ill-defined, and the paper does not stress-test that failure mode.
- Evaluated only on Qwen2.5-VL-7B at 1 fps — transfer to other VLMs, higher fps, or much longer (hour-scale) streams where long-term memory itself eventually saturates is unverified.
- The proactive trigger's single $\gamma$ and its precision/recall for *when* to speak are not quantified against dedicated proactive benchmarks in the numbers surfaced here.

*Verification: equations (Eq 1–4, Otsu) and all numbers checked against the paper's arXiv HTML (2603.02096) Sections 3.2–3.3 and Tables 1–5, and Figure 2 read from the rendered PDF page. Zotero not running (skipped). Numbers not independently confirmable beyond the paper's own tables are reported as stated.*

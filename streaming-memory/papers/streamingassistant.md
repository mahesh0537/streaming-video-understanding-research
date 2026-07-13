---
zotero_key: null
authors: Xinqi Jin, Hanxun Yu, Bohan Yu, Kebin Liu, Jian Liu, Keda Tao, Yixuan Pei, Huan Wang, Fan Dang, Jiangchuan Liu, Weiqiang Wang
year: 2025
arxiv: 2512.12560
pdf: https://arxiv.org/pdf/2512.12560
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# StreamingAssistant: Efficient Visual Token Pruning for Accelerating Online Video Understanding

**Lineage role:** A training-free *visual token pruning* front-end for online/streaming video MLLMs — it shrinks the per-frame visual token set (not the KV-cache directly) before tokens enter the streaming buffer, combining an existing temporal-redundancy drop with a novel *spatially-aware* spatial-redundancy metric, so the buffer that feeds the LLM stays small under continuous 1 fps ingestion.

## Problem — what was limited before this paper (short)
Online video understanding (surveillance, AI glasses) streams frames indefinitely, so an MLLM accumulates an enormous number of visual tokens, driving up GPU memory and per-step latency. Prior token-reduction work is one-sided or costly: temporal-only methods (e.g. TimeChat-Online's DTD) drop redundant frames but leave intra-frame spatial redundancy; spatial methods that rely on vision-encoder attention (CLS-token weights) are unreliable (the CLS attention reflects a classification objective, not the user's task) and are hard to compute under FlashAttention; similarity-based spatial methods (ToMe, FOLDER, DivPrune) ignore *position* — they can prune tokens that are visually similar but spatially far apart, blurring the positional signal the LLM's rotary embedding depends on. The goal: a cheap, position-aware pruner that removes both temporal and spatial redundancy with sub-millisecond overhead.

## Key idea — the core insight, 2-4 sentences
A token carries little new information only if it is redundant with a *spatially adjacent* neighbor, not just any similar token anywhere in the frame. StreamingAssistant therefore scores each token by its **Maximum Similarity to Spatially Adjacent Video Tokens (MSSAVT)** — cosine similarity to its 4-connected neighbors — and prunes high-redundancy tokens. To make this a single parallel pass (rather than an expensive iterative loop that re-updates redundancies), it uses a **checkerboard masked-pruning** strategy that only ever considers non-adjacent candidate tokens, guaranteeing two mutually-adjacent tokens are never both dropped. Temporal redundancy is handled first via the DTD algorithm borrowed from TimeChat-Online, and a token survives only if it is neither temporally nor spatially redundant.

![[streamingassistant.png]]
> **Crux (Figure 2).** Bottom: the streaming workflow — frames are encoded, passed through temporal pruning (Step 2) then spatial pruning (Step 3), the survivors buffered, and only at query time (Step 4) are text + buffered visual tokens fed to the LLM. Top: the MSSAVT spatial-pruning recipe (similarity → spatial-redundancy → checkerboard masking → parallel pruning) contrasted with the sub-optimal naive-parallel (unreliable) and sequential (slow) alternatives. *Jin et al. (2025), arXiv:2512.12560. Embedded for personal research reference.*

## Method + math — mechanism then the central equations
**System workflow.** Two dataflows run over the stream. An *always-on* dataflow fires at every timestamp $n\Delta t$: the incoming frame is turned into a video-token matrix $V^n \in \mathbb{R}^{W\times H\times D}$ by the vision encoder + projector (Step 1), then reduced by temporal pruning (Step 2) and spatial pruning (Step 3), and the survivors are appended to a **buffer**. A *query-triggered* dataflow fires only when the user asks at time $T$ (Step 4): the query is embedded and the text tokens are concatenated with all buffered video tokens and passed to the LLM to produce the answer. So compression happens continuously and cheaply; the expensive LLM forward pass happens only on demand.

**Temporal pruning (Step 2, adopted).** Uses the DTD (Dynamic Token Dropping) algorithm from [[timechat-online]]: compare $V^n$ against the previous frame $V^{n-1}$ to produce a boolean temporal dropping mask $M^n_t \in \{\text{True},\text{False}\}^{W\times H}$ (True = drop), with threshold $\tau_t$.

**Spatial redundancy = MSSAVT (Step 3, the contribution).** For the token $V^n[i,j]\in\mathbb{R}^d$ at row $i$, column $j$ of frame $n$, its spatial redundancy is the max cosine similarity to its 4-connected neighbors:

$$
R^n_s[i,j] = \max_{(\delta_i,\delta_j)\in\{(-1,0),(1,0),(0,-1),(0,1)\}} \Big\{ \mathrm{Sim}\big(V^n[i,j],\, V^n[i+\delta_i,\, j+\delta_j]\big) \Big\}
$$

where $\mathrm{Sim}(\cdot,\cdot)$ is cosine similarity. Restricting $(\delta_i,\delta_j)$ to immediate neighbors is the whole point: it keeps redundancy tied to spatial locality, unlike global similarity methods that ignore position.

**Why not the obvious pruners.** A *naive parallel* pruner drops every token whose $R_s>\tau_s$ in one shot — but two mutually-adjacent tokens can each name the other as their redundant neighbor and both get dropped (over-pruning; "unreliable"). A *sequential* pruner drops one token then re-computes redundancies in a loop — correct but slow. StreamingAssistant's **masked pruning (Sec. 3.3)** gets parallel speed with sequential safety.

**Masked pruning.** A fixed checkerboard candidate mask selects a non-adjacent half of the grid:

$$
M_p[i,j] = \begin{cases} \text{True} & (i+j)\bmod 2 = 1\\ \text{False} & \text{otherwise}\end{cases}
$$

Only masked-in tokens are eligible to be dropped, and since no two checkerboard cells are 4-adjacent, an adjacent pair can never both be removed. The spatial dropping mask is the conjunction of "redundant enough" and "eligible":

$$
M^n_s = (R^n_s > \tau_s)\ \&\ M_p
$$

**Final decision.** A token is dropped if it is temporally *or* spatially redundant — the final dropping mask is the union:

$$
M^n = M^n_t \ |\ M^n_s
$$

i.e. a token is retained only if it survives both. Pruning is a pure similarity/threshold/masking computation with no attention scores and no iteration, giving the reported sub-millisecond latency.

## Explicit design choices
- **Backbone / setting:** TimeChat-Online-7B (a fine-tuned Qwen2.5-VL-7B) as the streaming MLLM; video ingested at **1 fps**; training-free (pruning is a plug-in, no re-training).
- **Redundancy signal:** token-to-token cosine similarity in the projected visual embedding space, *not* vision-encoder CLS attention (rejected as objective-mismatched and FlashAttention-incompatible) and *not* global similarity (rejected for discarding position).
- **Neighborhood:** 4-connected (up/down/left/right) max-similarity; deliberately local to preserve positional structure the LLM's rotary position embedding relies on.
- **Parallelism guarantee:** fixed checkerboard candidate mask $M_p$ so pruning is one parallel pass yet never drops a mutually-adjacent pair — trades the sequential loop's cost for a static mask.
- **Order of operations:** temporal pruning (DTD) first, then spatial pruning, combined by mask union $M^n_t | M^n_s$.
- **Thresholds:** temporal $\tau_t=0.2$; spatial $\tau_s\in\{0.2,\,0.5\}$ tuning the dropping ratio.
- **Buffer semantics:** survivors accumulate in a buffer; the LLM only runs at query time over text + buffered tokens (decouples continuous compression from on-demand inference).
- **Variant:** StreamingAssistant-IA (an "iterative/adaptive" variant) is reported for latency comparison — higher but still bounded cost (~23.6 ms max) vs. the main method's <1 ms.

## Key results / what to remember
Operating point is a *very high* token dropping ratio (~92–95%), i.e. keep only ~5–8% of visual tokens.

- **Average over all benchmarks (Table 1):** no-dropping baseline **59.83** (100%). StreamingAssistant at **92.5% dropping → 57.35** (95.85% relative) and at **93.5% dropping → 57.00** (95.27% relative), versus DTD at 92.3% dropping → **56.17** (93.88% relative). So at comparable/higher dropping, StreamingAssistant retains ~2 relative-points more than the temporal-only DTD.
- **StreamingBench, real-time visual understanding (Table 2):** **70.24** at 91.5% dropping and **69.16** at 92.8% dropping, vs. DivPrune **68.96 / 68.48** — StreamingAssistant ahead by ~0.7–1.3 points.
- **OVO-Bench (Table 3):** **43.49** at 93.0% dropping / **43.48** at 94.0% dropping, vs. DivPrune **43.90 / 43.89** — here DivPrune is marginally ahead (~0.4 pt); roughly a tie.
- **VideoMME (Table 4):** overall **61.6** at 92.5% dropping (**53.1** on long videos) and **61.3** at 93.6% dropping (**53.0** long).
- **LongVideoBench (Table 5):** **56.2** at 93.6% dropping and **56.2** at 94.7% dropping, vs. DivPrune **55.7 / 54.3** — gap widens to ~1.9 pt as the dropping ratio climbs, i.e. StreamingAssistant degrades more gracefully.
- **Efficiency:** pruning latency **<1 ms** and essentially independent of dropping ratio for the main method; the -IA variant peaks at **~23.6 ms**. Abstract headline: up to **~4%** accuracy improvement at **<1 ms** overhead.

No Zotero highlights present.

Takeaways: position-aware local similarity (MSSAVT) beats global similarity for spatial pruning; a static checkerboard mask buys parallel-safe pruning without an iterative loop; temporal + spatial pruning are complementary (union of masks); the method is a cheap training-free plug-in whose edge over baselines grows at extreme compression and on long-video benchmarks, while being roughly tied on OVO-Bench.

## How it connects (evolution)
- [[timechat-online]] — direct parent: StreamingAssistant reuses its DTD temporal-pruning algorithm and uses TimeChat-Online-7B as the backbone; it *adds* the spatial pruning TimeChat-Online lacks.
- [[streamingbench]], [[ovo-bench]] — the streaming evaluation suites it reports on (real-time visual understanding / online video QA).
- [[streamingvlm]] — sibling in the streaming-efficiency line (keeping streaming inference cheap under unbounded video).
- [[streamkv]], [[hermes-kv]] — adjacent memory-compression approaches that instead prune/compress the *KV-cache*; StreamingAssistant compresses upstream at the visual-token level.
- [[flash-vstream]], [[rekv]] — streaming-memory systems that manage what visual information persists for later querying, the same buffer-then-query pattern.

## Open questions / limitations
- Extremely high dropping ratios (~92–95%) are the sweet spot reported; how accuracy holds at *moderate* compression, or whether these thresholds generalize beyond the TimeChat-Online-7B/Qwen2.5-VL backbone, is not shown.
- The checkerboard mask caps the removable fraction of adjacent structure and is content-agnostic (fixed parity), so it may over-retain in uniform regions and under-prune where a coarser candidate pattern would help.
- On OVO-Bench it is essentially tied with (slightly behind) DivPrune, suggesting the position-aware gain is task/benchmark dependent rather than universal.
- Redundancy is computed purely from visual-embedding similarity with no query awareness — tokens relevant to a future user question could be pruned before the query arrives (an inherent risk of compress-then-query streaming).

*Verification: equations (MSSAVT Eq. 1, checkerboard mask, $M^n=M^n_t|M^n_s$) checked against the rendered method text on p.3–4 of the PDF; all headline numbers (Tables 1–5, latency, thresholds, backbone) cross-checked against the arXiv HTML extraction of the paper's own tables; title/authors from the arXiv:2512.12560 abstract page.*

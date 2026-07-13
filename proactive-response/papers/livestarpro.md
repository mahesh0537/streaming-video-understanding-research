---
zotero_key: null
authors: Zhenyu Yang, Kairui Zhang, Bing Wang, Shengsheng Qian, Changsheng Xu (MAIS/CASIA — Institute of Automation, Chinese Academy of Sciences)
year: 2026
arxiv: 2606.17798
pdf: https://arxiv.org/pdf/2606.17798
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# LiveStarPro: Proactive Streaming Video Understanding with Hierarchical Memory for Long-Horizon Streams

**Lineage role:** Journal extension of LiveStar that keeps the perplexity-verification response-timing gate (SVeD) but bolts on a tree-structured hierarchical memory (TSHM) so proactive narration survives long-horizon streams beyond a fixed context window.

## Problem — what was limited before this paper (short)
An always-on streaming assistant must decide *when* to speak on an unbounded frame-by-frame stream, and prior online Video-LLMs mostly attach an explicit EOS / silence token and decode every frame — costly, and the added vocabulary token contaminates language modeling and yields unstable, repetitive or delayed responses. A second, orthogonal failure is memory: a continuous stream eventually overruns any fixed context window (e.g. 8192 tokens), and the usual eviction strategy (FIFO) discards history indiscriminately (catastrophic forgetting), while external memory banks are flat collections whose retrieval cost and footprint both grow linearly with stream duration. Existing benchmarks are also narrow (heavy Ego4D first-person bias; mostly QA-only synchronous settings), leaving live narration, temporal grounding, and multi-turn interaction under-evaluated.

## Key idea — the core insight, 2-4 sentences
Reframe response timing as *verification, not generation*: at each new frame run a single forward pass and use the perplexity of the current caption under the newly arrived frame as a confidence signal — if the frame makes the ongoing caption much more surprising than it was, the scene has changed enough to warrant a new utterance; otherwise stay silent. This "verify-then-generate" gate (SVeD) needs no silence token, and a matched training scheme (SCAM) shapes causal attention masks over interleaved frame-caption sequences so the model learns the incremental video-language alignment SVeD relies on. For long horizons, a Tree-Structured Hierarchical Memory (TSHM) compresses the active window by a Peak-End rule and offloads evicted events into a recursive event tree with sublinear (beam-descent) retrieval, so distant history stays recallable without saturating the context window.

![[livestarpro.png]]
> **Crux (Fig. 2).** The SVeD inference framework: for each incoming frame the frozen VLM does one forward pass, compares the output perplexity of the current caption against a scaled decode threshold, and either appends a new caption (green ✓) or emits "Silent" and swaps the frame into the context buffer — a dynamic response-silence schedule that avoids per-frame autoregressive decoding and EOS tokens. *Yang et al. (2026), arXiv:2606.17798. Embedded for personal research reference.*

![[livestarpro-tshm.png]]
> **Crux (Fig. 4).** Tree-Structured Hierarchical Memory: (a) short-term working memory compresses frames via the Peak-End rule (keeping low-perplexity "peak" keyframes + the clip's summary caption); (b) evicted units are offloaded into a recursive event tree that attaches semantically similar events as children ($\text{Sim}\ge\tau$) or spawns new branches; (c) a query does a hierarchical Top-K retrieval to fetch relevant event chains that augment generation. *Yang et al. (2026), arXiv:2606.17798. Embedded for personal research reference.*

## Method + math — the mechanism, then the objective/equations in full

**1. Streaming Verification Decoding (SVeD) — the response-silence gate.**
Let $Ctx^{<t_i}$ be the running context (past frames + emitted captions) and $Frm^{t_i}$ the frame at time $t_i$. The confidence that the *current* caption $Dec$ still describes the stream at frame $t_i$ is measured by its length-normalized perplexity over $N$ tokens:
$$
\mathrm{PPL}^{t_i}(Dec) \;=\; \Big(\tfrac{1}{P\big(Dec \,\mid\, Ctx^{<t_i},\, Frm^{t_i}\big)}\Big)^{1/N}.
$$
As a new frame $t_j>t_i$ arrives, a single forward pass yields $\mathrm{PPL}^{t_j}(Dec)$. The decoding gate fires (generate a new caption) iff the new frame makes the old caption sufficiently more surprising:
$$
\text{generate at } t_j \iff \mathrm{PPL}^{t_j}(Dec) \;>\; \alpha\cdot \mathrm{PPL}^{t_i}(Dec),
$$
where $\alpha$ (default $\approx 1.03$) is the adaptive response-silence threshold. If the test fails, the model stays **silent** and performs a logical *Swap*: the just-arrived frame is folded into the context buffer next to the still-valid caption ("Swap $Frm^{t_j}$ and $Dec^k$" in Fig. 2), so no autoregressive decode is spent. This replaces per-frame generation with per-frame *verification* and removes the EOS/silence vocabulary token entirely.

**2. Streaming Causal Attention Masks (SCAM) — the training objective.**
To make the model produce the incremental alignment SVeD verifies, training organizes data as interleaved frame–caption sequences grouped into *semantic clips* $C_k$ (contiguous frames that share one event description $Txt^k$). The reformulated objective forces every frame in a clip to independently support the clip's caption:
$$
\max\; P\big(Txt^{k}\,\mid\, Ctx^{<t_i},\, Frm^{t_i}\big),\quad \forall\, t_i \in C_k.
$$
The causal mask **hides the preceding captions within the same clip** (so the caption must be grounded in the *current* visual features, not copied from an earlier identical caption) while keeping the *terminal* captions of prior clips $\{C_1,\dots,C_{k-1}\}$ fully visible (so cross-event narrative context is preserved). Captions are sampled stochastically from $M$ paraphrased variants per frame to prevent memorizing surface forms.

**3. Tree-Structured Hierarchical Memory (TSHM) — long-horizon memory.**
*Short-term working memory (Peak-End compression).* Reuse the SVeD perplexity as a semantic-divergence score $S(t)=\mathrm{PPL}^t(Dec)$: **low** perplexity marks a "peak" keyframe strongly aligned with the ongoing description. When the token budget $L_{\max}$ is hit, each clip $C_k$ computes a per-clip median threshold $\tau_k$ and keeps only the low-divergence frames:
$$
\mathcal{T}_{keep}^{k} \;=\; \{\, t \in C_k \;\mid\; S(t) \le \tau_k \,\}, \tag{6}
$$
pruning roughly the higher-divergence ~50% while always retaining the clip's summary caption (the "End"). Older clips condense progressively; fully-condensed oldest clips are evicted to long-term memory.

*Long-term recursive event tree.* Each evicted event is a memory unit $U_i=\{c_i, v_i, \mathcal{E}_i, \mathcal{C}_i\}$ (caption $c_i$, peak-frame visual tokens $v_i$, semantic embedding $\mathcal{E}_i$, children $\mathcal{C}_i$). A new unit $U_{new}$ is placed under the most-similar existing node $U_{best}$ if $\text{sim}(\mathcal{E}_{new},\mathcal{E}_{best})>\sigma$ (a refinement/continuation of that event thread), else it becomes a new root. Parent embeddings are kept as the semantic centroid of their subtree via momentum aggregation:
$$
\mathcal{E}_{parent} \;\leftarrow\; \mathrm{Normalize}\big((1-\beta)\,\mathcal{E}_{parent} + \beta\,\mathcal{E}_{child}\big), \tag{7}
$$
with update rate $\beta$.

*Memory-augmented generation (hierarchical beam descent).* Because each parent is a subtree centroid, retrieval need not scan all $n$ stored units. Given query $q$ (the textual embedding of an explicit question, or an implicit query aggregated from recent short-term visual embeddings for open narration), starting from the roots the search scores only the immediate children of the current frontier by cosine similarity and keeps the Top-$k$:
$$
F_{d+1} \;=\; \operatorname*{Top\text{-}k}_{U_i \in \mathrm{Child}(F_d)}\; \frac{q\cdot \mathcal{E}_i}{\lVert q\rVert\,\lVert \mathcal{E}_i\rVert},
$$
giving expected complexity $O(k\,b\,\log_b n)$ for beam width $k$, branching factor $b$. The retrieved event chains (captions + visual tokens along the retrieval paths) $M_{retrieved}=\{(c_j,v_j)\mid j\in I\cup \mathrm{Path}(I)\}$ are injected back into the attention window to condition generation.

## Explicit design choices
- **Timing = perplexity verification, not an EOS classifier.** No silence/EOS token added to the vocabulary; one forward pass per frame; a *Swap* op folds silent frames into context instead of decoding.
- **Threshold is relative and adaptive:** gate compares new PPL to $\alpha\times$ the previous frame's PPL (default $\alpha\approx1.03$), not an absolute cutoff.
- **Semantic-clip supervision (SCAM):** intra-clip captions masked, prior-clip terminal captions visible; forces per-frame grounding while retaining narrative continuity.
- **Paraphrase augmentation:** $M$ caption variants per frame, stochastically sampled, to avoid overfitting caption surface forms.
- **One score reused three ways:** the SVeD perplexity is the timing gate *and* the keyframe-salience score $S(t)$ for Peak-End compression.
- **Peak-End compression:** keep low-perplexity peak frames ($S(t)\le\tau_k$, per-clip median) + clip summary caption; prune ~50% per pruning cycle.
- **Recursive event tree** with similarity-thresholded insertion ($\sigma$), momentum centroid parents (Eq. 7), and **sublinear beam-descent retrieval** — replaces flat $k$-NN memory banks whose cost grows linearly with stream length.
- **Backbone:** InternVideo2.5 architecture — InternViT vision encoder + MLP projector + LLM.
- **Two-phase progressive training on ~83K samples:** Phase I temporal-alignment pretraining = 63K curated segments (ActivityNet Captions 9K, Shot2Story 33K, Ego4D Narration Stream 20K, MVBench 1K); Phase II multi-task online adaptation = 20K OmniStarPro samples over the five online tasks.
- **New benchmark OmniStarPro** (data construction pipeline): 15 real-world scenarios / 46 fine-grained categories, Whisper-based speech filtering in preprocessing, GPT-4o-scored generative metrics (SemCor, SumFluen). Split into **OmniStarPro-Live** (5 short-horizon tasks) and **OmniStarPro-Long** (3 memory-centric tasks).

**Eval protocol / task taxonomy (benchmark side).**
- *OmniStarPro-Live (short-horizon):* RNG (Real-time Narration Generation — decide when to speak, penalized for latency), OTG (Online Temporal Grounding — localize event span as soon as possible), FDQ (Frame-level Dense QA), COQ (Contextual Online QA — recent memory + causal reasoning), MIQ (Multi-turn Interactive QA).
- *OmniStarPro-Long (memory-centric):* LMR (Long-range Memory Recall of an evicted entity attribute), CDQ (Cross-event Difference Query — contrast two distant moments), TBR (Temporal Backtracking — find the most recent past occurrence and report its timestamp). Reported per span bucket S (<10 min) / M (10–30 min) / L (>30 min).
- *Metrics:* PPL (↓), TokAcc / LM-Correctness (↑), TimeDiff (↓, timing error), TimRedun (↓), TimCover (↑), plus generative SemCor (↑) and SumFluen (↑) scored by GPT-4o; long-form recall reported as accuracy % per bucket.

## Key results / what to remember
Verified against the paper's own tables/text (PDF):
- **Long-form recall (Table III, OmniStarPro-Long, accuracy %), LiveStarPro:** LMR **63.4 / 49.7 / 37.2** (S/M/L), CDQ **55.1 / 42.8 / 31.5**, TBR **59.8 / 46.3 / 34.6**. Baselines on the *long* (>30 min) bucket: VideoLLM-online LMR 6.4, VideoLLM-MoD 6.9, MMDuet 9.1, LiveStar 21.1 — LiveStarPro's 37.2 is the standout, and the degradation from S→L is far gentler than baselines.
- **Flat vs. tree memory:** a flat $k$-NN external-bank baseline reaches **21.3%** on the long bucket vs TSHM's **37.2%** on the same partition — the recursive event tree, not just more memory, drives the gain.
- **Offline RNG controlled signal (Table I text):** on the *same* InternVideo2.5 backbone under identical fixed decoding, the backbone without streaming fine-tuning scores **SemCor 4.32** vs LiveStarPro **4.62** — isolating the SVeD+SCAM contribution from backbone quality.
- **SVBench (zero-shot, Table IV):** LiveStarPro average **52.20**, exceeding the InternVideo2.5 backbone (49.16) and LiveStar (49.76); still below general-purpose Qwen2.5-VL (55.21) and MiniCPM-V 2.6 (53.15). Fine-tuned LiveStar† rises **49.76 → 57.41** (a 15.37% relative gain).
- **Ego4D streaming (Table V):** LiveStarPro attains **18.1% higher TokAcc** than the second-best online assistant LION-FS.
- **Offline general benchmarks (Table VI):** **69.8** MVBench, **56.3** LongVideoBench, **60.8** VideoMME (w/o subtitles); exceeds VideoChat-Online by **+8.0** points on VideoMME — i.e. the streaming specialization causes a modest, not catastrophic, offline decline.
- **Dataset scale:** OmniStarPro-Live 20,137 expert-annotated streams (19,137 train / 1,000 test; avg 14.5 QA pairs, 8.2 caption segments; 45.54% exceed 100 s); OmniStarPro-Long 2,108 streams (avg 34.7 min, up to >60 min), 12,704 memory-centric queries, memory span avg 18.6 min / max 71.3 min, and 73.4% of queries span beyond the active context window. Total 22,245 streams.
- **Abstract headline (as stated, not all table-verified):** the abstract reports "28.9% improvement in semantic correctness," "18.2% reduction in timing error," and "1.58× inference speedup" vs prior online Video-LLMs. (n/r) I could not pin these exact three percentages to a single table; the closely related Ego4D **+18.1% TokAcc** and the SemCor/TimDiff table entries are the verified anchors, so treat the abstract's rounded figures as summary claims.
- Online RNG SemCor/TimDiff of **3.27 / 1.89** (Table II) appears in the fetched HTML but I did **not** independently confirm every Table I/II cell from the PDF: (n/r) for those specific per-cell values.

No Zotero highlights present.

Takeaways: (1) perplexity of the *ongoing* caption under each new frame is a clean, token-free timing signal — and doubles as a keyframe-salience score; (2) the real long-horizon win comes from a *structured* (tree, sublinear) memory over a flat bank, most visible on >30 min streams; (3) streaming specialization here costs only a modest offline-benchmark decline.

## How it connects (evolution)
- [[livestar]] — the NeurIPS 2025 conference paper this directly extends; SVeD's perplexity-verification timing gate is inherited, TSHM + OmniStarPro-Long are the journal additions.
- [[mmduet]] — a key streaming-response baseline it compares against on RNG and long-form recall.
- [[videollm-online]] — the EOS/silence-token online-decoding paradigm SVeD is explicitly designed to replace (used as a baseline).
- [[streammind]] — prior proactive/always-on streaming assistant in the same response-timing lineage.
- [[streaming-memory]] — sub-topic hub for the hierarchical/long-horizon memory thread (TSHM belongs here as much as to proactive timing).
- [[proactive-response]] — the sub-topic this note lives in: when-to-speak / response-silence for streaming assistants.

## Open questions / limitations
- **Threshold sensitivity:** the relative gate $\alpha\approx1.03$ is small and likely task-/domain-specific; robustness of the perplexity signal under noisy or slowly-drifting scenes is untested.
- **Tree assumptions:** the sublinear $O(kb\log_b n)$ retrieval assumes a balanced, bounded-growth tree; highly skewed event distributions (many near-duplicate events under one $\sigma$ threshold) can degrade toward linear scan.
- **Modality:** despite Whisper-based speech filtering in preprocessing, the model consumes only video frames — no explicit audio/ASR channel at inference.
- **Fixed window still bites:** an 8K-token active window plus aggressive Peak-End pruning means very long streams rely on compression that can drop details the tree never stored; generalization beyond the 83K-sample training distribution is unverified.

*Verification: equations (SVeD perplexity/gate, SCAM objective, Eq. 6 Peak-End, Eq. 7 momentum, beam-descent) and all headline numbers checked against the arXiv PDF (2606.17798) rendered pages — Table III (p.13), SVBench/Ego4D/offline text (p.13), dataset stats (p.10), method (pp.3–7). Note: the arXiv-HTML WebFetch mis-reported Table III (e.g. LMR 85.7 vs. the PDF's 63.4); PDF values were used and WebFetch-only cells are marked (n/r).*

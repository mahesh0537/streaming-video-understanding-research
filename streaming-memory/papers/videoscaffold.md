---
zotero_key: null
authors: Naishan Zheng, Jie Huang, Qingpei Guo, Feng Zhao (USTC & Ant Group)
year: 2025
arxiv: 2512.22226
pdf: https://arxiv.org/pdf/2512.22226
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# VideoScaffold: Elastic-Scale Visual Hierarchies for Streaming Video Understanding in MLLMs

**Lineage role:** builds streaming memory as an *elastic-scale event tree* — a next-frame-prediction segmenter grows a multi-level event hierarchy online, whose depth/granularity self-scales with clip duration, then consolidates bottom-up at query time.

## Problem — what was limited before this paper (short)
Video MLLMs mostly assume the whole clip is available and sample frames **uniformly** (offline) or keep a **fixed-size** rolling window / token buffer (streaming). Uniform sampling wastes tokens on static stretches and misses fast events; a single fixed granularity cannot serve both a 3-second clip and a 1-hour stream. Under the causal streaming constraint (no future frames) the model must decide *now*, without lookahead, where an event begins and ends and how coarsely to summarize it — existing methods lack a granularity that adapts to how much has actually happened.

## Key idea — the core insight, 2-4 sentences
Borrow Event Segmentation Theory: humans mark an event boundary exactly when their prediction of the next moment fails. VideoScaffold runs a learned **next-frame prediction** module at every abstraction level; a large prediction error (cosine distance between predicted and actual embedding) triggers a boundary and finalizes the current latent token, so the *number and length* of events emerge from the video's own dynamics rather than a fixed clock. Stacking predictors gives a hierarchy — lower levels capture short spans, higher levels abstract long-range semantic transitions — producing an **elastic-scale** tree that deepens for long streams. On a user query, **Hierarchical Event Consolidation** walks this tree bottom-up, picking each sub-event's most-surprising ("essential") frame and cross-attending to summarize, emitting both coarse and fine-grained event embeddings to the LLM.

![[videoscaffold.png]]
> **Crux (Figure 2).** The two-stage pipeline: streaming frames $v_t$ feed a next-frame-prediction segmenter (Elastic-Scale Event Segmentation) that grows a multi-level event tree online; on a user instruction, Hierarchical Event Consolidation summarizes it bottom-up into coarse + fine-grained event tokens for the LLM. *Zheng et al. (2025), arXiv:2512.22226. Embedded for personal research reference.*

## Method + math — the mechanism, then the objective in full

**Streaming constraint.** At time $t$ the model sees only $V_{1:t}=\{v_1,\dots,v_t\}$ (frame embeddings), never future frames.

**Base-level abstraction.** A temporal window of the last $m$ frames is compressed to one latent token by a shared learnable module $\Phi^{(1)}$:
$$\mathcal{Z}^{(1)}_t=\{v_{t-m+1},\dots,v_t\},\qquad z^{(1)}_t=\Phi^{(1)}\!\big(\mathcal{Z}^{(1)}_t;\phi^{(1)}\big),\quad z^{(1)}_t\in\mathbb{R}^d.$$

**Next-frame prediction.** A predictor $\Psi^{(1)}$ maps the current latent to the predicted next embedding:
$$\hat v_{t+1}=\Psi^{(1)}\!\big(z^{(1)}_t;\psi^{(1)}\big),\qquad \hat v_{t+1}\in\mathbb{R}^d.$$

**Hierarchical abstraction.** Level $l>1$ operates on the tokens of level $l-1$, so higher layers span progressively longer time:
$$\mathcal{Z}^{(l)}_t=\{z^{(l-1)}_{t-m_l+1},\dots,z^{(l-1)}_t\},\qquad z^{(l)}_t=\Phi^{(l)}\!\big(\mathcal{Z}^{(l)}_t;\phi^{(l)}\big).$$

**Hierarchical prediction.** The level-$l$ prediction is conditioned on the cumulative representations from *all* preceding layers (grounds abstract predictions in low-level cues):
$$\hat z^{(l)}_{t+1}=\Psi^{(l)}\!\big(z^{(1)}_t,z^{(2)}_t,\dots,z^{(l)}_t;\psi^{(l)}\big).$$

**Boundary refinement (the elastic trigger).** Prediction error is cosine distance between predicted and actual embedding:
$$\mathcal{E}^{(l)}_{t+1}=1-\cos\!\big(\hat z^{(l)}_{t+1},\,z^{(l)}_{t+1}\big)\in[0,2].$$
A boundary at level $l$ fires when $\mathcal{E}^{(l)}_{t+1}>\epsilon^{(l)}$. On firing, the current token $z^{(l)}_t$ is **finalized and appended** to the next level's sequence, $\mathcal{Z}^{(l+1)}_{t}\leftarrow\mathcal{Z}^{(l+1)}_{t}\cup\{z^{(l)}_t\}$; if $\mathcal{E}^{(l)}_{t+1}\le\epsilon^{(l)}$ the current event continues and no token is emitted. This is what makes segment count/length *elastic* — dynamics, not a clock, decide.

**Hierarchical Event Consolidation (HEC).** After EES the stream is a tree; the top level is $L$ and each event segment at level $l$ is an ordered set $\mathcal{S}^{(l)}=\{z^{(l)}_s,\dots,z^{(l)}_t\}$. On a query, consolidate bottom-up:
- *Essential element identification* — pick the sub-event's most-surprising frame (max prediction error) as the summary anchor: $z^{(l)}_{ess}=z^{(l)}_{i^\*},\ i^\*=\arg\max_i \mathcal{E}^{(l)}_i.$
- *Intra-layer aggregation* — cross-attend from that anchor to the rest of the segment: $\tilde z^{(l)}=\mathrm{CrossAttn}\big(Q=z^{(l)}_{ess},\,K{=}V=\mathcal{S}^{(l)}\setminus\{z^{(l)}_{ess}\}\big).$
- *Cross-layer aggregation* — carry the summary upward: $\tilde z^{(l+1)}=\mathrm{CrossAttn}\big(Q=\tilde z^{(l+1)}_{ess},\,K{=}V=\mathcal{V}^{(l)}\big).$
- Two **complementary event embeddings** feed the LLM: coarse (mean over top-level segment) $e_{coarse}=\frac{1}{|\mathcal{S}^{(L)}|}\sum z$, and fine-grained (top-level max-error anchor) $e_{fine}=z^{(L)}_{i^\*}$.

**Backbone / training.** EVA-CLIP vision encoder + Vicuna-7B LLM, two-stage (alignment then instruction tuning); VideoScaffold is a **plug-and-play** module (also bolted onto LLaVA as "LLaVA-SFT + Ours"). Prediction is done in the **latent space on patch tokens** (ablation-justified over pixel-space or CLS-token prediction).

## Explicit design choices
- Prediction-error boundary detection (cosine distance > threshold $\epsilon$), not fixed-stride or uniform sampling — zero-shot, streaming-compatible, no future frames.
- Multi-level predictors; higher-level prediction conditioned on the concatenation of all lower-level latents (keeps abstract levels grounded).
- **3 abstraction layers, threshold $\epsilon=0.4$** as the sweet spot (ablation Table 6): more layers give marginal gains + latency; smaller $\epsilon$ over-fragments, larger $\epsilon$ merges distinct scenes.
- Predict in **latent space with patch tokens** (best in Table 5) rather than raw pixels or CLS tokens.
- "Essential" frame = **max prediction error** per sub-event (best vs. random/middle-frame, Fig. 4b) — the most surprising frame anchors the summary.
- HEC uses **cross-attention** aggregation (beats Q-Former and distance-weighted pooling, Fig. 4a).
- Emit **two granularities** (coarse mean + fine-grained anchor) to the LLM.
- 60 frames used across all offline datasets; 0.5 fps in the streaming setting.

## Key results / what to remember
Model is 7B (Vicuna-7B). All verified against the paper's tables.

- **Short-form VideoQA (Table 1, zero-shot Acc):** MSVD-QA **72.5**, MSRVTT-QA **58.4**, ActivityNet-QA **48.9** — beats 13B LLaMA-VID (70.0 / 58.9 / 47.5) on MSVD & ActivityNet at half the LLM size.
- **LV-Bench (Table 1, >1h videos):** Overall **31.5**, best among all 7B baselines (LWM 25.5, LLaVA-NeXT* 32.2 uses larger-scale training).
- **MLVU (Table 2, 60 frames):** M-Avg **49.5**, G-Avg **4.37** — tops LLaVA-OV* M-Avg 44.2 despite OV's larger training; beats MA-LMM (36.4 with 1000 frames) and MovieChat (25.8 with 2048 frames) using only 60 frames.
- **VideoMME (Table 3, 60 frames):** w/o subtitles All **43.3** (Short 47.2 / Mid 43.6 / Long 39.2); w/ subtitles All **47.2**. (LLaVA-OneVision* reaches 58.5 but is trained on substantially larger data, marked \*.)
- **StreamingBench real-time (Table 4, 7B, 0.5 fps):** Overall **41.0** — **highest among streaming MLLMs**, beating VideoLLM-online (36.0, 8B, 2 fps) and Flash-VStream (23.2). Offline LLaVA-OneVision* still leads at 71.1 (non-streaming, 32 frames, larger data).
- **Plug-and-play gains:** adding VideoScaffold to LLaVA lifts short-QA by +2.5 / +1.6 / +1.4 pts (MSVD/MSRVTT/ActivityNet) and long-video benchmarks by +6.0 / +2.1 / +3.9 (LV-Bench / MLVU / VideoMME).
- **Ablation — elasticity matters (Table 5):** disabling the hierarchy (single fine-grained level) drops LV-Bench 31.5 → **27.3** while short-video MSVD barely moves (72.5 → 71.0), confirming hierarchy is what buys long-video gains.
- **Ablation — hyperparameters (Table 6):** 3 layers @ $\epsilon=0.4$ is best (LV-Bench 31.5); 2 layers = 27.7, 4 layers = 31.6 (marginal).

No Zotero highlights present.

Takeaways: prediction-error segmentation is a clean, lookahead-free way to get *content-adaptive* event boundaries in a stream; stacking it yields a memory whose depth self-scales with duration; consolidation is deferred to query time and returns multi-granularity tokens. The streaming absolute score (41.0) is modest vs. large offline models, so the win is efficiency + adaptivity under the causal constraint, not raw accuracy.

## How it connects (evolution)
- [[streaming-memory]] — sub-topic hub; VideoScaffold is a *hierarchical event-tree* memory whose granularity is elastic.
- [[streamingbench]] — the real-time benchmark VideoScaffold is evaluated on (Table 4).
- [[videollm-online]] — streaming baseline it outperforms (41.0 vs 36.0) with fewer fps.
- [[flash-vstream]] — streaming-memory baseline in the same StreamingBench table (23.2).
- [[streamforest]] — related tree/forest-structured streaming memory; contrast the segmentation criterion.
- [[streammem]] — memory-consolidation-on-query line VideoScaffold's HEC belongs to.

## Open questions / limitations
- Streaming absolute accuracy (41.0 on StreamingBench) trails strong offline models by a large margin — the method's value is adaptivity/efficiency, not headroom; unclear how it scales with a stronger LLM backbone than Vicuna-7B.
- Thresholds $\epsilon^{(l)}$ and layer count are global hyperparameters (fixed at 3 / 0.4); no per-stream or learned adaptation of the trigger itself.
- The next-frame predictor's quality bounds boundary quality; failure modes on low-motion or highly repetitive footage (where prediction error stays low) aren't analyzed.
- Memory-growth / latency at hour-plus streams is only argued qualitatively (deeper layers "introduce redundancy") — no explicit token-budget or wall-clock streaming-cost curve.

*Verification: all headline numbers checked directly against the paper's rendered Tables 1–6 (PDF pages 5–7) via figtool page renders; equations transcribed from PDF page 3 (Fig. 2 + Eqs. 2–14). WebFetch of the arXiv HTML was used for structure but its StreamingBench and MLVU figures were wrong and were corrected from the tables.*

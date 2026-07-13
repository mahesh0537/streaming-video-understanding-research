---
zotero_key: null
authors: Xiangyu Zeng et al. (Nanjing University, Shanghai AI Laboratory; with Zhejiang University, Huawei Noah's Ark Lab)
year: 2025
arxiv: 2509.24871
pdf: https://arxiv.org/pdf/2509.24871
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# StreamForest: Efficient Online Video Understanding with Persistent Event Memory

**Lineage role:** A persistent *event-level* tree-forest memory (PEMF) for streaming video LLMs — instead of frame-wise similarity merging or a fixed hierarchy, frames are grouped into events and events are hierarchically merged under three penalty terms; paired with the OnlineIT instruction data and the ODV-Bench (autonomous-driving online) benchmark.

## Problem — what was limited before this paper (short)
Offline video MLLMs read the whole clip at once, but streaming/online settings force the model to (1) store an ever-growing history of visual features under a fixed token budget and (2) reason about fine-grained spatiotemporal detail in real time at the query timestamp. Prior compression strategies are lossy in complementary ways: dropping/uniform-sampling frames throws away fine-grained actions, while pure inter-frame **similarity merging** collapses distinct-but-similar moments and loses critical short events. Neither preserves an event-structured, recency-aware memory that stays bounded as the stream grows.

## Key idea — the core insight, 2-4 sentences
Split memory into two complementary parts. A **Fine-grained Spatiotemporal Window (FSTW)** holds high-resolution real-time perception of the current frame plus a short-term buffer, so the query timestamp is seen sharply. A **Persistent Event Memory Forest (PEMF)** organizes all older history into *event-level* tree structures and, whenever the long-term token budget $L_q$ is exceeded, adaptively merges the least-important adjacent event nodes. The choice of what to merge is governed by three penalty functions — content similarity, historical merge count, and temporal distance to the query — so recent, distinctive, and not-yet-degraded events are preserved while redundant old ones are compressed.

![[streamforest.png]]
> **Crux (Figure 2).** Overview of StreamForest: the vision stream feeds a Fine-grained Spatiotemporal Window (real-time perception + short-term memory) and a Persistent Event Memory Forest whose event nodes are hierarchically merged under the three penalties shown below (similarity, temporal distance, merge count); the LLM answers the query at the current timestamp. *Zeng et al. (2025), arXiv:2509.24871. Embedded for personal research reference.*

## Method + math — mechanism then the objective in full

**Two-branch memory.** Incoming frames are encoded by a vision encoder (SigLIP-so400M). The **FSTW** keeps the current frame at full resolution (729 tokens of "real-time perception") plus a short-term spatiotemporal buffer (18 frames × 128 tokens). Within the short-term buffer, frames are automatically segmented into **meta-events** at inter-frame similarity minima (boundaries where consecutive frames are least similar). Consolidated meta-events graduate into the long-term **PEMF**.

**PEMF as an event forest.** PEMF is a hierarchical, tree-structured memory over event nodes (not frames). An upper limit $L_q$ caps the number of long-term tokens. When the running total would exceed $L_q$, PEMF performs **hierarchical memory consolidation**: it scores every pair of *adjacent* event nodes with a combined penalty and merges the lowest-penalty pair into a single parent node, halving their combined token count via **ToMe** (bipartite token merging). Repeated merging builds trees; the set of trees is the "forest."

Let $X_i \in \mathbb{R}^{n_i \times d}$ be the visual token features of event node $x_i$ ($n_i$ tokens). For adjacent nodes $x_i, x_{i+1}$ compute the pairwise cosine-similarity matrix $S_i = \mathrm{sim}(X_i, X_{i+1}) \in \mathbb{R}^{n_i \times n_{i+1}}$ and take its top-$k_i$ entries.

**Similarity penalty** — penalize merging highly-similar (redundant) nodes less, i.e. a *low* penalty means "safe/desirable to merge":
$$P_s(x_i, x_{i+1}) = 1 - \frac{1}{k_i}\sum_{(p,q)\in\mathcal{T}_i} S_i^{(p,q)}, \qquad \mathcal{T}_i = \arg\mathrm{TopK}_{(p,q)}\!\left(S_i^{(p,q)}, k_i\right).$$
So high top-$k$ similarity ⇒ small $P_s$ ⇒ preferential merging of redundant neighbors.

**Merge count penalty** — repeatedly-merged nodes have already lost detail; discourage merging them again. With $c_i$ the historical merge count of $x_i$ and $c_{\max}$ its max:
$$P_m(x_i, x_{i+1}) = \frac{c_i + c_{i+1}}{2\, c_{\max}}.$$
Frequently-merged nodes get a *higher* penalty, protecting them from further degradation.

**Temporal distance penalty** — preserve recent events. With query time $t_q$ and event time $t_i$, let $d_i = (t_q - t_i)/t_q$ be the normalized age:
$$P_t(x_i, x_{i+1}) = 1 - \frac{d_i + d_{i+1}}{2}.$$
Recent events (small age $d$) get a *higher* penalty ⇒ are less likely to be merged; older events get merged first.

**Combined penalty and rule.**
$$P(x_i, x_{i+1}) = w_s P_s + w_m P_m + w_t P_t, \qquad (w_s, w_m, w_t) = (0.4, 0.4, 0.2).$$
At each consolidation step PEMF merges the adjacent pair with the **minimum** $P$, halving its token count, until the budget $L_q$ is satisfied. Cost is small because scoring is over the bounded set of current event nodes, not all frames.

**ODV-Bench eval protocol (this is also a benchmark paper).** ODV-Bench targets *online* autonomous-driving video understanding: answer at the current timestamp using only content available up to then. Task taxonomy — **12 tasks in 3 groups**: Static Target (RTP real-time perception, HD hallucination detection, KIE key-info extraction, TCD traffic-change detection, DDM driving decision, PTM past-memory), Dynamic Target (AP action / LP location / DP distance prediction, plus risk), and Event-Oriented multi-agent interaction (RP risk perception, RA/ARA accident reasoning). Construction pipeline: collect from 6 driving datasets (e.g. BDD100K, Waymo) → semi-automatic clip selection (YOLO detection + filtering) → meta-annotations fused from dataset labels + a VLLM + manual verification → template-based MCQ generation with an option pool that injects plausible distractors → multi-round human quality control and video-length-proportional sampling. **Metric** = multiple-choice **accuracy** per task, averaged within each group and overall.

## Explicit design choices
- **Event-level, not frame-level, memory.** Compress and organize by *events* (meta-events from short-term similarity minima), giving tree nodes with semantic granularity.
- **Two-branch split:** sharp real-time FSTW (729 real-time tokens + 18×128 short-term) vs. bounded long-term PEMF forest — separates "see the query moment clearly" from "remember the history compactly."
- **Bounded budget $L_q$** with lazy consolidation: only merge when over budget, so memory update is $O(\text{event nodes})$ not $O(\text{frames})$.
- **Three orthogonal merge criteria** (similarity / merge-count / temporal) combined linearly with fixed weights $0.4/0.4/0.2$; merge-count regularizer explicitly protects already-degraded nodes.
- **ToMe bipartite matching** for the actual token merge (halves combined tokens); similarity uses top-$k$ token pairs across adjacent event sets to handle unequal event lengths.
- **Backbone:** SigLIP-so400M vision encoder + MLP projector + Qwen2-7B LLM; default visual-token cap **8192**; evaluated at **1 FPS**.
- **Five-stage training on 32 A100s:** stages 1–3 offline long-video pretraining (VideoChat-Flash, LLaVA-Video, LLaVA-OneVision data), stage 4 streaming fine-tuning → base StreamForest, optional stage 5 on OnlineIT-Drive → StreamForest(FT-drive).
- **OnlineIT instruction data:** OnlineIT-general (~400k, spatial/temporal/spatiotemporal/event perception) + OnlineIT-drive (89k driving: static/dynamic target + event reasoning), sourced from RefCOCO, LaSOT, Charades-STA, ActivityNet, Visual Genome, D²-City, TT100k, Road-Waymo, MM-AU, etc.

## Key results / what to remember
Verified against the paper's Tables 1 and 2 (7B, 1 FPS).

- **ODV-Bench (Table 1, accuracy):** StreamForest base **59.9** overall; StreamForest(FT-drive) **71.2** overall — vs. best prior open-source offline MLLM Qwen2.5-VL 55.6 and online VideoChat-Online 54.5. Human ceiling **91.4**.
- **StreamingBench (Table 2, Real-Time All):** **77.3** (FT-drive 76.8) — vs. Qwen2-VL 69.0, InternVL2 63.7.
- **OVBench (Table 2, Avg):** **60.5** (FT-drive 61.6) — vs. VideoChat-Online 54.9.
- **OVO-Bench (Table 2, Overall):** **55.6** (FT-drive 55.6) — vs. LLaVA-Video 53.1.
- **Offline generalization (Table 2):** VideoMME (w/o sub.) **61.4**, MLVU (M-Avg) **70.0**, MVBench **70.2**, PerceptionTest (Val) **73.1**.
- **PEMF ablation (Table 3, OVBench/OVO-Bench/MLVU):** PEMF **60.5 / 55.6 / 70.0** beats Similarity-Merge 60.3/53.4/68.0, FIFO 58.7/52.9/56.7, Uniform 58.2/52.7/69.4.
- **Component ablation (Table 4):** full model 60.5/55.6/70.0 vs. w/o both FSTW+PEMF 58.0/52.5/51.8 — PEMF is what rescues long-video MLVU (51.8 → 70.0).
- **Efficiency / robustness (from text & Figs 4–5, Table 6):** memory update adds ~0.17 s over 500 frames; GPU memory ~17 GB stable regardless of stream length; retains ~**96.8%** of avg accuracy even under an extreme ~1K-token budget vs. the 8K default (reported in text; read off Fig. 4, not a numbered table cell). NeurIPS 2025 Spotlight per the lineage hint (n/r in the fetched pages).

No Zotero highlights present.

**Takeaways:** (i) organizing streaming memory by *events in a tree forest* + a recency/redundancy/degradation-aware merge rule beats frame-wise similarity merging and FIFO, especially on long-video MLVU; (ii) separating a sharp real-time window from a bounded long-term forest gives strong online numbers while *keeping* offline VQA competitive; (iii) domain instruction data (OnlineIT-drive) lifts ODV-Bench from 59.9 → 71.2 with little cost to general benchmarks.

## How it connects (evolution)
- [[flash-vstream]] / [[rekv]] / [[infinipot-v]] — prior bounded/compressed streaming KV & feature memories that PEMF's event-forest supersedes with structured merging.
- [[eventmemagent]] — also organizes streaming memory by *events*; closest conceptual sibling on event-level memory.
- [[dispider]] / [[ovbench-videochat-online]] — online video MLLM baselines it compares against in Tables 1–2.
- [[streamingbench]] / [[ovo-bench]] — the online benchmarks it reports on; ODV-Bench extends this benchmark line into autonomous driving.
- [[streaming-memory]] — the sub-topic hub this note lives under.

## Open questions / limitations
- Fixed penalty weights $(0.4,0.4,0.2)$ and a fixed $L_q$ are hand-tuned; no learned/adaptive controller — robustness to very different stream statistics is untested here.
- Merging is only over *adjacent* event nodes, so semantically related but temporally distant events cannot be co-compressed; the "forest" is temporally local.
- Strong ODV-Bench gains depend on domain-specific OnlineIT-drive data; how much of the online advantage transfers to non-driving open-world streams beyond the reported benchmarks is unclear.
- The 96.8%-under-1K-token robustness claim is read from a figure, not a numbered table, so the exact operating point is approximate.

*Verification: equations (1)–(4), penalty weights, backbone, and token budgets taken from the arXiv HTML/PDF method section (Sec. 3.1.2); all headline numbers cross-checked against the rendered Table 1 (ODV-Bench) and Table 2 (StreamingBench/OVBench/OVO-Bench + offline VQA) on PDF pages 6–7. Efficiency/robustness figures are from the paper's text/Figs 4–6 and marked where not a table cell.*

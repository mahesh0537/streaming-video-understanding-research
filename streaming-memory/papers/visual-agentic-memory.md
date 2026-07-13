---
zotero_key: null
authors: Aiden Yiliu Li, Nels Numan, Anthony Steed (University College London)
year: 2026
arxiv: 2605.16481
pdf: https://arxiv.org/pdf/2605.16481
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# Visual Agentic Memory: Enabling Online Long Video Understanding via Online Indexing, Hierarchical Memory, and Agentic Retrieval

**Lineage role:** the retrieval-first, agentic/indexed-memory point in streaming video memory — it builds a searchable index online while the stream runs, then lets an MLLM *agent* plan, retrieve, inspect and verify raw evidence at query time, instead of compressing everything into a fixed working-memory state.

## Problem — what was limited before this paper (short)
Online long-video understanding forces a tension: a stream can span hours, days or months, but the model must answer at any moment under bounded compute. Two prior families both lose fine-grained evidence. End-to-end streaming MLLMs push frames into a bounded KV/working-memory state and hit a working-memory bottleneck on month-scale streams (human–model gap >70 points on MM-Lifelong). Compact-memory / summarisation schemes stay tractable but discard the raw pixels, so once fine detail is gone it cannot be recovered — grounding and verification collapse. What was missing is a memory that is committed *online* under streaming constraints yet remains *auditable and recoverable* down to the raw frame at query time.

## Key idea — the core insight, 2-4 sentences
VAM decouples continuous memory *construction* (online indexing as the stream evolves) from query-time *reasoning* (an agentic retriever), joined by a shared hierarchical memory. Indexing keeps only visually significant transitions as frame-level *moments* (raw RGB + embedding + timestamp), groups contiguous moments into *events* with MLLM-written summaries, and stores both in a **Parallel Representation**: a temporal pathway (event summaries + timestamps) and a spatial pathway (raw frames + frame embeddings) sharing metadata. At query time an MLLM acts as an agent — it plans sub-queries, does hybrid temporal+spatial retrieval, then *inspects the actual raw frames* to verify evidence before emitting a grounded, cited answer. The retrieval-first stance (retain raw frames alongside summaries) is what enables direct visual verification rather than trusting compressed text.

![[visual-agentic-memory.png]]
> **Crux (Figure 3).** The three coupled layers — Online Indexing (filter → embed → boundary-detect → commit moments/events), Hierarchical Memory (parallel Temporal and Spatial representations), and Agentic Retrieval (plan → search/rerank → inspect raw frames → grounded cited answer, with a "refine if needed" feedback loop) — showing how a continuous stream becomes a searchable, verifiable long-horizon memory trace. *Li, Numan & Steed (2026), arXiv:2605.16481. Embedded for personal research reference.*

## Method + math — mechanism then equations in full
The pipeline is training-free and orchestrated asynchronously: **online indexing** runs continuously over the stream; **agentic retrieval** runs on demand; both operate over the shared hierarchical memory.

### 1. Online Indexing
**Frame filtering (two gates).** Each incoming frame $I_t$ passes a sharpness gate and a redundancy gate before any neural cost is paid. Sharpness is the variance of the Laplacian of the grayscale frame:
$$\mathrm{Sharpness}(I_t) = \mathrm{Var}\!\big(\nabla^2 I_{t,\mathrm{gray}}\big)$$
Frames with $\mathrm{Sharpness}(I_t) < \tau_{\mathrm{blur}}$ are discarded as blurry; visually near-identical frames are also dropped. Only frames passing both gates reach embedding extraction. Stream is sampled at an initial ~0.5 fps with adaptive retention on top.

**Embedding extraction.** A surviving frame gets a dense embedding $\mathbf{e}_t \in \mathbb{R}^d$ from a pretrained multimodal encoder (Gemini Embedding 2 in experiments). The embedding is the *retrieval-facing* representation; the heavier MLLM is reserved for event summarisation and inspection.

**Adaptive deduplication + boundary-preserving moment commit.** Each new embedding is compared to a running reference $\mathbf{e}_{\mathrm{ref}}$ by cosine distance:
$$d_c(\mathbf{e}_t, \mathbf{e}_{\mathrm{ref}}) = 1 - \frac{\mathbf{e}_t \cdot \mathbf{e}_{\mathrm{ref}}}{\lVert \mathbf{e}_t\rVert\,\lVert \mathbf{e}_{\mathrm{ref}}\rVert}$$
If $d_c \le \tau_{\mathrm{dedup}}$ the frame is buffered as redundant context; when $d_c > \tau_{\mathrm{dedup}}$ the buffered endpoint is **committed as a moment** and the new frame becomes the reference (so the *last* frame of a redundant run — the transition — is the one kept). The threshold is not fixed: it is derived by **Otsu partitioning** of a sliding distance history $\mathcal{H}_d$,
$$\tau_{\mathrm{dedup}} \approx \mathrm{AdaptiveThreshold}(\mathcal{H}_d),$$
so retention adapts to how fast the scene is changing. The same Otsu-based change analysis defines **event boundaries**: distances between consecutive committed moments are scanned for local peaks above a relaxed threshold, with a duration cap preventing unbounded events.

### 2. Hierarchical Memory (Parallel Representation)
Contiguous moments between two boundaries form an event $E_i = \{m_{s_i}, \dots, m_{e_i}\}$, summarised by the MLLM: $C_i = \mathrm{MLLM}(E_i)$. Memory stores two coupled pathways that share metadata:
- **Temporal representation** — events with summaries $C_i$ and time spans $t_{is}\!-\!t_{ie}$; supports coarse, time-aware retrieval.
- **Spatial representation** — the retained raw RGB frames and their embeddings $\mathbf{e}_t$; supports fine-grained visual retrieval and direct inspection.

**Age-aware storage.** Retained items sit in *recent / mid / long* tiers by age $\Delta t$; compression strength and minimum spacing between kept items increase with age, so older history is thinned but never fully alignment-lost — retrieval can move from an event-level temporal hypothesis down to frame-level spatial evidence without misalignment.

### 3. Agentic Retrieval (query time)
A query turn produces an operational spec; complex requests are decomposed into executable sub-queries. The planner MLLM chooses from actions $\{\texttt{search}, \texttt{inspect}, \texttt{summarize}, \texttt{answer}\}$ over the shared memory.

**Hybrid retrieval.** For query embedding $\mathbf{q}_k$, temporal retrieval scores event descriptors and spatial retrieval scores frame embeddings:
$$S_{\mathrm{TEMP}}(d_i) = \cos(\mathbf{q}_k, \mathbf{v}_{d_i}), \qquad S_{\mathrm{VIS}}(F_t) = \cos(\mathbf{q}_k, \mathbf{e}_t).$$
**Grounded inspection.** Retrieved candidates are provisional. The MLLM calls `inspect` (or `joint_inspection` to compare temporally related candidates) to look at the actual evidence: $O_k = \mathrm{Inspector}(\mathcal{X}^{\mathrm{vis}}_k, \mathcal{X}^{\mathrm{temp}}_k, Q)$. A dashed "refine if needed" loop lets inspection results drive additional retrieval.
**Termination** is an explicit planner action: `answer` fires when evidence is sufficient, the turn budget is nearly exhausted, or successive turns stop improving. Output preserves the answer text, an evidence-sufficiency flag, the principal supporting candidate, and the full multi-turn trace (cited video intervals, e.g. "collision at 12.4–13.1s").

### Evaluation protocol (benchmarks used)
- **OVO-Bench** (Niu et al. 2025): 644 videos, ~3,100 queries, minutes to ~30 min; strict online eval across three modes — Real-Time Visual Perception (RT), Backward Tracing (BT), Forward Active Responding.
- **MM-Lifelong train@month** (Chen et al. 2026a): a single month-scale stream, 105.6 hours over 51 days, 266 questions; human accuracy 82.5% (human–model gap >70 points). Reports accuracy and **Ref@300** (a reference/grounding score at 300, reflecting event-level localisation granularity).

## Explicit design choices
- **Decouple indexing from retrieval**, coordinated only through the shared memory + asynchronous orchestration — construction runs with the stream, reasoning runs on demand.
- **Retrieval-first, evidence-preserving**: keep raw RGB frames alongside event summaries so answers can be *visually verified*, not just inferred from compressed text (explicit contrast with recursive-reasoning schemes like ReMA).
- **Cheap-first cascade**: Laplacian-variance blur gate + redundancy gate + embedding-cosine dedup filter frames *before* spending MLLM tokens; the MLLM is reserved for summarisation and inspection.
- **Adaptive, data-driven thresholds** via Otsu partitioning of a sliding distance history for both dedup and event boundaries — no hand-tuned global threshold; a duration cap bounds event length.
- **Boundary-preserving commit**: commit the transition frame (buffered endpoint) as the moment, so scene changes are the retained anchors.
- **Parallel temporal/spatial memory sharing metadata**, with **age-tiered** (recent/mid/long) compression so month-scale storage stays tiny.
- **Bounded agent**: explicit turn budgets, candidate-set sizes and inspection batch limits keep query cost predictable; termination is an explicit action.
- **Training-free**, model-family-shared: Gemini Embedding 2 for indexing, Gemini 3 Flash as the query-time MLLM agent (same family writes summaries and inspects).

## Key results / what to remember
No Zotero highlights present.

- **OVO-Bench, RT+BT average: 68.41** for VAM vs **67.46** for the Gemini 3 Flash end-to-end baseline (+0.95). Sub-scores reported: Real-Time average **78.94**, Backward Tracing **57.89**, Spatial Understanding (STU) **82.19** (Table 1).
- **MM-Lifelong storage efficiency (month run, 105.6 h / 51 days):** raw video = 11,406,923 frames (100%); uniform 1 fps = 380,304 (3.33%); uniform 0.5 fps = 190,152 (1.67%); **VAM output = 6,876 frames = 0.06% of raw** — ~1659× fewer than raw and ~55× fewer than 1 fps (Table 2).
- **MM-Lifelong train@month accuracy:** VAM **17.11%** (Ref@300 = **3.65**), second-highest reported; ReMA + GPT-5 = 17.62% (Ref@300 9.91), ReMA + Qwen3VL-A22B = 14.23%, GPT-5 = 10.15%, Qwen3-VL-235B = 9.09%. Human = 82.5% — the >70-point human–model gap is the headline motivation (Table 3).
- Takeaways: (1) the win over the strong end-to-end base on OVO-Bench is modest (+0.95) but comes with *auditable, cited* evidence intervals; (2) the real story is storage — three orders of magnitude compression while staying frame-recoverable; (3) on month-scale reasoning even the best systems are far below human, so long-horizon *global localisation* is still open.
- No explicit ablation table is reported in the paper; efficiency and accuracy comparisons stand in for it.

## How it connects (evolution)
- [[streaming-memory]] — this is a canonical member of the sub-topic: memory that is indexed online and queried later.
- [[streammem]], [[streamkv]], [[hermes-kv]] — contrast: KV/working-memory compression that keeps a bounded state; VAM instead retains recoverable raw frames and searches them.
- [[eventmemagent]], [[omnimem]], [[savemem]] — sibling *agentic / event-structured memory* systems; VAM's event grouping + agent planner is the same family, retrieval-first.
- [[rekv]], [[flash-vstream]], [[infinipot-v]] — retrieval-augmented streaming memory that also indexes-then-retrieves; VAM adds explicit inspection/verification of raw evidence.
- [[streamingbench]], [[ovbench-videochat-online]] — VAM evaluates on the OVO-Bench online protocol; useful for the eval-protocol lineage.

## Open questions / limitations
- OVO-Bench gain over the end-to-end Gemini 3 Flash base is only +0.95 — the agentic pipeline's benefit is small on minute-scale clips and mostly shows as auditability, not raw accuracy.
- On MM-Lifelong month scale VAM (17.11%) trails ReMA+GPT-5 (17.62%) and is ~65 points below humans; VAM's coarser Ref@300 (3.65 vs 9.91) suggests weaker fine localisation than its retrieval story implies.
- Heavy dependence on proprietary Gemini components (Embedding 2 + Gemini 3 Flash); no ablation isolates which of filtering / adaptive thresholds / inspection actually drives results, so the contribution attribution is unclear.
- Adaptive Otsu thresholds, duration caps, blur/dedup gates and turn/batch budgets are numerous hyperparameters whose sensitivity is not characterised.

*Verification: equations (Laplacian-variance sharpness, cosine-distance dedup, Otsu adaptive threshold, temporal/spatial cosine scores) and all numbers (OVO-Bench 68.41 vs 67.46; storage 6,876 frames = 0.06%; MM-Lifelong 17.11% acc / Ref@300 3.65) checked against the arXiv HTML method section and Tables 1–3, cross-read with the PDF (arXiv:2605.16481) rendered pages 3–4.*

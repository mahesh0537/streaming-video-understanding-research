---
zotero_key: null
authors: Zhenpeng Huang, Xinhao Li, Jiaqi Li, et al. (Nanjing University; China Mobile Research Institute; OpenGVLab, Shanghai AI Lab)
year: 2025
arxiv: 2501.00584
pdf: https://arxiv.org/pdf/2501.00584
tier: deep
subtopics: [streaming-benchmarks]
tags: [streaming-video-understanding, streaming-benchmarks]
---
# Online Video Understanding: OVBench and VideoChat-Online

**Lineage role:** A dual contribution (CVPR 2025) — **OVBench**, a timestamp-anchored online video QA benchmark with a past/current/future temporal taxonomy, paired with **VideoChat-Online**, a streaming baseline whose **Pyramid Memory Bank** hierarchical KV-compatible memory ties into the streaming-memory lineage.

## Problem — what was limited before this paper
Offline video MLLMs ingest a whole clip and answer once, but real-world streaming (driving, robot assistants, surveillance) needs answers *at a moving present timestamp* over an unbounded incoming stream. Prior video benchmarks miss the traits that make streaming hard: (1) the answer to the *same* question depends on *when* it is asked (an action "happening now" later becomes "in the past"); (2) contexts evolve as time passes; (3) responses must be real-time; (4) the visual history grows without bound, so a model cannot keep every frame at full resolution. No existing benchmark scored a model's ability to *perceive the present, remember the past, and anticipate the future* from a single continuously-growing stream, and existing memory-compression baselines (MovieChat, Flash-VStream) reprocess the whole compressed memory each step, which is too slow for streaming.

## Key idea
Define an explicit **online temporal context** — Past (P), Current (C), Future (F) — and build a benchmark whose questions are anchored to a query timestamp so the "correct" answer is a function of when you ask. OVBench organizes **6 core spatiotemporal tasks → 16 subtasks** across these contexts (spatial/temporal/spatio-temporal perception of the present, past memory recall, temporal-hallucination verification, future prediction). On the model side, a **Pyramid Memory Bank (PMB)** keeps a *hierarchy* of frame-token queues: a few recent frames at full spatial resolution, plus progressively higher-frame-rate / lower-resolution queues for long-range temporal context. Evicted frames are *down-written* (average-pooled to a coarser layer) rather than dropped, and the memory is kept **synchronized with the LLM's KV-cache** so tokens are precomputed once and never reprocessed — giving real-time streaming inference at ~4B params.

![[ovbench-videochat-online.png]]
> **Crux (Figure 3).** The Pyramid Memory Bank: three stacked queues — $m_s$ (static, high spatial detail, few frames), $m_{main}$ (balanced), $m_t$ (dynamic, high-frame-rate, low resolution) — where evicted frames flow *downward* via average-pooling; reads are concatenated in time order and the memory stays synchronized with the LLM KV-cache for streaming prefill/decode. *Huang et al. (2025), arXiv:2501.00584. Embedded for personal research reference.*

## Method + math

### OVBench evaluation protocol (the "math" of a benchmark)
**Temporal-context taxonomy.** Each question fixes a *query timestamp* $t_q$ and belongs to one of three context patterns relating the answer window to $t_q$: perceive the **Current** ($C$), reason from **Past into Current** ($C{\to}P\cup C$ / $C{\to}P$), verify **Past-vs-Current** consistency ($P{\leftrightarrow}C$, hallucination check), or predict the **Future** from observed history ($P\cup C{\to}F$). The **6 core tasks** are Spatial Perception, Temporal Perception, Spatio-Temporal Perception, Past Memory, Temporal Hallucination Verification, Future Prediction, expanded to **16 subtasks** (e.g. Action Location, Object Position; Action Sequence, Step Localization, Object Existence State; Action/Object Trajectory; Action Retrieval, Procedure Recall, Trajectory Retrieval; Action Persistence, Step Verification, Object Presence; Action Anticipation, Goal/Step Prediction, Movement Prediction).

**Data-construction pipeline.** Reuse the dense spatiotemporal annotations of **8 source datasets across 6 domains** (Movie/AVA, Instructional/HiREST+COIN, Road/TAO+ArgoVerse+BDD, indoor scenes/HACS, outdoor, open-domain). Draw only from validation/test splits to avoid training leakage. Human annotators produced ~**7,000** verified QA items; questions carry a specific timestamp (and bounding boxes where spatial). **Multiple-choice distractors are mined from *other timestamps in the same video*** — the key design that forces genuine temporal grounding rather than static scene recognition. Video context is trimmed to the maximum range implied by the question's earliest reference; items are balanced across task types and durations, then manually checked for clarity/ambiguity/accuracy.

**Metric.** Scoring is **multiple-choice accuracy** (predicted option vs. ground-truth option), reported per core task and averaged. Two evaluation regimes: a **sliding-window setting** giving offline models a fixed 32 s context at 2 fps ending at $t_q$, and a **streaming setting** feeding *all* frames from clip start to $t_q$ at 2 fps (the realistic online regime).

### VideoChat-Online model
Base: **InternVL2-4B** (InternViT-300M vision encoder at 448×448, Phi-3 LLM, MLP language connector). Training samples 1 fps, up to 64 frames.

**Pyramid Memory Bank.** Split memory into $n$ layers $\{m_i\}_{i=1}^{n}$; deeper layers trade spatial detail for temporal coverage. Each layer $i$ has a **sampling rate** $r_i$ (increasing with depth) and a **resolution** that decays geometrically:
$$\text{Res}_i = \frac{\text{Res}_1}{\beta^{\,i-1}}, \qquad \beta = 2 .$$
Each layer runs three operations:
1. **Streaming frame writing** — accept frames at rate $r_i$ into a queue of capacity $C_i$; when full, trigger eviction.
2. **Frame eviction & down-writing** — find the most similar *adjacent* frame pair $(f_a^i, f_b^i)$ by cosine similarity of average-pooled frame features, **evict the older frame**, and pass a coarser copy down to layer $i{+}1$:
$$f_{\text{next}}^{\,i+1} = \text{AvgPool2d}\!\left(f_{\text{evicted}}^{\,i},\ \text{Res}_{i+1}\right).$$
3. **Readout** — at query time, all stored frames across all layers are concatenated **in temporal order** and prefilled.

**KV-cache synchronization.** Because each new frame updates the bank, naive memory-compression methods must reprocess the whole compressed memory per step. Instead PMB aligns with the LLM KV-cache: after evicting frames at timestamps $t_{f_a}, t_{f_b}$, it drops exactly the cached tokens after the earlier of the two:
$$\text{KVCache} \leftarrow \text{KVCache} \setminus \{\, t_i \mid t_i > \min(t_{f_a}, t_{f_b}) \,\}.$$
So frame tokens are computed once and cached; only invalidated suffixes are recomputed, enabling real-time prefill/decode.

**Inference memory config (Table 16).** Three layers — $m_s$: 1 fps, 448×448, capacity 2 frames, 256 tokens/frame; $m_{main}$: 2 fps, 224×224, cap 2, 64 tokens/frame; $m_t$: 8 fps, 112×112, cap 12, 16 tokens/frame — totalling **832 tokens** resident (vs ~9,984 tokens for the dense offline evaluation).

**Offline-to-online learning.** Two-stage progressive training: first offline instruction tuning (VideoChat2-IT, STAR, PerceptionTest, ShareGPT4V/4o, LLaVA-OneVision), then online tuning on ~96K samples (5 task types, 12 datasets; ~6% of total data) reformatted into an **interleaved multi-turn dialogue** where visual frames and QA turns are threaded along the timeline (Fig. 4). Progressive (offline→online) beats joint training.

## Explicit design choices
- **Timestamp-anchored MCQ** with distractors drawn from *other timestamps of the same video* — the mechanism that makes the benchmark test temporal grounding, not static recognition.
- **Three explicit temporal contexts (Past / Current / Future)** as the top-level axis; 6 tasks → 16 subtasks below it.
- **Two eval regimes** — sliding-window (32 s @ 2 fps, fair to offline models) and streaming (all frames to $t_q$ @ 2 fps, the online test).
- **Benchmark reuses existing dense spatiotemporal annotations** from val/test splits (no leakage) instead of collecting raw video, keeping annotation quality high (~7K human-verified items).
- **Hierarchical (pyramidal) memory**: recent = high-res/low-fps; distant = low-res/high-fps, geometric resolution decay $\beta=2$.
- **Eviction by adjacent-frame cosine similarity** (drop the more redundant of the closest pair) rather than FIFO / uniform / token-merge.
- **Down-writing (average-pool to next layer)** instead of hard dropping evicted frames — preserves coarse long-range signal.
- **KV-cache-aligned memory update** so tokens are precomputed once; only suffix after an eviction is invalidated — the real-time enabler.
- **Small 4B backbone** (InternVL2-4B / Phi-3) chosen to demonstrate efficiency, not scale.
- **Interleaved multi-turn dialogue format + progressive offline→online training** as the data/training recipe.

## Key results / what to remember
No Zotero highlights present.

- **OVBench streaming setting:** VideoChat-Online (4B) averages **54.9%**, vs Flash-VStream (7B) **31.2%** and MovieChat (7B) **30.9%** — a ~**+23.7** point gain over Flash-VStream at *smaller* size; VideoLLM-Online scores only ~9.6% (struggles to emit valid options).
- **OVBench sliding-window setting:** VideoChat-Online (4B) **53.9%** average, edging Qwen2-VL-7B (~49.7%) and Gemini-1.5-Flash (~50.7%) — i.e. beats a 7B model by ~4 points at 4B.
- **Efficiency (Table 7):** VideoChat-Online runs OVBench at **8.71 GB** VRAM vs Flash-VStream **16.03 GB** and MovieChat **16.90 GB**; the dense InternVL2-4B baseline OOMs on the streaming input.
- **Memory-structure ablation (Table 4):** full PMB **54.9%** overall; removing $m_t$ 54.2, $m_{main}$ 54.3, $m_s$ 53.7 — each layer contributes, with the biggest spatial-perception drop when $m_s$ is removed.
- **Update-policy ablation (Table 5):** proposed eviction **54.9%** OVBench / **47.1%** VideoMME-Long, beating FIFO (54.0 / 41.3), Uniform (52.1 / 45.0), Token-Merge (51.5 / 43.9); no-compression OOMs on VideoMME-Long.
- **Training-strategy ablation (Table 6):** offline-only 44.1% → +online data 45.2% → +interleaved format 51.8% → +progressive 53.9% (**+9.8** total; the interleaved format is the largest single jump).
- **Offline generalization (Table 3):** competitive at 4B — MVBench ~65.2%, MLVU ~60.8%, VideoMME overall ~54.4% / long ~47.1%, EgoSchema ~54.7% (exact peer-comparison rows partly (n/r) here — verify against Table 3 before quoting head-to-head).

## How it connects (evolution)
- [[streaming-benchmarks]] / [[streamingbench]] / [[ovo-bench]] / [[svbench]] — sibling online/streaming video QA benchmarks; OVBench's distinctive axis is the timestamp-anchored Past/Current/Future taxonomy with same-video distractors.
- [[streaming-memory]] — the PMB is a hierarchical, KV-cache-synchronized memory; this is the note's cross-link into the memory lineage.
- [[flash-vstream]] / [[rekv]] — memory-compression streaming baselines OVBench outperforms; PMB's KV-sync directly targets their per-step reprocessing bottleneck.
- [[videollm-online]] — an early streaming dialogue model used as a baseline (weak at MCQ emission on OVBench).
- [[dispider]] / [[mmduet]] — contemporaneous streaming interaction models addressing the same "answer at the right moment" problem.

## Open questions / limitations
- **MCQ-only scoring** measures option selection, not free-form streaming dialogue quality or precise temporal localization; open-ended online generation is untested here.
- **Fixed hand-tuned pyramid** (3 layers, set capacities/fps/resolutions) — no learned or adaptive allocation of the memory budget to scene difficulty.
- **Redundancy-based eviction** (adjacent cosine similarity) can discard rare-but-important transient events that happen to sit next to a similar frame.
- **Benchmark built on reused val/test annotations** inherits those datasets' domain/label biases; the 6-domain coverage is broad but not exhaustive of real streaming settings.

*Verification: equations (resolution decay $\beta{=}2$, down-writing AvgPool2d, KV-cache eviction rule) and PMB layer config transcribed from the page-5 method text + Table 16 as rendered from the PDF; headline OVBench numbers (streaming 54.9, sliding-window 53.9, VRAM 8.71 GB) and ablation deltas cross-checked against the arXiv HTML extraction of Tables 2/4/5/6/7. A few offline-benchmark peer rows in Table 3 could not be fully disambiguated from the HTML and are marked (n/r).*

---
zotero_key: null
authors: Yifei Li, Junbo Niu et al. (Shanghai AI Laboratory / Tsinghua / Beihang / CUHK / SenseTime)
year: 2025
arxiv: 2501.05510
pdf: https://arxiv.org/pdf/2501.05510
tier: deep
subtopics: [streaming-benchmarks]
tags: [streaming-video-understanding, streaming-benchmarks]
---
# OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding?

**Lineage role:** The temporal-awareness streaming benchmark built around three timestamp-conditioned regimes — **backward tracing**, **real-time perception**, **forward active responding** — that exposes a ~30-point human-vs-model accuracy gap (best model 63.00% vs human 92.81%); CVPR 2025, a sibling of [[streamingbench]] in the online-video evaluation family.

## Problem — what was limited before this paper
Offline video-LLMs are scored on the *whole* video for a static, post-hoc answer, so their benchmarks never test **temporal awareness**: the ability to answer differently depending on *when* the question is asked along a still-arriving stream. A real online assistant must instead do three things at once — recall what already happened, judge what is happening right now, and sometimes *wait* until enough evidence arrives before answering. Prior streaming benchmarks (e.g. VStream-QA, [[streamingbench]]) only partially capture this: they largely reuse offline-style QA and do not systematically tie each query to a timestamp with clues distributed across past/present/future. OVO-Bench fills that gap.

## Key idea
Anchor every query to an explicit **timestamp $t$** and feed the model only the video prefix $\text{Video}[0{:}t]$, simulating a live stream. Organize tasks by *when the answer-bearing evidence lives relative to $t$*: **Backward Tracing** (evidence in the past), **Real-Time Visual Perception** (evidence at the current moment), and **Forward Active Responding** (evidence has not arrived yet, so the model must decide whether to answer now or keep watching — a "Video Chain-of-Time" analog of chain-of-thought). This turns *timing of the response* itself into part of the evaluation, not just answer correctness.

![[ovo-bench.png]]
> **Crux (Figure 2).** The task taxonomy: 12 fine-grained subtasks grouped into the three online-perception regimes (Backward Tracing / Real-Time Visual Perception / Forward Active Responding), each with a timestamped example query and a filmstrip showing where "clues", "misleading" scenes, and the "query" moment fall along the stream. *Li, Niu et al. (2025), arXiv:2501.05510. Embedded for personal research reference.* (The printed caption says "14 tasks"; the taxonomy itself lists 12 subtasks.)

## Method + math — the evaluation protocol

**Task taxonomy (3 regimes, 12 subtasks).**
- **Backward Tracing** — answer needs past events: **[EPM]** Episodic Memory (retrieve a key past moment), **[ASI]** Action Sequence Identification (recover correct order of past actions), **[HLD]** Hallucination Detection (reject a question about something never shown — correct answer is "unable to answer").
- **Real-Time Visual Perception** — answer needs the current frame(s): **[STU]** Spatial Understanding, **[OJR]** Object Recognition, **[ATR]** Attribute Recognition, **[ACR]** Action Recognition, **[OCR]** text reading, **[FPD]** Future Prediction (next probable phase).
- **Forward Active Responding** — answer's evidence is still in the future: **[REC]** Repetition Event Count (fire each time an event recurs), **[SSR]** Sequential Steps Recognition (respond as each procedural step is completed), **[CRR]** Clues Reveal Responding (withhold until the disambiguating clue appears).

**Streaming query construction.** For a Backward/Real-Time query posed at $t_i$, the visual input is the prefix clip $\text{Video}[0{:}t_i]$. To make questions harder in the main table, the query time for **[EPM]** and **[ASI]** is pushed to the *end* of the video, maximizing the temporal gap between the clue and the question.

**Scoring — Backward Tracing & Real-Time Visual Perception.** Standard multiple-choice accuracy against the ground-truth option; options are visually grounded so text priors cannot leak the answer, and the number of choices varies (2–5) rather than being fixed at four.

**Scoring — Forward Active Responding.** The model is queried *densely* along the temporal axis (multiple-triggering pipeline); it must respond at the right moment $m'$ relative to the ground-truth moment $m$. Two metric families:

Accuracy-based (used in the main Table 1):
$$\text{Acc} = \frac{1}{N}\sum_{m} \mathbb{F}(R_{m'}, A_{m})$$
where $\mathbb{F}=1$ if the response matches the answer (semantic match for open-ended tasks judged by GPT-4o), else $0$.

Score-based (rewards responding close to the right time, and — for [REC] — larger correct counts). For **[REC]**:
$$\text{Score} = \sum_{m} e^{\,i\cdot p_1}\cdot \mathbb{F}(R_{m'},A_m)\cdot 2^{-(m'-m)\,p_2},\qquad p_1=0.2,\ p_2=0.05$$
For **[SSR]** and **[CRR]**:
$$\text{Score} = \sum_{m} \mathbb{F}(R_{m'},A_m)\cdot 2^{-(m'-m)\,p},\qquad p=0.5$$
The $2^{-(m'-m)p}$ term is a temporal-locality decay: an answer that is correct but late (large $m'-m$) is discounted; the $e^{i p_1}$ term rewards higher counting indices $i$ in [REC].

**Data construction pipeline (Figure 3).** (1) Collect 644 unique videos across 7 domains (sports, video games, tutorials, ego-centric, etc.). (2) Meta-annotate three ways — repurpose existing dense-timestamp annotations; semi-automatic (Gemini-1.5 proposes coarse timestamps, humans refine); and full human annotation for [SSR]/[CRR]. (3) Generate QA via GPT-4o with rule-based transforms plus human curation, with visually grounded distractors. Result: ~2,800 fine-grained timestamped QA/meta-annotations; average query timepoint ≈ 428.89 s (videos range up to ~30 min).

## Explicit design choices
- **Timestamp-conditioned input** ($\text{Video}[0{:}t]$) is the core primitive — the same video yields different correct answers at different $t$.
- **Regime = location of evidence in time** (past / present / future) rather than by semantic topic — this is what makes it a *streaming* benchmark.
- **A "no-answer" option for [HLD]** ("Unable to answer") to directly probe temporal hallucination instead of forcing a guess.
- **Forward tasks demand response *timing*, not just content** — dense multiple-triggering querying + time-decayed scoring; an offline model that always outputs "Yes" can still score nonzero (a fairness caveat the authors flag).
- **Visually grounded distractors + variable option count (2–5)** to block language-prior shortcuts.
- **Consistency principle**: same frame count / fps across all evaluated models; offline models get prefix clips segmented at the query timestamp.
- **[EPM]/[ASI] queries placed at video end** to widen the clue-to-query gap.

## Key results / what to remember
All numbers verified against Table 1 (accuracy-based; higher = better).
- **Headline gap:** best model **Gemini 1.5 Pro = 63.00% overall** vs **Human Agents = 92.81%** → a **~29.8-point** human-model gap (the "~30 pt" lineage claim, confirmed).
- **Per-regime, Gemini 1.5 Pro / Human:** Real-Time **69.32 / 93.20**; Backward Tracing **62.54 / 92.33**; Forward Active **57.15 / 92.90**.
- **GPT-4o (64 frames):** overall **59.54** (Real-Time 64.46, Backward 60.75, Forward 53.40).
- **Other offline models (overall):** Qwen2-VL-72B **56.27**, LLaVA-Video-7B **52.91**, LLaVA-OneVision-7B **52.74**, Qwen2-VL-7B **50.39**, InternVL-V2-8B **50.15**, LongVU-7B **46.71**.
- **Online models underperform:** Dispider **41.78**, Flash-VStream-7B **33.61** (its Real-Time is only **28.37** vs Gemini's 69.32); VideoLLM-online-8B managed only Real-Time **20.79** / Backward **17.73** (overall n/r — dashes for Forward Active).
- **Hardest real-time skills:** even the best proprietary model scores **STU 58.43%** and **ACR 66.97%** vs human 92.70 / 92.57 — the widest current-perception gaps.
- **Hallucination [HLD]:** Gemini 1.5 Pro **52.64%** (text rounds to 52.69%) vs human **91.37%**; proprietary > open-source > online here.
- **Latency wall:** at 64 frames even efficient models (Qwen2-VL-7B, Flash-VStream) need **~4 s** per response — impractical for real-time dialogue; latency grows roughly exponentially with frame count (Figure 6).
- **Offline capability partially transfers:** offline models are competitive on Real-Time Perception but collapse on Forward Active Responding (they were never built to *wait*).

No Zotero highlights present.

Takeaways: online video understanding is far from solved; offline strength does not buy you *timing*; the benchmark is explicitly positioned as a target for **future** online models rather than a fair test of today's streaming systems.

## How it connects (evolution)
- [[streamingbench]] — the closest sibling streaming benchmark; OVO-Bench's timestamp-regime framing sharpens what StreamingBench measures.
- [[dispider]] and [[videollm-online]] — the online/streaming models evaluated here, both of which trail offline LLMs badly (Dispider 41.78, VideoLLM-online real-time 20.79).
- [[flash-vstream]] — the streaming-memory model whose real-time score (28.37) exemplifies the accuracy-vs-latency trade-off OVO-Bench exposes.
- [[svbench]] and [[proactivevideoqa]] — parallel efforts probing temporal/streaming and proactive-response evaluation.
- [[streaming-benchmarks]] — the sub-topic hub this note anchors.
- [[streaming-video-understanding]] — the topic hub.

## Open questions / limitations
- **Data diversity/bias:** scarcity of fitting dense-timestamp annotations, weak automatic QA generation, and high human-labeling cost limit domain coverage and may introduce bias.
- **Forward Active fairness:** offline models weren't designed to withhold responses, so they often random-guess; a degenerate "always say Yes" policy still scores nonzero — the metric only partly controls for this.
- **No strong online model exists yet:** the benchmark is more a signpost for future systems than a discriminative test of current online architectures.
- **Latency not in the headline score:** the ~4 s/response cost is reported separately (Figure 6), so a model can "pass" on accuracy while being unusable live.

*Verification: all Table 1 accuracy numbers, the human 92.81 / best-model 63.00 gap, per-regime and per-task figures (STU 58.43, ACR 66.97, HLD 52.64), and the Forward Active scoring formulas were read directly from the rendered PDF (arXiv:2501.05510v2, pages 6–7 incl. Table 1 and Figure 2 taxonomy); dataset counts (644 videos, ~2,800 QAs, 428.89 s avg query time) are from the paper text/abstract via the arXiv HTML.*

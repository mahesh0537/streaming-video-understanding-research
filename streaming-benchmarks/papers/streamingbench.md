---
zotero_key: null
authors: Junming Lin, Zheng Fang, Chi Chen, et al. (Tsinghua University / AIR / BUPT)
year: 2024
arxiv: 2411.03628
pdf: https://arxiv.org/pdf/2411.03628
tier: deep
subtopics: [streaming-benchmarks]
tags: [streaming-video-understanding, streaming-benchmarks]
---
# StreamingBench: Assessing the Gap for MLLMs to Achieve Streaming Video Understanding

**Lineage role:** The foundational timestamped streaming-video eval — it first formalizes the three axes (real-time / omni-source / contextual) that later streaming benchmarks inherit and extend, and exposes that even frontier MLLMs sit ~25 points below humans on streaming (query-at-a-moment) video QA.

## Problem — what was limited before this paper (short)
Video MLLMs were benchmarked almost entirely *offline*: the whole clip is preloaded, then a single question is asked "with the entire video visible." That protocol cannot test the things that make streaming hard — (1) a query can arrive at *any* moment, so the correct answer depends on how much video has elapsed; (2) audio and vision arrive synchronized in real time; (3) long streams accumulate redundant context and interaction history that must be reasoned over. The only prior streaming-ish benchmark, VStream-QA, was narrow (32 videos from Ego4d/MovieNet, 5 task types, vision-only, questions mutually independent) — too small and single-modal to assess streaming ability broadly.

![[streamingbench.png]]
> **Crux (Figure 1).** Offline benchmarks ask one question about the *entire* visible video; StreamingBench instead poses questions pinned to a specific "current time" and splits streaming competence into three task families — Real-Time Visual (answer changes with when you ask), Omni-Source (must fuse concurrent audio+vision), and Contextual (must use prior QA/interaction history). *Lin et al. (2024), arXiv:2411.03628. Embedded for personal research reference.*

## Key idea — the core insight
Turn "when the question is asked" into a first-class variable. Each QA pair carries a timestamp, and a model may only see the stream *up to that timestamp* — so identical questions ("What words are currently shown?") have different correct answers at 00:24 vs 01:26. On top of this timestamped protocol, StreamingBench decomposes streaming understanding into three orthogonal capability groups (real-time perception, omni-source audio-visual fusion, and context/history use) spanning 18 fine-grained tasks, giving the first comprehensive picture of *where* MLLMs fail in a streaming setting rather than a single aggregate score.

## Method + math — the eval protocol
For a benchmark the "math" is the construction pipeline, task taxonomy, timing model, and scoring rules.

**Scale & sources.** 900 human-curated YouTube videos across 8 categories (life record, competition, education, TV show, video games, documentary, animation & movie, unusual event); durations 3 s to ~24 min. 4,500 multiple-choice QA pairs (4 options each), 5 questions per video. Split: real-time visual = 500 videos / 2,500 Q; omni-source = 200 videos / 1,000 Q; contextual = 200 videos / 800 Q (plus 250 proactive-output questions created, of which 50 evaluated in this release).

**Task taxonomy (18 tasks in 3 groups).**
- *Real-Time Visual Understanding (10):* Object Perception (OP), Causal Reasoning (CR), Clips Summarization (CS), Attribute Perception (ATP), Event Understanding (EU), Text-Rich Understanding (TR), Prospective Reasoning (PR), Spatial Understanding (SU), Action Perception (ACP), Counting (CT).
- *Omni-Source Understanding (4):* Emotion Recognition (ER), Scene Understanding (SCU), Source Discrimination (SD), Multimodal Alignment (MA) — all require concurrent audio+vision.
- *Contextual Understanding (4):* Misleading Context Understanding (MCU), Anomaly Context Understanding (ACU), Sequential Question Answering (SQA), Proactive Output (PO).

**Construction pipeline (hybrid).** For real-time visual + proactive tasks: sample frames at 1 fps → stamp each frame's top-left corner with its time → GPT-4o captions every 20 frames, producing timestamped fine-grained captions (<20 s granularity) → GPT-4o generates QA pairs per task and auto-assigns each a query timestamp. For omni-source and other contextual tasks: human annotators write the QA pairs manually. Quality control: every pair (auto or manual) is human-verified for accuracy/clarity/relevance; pairs answerable *without* the video are discarded; options are shuffled for balance.

**Timing / evaluation model.** Let $t_q$ be a question's query timestamp and $V_{[0,t_q]}$ the stream up to it. The model is fed only $V_{[0,t_q]}$ and scored by multiple-choice accuracy
$$\text{Acc} = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}\!\left[\hat{a}_i = a_i^{\star}\right].$$
- *Real-time & omni-source:* clip the video at $t_q$, answer offline on that prefix (omni-source additionally supplies the concurrent audio).
- *SQA (contextual):* prior QA pairs are appended as text, formatted `{Timestamp1}: {QA1}; …; Answer accordingly: {current question}`, so the model must resolve references to earlier interactions.
- *PO (proactive output):* a polling scheme — within $\pm 4$ s around the ground-truth timestamp $t^\star$ the model is queried every second "Is it the right time to output?"; a response counts correct only if it actually fires within a 2-second window of $t^\star$, i.e. $|t_{\text{out}} - t^\star| \le 2$.

**Temporal-clue analysis (Figure 4 / Table 4).** Questions are cross-classified on two axes: clue *position* relative to the query — Prior / Concurrent / Subsequent — and clue *count* — Single vs Multiple, used to diagnose which temporal regimes models handle.

## Explicit design choices
- **Timestamp as identity of a question**: model sees only the prefix up to $t_q$; forbids leaking future frames — the mechanism that makes "same question, different time → different answer."
- **Three-axis taxonomy** (real-time / omni-source / contextual) rather than one flat set — isolates perception vs audio-visual fusion vs context/history as separable skills.
- **Multiple-choice, 4 options** everywhere → cheap, deterministic accuracy scoring; option shuffling to kill positional bias.
- **Hybrid annotation**: GPT-4o auto-generation for the large real-time set (scales to 2,500 Q) but human labeling for audio-dependent and contextual tasks GPT-4o can't reliably produce.
- **On-frame timestamp burn-in** (time printed on each frame) so the captioner gets sub-20s temporal granularity.
- **Proactive output via ±4s polling with a 2s tolerance** — an explicit operationalization of "answer at the right moment," not just "answer correctly."
- **Omni-source requires audio**: some questions are unanswerable from vision alone (e.g. matching a spoken line to a character), forcing true multimodal fusion.
- **SQA reference-resolution**: earlier QA fed back as text history to test cross-turn coreference in a stream.

## Key results / what to remember
All numbers are StreamingBench overall accuracy (%) unless noted, verified against the paper's Table 2 / Table 6.
- **Human ceiling: 91.66%** overall (Table 6 human row). Best model trails by **24.59 points**.
- **Gemini 1.5 Pro: 67.07%** overall — the top model. By group: Real-Time Visual **75.69**, Omni-Source **60.22**, Contextual **48.73** — a clear monotone decline as tasks demand more fusion/history.
- **GPT-4o: 60.15%**; **Claude 3.5 Sonnet: 57.68%**.
- Best open-source: **LLaVA-OneVision (7B) 56.36**, Qwen2-VL (7B) 54.14, MiniCPM-V 2.6 (8B) 53.85, LLaVA-NeXT-Video (32B) 52.77, InternVL2 (8B) 51.40, Kangaroo (7B) 51.10.
- **Purpose-built streaming MLLMs do *worse* than general offline MLLMs** (Table 6): VideoLLM-online (8B) **32.48** overall, Flash-VStream (7B) **24.04** — both near or below chance-adjusted floors on many tasks (they often just emit "A" or flood redundant text). This is the paper's most pointed finding for the streaming-benchmarks lineage: existing "streaming" models were not actually competitive on comprehensive streaming QA.
- Contextual understanding (esp. proactive output and reference resolution) is the universal weak point across all models.

No Zotero highlights present.

Takeaways: (1) streaming ≠ offline — the timestamp-prefix protocol is the load-bearing idea; (2) a three-axis taxonomy is enough to localize failures (perception is near-solved, fusion + context are not); (3) as of late-2024, dedicated streaming architectures underperformed strong offline MLLMs on this eval, motivating the wave of later streaming models and richer benchmarks.

## How it connects (evolution)
- [[videollm-online]] — the canonical online/streaming MLLM; StreamingBench evaluates it (32.48) and shows it lags offline models, framing the gap later work chases.
- [[flash-vstream]] — the other dedicated streaming model benchmarked here (24.04), used as evidence that streaming-specific memory designs weren't yet enough.
- [[ovo-bench]], [[svbench]] — successor streaming benchmarks that extend/critique the timestamped real-time evaluation StreamingBench introduced.
- [[omnimmi]] — pushes the omni-source (audio+vision, multi-turn) axis StreamingBench opened.
- [[proactivevideoqa]] — deepens the proactive-output / "answer at the right moment" task defined here.
- [[streaming-benchmarks]] — sub-topic hub tying these evals together.

## Open questions / limitations
- **Auto-generated real-time QA inherits GPT-4o's blind spots**: captions and questions come from the same model family being tested, risking correlated errors despite human verification.
- **Multiple-choice only** — measures recognition among 4 options, not open-ended generation, latency, or streaming throughput; a model can score well while being unusable in real time.
- **Proactive output under-evaluated** — only 50 of 250 PO questions scored in this release, and the ±4s/2s polling scheme is a coarse proxy for genuine real-time triggering.
- **Prefix-clip protocol is still offline-executed**: most tasks clip the video and answer "offline on the prefix," so it tests *timestamp-conditioned* understanding more than true incremental/causal streaming inference.

*Verification: task taxonomy, construction pipeline, and timing/PO protocol checked against the arXiv HTML (Sec. 3) and the rendered PDF pages 1/5; all headline numbers (91.66 human, 67.07 Gemini 1.5 Pro and its 75.69/60.22/48.73 subgroups, 60.15 GPT-4o, open-source rows) checked against Table 2, and the streaming-model rows (VideoLLM-online 32.48, Flash-VStream 24.04) read directly off Table 6 on the rendered page.*

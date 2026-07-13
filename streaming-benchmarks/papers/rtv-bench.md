---
zotero_key: null
authors: Shuhang Xun, Sicheng Tao, Jungang Li, Yibo Shi, Zhixin Lin et al. (HIT, HKUST(GZ), HKUST, XJTU, SDU, CityU, HUST)
year: 2025
arxiv: 2505.02064
pdf: https://arxiv.org/pdf/2505.02064
tier: deep
subtopics: [streaming-benchmarks]
tags: [streaming-video-understanding, streaming-benchmarks]
---
# RTV-Bench: Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video

**Lineage role:** A streaming-video QA benchmark whose signature is *Multi-Timestamp Question Answering* — the same conceptual query re-asked as the scene evolves so the correct answer changes over time — built on sport / driving / egocentric footage to stress continuous (not snapshot) understanding.

## Problem — what was limited before this paper (short)
Prior video-QA benchmarks (including long-video and even earlier streaming ones like [[ovo-bench]] / [[streamingbench]]) largely test a snapshot: one query gets one static answer, so a model can succeed by locating a single relevant frame. They inadequately probe whether an MLLM can maintain a *coherent, continuously updated* internal state as a live stream evolves — the "memory-loss under state change" failure mode. Real-time responsiveness and answer-tracking across transitions were under-measured.

## Key idea — the core insight, 2-4 sentences
RTV-Bench builds three design principles into the data itself. (1) **Multi-Timestamp QA (MTQA):** the *same* underlying question is posed at different timestamps and its correct answer shifts as the video unfolds — the model must actively track and update, not just retrieve. (2) **Hierarchical question groups:** each group has foundational questions (q0, q1) and a harder advanced question (q2) that logically depends on them, blocking cognitive shortcuts. (3) **Multidimensional evaluation** across eight capability dimensions grouped into Perception / Understanding / Reasoning, rather than one aggregate score.

![[rtv-bench.png]]
> **Crux (Figure 1).** The task taxonomy in action: eight dimensions under Perception (TP/SP/VP), Understanding (IA/PU/GU), and Reasoning (SR/FP). Within each question group, ①② are static foundational questions (answers underlined) while ③ is a *dynamically answered* question whose correct choice depends on the query timestamp — the MTQA mechanism that is the benchmark's core. *Xun et al. (2025), arXiv:2505.02064. Embedded for personal research reference.*

## Method + math — eval protocol, taxonomy, metrics
**Data construction.** 552 videos, ~4,631 QA pairs, 167.2 total hours (18.2 min average), spanning **3 domains** — intelligent driving, sports events, egocentric footage — across **16 subcategories**. Sources include EgoSchema plus curated online videos. QA templates are LLM-generated then human-expert-refined; timestamps of answer changes are **manually verified**. Questions are organized into groups of **≥3 multiple-choice** questions with escalating difficulty; the information needed for q2 generally subsumes what q0/q1 need. Video-query positions are bucketed as Shallow (0–3 min), Moderate (3–10 min), Deep (10 min+).

**Eight task dimensions.**
- Perception: **TP** Temporal Perception, **SP** Scene Perception, **VP** Visual Perception
- Understanding: **PU** Phenomenological Understanding, **IA** Intent Analysis, **GU** Global Understanding
- Reasoning: **SR** Spatiotemporal Reasoning, **FP** Future Prediction

**MTQA formalization.** For a dynamic query, the ground-truth answer is time-dependent: $A^\*(Q, t_q)$ — the correct choice for conceptual query $Q$ depends on the query time $t_q$. The two model families are evaluated on a matched footing:
- **Online (streaming) models** ingest the stream $V[0,t]$ continuously, maintain state $S_{t}$, and answer at query time from that state:
$$A_{M_\text{online}} = M_\text{online}\!\left(Q,\, t_q \mid S_{t_q}\right)$$
- **Offline models** lack streaming state, so real-time interaction is *simulated*: at query time $t_{q,i}$ the relevant segment $V_i$ is extracted and the model answers from that isolated segment only:
$$A_{M_\text{offline},i} = M_\text{offline}\!\left(Q_i,\, V_i\right)$$

**Metric 1 — Accuracy.** Proportion of model choices matching ground truth over all questions.

**Metric 2 — Score (group-aware, conditional Q2).** Rewards advanced-question (q2) correctness *only when* the group's foundational questions (q0, q1) are all correct — penalizing superficial success on hard questions without the underlying basics:
$$\text{Score} = \frac{\sum_{i=1}^{N} B_i \cdot N^{\text{correct}}_{q2,i}}{\sum_{i=1}^{N} N^{\text{total}}_{q2,i}}$$
where $N$ is the number of valid groups (those containing a q2), $B_i \in \{0,1\}$ is the prerequisite indicator (1 iff all basic questions in group $i$ are correct), $N^{\text{correct}}_{q2,i}$ is the count of correct q2 answers in group $i$, and $N^{\text{total}}_{q2,i}$ the total q2 count. This aligns with hierarchical learning: complex skills must build on mastered simple ones.

**Reported splits.** Results are broken out per dimension and also as **FQA** (foundational QA, no multi-timestamp supervision) vs **MTQA** (multi-timestamp, time-varying answers) — the FQA→MTQA drop isolates the continuous-tracking difficulty.

## Explicit design choices
- **Answers deliberately change over time** for a fixed conceptual question — the benchmark annotates the correct option *per query timestamp*, so retrieval-of-one-frame does not suffice.
- **Hierarchical q0/q1/q2 groups** with a **conditional (gated) scoring rule** — advanced credit is void unless basics are correct.
- **Matched evaluation harness** for online vs offline models: offline gets an isolated relevant segment $V_i$; online must answer from its running state $S_{t_q}$ (no re-feeding the whole video).
- **Eight fine-grained dimensions** reported separately, plus a Perception/Understanding/Reasoning grouping — diagnostic rather than a single leaderboard number.
- **Domain focus on high-dynamics footage** (driving, sports, egocentric) where scene state genuinely evolves fast.
- **Human-verified timestamps** for every answer transition; LLM-drafted, expert-refined QA.
- **Visual-only** by design (no audio track used).

## Key results / what to remember
All numbers from Table 2 (Overall = Acc % / Score), verified against the paper.
- **GPT-4o (closed):** best overall, **50.02% Acc / 22.10 Score**; FQA 56.53% vs **MTQA 44.73%** — a clear ~12-pt FQA→MTQA drop even for the top model.
- **IXC2.5-OL (7B, online):** 47.33% / 15.40 — best open-source online model overall.
- **VideoChat-Online (4B, online):** 45.83% / 12.10.
- **VITA-1.5 (7B, online):** 44.51% / 11.80 — notably, this *lowest* online model still beats the offline VideoLLaMA2 (39.55%), supporting "streaming-designed > offline."
- **Gemini 2.0 Flash (closed):** 42.00% / 12.00.
- **Offline baselines (7B):** Qwen2.5-VL 40.41% / 7.13; VideoLLaMA2 39.55% / 7.90; LLaVA-Video 34.90% / 4.80; LLaVA-OneVision 34.49% / 4.40.
- **Headline findings:** (1) most models score **below 50%** overall; (2) performance correlates positively with model scale but with diminishing / non-monotonic returns; (3) increasing sampled frames (8→64) gives only marginal and sometimes *degrading* gains; (4) streaming-designed (online) models consistently outperform offline ones; (5) the online/offline gap is widest on MTQA.

No Zotero highlights present.
Takeaways: the FQA→MTQA gap is the reusable signal — it quantifies "can the model update as state changes" separately from raw perception. The gated Score is a transferable idea for penalizing shortcut-correct answers on compositional tasks.

## How it connects (evolution)
- [[ovo-bench]] — RTV-Bench explicitly contrasts with OVO-Bench, which poses *different* questions at different timestamps; RTV-Bench instead revisits the *same* conceptual query with a shifting answer.
- [[streamingbench]] — sibling streaming-QA benchmark; RTV-Bench pushes harder on continuous answer-tracking (MTQA) and gated hierarchical scoring.
- [[svbench]] / [[river-bench]] — related streaming/real-time video-understanding benchmarks in the same family of evolving-scene evaluation.
- [[ovbench-videochat-online]] / [[ovbench-videochat-online]] — VideoChat-Online is one of the strongest online models evaluated here.
- [[streaming-benchmarks]] — parent sub-topic hub for these evaluation suites.

## Open questions / limitations
- **Visual-only:** audio (commentary, engine/ambient sound) is ignored, though it carries real-time cues for exactly these domains.
- **No mechanistic explanation** for the counter-intuitive frame-scaling result (more frames ↛ better) — left as future work.
- **MTQA-as-metric coupling:** temporal sensitivity is measured only through the MTQA question design; there is no separate quantitative "answer-change tracking" score isolating *when* a model updates.
- **Multiple-choice format** may under-test open-ended generation and could permit lucky guessing that the gated Score only partly controls.

*Verification: metric formulas (Score, MTQA $A^\*(Q,t_q)$, online/offline answer definitions) transcribed from the rendered PDF method page (p.6); all headline numbers cross-checked against the paper's Table 2; dataset scale and taxonomy cross-checked against arXiv HTML (v1) and Figures 1–2. QA-pair count reported as ~4,631 (arXiv HTML v1); some abstract summaries list 4,608.*

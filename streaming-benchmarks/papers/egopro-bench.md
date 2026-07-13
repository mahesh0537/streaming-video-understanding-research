---
zotero_key: null
authors: Dongchuan Ran, Linyu Ou, Xueheng Li, Wenwen Tong, Chenxu Guo, Hewei Guo, Kaibing Wang, Lewei Lu
year: 2026
arxiv: 2605.07299
pdf: https://arxiv.org/pdf/2605.07299
tier: deep
subtopics: [streaming-benchmarks]
tags: [streaming-video-understanding, streaming-benchmarks]
---
# EgoPro-Bench: Benchmarking Personalized Proactive Interaction in Egocentric Video Streams

**Lineage role:** the first benchmark to score *personalized* proactive assistance in egocentric streams — pairing precise intervention-timing metrics (F1 / mIoU / GHA over ground-truth intervals) with a persona-conditioned intent branch, backed by 2,400 eval + 12K+ train videos and a companion "short thinking" streaming model (ProAct-Stream).

## Problem — what was limited before this paper (short)
Streaming/video MLLMs are still fundamentally **reactive**: they answer when explicitly queried and do not continuously watch the scene to decide *when* to speak on their own. Prior proactive benchmarks (e.g. [[omnimmi]], [[ovo-bench]], [[streamingbench]], [[proactivevideoqa]]) test whether a model can fire at the right moment, but they judge a generic "correct alert" and largely ignore **who the user is** — the same scene should trigger different assistance for a chef vs. a visually-impaired pedestrian vs. a tourist. There was no benchmark that jointly measured (a) temporally-precise proactive triggering *and* (b) whether the volunteered response is consistent with a persistent, personalized user memory.

## Key idea — the core insight, 2-4 sentences
EgoPro-Bench splits proactive interaction into two complementary branches: an **event-driven** branch (fire an alert the instant a transient object or action appears — pure timing/perception) and an **intent-driven** branch (volunteer help that matches a *synthesized user persona + memory* across 10 daily-living domains). Data is manufactured by a two-track synthesis pipeline: streaming visual analysis for event triggers, and hierarchical persona → domain → memory → response generation for personalized intent, with strict filtering and human checks. Alongside the benchmark the authors train **ProAct-Stream**, a Qwen3-VL-8B streaming model taught "short thinking, better interaction" — a tightly length-budgeted `<think>` before each decision so reasoning does not stall the real-time stream.

![[egopro-bench.png]]
> **Crux (Figure 3).** The two-branch data-synthesis pipeline: the event-driven track (top) collects egocentric video → filters/annotates objects & actions → grounds candidate times → human-verified event data; the intent-driven track (bottom) samples a persona attributes pool → injects personality → expands into a full persona, then couples domain descriptions with user memory to synthesize personalized-intent responses. This *is* the benchmark — it defines both what is asked and how personalization is baked in. *Ran et al. (2026), arXiv:2605.07299. Embedded for personal research reference.*

## Method + math — eval protocol, metrics, and data construction
For a benchmark the "math" is the **task taxonomy + metric definitions + construction pipeline + judging scheme**.

### Task taxonomy (12 domains, 2 branches)
- **Event-driven** (2 domains — proactive *alerting*): `object` (alert on a specific transient visual object) and `action` (alert on a scene event / human action). Emphasis is visual + temporal precision.
- **Intent-driven** (10 daily-living domains): `working`, `travel`, `sports`, `art`, `navigation` (obstacle avoidance tailored for the visually impaired), `dailylife`, `shopping`, `cooking`, `driving`, `entertainment`. Emphasis is persona-consistent, useful, well-timed help.
- Evaluation is organized into three scored subsets: **EgoPro-Action**, **EgoPro-Object**, **EgoPro-Intent**.

### Streaming evaluation protocol
The model is fed the stream **frame-by-frame** (1.0 FPS sampling) with historical context retained; at every step it decides whether to respond. Predicted response times are matched to ground-truth intervals; intent-driven tasks use **Hungarian (bipartite) matching** for one-to-one temporal correspondence between predicted and gold interventions.

### Objective timing/perception metrics
A prediction counts as correct if it lands inside a gold interval with the right content. Over matched predictions:

$$
\text{Precision}=\frac{TP}{TP+FP},\qquad
\text{Recall}=\frac{TP}{TP+FN},\qquad
\text{F1}=\frac{2\,\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}
$$

- **mIoU** — mean temporal Intersection-over-Union between predicted response intervals and ground-truth intervals (how tightly the *timing* is grounded, not just whether it overlaps).
- **GHA (Ground-truth Hit Accuracy)** — the fraction of ground-truth intervals that contain at least one correct response:

$$
\text{GHA}=\frac{\#\{\text{gold intervals with}\ \ge 1\ \text{correct response}\}}{\#\{\text{gold intervals}\}}
$$

### LLM-as-judge metrics (personalization quality, intent branch)
Because "good help" is open-ended, the intent branch adds two 1–5 LLM-judge scores:
- **Memory Consistency** — does the volunteered response align with the user's synthesized history/persona?
- **Response Quality** — is it actionable / effective for that user in that scene?

### Data construction pipeline (Figure 3)
Sourced from 8 egocentric datasets — EgoBlind, StreamGaze, EgoExoLearn, EgoTextVQA, EgoSchema, LLaVA-Video, Ego4D, EgoQA — **77,345 videos collected → 14,339 used** (Table 2).
- **Event-driven track:** 1.0 FPS sampling; Qwen3-VL-30B-A3B-Instruct filters for high-dynamic egocentric sequences (drops static / info-sparse clips); MLLM object & action annotation under a **transient-visibility constraint** (target objects must appear transiently, not be persistent background — guaranteeing a distinct temporal trigger); frame-by-frame time annotation with automated time-filtering + **human verification**.
- **Intent-driven track:** hierarchical **persona synthesis** (500+ professions, 300+ personality traits; sample name/age/gender/education/occupation → inject personality → expand to expertise/style/objectives/hobbies/preferences with a quality check) → **domain customization** (scene description + domain taxonomy) → **memory generation** (domain-adaptive, dual-filtering) → **response generation** where intervention timing is decided jointly by *user memory + scene context*.

### ProAct-Stream reward design ("short thinking, better interaction")
The companion model is trained SFT then RL with a composite reward $R = R_{\text{reason}} + R_{\text{resp}}$. Reasoning-side rewards enforce a *short, non-repetitive, on-topic* think:
- **Format:** $R_{\text{format}}=1$ if the output uses well-formed `<think></think>` tags, else 0.
- **Length (budget):** with defaults $L_{\min}=16,\ L_{\max}=22$ tokens,
$$R_{\text{len}}=1-\max\!\Big(0,\ \min\big(1,\tfrac{L-L_{\min}}{L_{\max}-L_{\min}}\big)\Big)$$
(rewards keeping the think short so it does not stall the stream).
- **Historical diversity:** $R_{\text{hist}}=1-\max_{h\in H}\{\,\text{sim}_{\text{text}}(s_{th},h)\,\}$ (penalize repeating past thoughts).
- **Semantic consistency:** $R_{\text{sem}}=\tfrac12\big(\text{sim}_{\text{text}}(s_{th},g_{th})+\text{sim}_{\text{sem}}(s_{th},g_{th})\big)$ (align the think with a gold reasoning trace).
- Similarity primitives: $\text{sim}_{\text{text}}(s_1,s_2)=\frac{2M}{L_1+L_2}$ (character-match ratio, $M$ matched chars) and $\text{sim}_{\text{sem}}(s_1,s_2)=\frac{\text{LLM}(s_1,s_2)-1}{3}$ (1–4 judge score mapped to $[0,1]$).
- Response-side: $R_{\text{resp}}=R_{\text{len}}+R_{\text{cont}}$ (length budget + content correctness).

## Explicit design choices
- **Two orthogonal branches** — separate "when to fire" (event, objective F1/mIoU/GHA) from "what personalized help to give" (intent, adds LLM-judge Memory Consistency + Response Quality). Keeps timing precision measurable while still scoring personalization.
- **Persona as a first-class variable** — synthesize user profiles + persistent memory and inject them into scenes, so the *same* video can demand different assistance. This is the axis missing from prior proactive benchmarks.
- **Transient-visibility constraint** on event objects — forces a crisp temporal trigger rather than "object is always on screen," making timing metrics meaningful.
- **Hungarian matching** of predicted vs. gold interventions on the intent branch — one-to-one assignment before scoring, avoiding double-counting.
- **Frame-by-frame streaming eval at 1.0 FPS** with retained history — mirrors real online deployment, not offline whole-video QA.
- **"Short thinking" length-budgeted reasoning** — an explicit token budget ($L_{\min}=16,L_{\max}=22$) on `<think>` so latency stays bounded in a stream; the paper finds thinking helps intent but can *hurt* pure event timing (see results).
- **Companion model + ablation design** — SFT vs. RL-without-think vs. RL-with-think, so the benchmark ships with an analysis of when reasoning helps.
- **Reuse of 8 egocentric sources** heavily filtered (77,345 → 14,339 curated) rather than fresh capture.

## Key results / what to remember
Numbers are verified against the paper's Tables 4–5 (ProAct-Stream on Qwen3-VL-8B base).

**On EgoPro-Bench subsets (Table 5), SFT / RL-w/o-Think / RL-w-Think:**
- **EgoPro-Object** — F1 88.00 / 88.07 / **91.19**; mIoU 84.01 / 84.12 / **87.01**. Thinking *helps* object alerting.
- **EgoPro-Action** — F1 84.78 / **85.03** / 79.66; mIoU 78.54 / **78.66** / 72.92. Thinking *hurts* action-timing (latency/over-reasoning), supporting "short thinking."
- **EgoPro-Intent** — F1 55.02 / 55.14 / **56.34**; Memory Consistency 3.81 / 3.81 / **4.10**; Response Quality 3.02 / 3.01 / **3.23** (1–5 judge). Thinking most clearly helps *personalized* intent.

**On external proactive benchmarks (Table 4, ProAct-Stream SFT, unified streaming protocol):**
- OmniMMI: F1 62.85, mIoU 51.05
- OvO-Bench: F1 52.98, mIoU 41.01
- StreamingBench: F1 35.43, mIoU 25.11

**Scale:** 2,400 eval videos (200 per domain × 12) and **12,000+ training videos** (SFT 10,285 + RL 1,800). Baselines span Qwen2.5-VL (3B–72B), Qwen3-VL (4B–30B), VideoChat-R1.5, VideoRFT-7B, and TimeChat-Online-7B (see [[timechat-online]]).

No Zotero highlights present.

Takeaways: (1) personalization is a *separable, scorable* axis of proactive interaction, not just "answer at the right time"; (2) "thinking" is not free in a stream — it helps intent/object but degrades fast action-alert timing, motivating a hard length budget; (3) a synthesized-persona pipeline can manufacture 12K+ personalized training videos from existing egocentric corpora.

## How it connects (evolution)
- [[proactivevideoqa]] — earlier proactive-QA benchmark; EgoPro adds the *personalization* + persona-memory axis on top of timing.
- [[omnimmi]], [[streamingbench]], [[ovo-bench]] — the prior proactive/streaming benchmarks EgoPro re-evaluates against under one protocol (its Table 4 baselines).
- [[proact-vl]] — proactive vision-language interaction; sibling in the proactive-agent line, EgoPro supplies the graded personalized eval.
- [[egospeak]] — egocentric proactive turn-taking (when to speak); EgoPro generalizes "when + what personalized help" to 12 domains.
- [[streamgaze]] — one of EgoPro's source datasets and a related egocentric-stream benchmark.
- [[timechat-online]] — a streaming baseline model EgoPro benchmarks (efficient online video LLM).

## Open questions / limitations
- **Synthetic personas & LLM-judge scoring** — Memory Consistency / Response Quality are judged by an LLM against *synthesized* memory, so both the ground truth and the metric are model-generated; real human preferences may diverge.
- **Reasoning length is hand-tuned** — the "short thinking" budget ($L_{\min}=16,L_{\max}=22$) is fixed; the object-vs-action divergence suggests the optimal budget is task-dependent and not adapted online.
- **Absolute intent scores are modest** — EgoPro-Intent F1 ~56 and Response Quality ~3.2/5 even for the tuned model, indicating personalized proactive help is far from solved.
- **Egocentric-only, curated sources** — heavy filtering (77,345 → 14,339) and reliance on 8 existing datasets may bias domain coverage and scene diversity.

*Verification: metric formulas (F1/mIoU/GHA and the ProAct-Stream reward terms) and all headline numbers checked against the paper's Tables 2, 3, 4, 5 and Figure 3 via the arXiv HTML (2605.07299v1) and the downloaded PDF; figure cropped from PDF page 4 (Figure 3). Affiliations not explicitly listed in the fetched text.*

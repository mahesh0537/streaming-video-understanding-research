---
zotero_key: null
authors: Ruixiang Zhao, Jie Yang et al. (Renmin University of China; WeChat Vision, Tencent)
year: 2026
arxiv: 2605.18577
pdf: https://arxiv.org/pdf/2605.18577
tier: deep
subtopics: [streaming-benchmarks]
tags: [streaming-video-understanding, streaming-benchmarks]
---
# OmniPro: A Comprehensive Benchmark for Omni-Proactive Streaming Video Understanding

**Lineage role:** The first benchmark to *jointly* stress omni-modal perception (84% of samples need audio), true proactive when-to-respond behavior, and diverse task coverage — evaluated under a **dual probe-vs-online protocol** that separates "does the model understand the content" from "does it decide *when* to speak on its own."

## Problem — what was limited before this paper (short)
Prior proactive streaming benchmarks fall short on three axes at once: (1) they lean almost entirely on **visual** signals, so they cannot tell an omni-modal model (that hears speech and non-speech sound) from a vision-only one; (2) they use **polling or fixed-timestamp querying** rather than letting the model autonomously decide when to answer — so "proactivity" is never actually measured; and (3) they cover a **narrow set of tasks**. No existing benchmark tests omni-modal perception, autonomous response timing, and task diversity together, so a genuinely good omni-proactive model cannot be reliably distinguished.

## Key idea — the core insight, 2-4 sentences
OmniPro is built so that most triggers are **inherently audio-dependent** (a whistle, a siren, spoken content), forcing models to use both sight and sound. It organizes **9 sub-tasks across three cognitive levels** (Perception / Comprehension / Reasoning) spanning 6 basic capabilities, and evaluates each sample under **two complementary modes**: a **Probe mode** that queries the model just before and just after a ground-truth trigger to score pure content understanding (no streaming skill needed), and an **Online mode** where the model watches frames one-by-one and must *decide on its own* when to emit each response. This factorization isolates "what to say" (content correctness) from "when to say it" (timing), which is exactly the pair that proactive streaming conflates.

![[omnipro.png]]

> **Crux (Figure 1).** The 9 sub-tasks organized into three cognitive levels covering 6 capabilities; each panel shows an instruction (Q), the time-aligned red-triangle triggers, and the expected proactive responses (A) — most triggers hinge on audio cues (whistle, siren, spoken word). *Zhao, Yang et al. (2026), arXiv:2605.18577. Embedded for personal research reference.*

## Method + math — eval protocol, metrics, and data pipeline

### Task taxonomy (3 cognitive levels, 9 sub-tasks, 6 capabilities)
- **Perception:** Instant Event Alert (Event-Alert), Real-time State Monitoring (State-Monitor), Snapshot Counting (Snap.-Count), Explicit Target Grounding (Target-Ground).
- **Comprehension:** Event Narration (Event-Narr.), Cumulative Counting (Cum.-Count), Semantic Condition Alert (Cond.-Alert).
- **Reasoning:** Deduplicated Counting (Dedup.-Count), Sequential Step Instruction (Step-Inst.).
- The 6 basic capabilities cut across levels: **Alert, Monitoring, Grounding, Counting, Narration, Prediction**.

Each sample has a user instruction issued at the start, and one or more **ground-truth (GT) triggers** — timestamps where a proactive response is expected — with the reference answer for each. Samples carry **modality-isolation labels** (which modality the trigger depends on: visual, speech, non-speech sound), enabling fine-grained ablation.

### Probe mode — content understanding, streaming-agnostic
For each GT trigger the evaluator queries the model **twice**: a **pre-probe** sampled in the window $[-5, -2]$ s before the trigger, and a **post-probe** in $[0, +3]$ s after. A trigger is scored correct **only if both probes are answered correctly** (the pre-probe should *not* yet fire / should reflect the not-yet state, the post-probe should reflect the triggered state). The score is the fraction of correctly answered triggers:

$$\text{Accuracy} = \frac{\#\{\text{triggers with both pre- and post-probe correct}\}}{\#\{\text{total triggers}\}}$$

This measures *what* the model perceives without requiring it to manage its own timing.

### Online mode — full proactive ability
The model gets the instruction at the video start, then **streams frames one-by-one** and **autonomously decides when to emit a response**. Emitted responses are matched to GT triggers by **greedy temporal alignment** with a tolerance of $\pm 3$ s (a response counts only if it lands within the window of an unmatched GT trigger *and* is content-correct). Then:

$$\text{Precision} = \frac{\#\text{valid matched responses}}{\#\text{total model responses}}, \qquad \text{Recall} = \frac{\#\text{valid matched triggers}}{\#\text{total GT triggers}}$$

$$\text{F1} = \frac{2\,\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}$$

Timing (Precision penalizes spurious/early responses; Recall penalizes misses) and content correctness are jointly folded into one score. A tolerance-window ablation (Fig. 5) varies the $\pm$ window to test sensitivity of the joint F1.

### Correctness judging
- **Structured outputs** (counts, coordinates, state labels): **exact match**.
- **Open-ended outputs** (narration, step instructions): an **LLM judge (Gemini-3-Flash)** scores each prediction on a **1–5** scale; **score $\geq 3$ = correct**.

### Data construction pipeline
1. **Source videos:** 1,171 from LongVALE + 600 from COIN = **1,771** source videos (audio-rich, temporally annotated domains).
2. **Dense captioning:** Gemini-3-Flash produces temporally-aligned, timestamped multimodal dense captions with four fields — `caption`, `visual`, `audio`, `speech`.
3. **QA synthesis:** original video + dense captions + a task-specific prompt → structured QA samples via Gemini-3-Flash (~1,000 per sub-task → ~9,000 raw).
4. **Human QC:** 9 annotators, **two review rounds**, checking question naturalness, trigger-time accuracy, response faithfulness, and modality correctness. **~30% retained → 2,700 samples across 1,262 videos.**

## Explicit design choices
- **Dual-mode by construction:** every sample is scored in both Probe (Accuracy) and Online (F1) modes, decoupling content understanding from response-timing ability.
- **Audio-first sampling:** triggers are deliberately chosen so **84%** of samples require audio (speech or non-speech sound) — the mechanism that makes it an *omni*-benchmark rather than vision-only.
- **Modality-isolation labels** per sample → supports audio-only / video-only / audio-visual ablations (Table 3) and per-modality bottleneck analysis (Fig. 4).
- **Three-level cognitive taxonomy** (Perception/Comprehension/Reasoning) × 6 capabilities → breadth beyond single-capability benchmarks.
- **Multi-trigger samples:** avg **3.4 responses per question**, so counting/monitoring/narration tasks demand sustained streaming, not one-shot answers.
- **Temporal-location stratification:** triggers grouped Short/Mid/Long-term along the timeline (Fig. 3) to expose long-horizon degradation; avg first-trigger time 54.1 s, avg video 189 s.
- **Greedy $\pm 3$ s matching + LLM-judge (≥3/5)** as the online scoring rule; exact-match for structured tasks.
- **License:** CC BY-NC 4.0 (non-commercial).

## Key results / what to remember
No Zotero highlights present.

- **Dataset scale:** 2,700 samples, 1,262 videos, 9 sub-tasks; avg duration 189 s, avg 3.4 responses/question, avg first-trigger 54.1 s; **84% of samples require audio**.
- **Overall performance is modest**, confirming the benchmark is hard. **Probe mode (Accuracy, Mean %):** Gemini-3-Flash **40.4** (best), MiniCPM-o 4.5 (9B) 25.8, Qwen3-Omni (30B) 22.6, video-SALMONN 2+ (7B) 22.1.
- **Online mode (F1, Mean %):** MiniCPM-o 4.5 (9B) **20.9** (best of the online models reported), MMDuet2 (3B) 11.3.
- **Online << Probe gap:** even the best online model collapses on generation-intensive tasks (MiniCPM-o Event-Narr. **6.9%**, Step-Inst. **7.9%** F1) — knowing the answer ≠ emitting it at the right moment.
- **Audio helps:** full audio-visual input beats video-only by **+2.4 to +11.1** across the five omni-models (Table 3).
- **Non-speech audio is the shared bottleneck:** all models weakest on visual+sound triggers (**15.3–22.3**), i.e., perceiving/using non-speech sound is unsolved.
- **Long-horizon degradation:** models retain on average only **~37%** of their Short-term performance by the Long-term bucket (Fig. 3).

## How it connects (evolution)
- [[streaming-benchmarks]] — this note is a node in that sub-topic's benchmark landscape.
- [[omnimmi]] — prior omni-modal (audio+video) streaming/interaction benchmark; OmniPro extends omni-modal eval to explicit proactive timing.
- [[proactivevideoqa]] — proactive-response QA benchmark; OmniPro adds the audio dependence and probe/online split.
- [[streamingbench]] / [[ovo-bench]] — earlier streaming benchmarks that OmniPro critiques as largely vision-only and poll/fixed-timestamp rather than truly proactive.
- [[mmduet2]] — evaluated as an online proactive model on OmniPro (11.3 mean F1), a representative token/response-driven streaming VLM.
- [[omni-duplexeval]] — sibling omni-modal duplex/proactive evaluation; complementary framing of when-to-speak.

## Open questions / limitations
- **Synthetic QA + LLM-judge dependence:** samples are generated and scored with Gemini-3-Flash; despite two human review rounds, judge bias and generation artifacts may inflate/deflate open-ended scores (the judge shares a family with a top-scoring model).
- **Sourced from LongVALE + COIN only:** domain coverage inherits those datasets' biases (instructional/how-to heavy via COIN), so "diverse tasks" is broad but the video distribution is not open-domain.
- **Probe pre-window heuristic:** the $[-5,-2]$/$[0,+3]$ s probe windows and $\pm 3$ s matching tolerance are fixed hyperparameters; Fig. 5 shows online F1 is tolerance-sensitive, so cross-benchmark comparisons hinge on this choice.
- **Small online model pool:** relatively few streaming models are actually run in Online mode, so the online leaderboard is thin.

*Verification: probe/online metric formulas, task taxonomy, and pipeline restated from the arXiv HTML (arxiv.org/html/2605.18577); headline numbers (Probe mean 40.4 / online mean 20.9, 84% audio, 2,700 samples / 1,262 videos, +2.4–11.1 audio gain, ~37% long-term retention) cross-checked against the paper's Table 2/Table 3 and Fig. 1/2/3 as summarized from the HTML; crux figure cropped from the downloaded PDF page 2.*

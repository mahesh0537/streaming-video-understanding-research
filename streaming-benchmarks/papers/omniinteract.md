---
zotero_key: null
authors: Xudong Lu et al. (CUHK MMLab, SJTU, NTU, McMaster, CityUHK, JUFE)
year: 2026
arxiv: 2605.26485
pdf: https://arxiv.org/pdf/2605.26485
tier: deep
subtopics: [streaming-benchmarks]
tags: [streaming-video-understanding, streaming-benchmarks]
---
# OmniInteract: Benchmarking Real-World Streaming Interaction for Real-Time Omnimodal Assistants

**Lineage role:** A streaming benchmark that keeps the *raw* audio-visual stream intact — spoken queries and event sounds live inside the timeline, so the model must itself detect triggers, decide *when* to speak, and answer while the stream is still unfolding, rather than being handed a text prompt at a known timestamp.

## Problem — what was limited before this paper (short)
Prior streaming video QA benchmarks (e.g. [[streamingbench]], [[ovo-bench]], [[svbench]]) still hand the model a *text* question at a marked timestamp: the timing decision is removed, and the audio channel — including the very act of the user speaking — is stripped out. That misses the two hardest parts of a real omnimodal assistant: (1) noticing from the native audio-visual stream that a response is even warranted, and (2) producing it in the valid window without spilling over into the next moment. Offline video LLMs score well on QA but nobody had measured whether that competence survives genuine online, full-duplex interaction over a live stream.

## Key idea — the core insight, 2-4 sentences
Distribute the benchmark as *native audio-visual recordings* with the queries and event triggers embedded in the stream, and replay them chronologically through each model's own real-time interface (frames/audio revealed only at their true timestamps, no future access, no ground-truth slot boundaries). Ground every expected response in a temporal **slot** `[t_start, t_a, t_end)` and align the model's free-flowing output chunks to those slots *after* inference. This lets one metric family jointly grade *what* was said (semantic quality), *when* it was said (timeliness, with a time-decay), and *whether it broke conversational flow* (early hallucination, spillover, unmatched chatter).

![[omniinteract.png]]
> **Crux (Figure 3).** The evaluation reduces to temporal *slots*: each expected response has an observation onset $t_{start}$, an earliest-valid-answer time $t_a$, and a window close $t_{end}$; model output chunks are assigned to slots and split at $t_a$ into an *early* segment (tentative acknowledgment) and a *core* segment (the real answer), across the real-time/proactive, nested, 1QnA, and interruption settings. *Xudong Lu et al. (2026), arXiv:2605.26485. Embedded for personal research reference.*

## Method + math — the eval protocol in full

### Interaction taxonomy (what a "response opportunity" can be)
- **1Q1A split** (1,062 slots / 210 videos): each opportunity yields one response. Three mutually-exclusive slot types:
  - **Real-time** (638 slots): an explicit spoken user query; answer immediately from currently-available context.
  - **Proactive** (184 slots): no explicit query — a salient multimodal *event* is the trigger; the answer only becomes available later, so the model must monitor and wait.
  - **Nested** (240 slots): a real-time query is inserted *inside* the response window of a proactive one, forcing a context switch and later resumption of the outer query.
- **1QnA split** (368 slots / 40 videos): a single spoken instruction demands *multiple* temporally-grounded responses as a task progresses (continuous task monitoring).
- **Interruptions** (192 slots, cross-cutting: 147 in 1Q1A, 45 in 1QnA): a user interruption or event-triggered shift preempts the current slot; the model must stop cleanly rather than spill over. An interruption is annotated when the interval $[t_a, t_{end}]$ is shorter than the TTS-estimated duration of the ground-truth answer.

### Slot definition and chunk assignment
Every expected response is a slot
$$\text{slot} = [\,t_{start},\; t_a,\; t_{end}\,)$$
with $t_{start}$ the observation onset, $t_a$ the earliest moment a valid *core* answer can be given, and $t_{end}$ the window close. In proactive chains, each visual event triggers the next slot (its $t_{start}$ and $t_a$ coincide) and the *next* slot's $t_{start}$ serves as the current slot's $t_{end}$. A model-generated text chunk is assigned to a slot if its start time falls in $[t_{start}, t_{end})$; on overlap (nested resumption) it maps to the slot with the **latest** $t_{start}$ (most recent context). A chunk straddling $t_a$ is split at the word level into an **early** segment ($t < t_a$) and a **core** segment ($t \ge t_a$). Chunks matching no slot are recorded as *unmatched* and penalized.

### Interaction-aware scoring (soft TP + discrete penalties)
Each slot is scored with a time-decay to reward promptness. The **early stage** ($t<t_a$) rewards a valid tentative acknowledgment by onset timing; an early *hallucination* (committing to a real answer too soon) scores 0. The **core stage** ($t\ge t_a$) rates semantic correctness/coverage $S_{core}\in[0,1]$ against the ground truth, discounted by a timeliness factor
$$T_{core} = \max\!\Big(0,\; 1 - \frac{t_{anchor} - t_a}{t_{end} - t_a}\Big),$$
where $t_{anchor}$ is the moment the judge identifies the answer as semantically delivered — so later answers within the window decay linearly toward 0 at $t_{end}$. The slot's **soft true positive** clamps the summed stage scores:
$$TP = \min\!\big(1,\; \text{Score}_{ack} + S_{core}\cdot T_{core}\big).$$
Failures become discrete penalties. A **false negative (FN)** is a non-interruption slot lacking a core answer ($TP\le 0$). A **false positive (FP)** aggregates four unwarranted behaviors: (1) unmatched chunks, (2) early hallucinations, (3) low-quality responses, (4) **spill** (output past $t_{end}$ that breaks conversational continuity). The global metric is a soft micro-F1 over all slots:
$$\text{IA-QTF1} = \frac{2\sum TP}{2\sum TP + \sum FP + \sum FN}.$$

### Extended metrics
- **Interruption Diagnostic Suite (IDS)** — because global IA-QTF1 treats every interruption as a pure boundary-control case (no reward for partial pre-empted answers), IDS adds: **NOR** (No-Output Rate, fraction of interrupted slots with no output at all), **PAQ** (Partial Answer Quality, LLM-judged usefulness of already-spoken content *without* an incompleteness penalty), and **CSM** (Conditional Spill Metrics: spill rate and average spill duration, computed only over interrupted slots that produced output).
- **Nested Chain Completion Score (NCCS)** — geometric mean over the outer/inner core-stage scores of a nested pair:
$$\text{NCCS} = \sqrt{\text{Score}_{outer}\cdot \text{Score}_{inner}}.$$
It requires answering the inner query *and then resuming* the outer — so it collapses to ~0 if the outer is dropped after the interruption.

### Inference protocol & judging
Each recording is replayed chronologically through the model's *native* real-time interface; frames/audio surface only at their true timestamps, the model conditions on past+present but never on future frames/audio or on ground-truth slot boundaries. Output chunks are timestamped during replay and aligned to slots *after* inference — deterministic and comparable across models. Open-ended answers are graded by **GPT-4o** as an external judge (early-stage: acknowledgment vs. early hallucination; core-stage: 0–1 correctness + the semantic anchor for timeliness; interruption: partial-answer usefulness with future-step spoiler flagging for 1QnA).

### Data construction
- **1Q1A (built from scratch, 210 videos):** self-recorded, two scenario groups — Chinese daily-life (150 videos: home, gym, museum, shopping) and English math problem-solving (60 videos with evolving on-screen problem context). Each slot manually annotated with trigger, valid response window, and target answer, verified against audio-visual evidence.
- **1QnA (converted from existing benchmarks, 40 videos):** procedural/task-guidance and egocentric error-detection sources; task goals rewritten into natural instructions, synthesized to speech via **Qwen3-TTS**, prepended to the original stream, and step annotations mapped to grounded response slots.

## Explicit design choices
- Ship the benchmark as *raw audio-visual streams* with embedded spoken queries and event sounds — the model must detect the trigger; timing is not given.
- Slot-based grading `[t_start, t_a, t_end)` with a per-slot early/core split at $t_a$ (word-level), so acknowledgment vs. committed answer are scored separately.
- Timeliness as a *linear time-decay* to $t_{end}$, not a hard deadline — being early-but-wrong (hallucination) and being late both cost score, but differently.
- Spillover past $t_{end}$ is a *false positive*, explicitly penalizing flow-breaking talkativeness — a behavior offline metrics never see.
- Proactive slots have $t_{start}=t_a$ and chain via each next event; nested = a real-time slot nested in a proactive window, graded with a dedicated geometric-mean NCCS to force outer-query *resumption*.
- Interruptions defined operationally (ground-truth answer longer than $[t_a,t_{end}]$) and diagnosed by a separate NOR/PAQ/CSM suite rather than folded into F1.
- Post-hoc chunk-to-slot alignment keeps replay deterministic while allowing free-form streaming output; overlap resolved to the most-recent $t_{start}$.
- External GPT-4o judge (not any tested model) to reduce evaluator self-bias.

## Key results / what to remember
Verified against the paper's Table 3 (rendered PDF, page 6) and the HTML tables.

- **Overall IA-QTF1 is low for everyone.** Best All-Global: **MiniCPM-o 4.5 = 0.368**; then AURA 0.363, Gemini 2.5 Flash Live 0.344, Qwen3.5-Omni Flash Realtime 0.323 (All-Global column, all slots).
- **1Q1A global:** AURA best at **0.467**, MiniCPM-o 0.456, Gemini 0.428, Qwen3.5-Omni 0.401.
- **By slot type (1Q1A):** real-time favors explicit-query handlers — Gemini **0.553**, Qwen3.5-Omni 0.524; proactive favors patient monitors — MiniCPM-o **0.607**, AURA 0.549 (Gemini 0.121, Qwen 0.108 collapse here); nested — MiniCPM-o **0.599**, AURA 0.596.
- **Continuous task monitoring (1QnA) is near-broken:** best is AURA **0.052**, then Gemini 0.028, Qwen 0.023, MiniCPM-o 0.015.
- **Nested resumption (Table 4, 120 pairs):** best NCCS MiniCPM-o **0.284**, AURA 0.270 — but Gemini and Qwen answer many inner queries (Inner IA-QTF1 0.595 / 0.702) yet **fail to resume the outer** in 119/120 and 116/120 cases (NCCS 0.001 / 0.012). Context switching without resumption.
- **Interruption diagnostics (Table 5):** distinct failure modes — Gemini is conservatively silent (NOR **85.94%**, lowest spill CSM-SR 40.74% / CSM-AS 0.312 s); MiniCPM-o gives the most useful partial answers (PAQ **0.571**) but spills badly (CSM-SR 83.15%, CSM-AS 10.067 s); AURA NOR 79.17%, Qwen NOR 71.35%.
- **Offline competence doesn't transfer (Table 6):** MiniCPM-o 4.5 pure-quality score drops from 0.6833 (offline) to 0.3475 (online), **Δ ≈ −0.336** — full-duplex real-time inference roughly halves answer quality.

No Zotero highlights present.

Takeaways: (1) the frontier of *real* streaming omnimodal assistants is far from solved — sub-0.37 F1 on the aggregate; (2) models trade off along a silence↔spillover axis and no model manages both timing and flow; (3) multi-response monitoring and nested-query resumption are the sharpest failure modes; (4) the slot + early/core + time-decay scoring is a reusable template for grading *when-to-speak* in any proactive streaming system.

## How it connects (evolution)
- [[streaming-benchmarks]] — this is a sub-topic hub member; extends the streaming-QA benchmark line to native audio-visual, self-timed interaction.
- [[omnimmi]] — closest sibling: multi-modal, multi-turn *omni* streaming benchmark; OmniInteract adds explicit slot timing, proactivity, and interruption diagnostics.
- [[streamingbench]] — canonical text-prompted streaming QA benchmark that OmniInteract argues is too easy because timing/audio are given.
- [[ovo-bench]] — online video benchmark probing "answer at the right moment"; OmniInteract generalizes this to full audio-visual streams with proactive/nested/interruption structure.
- [[proactivevideoqa]] — proactive-response evaluation; shares OmniInteract's proactive-slot idea (respond on event, not query).
- [[svbench]] — streaming long-context dialogue benchmark; contrast on interaction structure and metrics.

## Open questions / limitations
- The whole grade depends on a single GPT-4o judge for open-ended scoring and semantic-anchor timestamps; judge noise directly moves both quality and timeliness — no reported inter-judge or human-agreement calibration.
- Only four models are tested (AURA, Gemini 2.5 Flash Live, MiniCPM-o 4.5, Qwen3.5-Omni Flash Realtime); no open-weight streaming-VLM baselines from the memory/KV-cache line, so it's unclear whether the low scores are architectural or interface artifacts.
- 1QnA is only 40 videos (converted, TTS-prepended instructions), a thin and partly synthetic slice for the split where every model fails worst — the 0.05 ceiling may partly reflect the conversion rather than pure capability.
- Post-hoc chunk-to-slot alignment and the TTS-duration interruption rule bake in modeling assumptions (word-level split, linear decay) that could reward or punish specific streaming cadences unevenly.

*Verification: Metric formulas (IA-QTF1, $T_{core}$, NCCS, slot definition) and taxonomy taken from the arXiv HTML (Sec. 3.3) and cross-checked against the rendered PDF pages 5–7; all headline numbers verified against the paper's Table 3 (rendered PDF page 6) plus Tables 4–6 from the HTML. Figure 3 cropped from the PDF (page 5 of the render).*

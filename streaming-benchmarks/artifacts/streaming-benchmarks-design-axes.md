---
tags: [streaming-video-understanding, streaming-benchmarks, synthesis]
---

# Streaming Benchmarks — Design-Axis Comparison

The decision matrix for the [[streaming-benchmarks]] sub-topic: one row per benchmark,
grouped by idea-family, scored along the axes that actually separate these designs. The
single load-bearing axis is the **timing protocol** — whether the evaluator hands the model
the query timestamp (feed `Video[0:t_q]`, score MCQ accuracy) or forces the model to *own*
its response timing (self-detect the trigger, decide when to speak). That split is the
sub-topic's central storyline: the 2026 wave (families D–E) is a validity correction on the
2024–25 evaluator-timed benchmarks (families A–B).

Numbers are categorical design facts drawn from each paper's on-disk note; where a paper
does not report a value it is marked `n/r`. Scale is `#videos / #QA` unless noted.

## Reading the columns

- **Timing protocol** — *evaluator-timed prefix* (model is fed `Video[0:t_q]`, timestamp given) · *model-owned timing* (model self-detects when to speak, timing never given) · *poll+tolerance* (fixed firing window) · *protocol harness* (re-scores other benchmarks).
- **Temporal regimes** — where the evidence/answer lives: Backward(past) · Real-time(present) · Forward(proactive/future).
- **Modality** — Visual-only · Omni(audio+visual) · +Gaze. "Ego" flags egocentric sourcing.
- **Interaction** — single-turn QA · multi-timestamp re-ask (same query, shifting answer) · multi-turn dialogue · proactive slots (model fires interventions).
- **When-to-respond scored?** — does the metric grade *timing of the model's own output*, and how.
- **Metric type** — MCQ-accuracy · open-ended LLM-judge · timing-F1 · PAUC-area · dialogue-OS · composite.
- **Ships baseline model?** — does the paper ship a matched streaming model, or is it eval-only?

## The matrix

| Benchmark | Timing protocol | Temporal regimes | Modality | Interaction | When-to-respond scored? | Metric type | Scale (videos / QA) | Ships baseline model? |
|---|---|---|---|---|---|---|---|---|
| **A. Evaluator-timed streaming QA** ||||||||||
| [[streamingbench]] | evaluator-timed prefix `V[0:t_q]`; proactive via ±4s poll / 2s tolerance | Real-time + Contextual(past history); no forward | Omni (real-time visual + omni-source audio) | single-turn QA (+ prior-QA appended for contextual) | partial — 50 proactive Q, ±4s poll + 2s firing tolerance | MCQ accuracy | 900 / 4,500 MCQ | no (evals existing) |
| [[ovbench-videochat-online]] | evaluator-timed prefix; two regimes (sliding-window 32s vs all-frames-to-`t_q`) | Past / Current / Future (explicit axis) | Visual-only | single-turn timestamp-anchored MCQ (distractors from same video) | no | MCQ accuracy | n/r (6 tasks / 16 subtasks) | yes — VideoChat-Online 4B (Pyramid Memory Bank) |
| [[rtv-bench]] | evaluator-timed running state `S_{t_q}`; matched online-vs-offline harness | Real-time (MTQA re-ask); shallow/moderate/deep query positions | Visual-only (audio deliberately ignored) | multi-timestamp re-ask (same query) + hierarchical q0/q1/q2 | no | MCQ accuracy + gated group Score | 552 / ~4,631 (167.2h) | no |
| [[streamingcot]] | evaluator-timed segment-level (answer-per-segment, time-evolving) | Real-time / evolving-per-segment | Visual-only (dense subtitles, no audio task) | multi-timestamp / per-segment evolving answers + ST-CoT traces | no | MCQ (distractor-aware) — **no results table (dataset paper)** | 5,000 / 34,470 (dataset) | no (ships ST-CoT supervision) |
| **B. Temporal-regime benchmarks** ||||||||||
| [[ovo-bench]] | evaluator-timed prefix `V[0:t]` + forward-active dense multi-trigger | Backward / Real-time / Forward-active (all three) | Visual-only (7 domains) | single-turn timestamped QA + forward dense triggering | yes — forward-active, `2^{-(m'-m)p}` time-decay | MCQ accuracy + time-decay response score (GPT-4o judge) | 644 / ~2,800 | no |
| [[streameqa]] | evaluator-timed, query-timestamp anchoring `V[t_e:t_q]` | Backward / Real-time / Forward × 3 embodied levels | Visual-only, Ego (HD-EPIC) | single-turn 4-choice MCQ | no | MCQ accuracy | 156 / 20,731 | no |
| **C. Interactive-dialogue benchmarks** ||||||||||
| [[svbench]] | evaluator-timed dialogue walk vs streaming walk (80% jump to linked chain) | Present + cross-segment linkage (past recall) | Visual-only | multi-turn dialogue chains + typed cross-clip linkage | no | LLM-judge OS (5 dims SA/CC/LC/TU/IC), open-ended | 1,353 / 49,979 | yes — StreamingChat (InternVL2) |
| [[river-bench]] | evaluator-timed window-based; clue `t_V` vs question `t'` vs now `t`; multiple EOS = silence | Retro(past) / Live(present) / Proactive(future) — all three | Visual-only | multi-turn streaming; explicit forgetting-curve buckets | yes — hard-zero early, linear decay late | MC + OE accuracy + Pro-Response localization (mixed) | 1,067 / 4,278 | yes — LTM adapter + LLaMA3-8B fine-tune |
| **D. Proactive-timing + model-owned-timing protocols (the 2026 validity correction)** ||||||||||
| [[proactivevideoqa]] | model-owned timing — question fixed at `t=0`, model chooses when to reply | Forward / proactive (multi-turn open-ended) | Visual (WEB/EGO/TV/VAD, EGO ego) | multi-turn open-ended proactive | yes — **PAUC** area under correctness-vs-time | PAUC-area (GPT-4.1 judge, 0/1/2) | 1,377 / 1,427 | no |
| [[streamingeval]] | model-owned timing — strict causality, model reads only snapshot `M_{t1}`; wall-clock | reuses OVO-Bench + StreamingBench regimes | Visual (substrates) | protocol harness (re-scores existing benchmarks) | n/a — folds latency/throughput/memory instead | composite **StreamingScore** = MaxFPS·Acc / (TTFT·Mem) | protocol (re-scores OVO-Bench + StreamingBench) | no |
| [[streamgaze]] | evaluator-timed, 3 causal windows (Past/Present/Proactive) + gaze scanpath conditioning | Past / Present / Proactive | Visual **+ Gaze**, Ego | single-turn QA conditioned on scanpath | yes (proactive) — multi-triggering precision, penalizes premature/excess | MCQ accuracy (past/present) + multi-trigger precision (proactive) | 285 / 8,521 | no (gaze-prompt ablation only) |
| [[egopro-bench]] | model-owned timing — frame-by-frame @1 FPS, model fires interventions, Hungarian match to gold | Forward / proactive (event) + personalized intent | Visual, Ego | proactive intervention slots + persona-conditioned | yes — event F1 / mIoU / GHA | timing-F1/mIoU/GHA + LLM-judge intent (MemCons/RespQual) | 2,400 eval (+12K train) | yes — ProAct-Stream (Qwen3-VL-8B) |
| [[streaming-video-wild]] | model-owned timing — per-second `</response>`/`</silence>` decision | Backward / Present / Forward (SW-F1 per split) | Visual-only | per-second multi-turn speak-or-silence | yes — **SW-F1** windowed TP/FP/FN, asymmetric weights | timing-F1 (SW-F1) + narration pairwise win-rate | Streaming-Eval: 138 / 15 categories | yes — StreamingVLM + StreamingHarness (8B) |
| **E. Omni-modal interaction benchmarks** ||||||||||
| [[omnimmi]] | mixed — comprehension tasks (AP/SG/MD) evaluator-timed at query timestamps (queries as text+audio); proactive control (PA/PT/SI) model-owned/duplex | Streaming comprehension (present) + proactive control (future) | Omni (video+audio) | 6-task, proactive turn-taking / duplex; SG re-asks same query, cumulative 1st/2nd/3rd-turn scoring | yes — silence-as-correct (PT) + windowed precision/IoU (PA) | accuracy + silence/windowed timing | 1,121 / 2,290 | yes — M4 (Qwen2-7B omni) |
| [[omnipro]] | dual: Probe (content-only accuracy) + Online (autonomous timing) — model-owned in Online mode | Alert/Monitor/Grounding/Counting/Narration/Prediction; proactive | Omni (84% of samples need audio) | proactive, dual-mode probe-vs-online | yes — Online **F1**, greedy ±3s match | Probe accuracy + Online F1 (LLM-judge ≥3/5) | 1,262 / 2,700 | no |
| [[omni-duplexeval]] | native duplex inference — model-owned | Real-Time Description (present) + Proactive Reminder (future, Δ10s window) | Omni | duplex, open-ended | yes — `S_temporal` (4-window ±1-2s) + all-or-nothing proactive | 0.5 content + 0.5 temporal (LLM-as-judge) | 660 / n/r | no |
| [[omniinteract]] | model-owned — raw stream, self-detect trigger, timing never given; native-interface replay | Real-time / proactive / nested / continuous(1QnA) / interruption | Omni (spoken queries + event sounds) | slot-based, nested queries, interruptions | yes — **IA-QTF1** soft micro-F1 with linear time-decay | timing-F1 (IA-QTF1) + NCCS / NOR / PAQ / CSM | 250 / 1,430 slots | no |

<!-- Hero figures: crux protocol diagrams, one per family (embedded from existing figures/) -->

**A — evaluator-timed prefix (the timestamped-MCQ template):**
![[streamingbench.png]]
*StreamingBench: offline "whole video visible" vs the timestamped-prefix protocol, split into real-time / omni-source / contextual. [[streamingbench]]*

**B — the three-regime taxonomy (backward / real-time / forward-active):**
![[ovo-bench.png]]
*OVO-Bench: regimes defined by where evidence lives in time, forward-active scored on response timing. [[ovo-bench]]*

**D — the model-owned-timing correction:**
![[streaming-video-wild.png]]
*Streaming-in-the-Wild reframes the task as per-second speak-or-stay-silent, scored by SW-F1. [[streaming-video-wild]]*

**E — omni-modal, self-detected triggers with slot-based grading:**
![[omniinteract.png]]
*OmniInteract keeps the raw audio-visual stream intact; slots with early/core split and time-decay. [[omniinteract]]*

## Reading the axes — patterns the matrix exposes

1. **The timing protocol is the field's fault line, and it moved.** Families A–B and the
   dialogue benchmarks hand the model the query timestamp and feed `Video[0:t_q]` — an
   *evaluator-timed* protocol that reduces to MCQ accuracy ([[streamingbench]],
   [[ovbench-videochat-online]], [[rtv-bench]], [[streameqa]], [[ovo-bench]] backward/real-time).
   The 2026 wave inverts this: the model must *own* its timing, self-detecting the trigger
   with timing never given ([[proactivevideoqa]], [[egopro-bench]], [[streaming-video-wild]],
   [[omni-duplexeval]], [[omniinteract]]). This is the sub-topic's central validity
   correction — the older benchmarks measured "can you answer at a moment," the newer ones
   measure "do you know *when* the moment is."

2. **Every model-owned-timing benchmark had to invent its own timing metric — and none
   share one.** The moment timing becomes the model's job, plain accuracy dies and a bespoke
   timing-aware score appears: PAUC area ([[proactivevideoqa]]), SW-F1 ([[streaming-video-wild]]),
   IA-QTF1 ([[omniinteract]]), event F1/mIoU/GHA ([[egopro-bench]]), the 50/50 content-temporal
   split ([[omni-duplexeval]]), Online-F1 with ±3s matching ([[omnipro]]). They rhyme
   (windowed TP/FP/FN with a time-decay term) but no two are interchangeable — there is no
   agreed cross-benchmark timing metric, which is the clearest gap in the field.
   Worse, they flatly *contradict* on how much a false alarm should cost: [[river-bench]]
   zeroes early responses, [[omniinteract]] scores early hallucination 0 with spillover as FP,
   [[omni-duplexeval]] is all-or-nothing per proactive sample — while [[streaming-video-wild]]'s
   SW-F1 sets `w_FP=0.2` vs `w_TP=w_FN=2.0`, false alarms ten times cheaper than misses. The
   choice selects the winner: [[omni-duplexeval]]'s failure split divides models into over-firers
   ([[livecc]]-Base 91.1% wrong, StreamingVLM 96.7%) and over-silencers (MMDuet2 75.8% no-answer,
   MiniCPM-o 4.5 49.2%), so spam-tolerant and hard-zero metrics crown opposite phenotypes. Only
   [[proactivevideoqa]] validates its metric shape against human preference at all (ω=0.5 lifts
   Cohen κ 0.31/0.36 → 0.45/0.49 on VAD).

3. **The backward / real-time / forward triad is independently rediscovered five times.**
   [[ovo-bench]], [[river-bench]], [[streameqa]], [[streaming-video-wild]], and
   [[ovbench-videochat-online]] (as Past/Current/Future) all converge on the same
   temporal-regime axis without coordinating. Forward/proactive is consistently where the
   human-model gap is widest, which is exactly why families D–E concentrate there.
   By now the taxonomy is a commodity — the differentiating content lives elsewhere on each
   row, e.g. [[river-bench]] is the only one to *quantify* the past regime as a forgetting
   curve (GPT-4o MC 63.26 short-recall → 52.21 very-long).

4. **Omni-audio is the newest modality frontier and it clusters tightly.** Only family E plus
   [[streamingbench]]'s omni-source axis use audio; everything else is visual-only. Where
   audio is required it is also the bottleneck — [[omnipro]] reports 84% of samples need
   audio and non-speech-sound triggers are the hardest split. The visual-only benchmarks
   ([[rtv-bench]], [[streameqa]], [[ovbench-videochat-online]]) even in audio-rich domains
   (driving, sports) deliberately drop the audio track.

5. **Egocentric + proactive is a coherent emerging pocket, and gaze is a solo experiment.**
   [[streameqa]], [[egopro-bench]], and [[streamgaze]] all pair egocentric sourcing with
   embodied/proactive tasks — but only [[streamgaze]] conditions on eye-gaze scanpaths as a
   first-class modality. Nobody else evaluates gaze-conditioned streaming reasoning.
   The premise is fragile, though: [[streamgaze]]'s own ablation finds gaze prompting barely
   carries signal — Qwen2.5-VL-7B scores 0.446 with no gaze vs 0.429 with textual or
   visual-overlay gaze (both *hurt*), 0.454 with a salience map — so its real human-model gap
   (0.827 vs 0.535) is not yet evidence that gaze is the operative missing modality; current
   models either can't use it or don't need it.

6. **Shipping a baseline splits by whether the task needs a new interaction protocol.**
   Benchmarks that demand duplex/dialogue/model-owned behavior ship a matched model to prove
   the task is doable ([[svbench]] StreamingChat, [[omnimmi]] M4, [[ovbench-videochat-online]]
   VideoChat-Online, [[river-bench]] LTM adapter, [[egopro-bench]] ProAct-Stream,
   [[streaming-video-wild]] StreamingVLM). Pure eval suites that re-score existing models do
   not ([[ovo-bench]], [[rtv-bench]], [[streameqa]], [[proactivevideoqa]], [[omnipro]],
   [[omni-duplexeval]], [[omniinteract]]).

7. **Deployability is scored exactly once.** [[streamingeval]] is the only benchmark that folds
   wall-clock throughput, TTFT, and a byte-level memory budget into its score
   (StreamingScore = MaxFPS·Acc / (TTFT·Mem)) — surfacing failures invisible in
   accuracy-only tables (e.g. a 4B model sustaining only 0.14 MaxFPS). Every other benchmark
   scores correctness and/or timing but ignores the compute cost of keeping up with the stream.
   Charging costs inverts leaderboards: Flash-VStream-7B's improved variant lifts OVO accuracy
   33.15 → 50.31 while its StreamingScore collapses 2.34 → 0.74 (MaxFPS 8 → 1, TTFT 0.12 → 1.31).
   Cross-subtopic bite: memory/token-compression methods in [[streaming-memory]] and
   [[proactive-response]] ([[r3-streaming]] 76.36 StreamingBench at ~95% token drop,
   [[hermes-kv]] +6.13) report accuracy deltas plus self-reported latency, never harness-enforced
   throughput — [[streamingeval]]'s protocol would plausibly re-order those SOTA claims.

8. **Hold the timing axis fixed and the "streaming gap" nearly vanishes; free it and
   performance collapses.** Under evaluator-timed probing the pure online penalty is small —
   [[streameqa]] measures EgoGPT offline→streaming at 62.03% → 60.93% (−1.1) — and the MCQ
   human-model gaps cluster at 25–30 points ([[streamingbench]] 91.66 vs 67.07, [[ovo-bench]]
   92.81 vs 63.00, [[svbench]] streaming OS 80.24 vs 58.17). Under model-owned timing the gap
   explodes: [[omni-duplexeval]] Proactive Reminder best 20.0 vs Human-Duplex 92.8;
   [[omniinteract]] same-model offline→online quality halves (MiniCPM-o 4.5: 0.683 → 0.348),
   1QnA continuous monitoring near-broken at best 0.052. [[omnipro]]'s dual protocol makes this
   a within-benchmark finding: probe-mode best is Gemini-3-Flash (40.4 accuracy), online best is
   MiniCPM-o 4.5 (20.9 F1) — the *ranking changes* between modes. Family A–B leaderboards are
   largely long-video QA with extra steps; the timing column predicts the human-model gap better
   than any content axis.

9. **Both camps' headline wins are own-benchmark results — and benchmark metrics are becoming
   training rewards.** On evaluator-timed suites the streaming specialists lose to offline
   generalists ([[streamingbench]]: VideoLLM-online-8B 32.48, Flash-VStream-7B 24.04 vs
   LLaVA-OneVision-7B 56.36; [[streamingeval]]: offline Qwen3-VL-8B tops both substrates under
   the same 0.5 GB / 1 FPS budget). On model-owned protocols the specialists win by miles
   ([[streaming-video-wild]]: StreamingHarness-8B SW-F1 45.8 vs best baseline GPT 5.4 at 18.3,
   ~2.5×) — but that model is continue-trained on data matched to the eval's per-second format
   and scored by its own SW-F1. The Goodhart loop is closing: [[proactivevideoqa]]'s PAUC is the
   dominant GRPO reward term in [[mmduet2]], [[egopro-bench]] ships ProAct-Stream trained
   against its own F1/mIoU/GHA, and methods papers keep *rejecting* family-A metrics and minting
   new ones ([[evostreaming]] builds RealStreamEval because prefix-feeding and per-second
   polling leak the timing decision to the evaluator; [[em-garde]] re-scores OVO forward-active
   with its own online F1, arguing the original is gameable by degenerate policies). Every
   headline number in this matrix is protocol-relative.

10. **QA provenance and judging are circling back on the models being judged.** The newest
   benchmarks author their QA with frontier models that then sit on (or share a family with)
   the leaderboard: [[streaming-video-wild]] (GPT 5.4 + Claude Opus 4.6 generation),
   [[streameqa]] (GPT-5 meta-extraction and distractor refinement — GPT-5 is also its top
   model at 61.3%), [[omnipro]] (Gemini-3-Flash generates *and* judges, and is the best
   probe-mode model). Only [[omni-duplexeval]] calibrates its LLM judge against re-annotated
   human scores (Spearman ρ>0.9 content, ~0.8 temporal) and ships *two* human baselines —
   Offline 91.5 vs Duplex 81.8 — quantifying how much of the "human ceiling" is itself
   duplex-constrained, a control nobody else runs.

11. **Multi-timestamp re-ask is the underused middle ground.** [[rtv-bench]] (MTQA, same query
   re-asked), [[streamingcot]] (answer-per-segment), and [[ovbench-videochat-online]]
   (distractors mined from other timestamps) force temporal grounding *without* full duplex —
   a cheaper stress test than model-owned timing. Nobody yet combines multi-timestamp re-ask
   *with* model-owned timing, and nobody combines omni-audio + model-owned timing + egocentric
   gaze in one benchmark — the open corners of this design space. More empty cells:
   *model-owned timing × enforced wall-clock* has no full entry ([[streamingeval]] charges
   compute but reuses probe-timed substrates; [[streaming-video-wild]] enforces a <1 s real-time
   threshold but is vision-only, scored by its own-model-favoring metric); no benchmark is built
   of streams where the correct behavior is mostly *silence* (false-alarm rates surface only
   incidentally); and no paper scores one fixed set of model outputs under PAUC, SW-F1, and
   IA-QTF1 together to measure metric agreement — the cheapest high-value study nobody has done.

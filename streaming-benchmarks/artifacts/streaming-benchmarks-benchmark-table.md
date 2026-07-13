---
tags: [streaming-video-understanding, streaming-benchmarks, synthesis]
---

# Streaming Benchmarks — Verified Benchmark-of-Benchmarks Table

This sub-topic *is* the benchmarks, so the "verified eval table" is a **benchmark-of-benchmarks**: one row per streaming-video benchmark — what it measures, its scale, task regimes, its **timing protocol**, its metric, and the best-reported model versus the human baseline (where a note carries one). The second table is a **coverage matrix**: which external *model* papers (from the sibling [[proactive-response]] and [[streaming-memory]] sub-topics) actually report a score on each benchmark.

The organizing axis — and the story these 17 benchmarks tell chronologically — is the **timing protocol**:

- **Evaluator-timed probe.** The harness feeds the model the prefix `Video[0:t_q]` and asks a question pinned to `t_q`; the model answers on demand (usually MCQ). The clock belongs to the evaluator. This is the 2024–2025 mainstream ([[streamingbench]], [[ovo-bench]] backward/real-time, [[svbench]], [[streameqa]], [[ovbench-videochat-online]], [[rtv-bench]]).
- **Model-owned timing.** No query timestamp is given, or the model's own throughput/decision decides *when* it reads and speaks; correctness is inseparable from *when* the answer fires. This is the **2026 validity correction** — [[proactivevideoqa]], [[streamingeval]], [[egopro-bench]], [[streaming-video-wild]], [[omni-duplexeval]], [[omniinteract]] all argue that an evaluator-timed probe over-credits models by handing them the one thing a real streaming assistant must earn: the decision of when to act. Several benchmarks are **mixed** — an evaluator-timed comprehension half plus a model-owned proactive half ([[ovo-bench]]'s forward-active, [[river-bench]], [[streamgaze]], [[omnimmi]], [[omnipro]]).

![[ovo-bench.png]]
*The foundational human-vs-model gap that anchors the family: [[ovo-bench]]'s three timestamp-conditioned regimes (backward tracing / real-time perception / forward active responding). Best model Gemini 1.5 Pro 63.00% vs human 92.81% — a ~30-point gap. Source: [[ovo-bench]] Fig. 2.*

---

## Table 1 — The benchmarks (grouped by idea-family)

Scores are the **note-verified** best-reported model and human baseline. `n/r` = the note does not report that value. Baseline/companion models shipped *with* a benchmark are flagged **[companion]** (they are not an external SOTA claim).

### A. Evaluator-timed streaming QA (timestamped, passive-to-contextual)

| Benchmark | What it measures | Scale (videos / QA) | Task regimes | Timing protocol | Metric | Best model · human |
|---|---|---|---|---|---|---|
| [[streamingbench]] (2411.03628) | Streaming comprehension along 3 axes; the foundational timestamped eval | 900 / 4,500 MCQ | real-time visual · omni-source A+V · contextual history (18 tasks) | Evaluator-timed probe (prefix to `t_q`; proactive `\|t_out−t*\|≤2s`) | MCQ accuracy + proactive ±2s tolerance | Gemini 1.5 Pro **67.07%** · human **91.66%** (−24.6 pt) |
| [[ovbench-videochat-online]] (2501.00584) | Timestamp-anchored online QA over Past/Current/Future | n/r videos (8 source datasets) / ~7,000 QA | 6 tasks / 16 subtasks × Past/Current/Future | Evaluator-timed probe (same-video distractors mined from other timestamps) | MCQ accuracy | VideoChat-Online-4B **[companion]** 54.9% (streaming) · human n/r |
| [[rtv-bench]] (2505.02064) | Real-time QA whose answer *changes* over time (Multi-Timestamp QA) | 552 / ~4,631 (167.2 h) | perception · understanding · reasoning (8 dims / 16 subcats) | Evaluator-timed multi-timestamp probe (same query re-asked as scene evolves) | Gated hierarchical Acc / Score | GPT-4o **50.02% Acc / 22.10 Score** · human n/r |
| [[streamingcot]] (2510.25332) | *Dataset*: time-evolving per-segment answers + spatiotemporal CoT traces | 5,000 / 34,470 dynamic QA (68,940 segments) | 6 temporally-evolving question types | Evaluator-timed segment-level answers | none — dataset paper, no benchmark/leaderboard table | n/r (no results table by design) |

### B. Temporal-regime benchmarks (backward / real-time / forward-active)

| Benchmark | What it measures | Scale (videos / QA) | Task regimes | Timing protocol | Metric | Best model · human |
|---|---|---|---|---|---|---|
| [[ovo-bench]] (2501.05510) | Temporal awareness across 3 timestamp-conditioned regimes | 644 / ~2,800 | backward tracing · real-time perception · forward active (3 regimes / 12 subtasks) | **Mixed**: evaluator-timed MCQ (backward/RT) + model-owned timing (forward-active, `2^−(m'−m)p` decay) | MCQ acc + time-decayed forward-active score | Gemini 1.5 Pro **63.00%** · human **92.81%** (−29.8 pt) |
| [[streameqa]] (2512.04451) | Embodied cognition × streaming temporal scope | 156 / 20,731 | 3 embodied levels × 3 temporal scopes = 9 cat / 42 subtasks | Evaluator-timed probe (`V[t_e:t_q]`, query set after event) | 4-choice MCQ accuracy | GPT-5 **61.3%** · human n/r (EgoVLPv2 23.7% is below 25% chance) |

### C. Interactive-dialogue benchmarks

| Benchmark | What it measures | Scale (videos / QA) | Task regimes | Timing protocol | Metric | Best model · human |
|---|---|---|---|---|---|---|
| [[svbench]] (2502.10810) | Streaming as temporal multi-turn dialogue chains linked across clips | 1,353 / 49,979 | dialogue walk vs streaming walk (9-skill taxonomy) | Evaluator-timed dialogue turns (streaming walk = 80% stochastic jump to a linked chain) | GPT-4-judge Overall Score (5 dims SA/CC/LC/TU/IC) | GPT-4o Dialogue OS **66.29** / Streaming OS **58.17** · human **83.93 / 80.24** |
| [[river-bench]] (2603.03985) | Retrospective/live/proactive along 3 temporal axes + forgetting curve | 1,067 / 4,278 | past (retro-memory) · present (live-perception) · future (pro-response) | **Mixed**: evaluator-timed MC/OE (past/present) + model-owned latency-aware proactive | MC/OE accuracy + proactive Loc/timing (hard-zero early, linear-decay late) | GPT-4o Retro 59.56 MC / Live 61.05 MC / Pro-Resp Loc 1.63 · human n/r |

### D. Proactive-timing + model-owned-timing protocols (the 2026 validity correction)

| Benchmark | What it measures | Scale (videos / QA) | Task regimes | Timing protocol | Metric | Best model · human |
|---|---|---|---|---|---|---|
| [[proactivevideoqa]] (2507.09313) | Proactive QA scored by area-under-correctness-vs-time | 1,377 / 1,427 | 4 domains (WEB/EGO/TV/VAD) | **Model-owned** (question fixed at `t=0`; model chooses when to answer) | PAUC (ω=0.5, ×100) | GPT-4.1-mini **47.8/65.8/59.4/47.7** · human **38.6/38.2/47.0/53.6** (humans hampered, not a ceiling) |
| [[streamingeval]] (2603.21493) | Unified wall-clock + byte-budget re-scoring of OVO + StreamingBench | reuses OVO-Bench + StreamingBench (no new data) | throughput / accuracy / latency / memory folded into one score | **Model-owned** (strict time-causality; model throughput dictates the memory snapshot it reads) | StreamingScore = MaxFPS^wf·Acc^wa / (TTFT^wt·M^wr) | Qwen3-VL-8B (offline) **[companion baseline]** OVO 58.00 (Score 2.21) / SB 77.31 (2.37) · human n/r |
| [[streamgaze]] (2512.01707) | Gaze-scanpath-conditioned past/present/proactive QA | 285 / 8,521 | 3 temporal stances / 10 tasks | **Mixed**: evaluator-timed (past/present) + model-owned multi-triggering (proactive) | accuracy + proactive multi-trigger precision | GPT-4o **0.535** · human **0.827** |
| [[egopro-bench]] (2605.07299) | Personalized proactive egocentric interaction (event + intent branches) | 2,400 eval (+12K train) | event-driven (object/action) + intent-driven (12 domains) | **Model-owned** (fire the alert the instant a transient trigger appears) | F1 / mIoU / GHA + LLM-judge memory-consistency/response-quality | ProAct-Stream (Qwen3-VL-8B) **[companion]** Object F1 91.19 / Action F1 79.66 / Intent F1 56.34 · human n/r |
| [[streaming-video-wild]] (2606.08615) | In-the-wild per-second speak-or-stay-silent (Streaming-Eval) | 138 / 15 categories | backward/present/forward × interaction mode | **Model-owned** (per-second `</response>`/`</silence>` decision) | SW-F1 (windowed TP/FP/FN, asymmetric weights) + narration win-rate | StreamingHarness-8B **[companion]** 45.8 SW-F1 (QA) vs best baseline GPT 5.4 High 18.3 · human n/r |

### E. Omni-modal interaction benchmarks

| Benchmark | What it measures | Scale (videos / QA) | Task regimes | Timing protocol | Metric | Best model · human |
|---|---|---|---|---|---|---|
| [[omnimmi]] (2503.22952) | Streaming state-awareness + proactive turn-taking for omni (V+A) LLMs | 1,121 / 2,290 | 6 tasks (AP/SG/MD comprehension + PA/PT/SI proactive) | **Mixed**: evaluator-timed comprehension + model-owned proactive (silence-as-correct) | per-task acc + windowed IoU / silence-correct | best comprehension near-floor (SG 16.33 Gemini-1.5-Pro); M4 **[companion]** PA 25.50 / PT 62.00 · human n/r |
| [[omnipro]] (2605.18577) | Omni-modal proactive (84% of samples need audio); decouples what vs when | 1,262 / 2,700 | 3 cognitive levels × 9 subtasks × 6 capabilities | **Dual**: Probe (evaluator-timed, content-only) + Online F1 (model-owned timing) | Probe Accuracy + Online F1 (±3s greedy match) | Probe Gemini-3-Flash **40.4%**; Online MiniCPM-o 4.5 **20.9%** · human n/r |
| [[omni-duplexeval]] (2605.17360) | Duplex omni grading of *what* and *when* jointly | 660 (all <60s) / 9 subtasks | Real-Time Description (6) + Proactive Reminder (3) | **Model-owned** (native duplex; ±window multi-sampling for ~2s latency) | 0.5·content + 0.5·temporal | MiniCPM-o 4.5 **39.6** · Human-Duplex **81.8** / Human-Offline **91.5** (proactive 20.0 vs 92.8) |
| [[omniinteract]] (2605.26485) | Raw AV stream; model must self-detect triggers + answer in response slots | n/r videos (120 nested pairs) | real-time / proactive / nested / 1QnA / interruption | **Model-owned** (timing never given; slot `[t_start,t_a,t_end)` with time-decay) | IA-QTF1 (soft TP + linear timeliness decay) | MiniCPM-o 4.5 All-Global **0.368** (AURA 0.363, Gemini 2.5 Flash Live 0.344) · human n/r |

**What the timing column shows.** The evaluator-timed benchmarks (A, B-comprehension, C-dialogue) report near-humanlevel absolute numbers on comprehension (GPT-5 61.3% on [[streameqa]], GPT-4o 66.29 OS on [[svbench]]) but still trail humans ~25–30 pts. The moment scoring shifts to **model-owned timing**, absolute numbers collapse: [[omni-duplexeval]] proactive best 20.0 vs human 92.8; [[omniinteract]] best IA-QTF1 0.368; [[streamgaze]] proactive OAA GPT-4o 0.149 vs human 0.780. The 2026 papers' claim — that evaluator-timed probes were masking the real deficit (deciding *when*) — is borne out by the size of the drop. The collapse is also **bimodal, not uniform**: [[omni-duplexeval]]'s failure split (Table 4) shows commentary-trained models over-firing (LiveCC-Base 91.1% wrong, [[streamingvlm]] 96.7%) while the strongest models over-stay-silent ([[mmduet2]] 75.8% no-answer, MiniCPM-o 4.5 49.2%) — the two families fail in opposite directions, and neither reliably times a response.

[[streamingeval]]'s single controlled table adds the *resource* half of the verdict, since it is the only place offline and online models are compared under an identical byte budget and wall clock: an off-the-shelf offline Qwen3-VL-8B tops both re-scored benchmarks (OVO 58.00 / SB 77.31, MaxFPS 8, TTFT 0.20 s) while the best native-online model [[streamforest]] merely ties on accuracy (55.57 / 77.26) at half the throughput (MaxFPS 4, TTFT 0.98 s). Flash-VStream's accuracy-improved variant jumps OVO 33.15→50.31 but craters its StreamingScore 2.34→0.74 (MaxFPS 8→1), and VideoChatOnline-4B sustains **MaxFPS 0.14** — an "online" model that cannot keep up with a 1 fps stream, invisible in every accuracy-only table. As of this table, streaming-native design buys deployability and forward-active timing, not accuracy.

### The shared hard core — answers that drift over time

Four benchmarks independently converged on the same operative definition of "streaming" understanding: the correct answer to a fixed question **changes as the stream advances**. [[rtv-bench]] makes it explicit and prices it (GPT-4o FQA 56.53% vs MTQA 44.73% — re-asking the *same* question as the scene evolves costs ~12 pts even for the best model); [[ovo-bench]]'s timestamp-conditioning (same clip, different correct answer at different `t`), [[ovbench-videochat-online]]'s same-video temporal distractors, and [[streamingcot]]'s per-segment evolving answers are the same construct built three more ways. Answer drift — not video length — is what actually separates these benchmarks from offline long-video QA.

---

## Table 2 — Coverage matrix: which *model* papers report on which benchmark

Adopters are external model papers (from [[proactive-response]] and [[streaming-memory]]) that report a **verified score** on the benchmark in their own tables. `[companion]` benchmarks are excluded from their own adopter list. This exposes the field's concentration: **[[streamingbench]] and [[ovo-bench]] are the de-facto twin standard; the 2025–2026 timing-correction benchmarks are almost entirely un-adopted as external reporting targets** (they evaluate models in-house but no outside model paper reports on them yet).

| Benchmark | # external model-paper adopters | Adopting model papers |
|---|---|---|
| [[streamingbench]] | 31 | [[dispider]], [[mmduet]], [[mmduet2]], [[em-garde]], [[r3-streaming]], [[response-g1]], [[roma]], [[streamagent]], [[streambridge]], [[streamov]], [[streampro]], [[stride]], [[thinkstream]], [[timechat-online]], [[vispeak]], [[wat]], [[decouple-and-cache]], [[eventmemagent]], [[fluxmem]], [[hermes-kv]], [[infinipot-v]], [[livevlm]], [[omnimem]], [[streamforest]], [[streaming-model-remember]], [[streamingassistant]], [[streamkv]], [[tww]], [[videoscaffold]], [[vst]], [[weavetime]] |
| [[ovo-bench]] | 31 | [[dispider]], [[em-garde]], [[evostreaming]], [[livecc]], [[r3-streaming]], [[response-g1]], [[streamagent]], [[streambridge]], [[streamo]], [[streamov]], [[streampro]], [[stride]], [[thinkstream]], [[timechat-online]], [[videollm-eyewo]], [[vispeak]], [[wat]], [[decouple-and-cache]], [[eventmemagent]], [[fluxmem]], [[hermes-kv]], [[infinipot-v]], [[savemem]], [[streamforest]], [[streaming-model-remember]], [[streamingassistant]], [[streamingvlm]], [[tww]], [[visual-agentic-memory]], [[vst]], [[weavetime]] |
| [[svbench]] | 2 | [[livestar]], [[livestarpro]] |
| [[ovbench-videochat-online]] | 2 | [[lyrav-dont-pause]], [[streamforest]] |
| [[proactivevideoqa]] | 2 | [[em-garde]], [[mmduet2]] (also reused as [[mmduet2]]'s PAUC training reward); the PAUC metric family also spreads without the benchmark — [[proact-vl]]'s own Live Gaming Benchmark scores with PAUC/TimeDiff/F1 |
| [[omnimmi]] | 1 | [[roma]] (OmniMMI-style PA/PO/CRR/REC) |
| [[rtv-bench]] | 0 | — evaluates models in-house only |
| [[streameqa]] | 0 | — evaluates 13 models in-house only |
| [[river-bench]] | 0 | — evaluates models in-house only |
| [[streamgaze]] | 0 | — evaluates streaming models in-house ([[vispeak]] is fine-tuned on it, not a reporting adopter) |
| [[streamingcot]] | 0 | — dataset paper, no benchmark table |
| [[streamingeval]] | 0 | — re-scores OVO + StreamingBench (a protocol, not yet a reporting target) |
| [[egopro-bench]] | 0 | — ships companion ProAct-Stream; re-evaluates [[omnimmi]]/[[ovo-bench]]/[[streamingbench]] in-house |
| [[streaming-video-wild]] | 0 | — ships companion StreamingHarness |
| [[omnipro]] | 0 | — evaluates omni models in-house only |
| [[omni-duplexeval]] | 0 | — evaluates duplex omni models in-house only |
| [[omniinteract]] | 0 | — evaluates omni models in-house only |

**Reading of the matrix.** Two benchmarks carry the entire field's cross-paper comparability (31 adopters each), and they are the two *oldest* general streaming benchmarks. Every newer benchmark — including all six of the model-owned-timing validity-correction benchmarks (Group D) and all four omni-modal ones (Group E) — has **zero** external reporting adoption so far. The correction they diagnose has been *published* but not yet *adopted* as a reporting standard: model papers keep reporting StreamingBench/OVO-Bench accuracy even as those benchmarks' own evaluator-timed protocol is what the 2026 papers argue is invalid. This is the open coordination gap in the sub-topic.

Two footnotes to the concentration story. First, the timing correction does circulate *inside* model papers even where the correction benchmarks don't: [[evostreaming]]'s RealStreamEval re-scores OVO tasks under strict causality plus a verbosity penalty and **re-ranks the field** ([[dispider]] falls to 33.3 Overall vs EvoStreaming 54.6). Second, this matrix hides a parallel closed ecosystem in the memory sub-topic: ~10 memory papers ([[rekv]], [[cacheflow]], [[rlivs]], [[streamingtom]], [[streammem]], [[videoscan]], [[hermes-kv]], [[infinipot-v]], [[livevlm]], [[decouple-and-cache]]) report on the RVS-Ego/Movie streaming-QA splits shipped with [[flash-vstream]] — largely training-free methods on frozen LLaVA-OneVision backbones — and model families essentially never cross between the RVS cluster and the OVO/SB cluster.

---

## Caveats (what the numbers do and don't mean)

- **Cross-benchmark scores are not comparable.** Metrics differ by construction — MCQ accuracy ([[streamingbench]], [[ovo-bench]], [[streameqa]]), GPT-judge Overall Score ([[svbench]]), area-under-time PAUC ([[proactivevideoqa]]), composite StreamingScore ([[streamingeval]]), and slot-F1 variants (SW-F1 [[streaming-video-wild]], IA-QTF1 [[omniinteract]], joint content×temporal [[omni-duplexeval]]). A "best model" figure is only meaningful within its row.
- **Even one benchmark's name is not one number — the subset/overall collapse.** "[[streamingbench]]" in a model paper usually means the **Real-Time Visual subset** (GPT-4o 73.28, Gemini 1.5 Pro 75.69), not the 18-task overall in Table 1 above (GPT-4o 60.15, Gemini 67.07) — a ~13-pt gap. Headline adopter scores like [[streaming-model-remember]] 82.67, [[wat]] 77.70, [[r3-streaming]] 76.36 are real-time-subset numbers that cannot sit next to the 67.07 overall. Same on [[ovo-bench]]: [[streambridge]]'s 71.30 is a real-time-regime average (Gemini's RT is 69.32 vs its 63.00 overall), not an overall like [[streaming-model-remember]]'s 67.03. Cross-paper tables that silently mix the two inflate apparent progress by ~10 points.
- **Benchmarks disagree on the *sign* of timing error.** [[river-bench]] hard-zeros early responses and linearly decays late ones; [[ovo-bench]]'s `2^−` decay punishes only lateness; [[sdqes]] does the opposite — its Streaming Recall window *allows 5 s of anticipation* while capping latency at 10 s; PAUC ([[proactivevideoqa]]) rewards answering early-and-correct. A model tuned to one benchmark's asymmetry is mis-tuned for another's — "proactive timing" is not yet a single agreed skill definition.
- **Timing metrics have already been gamed once, measurably.** [[mmduet]]'s EGO PAUC of 46.0 came with a **99.4% duplicate-reply rate** — reply spam that PAUC credits; [[mmduet2]] fixed the spam (duplicates → 8.1%) and its PAUC *fell* to 33.6, revealing the earlier score was mostly artifact. A PAUC-family leaderboard without a duplicate/verbosity column next to it is unaudited.
- **Constructor-family contamination.** Four benchmarks are annotated/generated by a model family that then scores at or near the top of them: GPT-4o generates [[streamingbench]]'s 2,500 real-time QAs and is evaluated on them; Gemini-1.5 semi-annotates [[ovo-bench]]'s timestamps and Gemini 1.5 Pro tops it; Gemini-3-Flash generates *and* judges [[omnipro]] and is its best probe model (40.4); GPT-5 extracts [[streameqa]]'s meta-info and refines its distractors and is its best model (61.3). Each note flags the risk; nobody has measured the size of the same-family advantage.
- **Human baselines are inconsistent.** Some benchmarks report a strong human ceiling ([[ovo-bench]] 92.81, [[svbench]] 83.93/80.24, [[streamgaze]] 0.827, [[omni-duplexeval]] Human-Duplex 81.8); [[proactivevideoqa]]'s "human" is explicitly *not* a ceiling (annotators are hampered by having to pause + type at the exact moment, and several models beat it on WEB/EGO); many 2026 benchmarks report no human number at all. The [[proactivevideoqa]] anomaly matters beyond the row: since [[mmduet2]] uses PAUC as its RL *training reward*, optimizing a metric humans lose on optimizes away from human-like timing.
- **[[streamingeval]] vs [[streaming-video-wild]] name collision.** Both use "Streaming-Eval"-style naming but are distinct: [[streamingeval]] (2603.21493) is a wall-clock re-scoring *protocol* over existing benchmarks; [[streaming-video-wild]] (2606.08615) ships a new in-the-wild benchmark also called *Streaming-Eval* with its own SW-F1 metric. They are kept as separate rows.
- **LLM-judge dependence.** [[svbench]], [[proactivevideoqa]], [[omnipro]], [[omni-duplexeval]], [[omniinteract]], [[river-bench]] all rely on an LLM judge; several ([[omnipro]], [[omniinteract]]) note the judge shares a model family with a top scorer, a possible bias.
- **Companion models are lower bounds, not SOTA.** [[ovbench-videochat-online]]'s VideoChat-Online, [[omnimmi]]'s M4, [[egopro-bench]]'s ProAct-Stream, [[streaming-video-wild]]'s StreamingHarness, and [[streamingeval]]'s Qwen3-VL-8B baseline are shipped *with* their benchmark to demonstrate the task is learnable, not to claim a frontier.

**Siblings:** [[evolution-of-streaming-benchmarks]] (narrative), [[streaming-benchmarks-design-axes]] (axis comparison), [[streaming-benchmarks-concept-graph]] (map). Hub: [[streaming-benchmarks]].

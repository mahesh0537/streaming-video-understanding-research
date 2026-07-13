---
zotero_key: null
authors: Yansong Shi, Qingsong Zhao, Tianxiang Jiang et al. (USTC / Shanghai AI Lab / Fudan / Nanjing; Limin Wang group)
year: 2026
arxiv: 2603.03985
pdf: https://arxiv.org/pdf/2603.03985
tier: deep
subtopics: [streaming-benchmarks]
tags: [streaming-video-understanding, streaming-benchmarks]
---

# RIVER: A Real-Time Interaction Benchmark for Video LLMs

**Lineage role:** casts online video understanding as an interactive dialogue split into three temporal axes — retrospective memory (past), live-perception (present), and proactive response (future) — and, uniquely, quantifies the *memory forgetting curve* by bucketing recall questions across four temporal spans.

## Problem — what was limited before this paper (short)
Nearly all MLLMs are offline: they ingest the whole video, then answer. Existing "online" benchmarks (VStream-QA, StreamingBench, OV-Bench, OVO-Bench) attach timestamps to questions but still evaluate largely like offline QA. Two things went unmeasured: (1) how memory *degrades* as the recall interval grows (a forgetting curve), and (2) the *joint* accuracy-and-timing quality of a proactive response — waiting for a cue then answering as fast as possible without false alarms. OVO-Bench is the closest prior work but lacks fine-grained temporal segmentation of the clue/response intervals.

## Key idea — the core insight
Model online interaction as a window-based dialogue and organize every question by the temporal relationship between the *clue time* $t_V$ (when the queried event happens), the *question time* $t'$, and *now* $t$. This yields three task families — Retro-Memory ($t_V < t'$, answer from history), Live-Perception ($t' \le t_V \le t$, answer immediately), and Pro-Response ($t_V > t$, wait for the cue then respond). Retro-Memory is further sliced into short/medium/long/very-long recall intervals so the benchmark plots an explicit *forgetting curve*, and Pro-Response is scored with a latency-aware metric that penalizes early answers to zero and linearly decays late ones.

![[river-bench.png]]
> **Crux (Figure 1).** The four interaction subclasses laid out on a timeline — Query (▶), Cue (◆), Answer (●): Retro-Memory pulls the clue from the past, Live-Perception from the present (both demand an immediate answer), while Pro-Response must *wait* for the cue to appear on the stream and then respond as fast as possible (Instant = single answer, Streaming = continuous narration). *Shi et al. (2026), arXiv:2603.03985. Embedded for personal research reference.*

## Method + math — the eval protocol in full
**Formulation.** Online interaction is a window-based video-text-to-text task with the autoregressive objective
$$\mathcal{L} = -\log P_\theta\big(r_t \mid V_{t':t},\, q,\, h_{<t'},\, r_{<t}\big),$$
where $P_\theta$ is the online MLLM (oMLLM), $V_{t':t}$ is the streaming video from $t'$ to now $t$, $q$ the user query (words), $r_t$ the desired response, and $h_{<t'}$ the historical modeling of past context. Crucially, each response $r$ may contain **multiple EOS tokens at arbitrary positions** to simulate silence/pauses in a live conversation (the model can choose to say nothing at a given window).

**Task taxonomy** (by clue time $t_V$):
- **Retro-Memory** ($t_V < t'$): recall a past event; questions bucketed by recall interval — short (15–30 s), medium (30–60 s), long (300–900 s), very long (1800–3600 s). Each answer must be derivable from exactly one moment. ~1.5k MC pairs, built from Vript-RR, LVBench, LongVideoBench with clue timestamps inserted at chosen positions.
- **Live-Perception** ($t' \le t_V \le t$): comprehend the current short window — environment, actions, object attributes (color/quantity). ~0.4k samples, same sources.
- **Pro-Response** ($t_V > t$): monitor the stream and fire when a user-specified condition is met.
  - *Instant* (single answer, e.g. predict the next event / notify when an object appears): ~1.4k samples from Ego4D-Narration + QVHighlights, further split short/medium/long/very-long by interval.
  - *Streaming* (continuous real-time narration, akin to dense captioning but conversational): ~1.2k samples from Ego4D-Narration val, following VideoLLM-Online's construction.

The paper also studies how the interval $\Delta = \lVert t_V - t \rVert$ (with $\Delta = 0$ when $t' \le t_V \le t$) drives performance on retro-memory and pro-response — this is the forgetting-curve analysis.

**Data construction pipeline** (Figure 3): collection → filtering → generating/annotation → verification. Each item carries `basic` (video_path, duration), `clue` (query_time), `question` (time, question, choices), and `answer` (time, ground_truth). Quality control: (1) LLM filtering to drop questions answerable **without visual input** (removes language priors) and overly broad / excessively long / ambiguous-cue questions; (2) semantic-similarity curation to keep only *distinctive events* as unambiguous reference anchors, each paired with a precise timestamp.

**Metrics.**
- *Retro-Memory & Live-Perception*: regex-match the chosen letter for multiple-choice (**MC** accuracy); for failed matches / non-MC use open-ended (**OE**) judging with Qwen2.5-72B scoring consistency against the reference. Both MC and OE numbers are reported.
- *Pro-Response* — a **Response Accuracy Metric** aligned to a ground-truth timestamp $t_g$ with a tolerance window of length $w$ centered on $t_g$. Conceptually the per-response score $s(\hat t)$ behaves as
$$s(\hat t) = \begin{cases} 1, & |\hat t - t_g| \le w/2 \\ 0, & \hat t < t_g - w/2 \quad (\text{early} \to \text{false alarm, hard zero}) \\ \max\!\big(0,\ 1 - \tfrac{\hat t - (t_g + w/2)}{\tau}\big), & \hat t > t_g + w/2 \quad (\text{late} \to \text{linear decay}) \end{cases}$$
i.e. full credit inside the window, **zero for any early response** (to suppress false alarms), and a linear decay to zero for late responses as latency grows. A `Loc` score measures how precisely the response lands in the correct window; content is scored MC/OE on top.

## Explicit design choices
- **Three-axis taxonomy keyed on clue time** $t_V$ vs. question time $t'$ vs. now $t$ — past / present / future as one unified interactive formulation.
- **Forgetting curve as a first-class object**: retro-memory (and instant pro-response) explicitly bucketed short/medium/long/very-long so degradation is plotted, not averaged away.
- **Asymmetric proactive scoring**: early responses get a hard zero (false alarms are worse than lateness); late responses decay linearly — encodes human tolerance.
- **Multiple EOS tokens per response** so "staying silent" is a valid, modeled action in the streaming dialogue.
- **Data reuse over new capture**: restructures Vript-RR, LVBench, LongVideoBench (memory/perception), Ego4D-Narration + QVHighlights (proactive) into online form with precise clue/question/answer timestamps.
- **1,067 videos / 4,278 questions**; task mix ≈ Retro-Memory 42.9%, Live-Perception 39.0%, Proactive Anticipation 18.0%; durations reach up to ~120 min.
- **Language-prior scrub**: LLM filter deletes questions answerable without vision; semantic-similarity filter keeps only distinctive-event anchors.
- **Companion baseline method** (not the benchmark itself): a sliding-window long-short-term-memory adapter that turns offline MLLMs online — 1 fps sliding window, short-term memory = current-window frame tokens, long-term memory = $M$ fixed compressed slots filled by a **nearest-neighbor averaging** merge; timeline/timestamps injected into the system prompt. Plus a trained model: VideoLLM-Online-style (SigLIP-Large-Patch16 + 2-layer MLP + LLaMA3-8B, 4 fps, LoRA on all linear layers, 1 epoch, lr $3\times10^{-5}$, DeepSpeed ZeRO-2) fine-tuned on a RIVER pro-response training set with **randomized query timestamps** (not anchored at 0 s) for generalization.

## Key results / what to remember
All numbers % , verified against the paper's tables. "OE" = open-ended judge, "MC" = multiple-choice; "Loc" = temporal-localization score for pro-response.

- **GPT-4o (50 frames) is best overall** (Table 2): Retro-Memory 59.56 MC / 39.09 OE; Live-Perception 61.05 MC / 40.08 OE; Pro-Response Loc 1.63. It cannot do proactive Instant/Streaming (∅).
- **Gemini-1.5-pro (50 frames)** (Table 2): Retro 36.35 MC; Live 52.19 MC; Loc 1.51.
- **Best open-source online-adapted**: VideoChat-Flash @1fps (Table 2) — Retro 45.75 MC / 25.68 OE; Live 56.35 MC / 33.60 OE; Pro-Response Instant 20.24 MC, Streaming 35.90 OE, Loc 6.21. LLaVA-Video @1fps: Instant 19.50, Loc 6.21. InternVL2.5 @1fps: Live 58.84 MC, Instant 17.95.
- **Native streaming models underperform** (Table 2): Flash-VStream @1fps — Retro 27.28 MC, Live 29.28 MC, Loc 1.31; attributed to being tuned for long-video comprehension, not interactive QA.
- **Forgetting curve (Table 4, MC)**: performance decays with recall interval. GPT-4o 63.26 (short) → 63.26 (medium) → 58.01 (long) → 52.21 (very long). VideoChat-Flash@1fps 43.92 → 48.90 → 41.44 → 38.12. Adding the long-term-memory module reduces the decay (~12% less degradation vs. baseline, per text/Fig. 5).
- **Fine-tuning helps proactive (Table 3)**: VideoLLM-Online @2fps Instant 23.88 → **+RIVER 33.28** (@2fps) → **35.16** (@4fps); MC 6.67 → 9.84 → 10.53; Loc 4.41 → 5.03 → 5.47. Text reports an **11.28% pro-response accuracy improvement over baseline** from RIVER fine-tuning.
- **Hardest cue type**: causal cues (Table 5) — all methods weak; e.g. VideoChat-Flash@1fps Fine-Grained 50.39 / Causal 40.92 / Background 54.10 MC.

No Zotero highlights present.

Takeaways: offline MLLMs win single-shot QA but collapse under strict real-time constraints; memory degrades measurably with interval and a lightweight LTM slot module flattens the curve; proactive timing is near-floor for everyone (Loc ≤ ~6%), and targeted fine-tuning on interactive data is the biggest lever for it.

## How it connects (evolution)
- [[ovo-bench]] — the closest prior taxonomy (Backward Tracing / Real-Time Perception / Forward Active Responding); RIVER refines it with fine-grained clue/response interval segmentation and the forgetting curve.
- [[streamingbench]] — timestamp-linked online benchmark RIVER positions against for not formalizing interaction dynamics.
- [[ovbench-videochat-online]] — OV-Bench's current/past/future temporal split, a direct predecessor of RIVER's three axes.
- [[videollm-online]] — the proactive-response paradigm (LM-head special tokens for when-to-speak); RIVER borrows its streaming construction and fine-tunes it as the online baseline.
- [[mmduet]] — alternative proactive-turn mechanism (per-frame info/relevance scores) in the same online-interaction lineage.
- [[streaming-benchmarks]] — sub-topic hub.

## Open questions / limitations
- Proactive-response absolute scores are near the floor (Loc ≤ ~6%), so the benchmark currently discriminates mostly by *how badly* models fail at timing rather than resolving strong systems.
- The forgetting-curve construction reuses offline QA datasets with inserted timestamps; whether inserted clue positions faithfully mimic naturalistic streaming recall is unclear.
- The Pro-Response scoring window $w$, its early-zero / late-linear-decay shape, and the latency scale $\tau$ are design hyperparameters; the paper describes the shape but exact $w$/decay constants aren't pinned in the main text (n/r).
- Open-ended scores depend on a Qwen2.5-72B judge, inheriting that judge's biases.

*Verification: equations and metric semantics checked against the PDF §3 (formulation, task types, metrics) and Figure 1; all headline numbers cross-checked against Tables 1–5 (pages 5, 8, 9) of arXiv:2603.03985 (ICLR 2026 camera-ready). Exact Pro-Response window/decay constants not stated in main text — flagged n/r; the $s(\hat t)$ piecewise form encodes the described shape, not verbatim paper equations.*

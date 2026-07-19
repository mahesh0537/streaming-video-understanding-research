# M0 — frozen Qwen3-VL-8B on StreamingBench (official harness)

> **Subset headline (ctx=full): RT-All 69.21 micro** · Omni 41.45 (vision-only) ·
> Contextual-MCQ 38.79 · PO 28.95 · overall-MCQ-micro 57.25 (our defined aggregate).
> 15% stratified subset, 648 MCQ + 38 PO, 0 errors. 2026-07-18.
> ctx=60s variant ✅: beats ctx=full by +6.3 overall-MCQ. **FULL RUN ✅ (2026-07-19,
> ctx=full): RT-All 73.00 · Omni 40.00 · Contextual-MCQ 35.87 · PO 32.40 · overall-MCQ
> 58.68** (4,500 q, 0 errors). PO matches STRIDE's published 32.40 to the decimal.

## Protocol

Official harness (THUNLP-MT/StreamingBench @8929923) driven end-to-end: official prompts,
on-the-fly ffmpeg libx264 prefix re-encode per query, SQA context accumulation (with GT
answers of prior turns, per official code), PO 1-Hz polling over [t+1, gt+4] breaking on
first "yes" (±2 s window), scoring = **response[0] exact-match** via official count.py,
micro (question-weighted) per category. **Shipped-code behavior replicated including the
fps dead-code finding** ([findings](findings.md#streamingbench-fps-schedule-is-dead-code)):
fps=1.0 for ALL clip lengths. Context setting per row: ctx=full = shipped default
(prefix from t=0); ctx=60s = README-"Main" variant. Omni-Source: vision-only — clips
carry audio but the official offline path feeds no transcript; low ER/SCU/SD/MA is the
designed "gap". No LLM judge anywhere. Deviations: greedy pinned; response left-stripped.

## Results (subset 15%, seed 17)

| Group | ctx=full | ctx=60s |
|---|---|---|
| **Real-Time All (micro, n=380)** | **69.21** | **74.74** |
| **Omni-Source All (n=152)** | **41.45** | **46.05** |
| **Contextual MCQ All (n=116)** | **38.79** | **50.00** (SQA 47.5→70.0) |
| Proactive Output (n=38) | time 28.95 / ans 28.95 | n/a (ctx-independent) |
| **Overall MCQ micro (n=648)** | **57.25** | **63.58** |

Weakest RT subtasks: Counting 41.4, Spatial 56.8, Causal 60.0. Strongest: Prospective
88.2, Attribute 82.6. Contextual: MCU 31.6 / ACU 36.8 / SQA 47.5. Omni floor: Scene
Understanding 21.1.

### Full run (ctx=full, 2026-07-19)

| Group | acc | n |
|---|---|---|
| Real-Time All (micro) | **73.00** | 2500 |
| Omni-Source All | **40.00** | 1000 |
| Contextual MCQ All | **35.87** | 750 |
| Proactive Output | time **32.40** / ans 32.40 | 250 |
| Overall MCQ micro | **58.68** | 4250 |

Full-vs-subset: groups stable within ~4 pts; per-task varies more (Counting +17.7,
SQA −8.3) — subsets are for iteration, groups for tracking, full runs for milestones.
**PO 32.40 == STRIDE's published baseline row exactly** — our polling floor reproduces
the de-facto protocol; the M3 target band is 32.4 → 44.0 (Response-G1's mark).

## Conclusions (so far)

0. **The context setting is the biggest protocol lever measured in M0: ctx=60s > ctx=full
   by +5.5 RT / +4.6 Omni / +11.2 Contextual / +6.3 overall-MCQ.** Full prefixes at 1 fps
   dilute the recent frames most questions target. Exception: Counting prefers full
   context (41.4 vs 31.0) — it needs the whole stream (axis-A memory logic). Any SB
   number without a ctx label is uninterpretable.
1. **RT-All 69.21 is the comparable number for vision-only baselines** — Omni (audio-
   dependent) and Contextual drag every cross-category aggregate; "SB overall" without a
   label is meaningless (and has no official aggregator — see findings).
2. **PO floor ≈ 29** under the official break-on-first-yes polling — the M3 trigger axis
   baseline (published floor ≈32.4 [[stride]]; training-free ceiling on record 44.0
   [[response-g1]]).
3. [[streamingeval]]'s 77.31 for the same frozen backbone is its own memory-bounded causal
   harness, not this protocol — the ~8-pt gap between that and our 69.21 is protocol, not
   method, exactly the scatter M0 exists to pin down.
4. Counting/Spatial confirmed weak per the plan's expectations; Text-Rich did NOT
   underperform (75.5) on this backbone.

Raw runs: `/data/mahesh/svu-workspace/results/runs/m0-sb-subset15-qwen3vl8b/` (+ ctx60,
full). Harness anatomy incl. corrections:
`/data/mahesh/svu-workspace/docs/streamingbench-harness-notes.md`.

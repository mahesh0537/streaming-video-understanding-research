# M0 — frozen Qwen3-VL-8B on RIVER (first external baseline)

> **Subset headline: Retro+Live MC overall 42.07** (n=290; 16-frame official driver
> budget). Forgetting curve: second 43.1 / short 39.7 / middle 36.2 / long 48.3;
> Live-Perception 43.1. OE cached, judge deferred. 2026-07-18.
> [[river-bench]] has zero external adopters — these are the first non-author numbers.

## Protocol

Official annotations (OpenGVLab/RIVER); prefix-reencode per the shipped `longshort-off`
drivers: **16 uniform frames over [0, question_time]**, official MC/OE prompts verbatim,
two passes per question. MC letter extraction is our defined rule (no MC scorer ships).
OE judge deferred — paper pins Qwen2.5-72B but the shipped scorer defaults to 7B (a
protocol fork to label at scoring time). Pro-Response: the release ships only the
SCORER; the inference driver is missing — our polling-floor definition (16-s windows,
yes/no gate, break→answer) is documented as ours. Notable code excavation: the paper's
"unpublished" Pro-Response decay constant is **τ = w/2** in the shipped scorer
(w = 15/30/300/1800 s per bucket; early = hard 0, late linear to 0 at w past t_g).

## Results (subset 15%, seed 17)

| Bucket (recall Δ) | MC acc | n |
|---|---|---|
| second (15–30 s) | 43.10 | 58 |
| short (30–60 s) | 39.66 | 58 |
| middle (300–900 s) | 36.21 | 58 |
| long (1800–3600 s) | 48.28 | 58 |
| imme (Live) | 43.10 | 58 |
| **overall** | **42.07** | 290 |

Reference (authors' pipeline): GPT-4o Retro 59.56 MC / Live 61.05 MC. Plan targets:
Retro ≥ 60 MC (M2), Loc > 6.2 (M3).

## Conclusions (so far)

1. **~17 pts under GPT-4o's Retro MC with the official 16-frame budget** — the purest
   axis-A measurement in the program: 16 uniform frames over an hour of video is one
   frame per ~4 min. This is the number M2's memory methods exist to lift.
2. **Non-monotone forgetting curve on the subset** (long bucket highest) — suspect
   bucket×source confound at n=58; verify on the full run before interpreting.
3. RIVER is the cheapest benchmark to run (no ffmpeg re-encode; ~3.7 s/call) — good
   iteration target for M2 ablations.
4. Video coverage: Retro/Live ~98% assembled (LVBench unlock was the key move);
   Pro-Response pending QVHighlights (streaming in) + a sized Ego4D subset fetch.

Raw runs: `/data/mahesh/svu-workspace/results/runs/m0-river-retrolive-subset15-qwen3vl8b/`.
Anatomy: `/data/mahesh/svu-workspace/docs/river-harness-notes.md`.

# M0 — frozen Qwen3-VL-8B on OVO-Bench (official harness)

> **Headline: 51.59 total** (backward 43.06 / realtime 61.78 / forward 49.92).
> Full official run: 1,640 items / 3,035 queries, 0 inference errors, ~2 h on 1× H100.
> **First official-harness number on record for this backbone** — resolves the published
> scatter (see [findings](findings.md#published-scatter-resolved)). 2026-07-17.

## Protocol (label for every number here)

Official OVO-Bench *current edition* (JoeLeelyf/OVO-Bench; README warns it differs from the
arXiv paper): pre-chunked prefix clips Video[0:ceil(t)], **64-frame uniform sampling**
(min(64, len−2)), max_pixels 360·420, official prompts, **fully rule-based scoring — no
LLM judge exists in the current pipeline** (letter-substring MCQ; digit-extraction REC;
Yes/No substring SSR/CRR), macro-average regimes, total = mean of 3 regimes.
Deviations (recorded): greedy decoding pinned (HF config default is sampling t=0.7);
responses stored as strings not 1-element lists. Backbone frozen bf16, sdpa+forced-flash.

## Results (full run)

| | EPM | ASI | HLD | **Backward** | STU | OJR | ATR | ACR | OCR | FPD | **Realtime** | REC | SSR | CRR | **Forward** | **Total** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full 3035 q | 49.49 | 68.92 | 10.75 | **43.06** | 55.62 | 57.61 | 68.10 | 53.21 | 71.81 | 64.36 | **61.78** | 43.55 | 64.55 | 41.67 | **49.92** | **51.59** |
| subset 15% (528 q) | 48.89 | 60.87 | 10.71 | 40.16 | 55.56 | 57.14 | 66.67 | 41.18 | 78.26 | 68.75 | 61.26 | 49.54 | 68.06 | 42.00 | 53.20 | 51.54 |

Subset tracks full within 0.1 total (per-task drift up to ~12 pts on small tasks) →
iterate on subsets, report full.

## Conclusions

1. **HLD ≈ 10.7 — below the 25% random floor, verified genuine** (GT is always "Unable to
   answer"; the frozen model commits to a concrete hallucinated answer every time).
   The M1 axis-D HLD guard is the single cheapest large win: ~10+ pts of Backward
   macro-average sit behind it. Consistent with [[ovo-bench]]'s finding that HLD separates
   proprietary > open > online models.
2. **Backward (43.06) is the weakest regime** — the axis-A memory case, as [[savemem]] /
   [[hermes-kv]] evidence predicted. Realtime (61.78) is already strong.
3. **Forward = 49.92 in the current edition's rule-based scoring.** Note this is NOT the
   Table-1 GPT-judged FAR family from the paper — the current codebase replaced judge
   scoring entirely; comparisons to Response-G1 58.2 / STRIDE 46.30 need re-running those
   under this edition or labeling the difference.
4. Timing: video decode dominates inference 4:1 (1.6 s vs 0.4 s per query) — budget for
   M2 methods that re-decode windows.

Raw runs: `/data/mahesh/svu-workspace/results/runs/m0-ovo-{full,subset15}-qwen3vl8b/`.
Harness anatomy: `/data/mahesh/svu-workspace/docs/ovo-harness-notes.md`.

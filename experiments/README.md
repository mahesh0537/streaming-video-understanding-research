# Experiments — training-free baseline program (frozen Qwen3-VL-8B)

Results and conclusions from executing [[Jul_17th]] (ideation/Jul_17th.md). One directory
per milestone; every number carries a protocol label. Raw runs, code, and per-query
outputs live in the implementation workspace `/data/mahesh/svu-workspace` (its own git
repo); this directory holds the *findings* — what the vault needs to reason about
next steps.

## Status board

| Milestone | Benchmark | Status | Headline (protocol-labeled) |
|---|---|---|---|
| M0 baseline truth | [[ovo-bench]] | ✅ full run 2026-07-17 | **51.59** overall, official harness (first such number on record) |
| M0 baseline truth | [[streamingbench]] | ✅ COMPLETE (subset both ctx + full run) | Full: RT-All **73.00**, overall-MCQ **58.68**, PO **32.40** (=STRIDE row); ctx=60s subset RT 74.74 |
| M0 baseline truth | [[river-bench]] | ✅ retro/live FULL (98% cov) · ⏳ pro-response | **MC 43.78 / OE 29.42**, monotone forgetting curve — first external baseline ever |
| M1 prompt layer (axis D) | [[ovo-bench]] | 🔄 ONGOING — d1 ✅ d2 ✅ · d3/d4 interrupted | **d2 evidence gate: subset total 51.54 → 59.87** (HLD 10.71 → 85.71); d1 instruction-only failed. d2 confounded by HLD's single-label GT — see axis-d.md |
| M2 memory bake-off (axis A) | – | ⬜ | order: re-prefill → SAVEMem → HERMES |
| M3 triggers (axis C) | – | ⬜ | PO floor 32.40 (full, =STRIDE); RIVER Loc floors per-bucket recorded |
| M4 composition | – | ⬜ | d2 blocked from composition until its false-positive rate is measured |

### Ongoing / next runs

Workstation crashed 2026-07-20 09:00 UTC mid-M1; both repos were committed and pushed
first, so no code or completed result was lost. Queued, in order:

| Run | Config | State | Note |
|---|---|---|---|
| d3-spatial (OVO STU) | `m1-ovo-stu-d3spatial-qwen3vl8b` | ⏳ died at 5/178 | resumable — relaunch same config |
| d4-counting (SB Counting) | `m1-sb-counting-d4-qwen3vl8b` | ⏳ died at 1/173 | resumable; needs `sb_precut.py` first |
| d2 false-positive probe | not yet written | ⬜ | held-out answerable items carrying an unable-option; gates d2's promotion to M4 |

## Files

- [m0-baseline-truth/ovo-bench.md](m0-baseline-truth/ovo-bench.md) — full M0 OVO results + protocol
- [m0-baseline-truth/streamingbench.md](m0-baseline-truth/streamingbench.md) — SB subset results (full run pending)
- [m0-baseline-truth/river.md](m0-baseline-truth/river.md) — RIVER first-external-baseline results
- [m0-baseline-truth/findings.md](m0-baseline-truth/findings.md) — adversarial findings the plan must absorb
  (published-scatter resolution, official-harness bugs, judge facts)
- [m1-prompt-layer/axis-d.md](m1-prompt-layer/axis-d.md) — axis-D layer results: d1 negative, d2 win + its
  confound, d3/d4 in flight

## Conventions

- Protocol block embedded in every metrics.json in the workspace; summarized per table here.
- "SB" is never quoted without a group label (RT vs Omni vs Contextual vs overall-MCQ).
- OVO Forward numbers are Table-1-accuracy-family only; other families co-reported, never mixed.
- Subset mode = stratified 15%, seed 17 — validated within 0.1 total of the full run on OVO.

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
| M0 baseline truth | [[streamingbench]] | ✅ subset ctx=full · ⏳ ctx=60s variant · ⏳ full run | RT-All **69.21** micro, shipped-code protocol (fps dead-code replicated) |
| M0 baseline truth | [[river-bench]] | ✅ retro/live subset · ⏳ full + pro-response | Retro+Live MC **42.07** (first external baseline ever) |
| M1 prompt layer (axis D) | – | ⬜ | HLD guard is the confirmed cheapest win (see findings) |
| M2 memory bake-off (axis A) | – | ⬜ | order: re-prefill → SAVEMem → HERMES |
| M3 triggers (axis C) | – | ⬜ | PO floor measured at ≈29 (subset) |
| M4 composition | – | ⬜ | – |

## Files

- [m0-baseline-truth/ovo-bench.md](m0-baseline-truth/ovo-bench.md) — full M0 OVO results + protocol
- [m0-baseline-truth/streamingbench.md](m0-baseline-truth/streamingbench.md) — SB subset results (full run pending)
- [m0-baseline-truth/river.md](m0-baseline-truth/river.md) — RIVER first-external-baseline results
- [m0-baseline-truth/findings.md](m0-baseline-truth/findings.md) — adversarial findings the plan must absorb
  (published-scatter resolution, official-harness bugs, judge facts)

## Conventions

- Protocol block embedded in every metrics.json in the workspace; summarized per table here.
- "SB" is never quoted without a group label (RT vs Omni vs Contextual vs overall-MCQ).
- OVO Forward numbers are Table-1-accuracy-family only; other families co-reported, never mixed.
- Subset mode = stratified 15%, seed 17 — validated within 0.1 total of the full run on OVO.

---
tags: [streaming-video-understanding, streaming-benchmarks]
---

# Streaming benchmarks — measuring online video understanding

How the field measures streaming ability: timestamped QA (answer as-of a moment), backward/real-time/forward task splits, proactive-timing metrics, and — the 2026 correction — protocols where the *model* owns the timing decision and over-talking is penalized.

Siblings: [[proactive-response]] (the capability these benchmarks expose as weakest), [[streaming-memory]] (long-horizon degradation findings). Hub: [[streaming-video-understanding]] · [[streaming-video-understanding-overview]].

## Synthesis artifacts (start here)

- [[evolution-of-streaming-benchmarks]] — read this first: 2024/11 → 2026/06, how the eval question shifted from "can you answer at a moment" to "do you know when the moment is".
- [[streaming-benchmarks-design-axes]] — 17 benchmarks × 9 axes (timing protocol, regimes, modality, scoring, scale, ships-a-baseline?).
- [[streaming-benchmarks-benchmark-table]] — the benchmark-of-benchmarks + the adoption coverage matrix.
- [[streaming-benchmarks-concept-graph]] — typed lineage graph (extends / critiques / re-scores).

## Where the benchmarks agree / diverge

- **The fault line is the timing protocol.** Evaluator-timed prefix evals ([[streamingbench]], [[ovbench-videochat-online]], [[rtv-bench]], [[ovo-bench]] backward/real-time) hand the model the query timestamp; the 2026 wave ([[proactivevideoqa]], [[egopro-bench]], [[streaming-video-wild]], [[omni-duplexeval]], [[omniinteract]]) makes the model self-detect the trigger. This is the validity correction — older numbers overstate streaming ability.
- **The backward/real-time/forward triad is independently rediscovered five times** ([[ovo-bench]], [[river-bench]], [[streameqa]], [[streaming-video-wild]], [[ovbench-videochat-online]]); forward/proactive is always where the human gap is widest.
- **Every model-owned-timing benchmark invented its own timing metric** (PAUC, SW-F1, IA-QTF1, event-F1/GHA, Online-F1 ±3s). They rhyme — windowed TP/FP/FN with time decay — but none interoperate.
- **The adoption gap:** [[streamingbench]] and [[ovo-bench]] each have ~31 external adopters; *every* model-owned-timing and omni-modal benchmark has zero external adoption so far. The community measures on the older, easier protocol.
- **Audio clusters in one family** ([[omnimmi]], [[omnipro]], [[omni-duplexeval]], [[omniinteract]] + StreamingBench's omni axis); visual-only benchmarks drop audio even in audio-rich domains. [[omnipro]]: 84% of samples need audio; non-speech sound is the hardest split. [[streamgaze]] is the lone gaze-conditioned experiment.

## Questions worth attacking

1. **A unified timing metric** that the six bespoke ones reduce to — the single highest-leverage contribution available in this sub-topic (hint: write each as windowed TP/FP/FN + decay and see what's actually different).
2. **Why zero adoption of the honest protocols?** Cost, tooling, or leaderboard incentives — and what would make model-owned timing the default?
3. Human baselines exist only for a few (OVO-Bench 92.8); most 2026 protocols ship without one, so "the gap" is unmeasured exactly where it matters.

## First moves → `artifacts/`

- Tabulate the six timing metrics side-by-side in one notation and identify the degenerate cases each one mishandles (hints-only: the artifact table gives you the definitions; the reduction is yours to derive).

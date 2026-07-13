---
tags: [streaming-video-understanding, proactive-response]
---

# Proactive response — "when to speak"

The core question of this workspace: given a live, unbounded video stream, how does a model decide the *moment* to respond — flag an event, alert, narrate, answer — versus stay silent? This is the sub-area with the largest human–model gap (OVO-Bench: humans 92.8 vs best model 63.0) and the hottest 2026 output.

Siblings: [[streaming-benchmarks]] (how timing is measured), [[streaming-memory]] (the substrate triggers sit on). Hub: [[streaming-video-understanding]] · [[streaming-video-understanding-overview]].

## Synthesis artifacts (start here)

- [[evolution-of-proactive-response]] — read this first: the throughline from [[videollm-online]]'s streaming-EOS loss (2024/06) to the 2026 training-free trigger wave, one section per mechanism family.
- [[proactive-response-design-axes]] — the decision matrix: 35 papers × 8 axes (decision locus, trigger signal, states, training-free?, decoupled?, omni-audio?, …).
- [[proactive-response-benchmark-table]] — verified numbers grouped per benchmark, with the same-name-different-protocol traps flagged.
- [[proactive-response-concept-graph]] — the Mermaid lineage graph with typed edges + gateways into the sibling sub-topics.

## Reading order

[[videollm-online]] (the origin: when-to-speak as a learned per-frame EOS decision) → [[mmduet]] (the duet reframing + per-frame heads) → [[dispider]] (disentangled perception/decision/reaction — the decoupling archetype) → [[timechat-online]] (visual-change trigger, training-free direction) → [[streambridge]] + [[evostreaming]] (the adaptation thesis: offline models only lack a policy) → [[livestar]]/[[livestarpro]] (verification decoding, no silence tokens) → [[response-g1]] + [[querystream]] (relevance-conditioned, not change-conditioned) → [[lyrav-dont-pause]] (FSM controller, speak-while-watching) → [[mmduet2]]/[[streampro]]/[[thinkstream]] (RL replaces the threshold knob).

## Where the papers agree / diverge

- **Agree:** perception is not the bottleneck — the timing policy is ([[streambridge]], [[evostreaming]]). Binary speak/silent is too coarse; richer states (standby, ask, continue) keep appearing for timing precision, not content. Decoupling the cheap always-on decision path from the heavy generator is the strongest predictor of real-time FPS ([[streammind]] ~100 FPS, [[dispider]], [[stride]]).
- **Diverge — where the decision lives:** in-band token (Family A, pays full-LM per frame) vs dedicated head/policy (Family B) vs decoding-time signal the frozen model already emits (Family C: perplexity, similarity derivative, redundancy valleys, retrieval score). Family C gets timing "for free" but on a proxy signal — change ≠ relevance, the exact gap [[response-g1]] and [[querystream]] attack.
- **Diverge — supervision:** every trained trigger builds its own bespoke frame-level speak/silent labels (VAPDA-127K, PROASSIST, Stream-IT, Streamo-465K, …). There is no shared streaming-trigger training set; the training-free family exists partly because labels are that expensive.
- **The family taxonomy overstates mechanism diversity** — family boundaries track the claimed problem more than the decision mechanism; the in-band token recurs in every era (RL reframings, ternary variants, reasoning loops).

## Questions worth attacking

1. **The unbuilt corner of the matrix:** no system combines a *training-free* decoding-time trigger + *decoupled constant-cost* perception + *omni-audio* turn-taking. Closest partials: [[timechat-online]] (training-free + decoupled, vision-only), [[streamov]] (decoupled + omni, trained head). What signal would an audio-aware training-free trigger read? (Hint: start from what Family C already reads off frozen models, and ask what the audio channel's analogue is.)
2. **Change- vs relevance-conditioned triggering:** [[querystream]] and [[response-g1]] both argue visual change is the wrong gate. Is there a middle signal — cheap like change, query-aware like relevance — that doesn't need a scene graph?
3. **No shared trigger benchmark:** papers scatter across OVO-Bench FAR, StreamingBench PO, and bespoke evals — which currently hides how far apart these designs really are (see [[proactive-response-benchmark-table]] for the traps).

## First moves → `artifacts/`

- Reproduce the trigger-signal comparison on one clip set: extract Family C's signals (perplexity drift, frame-similarity derivative, redundancy valley) from one frozen backbone and eyeball where they fire vs ground-truth event onsets. (Hints-only from here — this is your derivation/experiment space.)
- Sketch the timing-metric you'd want before building anything (the benchmark side's fragmentation, [[streaming-benchmarks]], is the reason nothing composes).

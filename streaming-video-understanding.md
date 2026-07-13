---
tags: [streaming-video-understanding, hub]
---

# Streaming video understanding

**Start here:** [[streaming-video-understanding-overview]] — the cross-sub-topic connecting map (co-evolution timeline, trigger×memory design-space map, open problems, reading path). Then per sub-topic: [[evolution-of-proactive-response]] · [[evolution-of-streaming-benchmarks]] · [[evolution-of-streaming-memory]], each with a design-axes matrix, verified benchmark table, and concept graph beside it in `artifacts/`.

Models that watch video as it arrives — live, unbounded, causal (no rewind, no lookahead) — and must decide *when* and *what* to output while keeping compute bounded forever. Splits from offline video understanding along exactly two axes, which are the two model sub-topics here:

- **"When to act"** → [[proactive-response]] — the core focus of this topic: mechanisms for deciding the moment to speak / flag an event / stay silent. Temporal grounding is deliberately NOT a sub-topic; recent work treats localization as emergent from good event-level understanding.
- **"How to sustain"** → [[streaming-memory]] — KV-cache compression, hierarchical/event memory, retrieval: the infrastructure that bounds an ever-growing context.
- **How progress is measured** → [[streaming-benchmarks]] — timestamped/online eval, the proactive-timing benchmarks, and the 2026 eval-validity correction (model must own the timing decision).

## Sub-topics

| Sub-topic | Scope | Note |
|---|---|---|
| [[proactive-response]] | "when to speak" trigger mechanisms, origin (VideoLLM-online 2024) → frontier (LiveStarPro, LyraV 2026/06) | 34 deep notes |
| [[streaming-benchmarks]] | StreamingBench, OVO-Bench, RIVER, OmniPro, StreamingEval and the timing-eval wave | 17 deep notes |
| [[streaming-memory]] | KV eviction/retrieval, hierarchical & event memory, agentic memory | 37 deep notes |

## Cross-connections (the dots to connect)

- Every proactive mechanism sits on a memory substrate: event-gated triggers (StreamMind, TimeChat-Online) presuppose event-structured memory ([[streamforest]]-style); LiveStarPro couples hierarchical memory *to* its verification-decoding trigger.
- The benchmark axis drives the mechanism axis: OVO-Bench's Forward Active Responding category exposed the timing gap (humans ~93 vs best models ~63); the 2026 protocol correction (RealStreamEval, [[streamingeval]]) penalizes over-talking, which killed naive silence-token approaches.
- The field's cheapest big result: strong offline VideoLLMs already perceive well and only lack an *interaction policy* ([[streambridge]], [[evostreaming]]) — adaptation beats re-architecting.

## Status snapshot (July 2026)

Perception is near-solved (70s–80s on real-time QA columns); *timing* is not (models collapse ~30 pts on proactive/contextual tasks; a model tuned specifically for forward-active responding still scores ~20 on OVO-Bench). Trigger design space as of mid-2026: EOS/silence tokens → dedicated speak-heads (Dispider, ROMA) → perplexity verification (LiveStar) → visual-change/event triggers (TimeChat-Online) → scene-graph gating (Response-G1) → propose-match (Em-Garde) → training-free FSM controllers (LyraV). Industry: only Gemini ships live-video understanding as product; open frontier is Qwen-Omni + MiniCPM-o 4.5.

## Further reading (surveys, no vault notes)

- *Towards Online Interactors: A Comprehensive Survey on Streaming Video Understanding* — Preprints.org, 2026/06 (DOI 10.20944/preprints202606.1674.v1); by the maintainers of the Awesome-Streaming-Video-Understanding repo (the field's living taxonomy).
- *Watch, Remember, Reason* (arXiv 2606.07433) — broader video-MLLM framework survey; streaming is one memory strategy in it.
- *From Static Inference to Dynamic Interaction* (arXiv 2603.04592, ACL 2026 Findings) — streaming *LLM* survey; useful terminology, not video-specific.
- Tarsier2 (arXiv 2501.07888) — offline generalist lineage; the temporal-alignment + DPO recipe later streaming models borrow.

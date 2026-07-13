---
tags: [streaming-video-understanding, streaming-memory]
---

# Streaming memory — sustaining an unbounded stream

The infrastructure axis: bounding an ever-growing context so a model can run on video forever — KV-cache eviction/compression/retrieval, hierarchical and event-structured memory, agentic/indexed memory, and the open question of *what to remember at all*.

Siblings: [[proactive-response]] (triggers presuppose a memory substrate), [[streaming-benchmarks]] (long-horizon degradation is a benchmark finding). Hub: [[streaming-video-understanding]] · [[streaming-video-understanding-overview]].

## Synthesis artifacts (start here)

- [[evolution-of-streaming-memory]] — read this first: from [[streamingdvc]]'s clustering memory and [[flash-vstream]]'s STAR memory to event trees, TTT memory, and agentic retrieval.
- [[streaming-memory-design-axes]] — 37 papers × the substrate/operation axes (what is stored, evict vs compress vs retrieve, query-aware?, training-free?, bounded?).
- [[streaming-memory-benchmark-table]] — verified headline numbers per benchmark, with the non-comparability caveat.
- [[streaming-memory-concept-graph]] — typed lineage graph.

## Where the papers agree / diverge

- **Two substrates split the field, and training-free-ness picks the side.** KV-cache-native methods (compress/evict/retrieve the decoder's own cache) are almost all training-free ([[infinipot-v]], [[streammem]], [[rekv]], [[streamkv]], [[hermes-kv]], [[cacheflow]], [[livevlm]], [[v-rex]], [[fluxmem]]); any richer substrate — distilled tokens, text notes, event trees, parametric weights — pays for training ([[flash-vstream]], [[streamforest]], [[providellm]], [[tww]], [[video-salmonn-s]]).
- **Training-free ⟺ query-agnostic write.** Compressing before the question is known forces a stand-in query (proxy tokens, pseudo-question banks: [[streammem]], [[streamkv]], [[hermes-kv]], [[savemem]]); query-conditioned compression is deprecated because it forces re-prefill per question.
- **Compress-then-retrieve is the 2025+ default**; pure lossless offload ([[rekv]], [[cacheflow]], [[v-rex]]) and pure budgeted eviction ([[infinipot-v]], [[streammem]]) are the endpoints. The schism: lossy-fixed-budget on GPU vs lossless offload with transfer latency — argued explicitly both ways ([[infinipot-v]], [[livevlm]] vs [[rekv]]).
- **Bounded footprint is table stakes** — every serious method holds GPU state constant on unbounded streams; differentiation moved to *what survives* compression.
- **The principled question is open:** [[streaming-model-remember]] shows nobody has a theory of optimal retention; everything is heuristic (recency, similarity, event boundaries, penalties).

## Questions worth attacking

1. **Memory ↔ trigger coupling** (the bridge to [[proactive-response]]): event-structured memory ([[streamforest]], [[eventmemagent]]) and event-gated triggers ([[streammind]], [[timechat-online]]) are the same boundary-detection problem solved twice — can one event segmentation serve both? (Hint: compare what each side's boundary detector actually computes.)
2. **What should survive**: [[streaming-model-remember]] frames retention as an allocation problem; the design-axes table shows which heuristics recur — the gap between them is where a principled answer would land.
3. **Audio-visual memory is nascent** ([[video-salmonn-s]], [[omnimem]]) — compression that respects cross-modal redundancy rather than compressing streams independently.

## First moves → `artifacts/`

- Map the event-boundary detectors used by memory papers vs trigger papers into one table and mark which signals they share (hints-only beyond that — the unification is your derivation).

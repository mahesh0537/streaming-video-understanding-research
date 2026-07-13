# Streaming Video Understanding — Research Notes

A deep literature map of **streaming video understanding**: models that watch video as it
arrives — live, unbounded, causal (no rewind, no lookahead) — and must decide *when* and
*what* to output while keeping compute bounded forever.

These are research notes written as an [Obsidian](https://obsidian.md) vault, so links
between notes use `[[wikilink]]` syntax. They render as plain text on GitHub; open the
folder in Obsidian for full cross-linked navigation.

**Start here:** [`streaming-video-understanding-overview.md`](streaming-video-understanding-overview.md) —
the cross-sub-topic connecting map (co-evolution timeline, trigger × memory design-space
map, open problems, reading path).

## The three sub-topics

The field splits along two capability axes joined by a measurement referee:

| Sub-topic | Scope | Depth |
|---|---|---|
| [`proactive-response/`](proactive-response/) | **"When to act"** — trigger mechanisms for deciding the moment to speak / flag an event / stay silent. Origin (VideoLLM-online, 2024) → frontier (LiveStarPro, LyraV, 2026). | 34 deep notes |
| [`streaming-memory/`](streaming-memory/) | **"How to sustain"** — KV-cache compression, hierarchical/event memory, retrieval: the infrastructure that bounds an ever-growing context. | 37 deep notes |
| [`streaming-benchmarks/`](streaming-benchmarks/) | **How progress is measured** — timestamped/online eval, proactive-timing benchmarks, and the 2026 eval-validity correction (the model must own the timing decision). | 17 deep notes |

## Layout

Each sub-topic folder contains:

- `<sub-topic>.md` — the running understanding + links to related notes
- `papers/` — one deep note per paper (method + math, crux figure, verified results)
- `figures/` — crux figures/tables cropped from papers, embedded in the notes
- `artifacts/` — generated syntheses: the evolution narrative, design-axis comparison
  table, verified benchmark table, and concept graph

## Status snapshot (July 2026)

Perception is near-solved (70s–80s on real-time QA columns); *timing* is not (models
collapse ~30 points on proactive/contextual tasks). The trigger design space as of
mid-2026 runs: EOS/silence tokens → dedicated speak-heads (Dispider, ROMA) → perplexity
verification (LiveStar) → visual-change/event triggers (TimeChat-Online) → scene-graph
gating (Response-G1) → propose-match (Em-Garde) → training-free FSM controllers (LyraV).
Only Gemini ships live-video understanding as product; the open frontier is Qwen-Omni +
MiniCPM-o 4.5.

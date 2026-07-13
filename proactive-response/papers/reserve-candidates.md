---
tier: reserve
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response, reserve]
---

# Reserve candidates — proactive-response

Super-compact backlog (scanned 2026-07-12). Promotion = write/extend a real `papers/<slug>.md` at compact/deep tier and remove the block here.

## SDVC — Streamlined Dense Video Captioning (arXiv 1904.03870, 2019/04, CVPR 2019)
- core idea: dense captioning that models cross-event dependencies for coherent story-like output
- method: event-sequence-generation network *selects* which proposals to describe + RL-trained sequential captioner
- angle: offline proposal selection, not a causal online trigger
- why it might matter: earliest learned "which events deserve output" head — conceptual ancestor of when-to-emit
- kept reserve because: historical, non-streaming; cite as lineage origin only

## VideoLLM-MoD (arXiv 2408.16730, 2024/08, NeurIPS 2024)
- core idea: cut online-video-LLM compute by skipping (mixture-of-depths) rather than dropping vision tokens
- method: learn to bypass transformer layers for ~80% of vision tokens; ~42% time / ~30% memory savings
- angle: pure efficiency for the VideoLLM-online lineage
- why it might matter: compute backbone for proactive systems
- kept reserve because: no response-timing component

## OpenHOUSE (arXiv 2509.12145, 2025/09, ICCV 2025)
- core idea: open-ended hierarchical streaming understanding — real-time action detection + free-form description
- method: LLM-enriched hierarchical event annotations; streaming boundary detector between adjacent actions (~2× prior)
- angle: streaming temporal localization/captioning; trigger = action boundary, not user-facing response
- why it might matter: boundary detection as a when-to-emit signal
- kept reserve because: aimed at segmentation/description, not the speak decision

## StreamingClaw (arXiv 2603.22120, 2026/03, tech report)
- core idea: unified real-time embodied agent framework (perception-decision-action), OpenClaw-compatible
- method: five capabilities incl. future-event prediction w/ proactive interaction + hierarchical multi-agent memory; marked "under progress"
- angle: broad embodied-agent systems report
- why it might matter: future-event anticipation for timely action
- kept reserve because: proactivity incidental; no isolated, evaluated trigger mechanism

## ColorTrigger (arXiv 2603.22466, 2026/03, CVPR 2026)
- core idea: stream in grayscale continuously, capture color only when it counts (always-on sensing)
- method: training-free windowed grayscale-affinity QP detects chromatic redundancy causally; 91.6% of full-color performance at 8.1% RGB frames
- angle: input-side sensing trigger
- why it might matter: shares causal online-triggering machinery
- kept reserve because: decides when to *sense*, not when to *speak*

## Stream3D-VLM (arXiv 2606.06891, 2026/06)
- core idea: online 3D spatial understanding from streaming video with real-time response control
- method: autoregressive streaming control (next token decides when to respond) + incremental 3D geometry priors + geometry-adaptive voxel compression; >1M spatio-temporal 3D QA
- angle: 3D-spatial niche wrapped around a streaming-control module
- why it might matter: its autoregressive "when to generate" controller is a proactive mechanism
- kept reserve because: 3D-geometry pipeline is off-axis; extract the controller idea only if needed

## ToM — "Learning to Respond" (OpenReview gmpnSSiJt7, listed 2025/12)
- not locatable on arXiv or by search as of 2026-07-12; the awesome-repo links an OpenReview PDF only
- revisit if it lands on arXiv or gets a venue page

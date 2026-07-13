---
tier: reserve
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory, reserve]
---

# Reserve candidates — streaming-memory

Super-compact backlog (scanned 2026-07-12; the last three are repo-listed efficiency papers not yet individually scanned). Promotion = write/extend a real `papers/<slug>.md` at compact/deep tier and remove the block here.

## VideoNarrator / Synchronized Video Storytelling (arXiv 2405.14040, 2024/05, ACL 2024 Findings)
- core idea: narrations synchronized per clip — grounded, knowledge-integrated, length-matched to clip duration
- method: storyline-first structured generation; E-SyncVidStory benchmark + metrics
- angle: offline storyline-conditioned narration (ads/media), not stream-triggered
- why it might matter: output-length-vs-duration matching loosely rhymes with response pacing
- kept reserve because: offline; the "when" overlap is superficial

## VITA-1.5 (arXiv 2501.01957, 2025/01, NeurIPS 2025 Spotlight)
- core idea: GPT-4o-style end-to-end vision+speech real-time interaction, no external ASR/TTS
- method: progressive multi-stage training adding audio to a vision-text LLM
- angle: omni-modal speech-integration system
- why it might matter: real-time interaction plumbing context for deployment
- kept reserve because: speech-centric; not stream-timing or video memory

## StreamFormer (arXiv 2504.20041, 2025/04, ICCV 2025)
- core idea: streaming video *backbone* — frame-by-frame low latency via causal temporal attention in a pretrained ViT
- method: causal temporal attention + multitask visual-language alignment; online action detection / online VIS / VideoQA
- angle: representation level, below the LLM
- why it might matter: the causal encoder many streaming models could sit on
- kept reserve because: one layer below our focus

## Learning from Streaming Video with Orthogonal Gradients (arXiv 2504.01961, 2025/04, CVPR 2025)
- core idea: SSL from continuous streams where correlated sequential batches break IID training
- method: orthogonal-gradient constraint decorrelating sequential batches; optimizer-agnostic
- angle: training dynamics/optimization
- why it might matter: how to *train* on streams
- kept reserve because: orthogonal to inference-time memory/triggering

## Venus (arXiv 2512.07344, 2025/12, INFOCOM 2026)
- core idea: edge memory-and-retrieval *serving system* for online VLM video understanding
- method: edge-cloud disaggregation; scene-segmentation ingestion → hierarchical multimodal-embedding memory; progressive keyframe sampling; 15–131× latency speedup
- angle: systems/serving deployment
- why it might matter: the deployment-infra reference point (pair with [[v-rex]] which carries the algorithmic KV content)
- kept reserve because: serving architecture, not memory modeling

## OmniStream (arXiv 2603.12265, 2026/03) — not yet scanned
- repo-listed under computational efficiency / sparse computing; scan before any promotion

## AutoGaze (arXiv 2603.12254, 2026/03, CVPR 2026) — not yet scanned
- repo-listed under computational efficiency (gaze-driven sparse compute); scan before any promotion

## STC (arXiv 2512.00891, 2025/12, CVPR 2026) — not yet scanned
- repo-listed under computational efficiency / sparse computing; scan before any promotion

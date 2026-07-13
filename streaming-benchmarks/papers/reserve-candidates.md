---
tier: reserve
subtopics: [streaming-benchmarks]
tags: [streaming-video-understanding, streaming-benchmarks, reserve]
---

# Reserve candidates — streaming-benchmarks

Super-compact backlog (scanned 2026-07-12). Promotion = write/extend a real `papers/<slug>.md` at compact/deep tier and remove the block here.

## TemporalBench (arXiv 2410.10818, 2024/10)
- core idea: fine-grained temporal understanding benchmark; argues most video benchmarks are effectively static-image tests
- method: ~10K QA from ~2K human annotations (action frequency, event order); Multiple Binary Accuracy metric vs MCQ bias
- angle: offline, whole-clip protocol; SOTA ~38.5% vs human (~30 pt gap)
- why it might matter: temporal-order reasoning underlies streaming event flagging
- kept reserve because: offline protocol — not streaming/proactive

## OST-Bench (arXiv 2507.07984, 2025/07, NeurIPS 2025)
- core idea: online spatio-temporal 3D scene understanding over incrementally acquired agent observations
- method: 1.4K scenes / 10K QA from ScanNet, Matterport3D, ARKitScenes; agent-exploration setting
- angle: genuinely online but embodied-3D niche
- why it might matter: accuracy decays with exploration length — a clean memory-degradation data point for [[streaming-memory]]
- kept reserve because: 3D-spatial niche outside our scope

## QICD (no arXiv; listed 2025/11, NeurIPS 2025)
- core idea: live step-by-step task guidance — streaming dialogue with proactive response generation + timing
- not locatable on arXiv as of 2026-07-12; revisit via its NeurIPS 2025 page if needed

---
zotero_key: null
authors: Guowei Tang, Tianwen Qian et al. (East China Normal University)
year: 2026
arxiv: 2603.21493
pdf: https://arxiv.org/pdf/2603.21493
tier: deep
subtopics: [streaming-benchmarks]
tags: [streaming-video-understanding, streaming-benchmarks]
---
# StreamingEval: A Unified Evaluation Protocol towards Realistic Streaming Video Understanding

**Lineage role:** the 2026 eval-validity correction — replaces offline "pseudo-streaming" scoring with a wall-clock, model-owned-timing protocol that runs online and offline models under one byte-level resource budget and one composite efficiency-accuracy score.

## Problem — what was limited before this paper (short)
Prior streaming-video benchmarks ([[streamingbench]], [[ovo-bench]], and others) measure one axis at a time — usually QA accuracy under a truncated visual context — while the model still runs **offline**: the video is cut at the query timestamp and then processed at leisure with unlimited compute and memory. This "pseudo-streaming" setup ignores the constraints that actually decide deployability (can the model keep up with the frame rate, how long until it answers, how big is the state it must hold). There was no shared harness under which a natively-online model and a strong offline VideoLLM could be compared **fairly and reproducibly** on the same footing, so accuracy tables did not reflect real streaming behavior.

## Key idea — the core insight, 2-4 sentences
Make timing and memory **first-class, model-owned** quantities rather than free background assumptions. StreamingEval runs every model through one asynchronous three-stage pipeline — a Frame Player that emits frames at a fixed rate, a Memory Updater that maintains a **fixed-capacity** visual cache with strict time-causality, and a Responder whose latency is measured on the wall clock — and forces online and offline models to share an **identical byte-level memory budget**. It then folds sustainable frame rate, accuracy, latency, and memory into a single tunable **StreamingScore**, exposing the accuracy-vs-deployability trade-off that per-axis benchmarks hide.

![[streamingeval.png]]
> **Crux (Figure 2).** The unified StreamingEval pipeline: (top) Frame Player streams frames into a buffer at a set rate; (middle) Visual Encoder + Memory Updater enqueue each frame's tokens into a fixed-capacity FIFO memory bank; (bottom) on a query the Responder prefills the LLM over `{query, memory@t}`, and TTFT / Acc are read off the generated answer. This is the core because it defines the *causal, resource-bounded* substrate under which online and offline models are made comparable. *Tang, Qian et al. (2026), arXiv:2603.21493. Embedded for personal research reference.*

## Method + math — the protocol as the mechanism, then the objective in full

**Three decoupled asynchronous processes.**
1. **Frame Player** — samples raw frames from the source video and pushes them into a frames buffer at a **fixed interval** (1 FPS in the main experiments), popping the current frame $t_n$ to the encoder. This decouples the video's arrival clock from the model's processing clock.
2. **Encoder & Memory Updater** — each incoming frame $v_i$ is mapped to a visual representation
$$z_i = g_\theta(v_i),$$
where $g_\theta$ is the visual backbone. An online memory state $M$ is then updated by the **model-specific** write/evict rule $\mathcal U$ under a capacity constraint $B$:
$$M_{\tau_i^+} = \mathcal U\!\left(M_{\tau_i^-},\, z_i;\, B, \pi\right),$$
with $\tau_i^-,\tau_i^+$ the memory states just before/after frame $i$ arrives at time $\tau_i$, and $\pi$ the eviction policy (FIFO in the standardized offline path). For offline models, a projection layer maps features into the LLM embedding space and a **fixed-length visual context window with FIFO eviction** emulates a bounded memory so they too obey the budget.
3. **Responder** — a user query $q_{t_0}$ can arrive at any time $t_0$. The responder first encodes the query, finishing at time $t_1$ (so $t_1 > t_0$); it may only read the memory snapshot $M_{t_1}$ available **up to $t_1$** — this is the strict time-causality / **model-owned timing**: what the model can see is dictated by how fast *it* processed frames, not by an oracle cut at the query time. It then generates autoregressively:
$$R_{t_1} \sim p_\phi\!\left(\cdot \mid q_{t_0}, C_{t_1}, M_{t_1}\right),$$
where $C_{t_1}$ is the interaction/dialogue history up to $t_1$ and $p_\phi$ is the LLM backbone.

**"Two settings, one unified budget."** A fair online-vs-offline comparison is hard because online models run causally while offline models assume the full video, and because visual-token embeddings differ in dimensionality across models — so bounding history by *token count* gives inconsistent real footprints. StreamingEval instead bounds the **byte-level** memory. The per-model memory footprint for a budget of $B$ retained frames is (Appendix A.1):
$$\mathrm{Mem}_i(B) = B\, d_i\, s_{\text{emb}} \;+\; B\, 2 L_i\, h_i^{kv}\, s_{kv},$$
where $d_i$ is embedding dimension, $s_{\text{emb}}$ bytes per element, $L_i$ the number of LLM layers, $h_i^{kv}$ the KV channel width, and $s_{kv}$ bytes per KV element. The first term is the stored visual embeddings; the second is the KV cache.

**Core metrics (all measured, not judged).**
- **MaxFPS** — the maximum input frame rate the model can sustain **without a persistent backlog** in the frame buffer (the model's own throughput ceiling).
- **TTFT** — time-to-first-token, wall-clock elapsed from query arrival to the first generated token.
- **Memory_bank (Mem, GB)** — the enforced budget of online-available visual history cache.
- **Acc** — multiple-choice QA accuracy, scored directly against ground-truth labels following each benchmark's official definition (no LLM judge).

**The composite objective — StreamingScore (Eq. 4):**
$$\mathrm{StreamingScore}(w) \;\triangleq\; \frac{\mathrm{MaxFPS}^{\,w_f}\cdot \mathrm{Acc}^{\,w_a}}{\mathrm{TTFT}^{\,w_t}\cdot M^{\,w_r}}, \qquad M \triangleq \mathrm{Mem}\cdot \ln(\mathrm{Params}),$$
with weights constrained by $w_f + w_a + w_t + w_r = 1$ and a default of $w_f=w_a=w_t=w_r=0.25$. Higher throughput and accuracy push the score up; higher latency and (memory-scaled) model size push it down. The weights are tunable so a deployment can express its own priority (e.g. latency-critical edge vs accuracy-critical server).

**Substrates.** No new dataset is introduced; the protocol is applied over **OVO-Bench** (12 tasks, ~2,800 annotations, clustered as Backward Tracing / Real-Time Visual Perception / Forward Active Responding) and **StreamingBench** (18 tasks, 900 videos, 4,500 QA). All models run at 1 FPS, BF16, on a single RTX 4090-class GPU with unified prompting and decoding.

## Explicit design choices
- **Asynchronous decoupling** of frame arrival (Frame Player) from processing (Encoder/Memory) from answering (Responder) — the only way to measure real backlog and latency instead of assuming instant processing.
- **Fixed-capacity FIFO memory bank** for *all* models, including offline ones (via a projection + bounded visual context window), so nobody gets unlimited history.
- **Byte-level (GB) budget, not token-count budget** — normalizes across models whose token embeddings differ in width; budget default 0.5 GB, swept over {0.1, 0.3, 0.5, 1.0, 1.5} GB.
- **Strict time-causality:** query encoding finish time $t_1 > t_0$; model may only read $M_{t_1}$ — encoding cost eats into what can be seen (model-owned timing).
- **Wall-clock TTFT and backlog-based MaxFPS** — throughput and latency are the model's own emergent numbers, not harness-imposed.
- **Single tunable composite score** (StreamingScore) with size penalty $M=\mathrm{Mem}\cdot\ln(\mathrm{Params})$, letting one number rank deployability while remaining reweightable.
- **Direct-label QA scoring** (multiple-choice, official per-benchmark), avoiding LLM-judge variance; an open-ended E2E-latency setting is kept *separate* and not mixed into the MC ranking.
- **1 FPS, BF16, single consumer GPU (RTX 4090-class)** — an edge/mobile-realistic budget, deliberately capping models at 7B/8B scale.

## Key results / what to remember
Setup: 12 models, 0.5 GB memory budget, 1 FPS, RTX 4090-class GPU. Numbers below are read directly from **Table 1** (OVO-Bench per-cluster + Overall + its StreamingScore; StreamingBench Real-time + its StreamingScore).

**Offline VideoLLMs (OVO Overall / OVO StreamingScore | StreamingBench / SB StreamingScore, MaxFPS, TTFT):**
- Qwen3-VL-8B — OVO **58.00** (RT 78.73 / Back 51.82 / Fwd 43.46), OVO SS **2.21**; SB **77.31**, SS **2.37**; MaxFPS 8, TTFT 0.20 — top overall.
- MiniCPM-V4.5-8B — OVO 57.04 (Back best 54.52), SS 2.07; SB 76.55, SS 2.23; MaxFPS 6, TTFT 0.18.
- Llava-OV1.5-8B — OVO 55.58, SS 2.05; SB 76.19, SS 2.22; MaxFPS 7, TTFT 0.21.
- InternVL3.5-8B — OVO 53.78, SS 2.04; SB **77.96** (highest SB accuracy), SS 2.24; MaxFPS 7, TTFT 0.21.
- VideoChat-7B — OVO 49.95, SS 1.73; SB 72.22, SS 1.89; MaxFPS 9 (highest throughput).
- VideoLLaMA3-7B — OVO 48.44, SS 1.58; SB 68.90, SS 1.72.

**Online Video-LLMs:**
- StreamForest-7B — OVO **55.57** (Fwd best 53.49), SS 1.26; SB **77.26**, SS 1.37; MaxFPS 4, TTFT 0.98 — best native-online accuracy, near-offline.
- Flash-VStream-7B — OVO 33.15, SS **2.34** (highest OVO StreamingScore); SB 23.23, SS 2.18; MaxFPS 8, TTFT **0.12** — wins on speed, loses badly on accuracy.
- Flash-VStream-7B* (improved variant) — OVO 50.31, SS 0.74; SB 74.48, SS 0.81; MaxFPS 1, TTFT 1.31 — accuracy jumps but StreamingScore collapses (slow, heavier).
- TimeChat-Online-7B — OVO 45.67, SS 1.67; SB 73.64, SS 1.88; MaxFPS 7, TTFT 0.62.
- ReKV-7B — OVO 48.00, SS 1.18; SB 64.53, SS 1.27; MaxFPS 5, TTFT 1.29.
- VideoChatOnline-4B — OVO 40.40, SS 0.92; SB 58.81, SS 1.19; **MaxFPS 0.14** (cannot keep up with 1 FPS — deployment-blocking).

**Takeaways.**
- No Zotero highlights present.
- Under one budget, **offline VideoLLMs win aggregate accuracy** (stronger on Backward Tracing / long-horizon and Real-Time Perception), because full-context encoding better supports retrospective aggregation.
- **Online models are on par or better only on Forward Active Responding** (rapid decision-making), and some win on **StreamingScore** by answering faster and pacing more stably under causal constraints — the accuracy leader is *not* the deployability leader.
- The Flash-VStream vs Flash-VStream* pair is the headline lesson: pushing accuracy (33→50 OVO) can crater StreamingScore (2.34→0.74) when it costs throughput and latency — exactly the trade-off per-axis benchmarks miss.
- VideoChatOnline-4B's MaxFPS 0.14 shows an "online" model that literally cannot sustain 1 FPS — invisible in accuracy-only tables, glaring here.
- Accuracy nearly **saturates in the 1.0–1.5 GB memory regime**; the interesting differentiation is at tighter budgets (Figure 3).

## How it connects (evolution)
- [[streamingbench]] — one of the two evaluation substrates; StreamingEval re-scores it under wall-clock/byte-budget constraints instead of its native offline setting.
- [[ovo-bench]] — the second substrate (Backward / Real-Time / Forward clusters are reused as StreamingEval's task axes).
- [[streaming-benchmarks]] — the sub-topic hub this note anchors; StreamingEval is the eval-validity correction to the whole benchmark line.
- [[timechat-online]] — evaluated model (TimeChat-Online-7B); its token-dropping design is exactly what a byte-budget protocol stress-tests.
- [[flash-vstream]] — evaluated model whose fast-but-lossy memory makes it the throughput/accuracy trade-off exemplar here.
- [[rekv]] — evaluated online model (ReKV-7B), a retrieval-KV approach measured under the unified budget.
- [[streamingbench]] and [[proactivevideoqa]] — sibling benchmarks in the same validity conversation about realistic streaming evaluation.

## Open questions / limitations
- **Scale-capped:** experiments stay at 7B/8B on a single consumer GPU (edge focus), so conclusions about larger-model efficiency/accuracy trade-offs are untested.
- **Closed-source excluded:** proprietary models lack reproducible inference interfaces, so the protocol cannot standardize them — the ranking is open-source-only.
- **Metric-gaming risk:** the authors flag that a single composite StreamingScore could be over-optimized (or over-generalized as a universal standard) in ways that don't reflect real deployment; weights are a knob, not a settled truth.
- **Setting fragmentation:** end-to-end latency is measured in an open-ended generation setting distinct from the multiple-choice accuracy setup, so those two numbers aren't directly comparable.

*Verification: equations (1)-(4) and the memory formula transcribed from the rendered method page (PDF p.4) and Appendix A.1; all headline numbers (MaxFPS, Mem, TTFT, OVO clusters/Overall/StreamingScore, StreamingBench/StreamingScore) read directly off the rendered Table 1 (PDF p.6); cross-checked against the arXiv HTML full text. Code: github.com/wwgTang-111/StreamingEval1.*

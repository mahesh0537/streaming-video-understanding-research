---
zotero_key: null
authors: Zikang Liu et al. (CASIA / UCAS School of AI / OPPO AI Center)
year: 2026
arxiv: 2603.12938
pdf: https://arxiv.org/pdf/2603.12938
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# Thinking in Streaming Video

**Lineage role:** Brings explicit chain-of-thought *reasoning* into the proactive-response loop — the model thinks incrementally on every incoming chunk, and a timing-aware RL reward decides not just *what* to say but *when* to speak vs. stay silent, with reasoning traces doubling as compressed streaming memory.

## Problem — what was limited before this paper (short)
Streaming video assistants must respect strict causality (only past frames visible), keep computation/memory bounded over an unbounded stream, and interact at the right moment. Prior streaming VLMs mostly perceive-then-answer: they defer any deliberation until a query arrives, so they cannot reason as evidence accumulates, and proactive systems that *do* decide when to speak typically emit answers with little intermediate reasoning. Batch reasoning models are accurate but need the whole clip and blow the latency budget for real-time (e.g. 2 FPS) use.

## Key idea — the core insight, 2-4 sentences
Reformulate streaming understanding as a **Watch–Think–Speak** loop: on each new video chunk the model writes a short reasoning update inside `<think>…</think>` tags and then chooses an action `a_t` — emit a `<response>` or stay `<silent>`. Crucially the accumulated `<think>` traces are *not thrown away*: they act as **Reasoning-Compressed Streaming Memory (RCSM)**, compact semantic anchors that replace evicted raw visual tokens so context stays bounded. The whole thing is tuned with **streaming RL from verifiable rewards** whose reward includes an explicit response-timing term, teaching the model both correctness and *when* to speak.

![[thinkstream.png]]

> **Crux (Fig. 2).** The ThinkStream framework: (a) streaming Watch–Think–Speak rollouts trained by RLVR with reward $\mathcal{R}=\mathcal{R}_{format}+\mathcal{R}_{time}+\mathcal{R}_{acc}$; (b) Reasoning-Compressed Streaming Memory evicts stale dense video tokens but keeps compressed thinking/response tokens as long-term anchors; (c) a CUDA-Graph streaming backend with eager prefill + replayable decode/evict kernels. This one diagram is the paper — it fuses incremental reasoning, memory, and timing-aware RL. *Liu et al. (2026), arXiv:2603.12938. Embedded for personal research reference.*

## Method + math — the mechanism, then the central objective/equations in full

**Watch–Think–Speak output structure.** Video arrives as a stream of chunks. At streaming step $t$ (after observing new segment $v_t$) the model autoregressively produces a structured token block

$$\langle\text{think}\rangle\, r_t \,\langle/\text{think}\rangle\, a_t$$

where $r_t$ is a short time-grounded reasoning update and $a_t\in\{\langle\text{response}\rangle\,\text{text},\ \langle\text{silent}\rangle\}$. Reasoning is a lightweight *continuous* process: each chunk triggers a brief update that (1) summarizes newly observed events, (2) updates hypotheses about ongoing activity, or (3) refines earlier interpretation, and the model only acts when accumulated understanding is reliable enough.

**Reasoning-Compressed Streaming Memory (RCSM).** To stay bounded on an unbounded stream, the KV cache at step $t$ keeps only a sliding window of $W$ steps of *dense visual* KV but retains **all** reasoning+response KV from the start:

$$\mathcal{M}_t=\operatorname{Concat}\Big(\big\{KV(v_\tau)\big\}_{\tau=\max(1,\,t-W+1)}^{t},\ \big\{KV(r_\tau\oplus a_\tau)\big\}_{\tau=1}^{t}\Big)$$

Outdated dense video tokens are evicted from the cache; the highly compressed thinking/response tokens survive as long-term semantic anchors. This is the key bet: a few reasoning tokens summarize a segment far better than the raw visual tokens they replace (validated by the memory-ablation, Table 6). A causal attention mask lets current queries attend to the recent visual window plus the full reasoning history.

**Streaming RLVR objective.** Post-training uses **GRPO** (Group Relative Policy Optimization) with KL regularization over $G$ streaming rollouts. The verifiable reward has three additive parts:

$$\mathcal{R}=\mathcal{R}_{format}+\mathcal{R}_{time}+\mathcal{R}_{acc}$$

- $\mathcal{R}_{format}$: structured-output compliance (well-formed `<think>`/`<response>`/`<silent>` blocks).
- $\mathcal{R}_{time}$: timing reward penalizing premature or late responses,
$$\mathcal{R}_{time}=\max\!\Big(0,\ 1-\frac{|t_{resp}-t_{gt}|}{w}\Big)$$
i.e. a linear tolerance window $w$ around the ground-truth response time $t_{gt}$ — this is what makes proactivity *learnable* rather than heuristic.
- $\mathcal{R}_{acc}$: verifiable correctness against ground truth, restricted to deterministic answer formats (multiple-choice, binary yes/no, counting) so it can be auto-checked with no LLM judge.

**Data pipeline (the "ThinkStream Dataset").** ~110K cold-start SFT instances + ~9K RLVR instances, built in three stages: (1) segment videos with PySceneDetect and dense-caption each segment with Qwen3-VL-235B; (2) synthesize diverse instructions as a Cartesian product over interaction modes (real-time dialogue / event trigger / continuous output) × temporal scope (past / current / future) × 7 content semantic dimensions; (3) generate strictly causal, timestamp-synchronized time-grounded CoT traces.

**Inference backend (Algorithm 1).** A custom CUDA-Graph streaming engine: an **eager prefill** phase handles variable-length (ragged) visual tokens, then a **decode-and-prune** phase uses replayable CUDA graphs for both the decode kernel and the KV-eviction ("evict") kernel, doing in-place memory shifting each chunk. Uses FlexAttention for RCSM masking, FlashInfer for sampling, Liger Kernel for training.

## Explicit design choices — concrete decisions
- **Base model:** Qwen2.5-VL-3B (a 7B variant also evaluated); trained on 8× NVIDIA H20.
- **Output contract:** every step is `<think> r_t </think> a_t`; the *action head* is just a special token (`<response>` vs `<silent>`) — proactivity is folded into generation, not a separate classifier.
- **Memory = reasoning, not captions:** discard raw visual KV beyond window $W$; keep all reasoning/response KV. Ablation shows discrete-caption memory *hurts* (48.7) vs. cold-start CoT (60.5) vs. RLVR CoT (64.8).
- **Reasoning token budget capped ≈ 20 tokens / video-second** — sweet spot of accuracy vs. latency.
- **Visual KV window $W\approx$ 20 seconds** — best accuracy/efficiency trade-off.
- **RL = GRPO + KL**, group size 8; reward is additive format+time+accuracy; only deterministic answer types so rewards are verifiable without a judge model.
- **Cold-start SFT then RLVR:** SFT on 110K time-grounded CoT instances, then 9K RLVR.
- **Timing supervised explicitly** via $\mathcal{R}_{time}$ with a linear tolerance window $w$.
- **Systems co-design:** CUDA-Graph decode+evict kernels; eager prefill for ragged visual input; in-place KV shift per chunk.
- **Training hyperparams:** cold-start batch 64, LR 1e-5; RLVR batch 8, LR 2e-7, AdamW.

## Key results / what to remember — exact headline numbers with settings
No Zotero highlights present.

- **OVO-Bench (Table 1), overall avg:** ThinkStream-3B **59.66** vs. base Qwen2.5-VL-3B **51.00**; beats online baselines Streamo-3B (51.64), StreamForest-7B (56.61), Dispider-7B (45.31), Flash-VStream-7B (27.88). Sub-scores: Real-Time Visual Perception avg **67.03**, Backward Tracing avg **52.30**.
- **StreamingBench Real-Time (Table 2), avg:** ThinkStream-3B **75.00** — exceeds GPT-4o (73.28) and Dispider-7B (67.63), just under Gemini 1.5 pro (75.69) and Human (91.46).
- **Offline benchmarks (Table 3):** OVO Real-Time **67.0** (↑7.0 over base), OVO Backward **52.3** (↑10.4), VideoMME **61.9** (↑0.8, base 61.5), LongVideoBench **56.4** (↑1.8, base 54.2), overall avg **59.4** (↑5.0 over base 54.4) — offline ability preserved despite aggressive token eviction.
- **Memory-representation ablation (Table 6), avg:** no memory 56.9 → discrete captions 48.7 → cold-start CoT 60.5 → **RLVR-optimized CoT 64.8** (reasoning tokens beat naive captions by a wide margin; +4.3 from RLVR).
- **Reasoning-budget ablation (Table 4):** 0 tok/s → StreamingBench 69.6 / OVO-BW 41.8 / 130 ms; **20 tok/s → 75.0 / 52.3 / 380 ms** (chosen); 30 tok/s → 75.0 / 52.6 / 505 ms (marginal, +125 ms).
- **KV-window ablation (Table 5):** 5s→46.7, 10s→50.1, **20s→52.3** (OVO-BW; also best StreamingBench 75.0), 30s→51.6.
- **Efficiency (Fig. 3):** custom engine **154.07 tok/s** vs. baseline **30.06 tok/s** at batch 1 (>5× speedup); Fig. 4: latency stays <0.5s per video-second (baseline 1.0–1.4s) — meets 2 FPS real-time budget.

Takeaways: (1) intermediate reasoning traces are a *better* streaming memory than raw visual tokens or captions; (2) response timing can be trained directly as a verifiable RL reward; (3) a 3B model with this recipe matches/beats much larger and proprietary streaming baselines while staying real-time.

## How it connects (evolution)
- [[streammind]] — proactive "think before responding" streaming assistant; ThinkStream makes the thinking an explicit CoT + trains timing with RL.
- [[dispider]] — disentangles perception/decision/reaction for proactive response; a direct online-MLLM baseline ThinkStream compares against.
- [[mmduet]] — dense per-frame response-timing for proactive dialogue; ThinkStream's $\mathcal{R}_{time}$ is a learned counterpart to its timing heuristics.
- [[videollm-online]] — the streaming-dialogue "when to speak" formulation ThinkStream inherits and augments with reasoning.
- [[proactive-response]] — the sub-topic hub this note belongs to.
- [[streaming-memory]] — RCSM (reasoning-as-memory) is a distinctive point in the streaming-memory design space vs. KV-eviction / caption memory.

## Open questions / limitations
- $\mathcal{R}_{acc}$ is restricted to deterministic answer types (MCQ/binary/counting); open-ended proactive dialogue timing/quality is harder to verify and not directly rewarded.
- RCSM assumes a few reasoning tokens faithfully summarize an evicted segment — fine for the benchmarks, but long-horizon fine detail (rare objects, exact counts far back) may be irretrievable once visual KV is gone.
- Fixed 20 tok/s reasoning budget and 20s visual window are tuned globally; adaptive budgets for bursty vs. idle streams are unexplored.
- Timing evaluated against single ground-truth response times; robustness when multiple valid response moments exist is unclear.

*Verification: equations ($\mathcal{R}$, $\mathcal{R}_{time}$, $\mathcal{M}_t$) transcribed from the rendered Fig. 2 panel and the method text; all numbers cross-checked against the rendered pages of Tables 1–6 and Fig. 3/4 in the arXiv PDF (2603.12938v1). Author list/affiliations and dataset sizes from the arXiv HTML/abstract.*

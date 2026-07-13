---
zotero_key: null
authors: Jialiang Zhang, Junlong Tong, Junyan Lin et al. (Institute of Digital Twin, Eastern Institute of Technology, Ningbo; SJTU; LMU Munich)
year: 2026
arxiv: 2603.02872
pdf: https://arxiv.org/pdf/2603.02872
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# Think-as-You-See: Streaming Chain-of-Thought Reasoning for Large Vision-Language Models

**Lineage role:** couples KV-cache management to reasoning — a *dual, modality-split* KV cache (video vs. reasoning) that lets a video LVLM decode chain-of-thought tokens concurrently with frame ingestion instead of waiting for the whole clip.

## Problem — what was limited before this paper (short)
Almost all video-reasoning LVLMs run a **batch ("wait-and-see") paradigm**: encode the entire clip, then reason. That inflates latency (Time-to-First-Token grows with clip length) and causes **temporal drift** — by the time the model reasons, early cues are stale or hallucinated. The obvious fix, **naive interleaving** (frame → reason → frame → reason in one causal stream), is serial: because visual and text tokens share one monolithic cache, new frames cannot be encoded while reasoning tokens are still being generated (and vice-versa), so perception blocks on reasoning. This entangled ordering also drifts from the pretraining distribution of LVLMs, which factorize visual encoding and text decoding.

## Key idea — the core insight, 2-4 sentences
Turn reasoning into a **continuous process synchronized with the stream** rather than a terminal step. TaYS keeps perception and reasoning *decoupled but temporally causal* via three mechanisms: a **streaming attention mask** (each reasoning step sees only frames up to its time, via a sliding window), a **decoupled RoPE positional encoding** (vision and reasoning get independent position axes so relative distances stay stable as the visual sequence grows), and a **parallel dual KV-cache** — a read-heavy video cache and a dynamic reasoning cache — fused at decode time by pointer-level (zero-copy) *merge*, then *split* again. The result: frames keep flowing into the video cache while the reasoning cache autoregressively decodes, so the reasoning path is never stalled by encoding.

![[tays.png]]
> **Crux (Figure 3).** The streaming reasoning framework: (a) parallel video + reasoning KV caches joined by a recursive *merge & split* loop so visual encoding and CoT decoding run concurrently; (b) the streaming attention mask (sliding window over frames per reasoning step) used during training; (c) inference information flow, showing TaYS's parallel paths avoid the sequential "flow blocking" of the interleaved paradigm. *Zhang et al. (2026), arXiv:2603.02872. Embedded for personal research reference.*

## Method + math — the mechanism, then the central objective

**Task formalization.** A stream is frames $V=\{F_t\mid 1\le t\le T\}$ with accumulated multimodal context $C_{<t}$. Offline CoT assumes global access and only decodes at $t=T$:
$$h_i=\text{Decoder}\big(y_{<i};\ \text{Enc}(V)\big),\qquad \hat y_i\sim P_\theta(y_i\mid V,y_{<i}).$$
Streaming CoT instead sees only the partial prefix $V_{\le t}=\{F_1,\dots,F_t\}$ and decodes as frames arrive:
$$h_i^t=\text{Decoder}\big(y_{<i}^t;\ \text{Enc}(V_{\le t}),\ C_{<t}\big),\qquad \hat y_i^t\sim P_\theta\!\big(y_i^t\mid V_{\le t},\,y_{<i}^t,\,C_{<t}\big).$$
The training objective maximizes the **cumulative** likelihood up to each time $t$ (future frames $\{F_{t+1},\dots,F_T\}$ are strictly masked out), i.e. the central equation:
$$\max_\theta\ P_\theta(Y_{\le t}\mid V_{\le t})=\prod_{i=1}^{N_t} P_\theta\!\big(y_i^t\mid V_{\le t},\,y_{<i}^t,\,C_{<t}\big).$$
Offline CoT is the degenerate case where all reasoning is deferred to $t=T$.

**Streaming attention mask.** For visual length $N_v$ and reasoning length $N_r$, query position $i$, key position $j$:
$$\widetilde M(i,j)=\begin{cases}-\infty, & i>N_v,\ j<N_v,\ j> i-N_v,\\[2pt] M_{\text{causal}}(i,j), & \text{otherwise},\end{cases}$$
where $M_{\text{causal}}$ is the standard autoregressive mask. The condition $j> i-N_v$ makes a **sliding window over visual tokens** relative to the current reasoning step, so a reasoning token integrates only frames within its temporal window — no leakage from future frames.

**Decoupled positional encoding.** Under monolithic RoPE indexing, a reasoning token $r_t$ is offset by the full visual length: $(R_{N_v+t}q_{r_t})^\top(R_s k_{v_s})=q_{r_t}^\top R_{(N_v+t)-s}^\top k_{v_s}$. As $N_v$ grows continuously in a stream, this shifts relative positions and destabilizes temporal perception. TaYS assigns **independent axes** $\text{pos}(v_s)=s$, $\text{pos}(r_t)=t$, so the interaction becomes
$$(R_t q_{r_t})^\top(R_s k_{v_s})=q_{r_t}^\top R_{t-s}^\top k_{v_s},$$
keeping the relative temporal distance $(t-s)$ semantically stable regardless of sequence growth.

**Parallel dual KV cache (the merge–generate–split loop).** Two modality-specific caches: read-heavy video cache $C_v$, dynamic text/reasoning cache $C_r$. Each arriving frame is appended non-blockingly: $C_v^{(t)}=C_v^{(t-1)}\cup \text{Enc}(F_t)$. At decode, attention runs over a **logical concatenation** of $C_v^{(t)}$ and $C_r^{(t-1)}$ implemented by **pointer-level composition (zero-copy)**. After the reasoning segment $R_t$ is generated, only the text cache updates: $C_r^{(t)}=C_r^{(t-1)}\cup \text{Dec}(R_t)$; the video cache stays immutable during that step, then a *split* restores the modality-specific views. Because $C_r$ decodes while new frames flow into $C_v$ independently, reasoning is never stalled by visual encoding.

**Streaming CoT data construction (2-step, on VideoEspresso train split).** (1) *Frame-ID alignment*: resample every video to **2 FPS**; for each target timestamp $\tau'_{t'}=0.5(t'-1)$ s, pick the annotated **keyframe** if the timestamp falls in its labeled interval, else the nearest frame:
$$F_{t'}=\begin{cases}F_k, & \tau'_{t'}\in[\tau_k^{\text{start}},\tau_k^{\text{end}}]\ \&\ F_k\ \text{keyframe},\\[2pt] \arg\min_{F_t}\lvert \tau_t-0.5(t'-1)\rvert, & \text{otherwise},\end{cases}$$
preserving annotated moments while keeping temporal regularity. (2) *Trajectory construction + QA*: prompt **GPT-4o** to emit per-keyframe triplets $(Q_t,R_t,A_t)$ (temporally grounded question, reasoning step, answer). *Quality control* keeps only samples whose question/reasoning cosine similarity is high, using **BGE-M3** embeddings:
$$\text{consistency}(Q_t,R_t)=\frac{v_Q\cdot v_R}{\lVert v_Q\rVert\,\lVert v_R\rVert}.$$
Sentence-level **`<EOT>`** boundary tokens delimit minimal reasoning units so the model emits causally ordered, frame-consistent output.

## Explicit design choices
- **SFT on Qwen2.5-VL-3B/7B-Instruct** — TaYS is a fine-tuning framework, not a new backbone.
- **Two physically separate KV caches** (video vs. reasoning) instead of one monolithic cache — this is the concrete lever that couples KV management to reasoning and unblocks parallelism.
- **Pointer/zero-copy merge** at decode (logical concat), immutable video cache during a reasoning step, then split — avoids tensor copy overhead on the critical path.
- **Sliding-window streaming mask** ($j>i-N_v$) enforces per-step causality during training (not just inference).
- **Modality-decoupled RoPE** ($\text{pos}(v_s)=s$, $\text{pos}(r_t)=t$) to fight index drift as $N_v$ grows.
- **Timestamp-based resampling to 2 FPS with keyframe pinning** (not uniform sampling) to align frames with annotated reasoning units; clips truncated to model max length (≤30 s).
- **GPT-4o for QRA triplet generation, BGE-M3 for consistency filtering, `<EOT>` unit delimiters** — the streaming-CoT data recipe.
- **Baselines isolate the design**: Batch w/o-thinking, Batch w/-thinking, Batch SFT, and **Interleaved SFT** (streaming but single monolithic cache, no parallel caching) — so gains attribute to *parallel* streaming.
- **Evaluation on an *extended* VideoEspresso protocol** with 14 task columns (Narrative, Event, Ingredient, Causal, Theme, Contextual, Influence, Role, Interaction, Behavior, Emotion, Cooking, Traffic, Situational).

## Key results / what to remember
Numbers verified against Table 1 (avg accuracy, extended VideoEspresso) and Table 2 (latency @ FPS).

- **Objective accuracy (avg, ↑).** 3B: Batch w/o-thinking **27.99**, Batch w/-thinking **28.16**, Batch SFT **29.18**, Interleaved SFT **33.96**, **TaYS 33.45**. 7B: Batch w/o-thinking **28.89**, Batch w/-thinking **31.57**, Batch SFT **30.38**, Interleaved SFT **34.98**, **TaYS 36.86**. Abstract headline: **+2.9% over batch CoT baselines**. Honest caveat: on the *objective* metric the **Interleaved** model is essentially tied with (3B: 33.96 vs 33.45) or below (7B) TaYS — the paper itself concedes objective metrics don't separate the two streaming paradigms, and leans on the subjective eval for TaYS's edge.
- **Subjective (GPT-5 win rate, ↑).** TaYS **43.7%** vs Batch **31.4%** vs Interleaved **21.7%**; TaYS wins **61.1%** of Cooking-Process samples (vs 11.1% Interleaved) and **75.0%** of Preparation-Steps.
- **Latency (Table 2, FPS=2).** TTFT: Batch **10.48s**, Interleaved **0.0295s**, **TaYS 9.2e-7s** (~near-zero, warm-start). End-to-end delay: Batch **13.90s**, Interleaved **14.19s**, **TaYS 12.19s**; TaYS delay stays ~**12s stable across FPS 1–5** while Interleaved delay climbs to **20.13s** at FPS=5. TaYS accuracy peaks **36.01** at FPS=3. Abstract phrasing: TTFT "**10.6s → near-zero**."
- **Temporal grounding.** Mean reasoning-to-keyframe deviation $\Delta t$: **TaYS 0.69s vs Interleaved 1.52s** (≈**55% reduction**); **86.0%** of TaYS reasoning within 1s of a keyframe vs **62.4%** Interleaved. TaYS also shows smoother consecutive-step similarity (less redundant/looping reasoning).

No Zotero highlights present.

Takeaways: the transferable idea is **splitting the KV cache by modality and fusing by zero-copy pointer merge** so a streaming LVLM decodes CoT without pausing frame ingestion; the accompanying **sliding-window mask + decoupled RoPE** are the training-side pieces that make partial-prefix reasoning stable. The win is mostly **responsiveness + temporal alignment**, not raw accuracy over interleaving.

## How it connects (evolution)
- [[streamingcot]] — the closest sibling: streaming chain-of-thought reasoning; TaYS is the parallel-cache instantiation of that goal.
- [[thinkstream]] — think-while-streaming line; shares the "reason as frames arrive" premise.
- [[dispider]] — disentangling perception from decision/reaction in streaming; TaYS's perception/reasoning decoupling is the KV-cache analogue.
- [[decouple-and-cache]] — decoupling + caching for streaming; directly related to the dual-cache mechanism.
- [[streamingvlm]] — streaming VLM with explicit KV-cache management under continuous input.
- [[videollm-online]] — foundational online/streaming video-LLM baseline that TaYS's paradigm shift is arguing past.

## Open questions / limitations
- **Objective accuracy doesn't beat interleaving** (3B tie / marginal), so the paradigm's value rests heavily on subjective GPT-5 judging and latency — vulnerable to judge bias; no human study reported.
- **Narrow benchmark**: only extended VideoEspresso, clips ≤30 s, 2 FPS — unclear how the dual cache scales to genuinely long/unbounded streams where the *video* cache itself would need eviction/compression (TaYS deliberately does **not** compress temporal structure).
- **Data pipeline leans on GPT-4o + BGE-M3 filtering** to synthesize streaming-CoT trajectories; quality and bias of that supervision are unquantified.
- **"Near-zero" TTFT is a warm-start, decoder-level number** (~1e-6s) that excludes per-frame visual encoding cost; the honest system latency is the ~12s end-to-end delay.

*Verification: equations (Eqs. for task def, streaming mask, decoupled RoPE, dual-cache updates, frame-ID alignment, BGE-M3 consistency) transcribed from the arXiv HTML + PDF method (pages 3–6); all accuracy/latency/temporal numbers checked against the paper's own Table 1 and Table 2 and the §4.2–4.4 text. Crux figure = Figure 3, page 5 of the PDF.*

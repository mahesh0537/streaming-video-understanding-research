---
zotero_key: null
authors: Haolin Yang, Feilong Tang et al. (MBZUAI; Shanghai AI Laboratory; DeepGlint; SJTU; Monash)
year: 2025
arxiv: 2508.01875
pdf: https://arxiv.org/pdf/2508.01875
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# StreamAgent: Towards Anticipatory Agents for Streaming Video Understanding

**Lineage role:** Recasts streaming VideoQA as an *anticipatory agent* that predicts **when/where** future task-relevant evidence will appear (defer-and-hunt via three temporal horizons + tools), backed by a hierarchical streaming KV-cache memory for cheap selective recall — cross-link [[streaming-memory]].

## Problem — what was limited before this paper (short)
Streaming VideoQA needs continuous perception plus a *timing* decision: for a question posed at time $t$, is the evidence seen so far enough to answer, or should the model wait for future frames? Prior online video LLMs handle this reactively. Single-LLM alternating designs (e.g. [[videollm-online]]) fold perception and generation into one autoregressive pass and are slow; asynchronous binary-trigger designs (e.g. [[dispider]]) separate "decide" from "respond" but only fire on the *current* state — they lack task-driven planning and future anticipation, so they respond prematurely with incomplete evidence (answer "1 goal" right after the first goal in a match). The missing capability is **anticipation**: proactively predicting and actively acquiring the future information a query actually requires.

## Key idea — the core insight
StreamAgent frames streaming understanding as a planning agent that, at each timestep, forecasts future task-relevant events across three temporal horizons — **Reactive** ($h{=}0$, answer now), **Proactive** ($h{=}\delta$, near-future), **Speculative** ($h{=}\Delta$, long-range) — generates candidate plans, and scores them with an LLM-as-a-judge on *sufficiency of current evidence* vs *value of future observation*. The winning plan either triggers an asynchronous response or launches goal-directed **information hunting**: invoking tools (e.g. zoom/crop into a predicted bounding box, or persistently track a region across incoming frames) to actively gather the missing evidence. A hierarchical **streaming KV-cache** (long-term on CPU, layer-adaptive selective recall to GPU) preserves original visual tokens so answers can be generated cheaply without re-encoding the whole stream.

![[streamagent.png]]
> **Crux (Figure 2).** The full StreamAgent loop: incremental **Memory** update (captioner compresses clip $v_t$ into $m_t$) → **Proactive Anticipation** where an anticipatory agent proposes Reactive/Proactive/Speculative plans, a Plan Evaluation judge selects one, and it either reacts asynchronously (streaming KV-cache) or → **Action** (tool selection, e.g. zoom into a bbox in future frames). Shows anticipation is the core: predict *when/where* evidence appears, then hunt it. *Yang, Tang et al. (2025), arXiv:2508.01875. Embedded for personal research reference.*

## Method + math — the mechanism, then the objective in full
**Problem formulation.** A video stream is $\mathcal{V}^T = \{v_t\}_{t=1}^T$ with $v_t$ the $t$-th clip; a question $Q_t$ arrives at timestamp $t$. Goal: use current clip $v_t$ and past video $\mathcal{V}_{1:t-1}$ to answer $Q_t$. When $\mathcal{V}_{1:t}$ is insufficient, the model should **defer** and respond at a future $t'$ ($t \le t' \le T$) once $\mathcal{V}_{t+1:t'}$ supplies sufficient evidence. A decision function $D(Q_t, \mathcal{V}_{1:t})$ chooses wait vs respond, and a response function $f$ is invoked only when enough information has accumulated.

**1. Incremental memory update.** Continuously encoding every frame makes visual tokens grow linearly (prohibitive for real-time decisions). Instead, the visual token sequence $\mathcal{X}_t = \{x_i\}_{i=1}^{N_t}$ of each incoming clip is compressed into a compact textual caption $\mathcal{C}_t = \{c_i\}_{i=1}^{n_t}$ with $N_t \gg n_t$, and a memory state is recurrently updated:
$$m_t = \mathcal{U}(m_{t-1}, \mathcal{C}_t), \tag{1}$$
where $\mathcal{U}(\cdot)$ is the captioning-model update. Only $m_{t-1}$ and $\mathcal{C}_t$ are needed each step → **constant** memory cost regardless of stream length. (Compression loses early fine-grained detail; this is compensated by the streaming KV-cache, which keeps the original visual tokens for recall.)

**2. Proactive anticipation with plan evaluation.** A lightweight MLLM agent $A(\cdot)$ (the "anticipatory stream agent") forecasts temporal dynamics and spatial distribution of task-relevant events. Current state $S_t = A(m_t, v_t)$ fuses text memory + current video. From $S_t$ the agent reasons about future event trajectories $E_t^{(h)} = \{e_i\}_{i=t+1}^{t+h}$, where $h$ is the temporal reasoning depth. Three complementary perspectives generate candidate plans:
- **Reactive** $h{=}0$ — decide solely on $S_t$, no forecasting (favor immediate response when evidence is present).
- **Proactive** $h{=}\delta$ — extrapolate to anticipate near-future task-relevant events.
- **Speculative** $h{=}\Delta$, $\Delta \gg \delta$ — explore long-range future possibilities for queries needing distant evidence.

Each perspective yields one or more candidate plans:
$$P_j = A\!\left(E_t^{(h_j)}, S_t \mid m_t, v_t\right), \qquad \mathcal{P} = \{P_j\}_{j=1}^{k}, \tag{2}$$
with $h_j \in \{0, \delta, \Delta\}$ and $k$ the total number of candidate plans. The **same agent** $A(\cdot)$ then scores each plan with a separate prompt (LLM-as-a-judge), on two complementary criteria: (i) whether currently accumulated evidence is sufficient to answer $Q$, and (ii) whether further observation of future frames is likely to provide critical missing information — and selects the most appropriate plan $\hat{P}$.

**3. Tool-augmented action.** Beyond passive frame consumption, StreamAgent decides *when, where, and how* to acquire evidence. Given $\hat{P}$, $S_t$, and $Q$, the agent outputs a structured action spec: which tool to invoke and its spatio-temporal parameters (e.g. a bbox for ROI crop, or a target region to keep tracking). At timestep $t$ it selects a tool subset $\mathcal{T}_t' \subseteq \mathcal{T}$ and applies each $\phi_j \in \mathcal{T}_t'$ to a sub-region $v_{t+1}^j \subseteq v_{t+1}$ of the incoming frame, producing $R_j = \phi_j(v_{t+1}^j)$. The perception state updates:
$$S_{t+1} = A\!\left(m_t, \{R_j\}_{j=1}^{|\mathcal{T}_t'|}\right). \tag{3}$$
Tool invocations are **not one-shot**: $\hat{P}$ can instruct persistent monitoring of a spatial region / object tracking across $v_{t+1}, v_{t+2}, \dots$ until the target evidence is captured.

**4. Hierarchical streaming KV-cache (the memory backbone).** To answer without re-encoding, StreamAgent keeps original visual KV entries in a three-stage hierarchy (Fig. 3):
- **Long-term memory (continuous perception).** As the stream is encoded, per-frame KV caches are offloaded to **CPU**. Clips are prefilled in chunks: an incoming clip is split into chunks $\{Z_j^{v_i}\}_{j=1}^{\mu}$, sequentially prefilled, with attention concatenating clip / chunk / current KV:
$$K = [H_{\text{clip}}^{k}, H_{\text{chunk}}^{k}, X W_K], \quad V = [H_{\text{clip}}^{v}, H_{\text{chunk}}^{v}, X W_V]. \tag{4}$$
- **Selective recall (dynamic, layer-adaptive).** On a query, relevant KV entries are retrieved **per layer** by attention score. Frame descriptor = mean key $\tfrac{1}{T_f}\sum_{j=1}^{T_f} k_j$; query descriptor = mean query $\tfrac{1}{T_q}\sum_{j=1}^{T_q} q_j$; their dot product gives per-frame score $S_{h,j}$ at layer $h$. Layer-adaptive selection keeps every frame within margin $\alpha$ of the layer max:
$$J_h = \Big\{ j \in \{1,\dots,L_k\} \;\big|\; \max_{j'}(S_h) - S_{h,j} \le \alpha \Big\}, \tag{5}$$
$$\beta_i = \frac{e^{S_{h,i}}}{\sum_{j=1}^{l} e^{S_{h,j}}}. \tag{6}$$
Because $\alpha$ is a fixed *margin* (default $\alpha{=}3$), each transformer layer retrieves a **variable** number of KV entries tuned to its own attention behavior — unlike static top-$k$ pruning. Selected KV are reloaded to GPU as **short-term memory** and combined with the streaming input for answer generation:
$$K = [H_{\text{out}}^{k}, X_q W_K], \quad V = [H_{\text{out}}^{v}, X_q W_V]. \tag{7}$$

## Explicit design choices
- **Two-model split:** planner/tool-coordinator = **Qwen2.5-VL-3B** (lightweight, does anticipation + plan evaluation + tool selection); responder = **Qwen2.5-VL-7B** (precise final answers).
- **Text memory + visual KV are decoupled:** captions give O(1) rolling context for *decisions*; original visual tokens are preserved in the KV-cache only for *answer generation* (recovers the detail captioning drops).
- **Three anticipation horizons as an explicit plan set** ($h\in\{0,\delta,\Delta\}$) rather than a single binary trigger — heuristic selection among them beats any fixed strategy.
- **LLM-as-a-judge plan evaluation** on two axes: current-evidence sufficiency vs future-information value; principle prompt = "1. current state summary, 2. future planning."
- **Tools do active perception:** zoom/crop into predicted bbox, persistent region monitoring / object tracking across future frames — "information hunting," not passive consumption.
- **KV hierarchy:** GPU→CPU offload when a token threshold is hit; **layer-adaptive margin selection** ($\alpha{=}3$) not global top-$k$; chunked prefill for incremental encoding.
- **Inputs:** frames resized to 224×224 at **1 FPS** (following Dispider). Hardware: NVIDIA A800 (80GB), FP16.

## Key results / what to remember
Verified against the paper's own tables (Tables 1, 3, 4 rendered from the PDF).

**OVO-Bench (Table 1), 1 fps.** StreamAgent-7B **Overall 49.4** — best among *online* models and above several offline ones. Sub-scores: Real-Time Visual Perception **61.3**, Backward Tracing **41.7**, Forward Active Responding **45.4**. Dispider (1 fps) = 41.8 overall / 34.7 forward → StreamAgent improves Forward Active Responding by **+10.7** over Dispider (45.4 vs 34.7). TimeChat-Online-7B = 45.6 overall. Ceilings: Human 92.8, Gemini 1.5 Pro 65.3, GPT-4o 58.6.

**StreamingBench (Table 4), 1 fps.** StreamAgent-7B **Overall 57.02**, vs Dispider-7B 53.12 → **+3.90**. Category "All" scores: Real-Time Visual Understanding **74.28**, Omni-Source Understanding **36.24**, Contextual Understanding **34.62**. Beats Claude 3.5 Sonnet (57.68 is above; actually Claude 57.68 > 57.02 — StreamAgent trails proprietary Claude/GPT-4o 60.15/Gemini 67.07 but leads all open online models). Human 91.66.

**Offline long-video (Table 3), 1 fps.** StreamAgent-7B: MLVU **67.2**, LongVideoBench **57.9**, VideoMME overall **62.9** / long **50.6**. Its own backbone Qwen2.5-VL-7B: MLVU 66.9, LongVideoBench 61.5, VideoMME 63.2 / long 50.4 — so StreamAgent matches/slightly edges the backbone on MLVU and VideoMME-long while trailing on LongVideoBench, i.e. the streaming machinery does not hurt offline competence.

**Ablations (Figures 5–6, Table 2).** (a) Memory module gives significant gains on longer videos vs concatenation-based captions; (b) heuristic (three-horizon) planning consistently beats fixed strategies, though the Speculative horizon alone stays limited due to long-horizon uncertainty; (c) tool-based perception improves accuracy with minimal overhead. Streaming KV-cache: ~**30% faster** inference than ReKV on Qwen2.5-VL-7B with GPU memory on par with sampled-frames baseline (Fig. 6). (Exact ablation deltas are shown only as bar plots — n/r as numbers.)

No Zotero highlights present.

Takeaways: (1) the win is *timing* — deferring the answer until anticipated evidence arrives (+10.7 forward on OVO-Bench) is where prior binary-trigger streamers fail; (2) anticipation is cheap because it runs on a 3B planner + rolling text memory, with the 7B responder only reading a layer-adaptively recalled KV subset; (3) tools turn perception *active* (predict a bbox, then zoom/track), which is the concrete lever for "where" future evidence lives.

## How it connects (evolution)
- [[dispider]] — the immediate foil: asynchronous decide/respond but no future planning; StreamAgent's forecasting is the answer to its premature-response failure.
- [[videollm-online]] — the single-LLM alternating baseline whose slowness/reactivity motivates the two-model anticipatory split.
- [[timechat-online]] — contemporaneous online video LLM baseline on the same OVO-Bench / long-video tables.
- [[rekv]] — the streaming KV-cache method StreamAgent benchmarks against (~30% faster than ReKV); shares the offload-and-recall lineage.
- [[streaming-memory]] — the sub-topic hub for hierarchical KV / long-term memory that this paper's Fig. 3 mechanism belongs to.
- [[proactive-response]] — sub-topic hub: anticipatory "when to answer" timing is the defining problem here.

## Open questions / limitations
- Caption-based incremental memory admittedly loses early fine-grained detail; recovery hinges entirely on the KV-cache still holding the right visual tokens — failure modes when the needed frame was already offloaded/evicted are not quantified.
- The Speculative ($h{=}\Delta$) horizon is acknowledged to be weak in isolation (long-horizon prediction uncertainty); the paper leans on heuristic selection rather than showing reliable long-range forecasting.
- Overall accuracy (49.4 OVO-Bench, 57.02 StreamingBench) still trails proprietary models (Gemini 65.3 / 67.07); the anticipatory design leads open *online* models but does not close the gap to strong offline/proprietary systems.
- Tool set (zoom/crop, track) is narrow and hand-picked; generality of "information hunting" to non-spatial evidence (audio, dialogue, off-screen reasoning) is untested.

*Verification: Eqs (1)–(7), the three-horizon plan formulation, and the two-model config were read from the rendered PDF pages 2–3; headline numbers (OVO-Bench Table 1, StreamingBench Table 4, offline Table 3) were verified against the paper's own rendered tables — Forward +10.7 and StreamingBench +3.90 recomputed from the table cells. Ablation magnitudes shown only as bar/radar plots are marked n/r.*

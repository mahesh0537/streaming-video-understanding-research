---
zotero_key: null
authors: Dingyu Yao et al. (Institute of Information Engineering, CAS / UCAS / JD.COM; with Nan Duan, Jiaqi Wang)
year: 2026
arxiv: 2606.08615
pdf: https://arxiv.org/pdf/2606.08615
tier: deep
subtopics: [streaming-benchmarks]
tags: [streaming-video-understanding, streaming-benchmarks]
---

# Harnessing Streaming Video in the Wild

**Lineage role:** Jun-2026 frontier work that reframes streaming video understanding as **multi-turn online interaction** and ships a matched triple — a streaming-native VLM (trained with a silence/response trigger loss), a plug-and-play deployment harness (three-tier memory + prefix-aware KV cache), and an **in-the-wild evaluation benchmark, Streaming-Eval**, that jointly scores answer correctness *and* response timeliness. The benchmark is the anchor for this sub-topic.

## Problem — what was limited before this paper (short)
Offline video VLMs consume the entire clip up front and answer once; streaming deployment instead reveals frames one at a time and demands the model decide *when* to speak. Prior streaming benchmarks (they cite StreamingBench / OVO-Bench / SVBench-style suites) mostly grade **correctness only**, and they collapse inference into a single-turn offline-style paradigm that forces an immediate answer with questions restricted to already-observed (past/present) content. That misses the three things real streams need at once: **proactive interaction** (choose to respond vs. stay silent, and even wait for *future* evidence), **long-horizon memory** (hours, not the minutes of training clips), and **sub-second real-time latency**. Existing methods tackle these in isolation.

## Key idea — the core insight
Treat streaming as a **causal, multi-turn dialogue over a sequential frame stream**: at every one-second step the model emits either a `</response>` utterance or a `</silence>` control token, conditioned only on past frames and dialogue history. To make a base VLM behave this way, (1) **train** it with a role-weighted "trigger" loss that fixes the silence↔response class imbalance, (2) **deploy** it behind a harness that bounds memory to a fixed budget via a three-tier (short/mid/long) compression hierarchy and keeps latency sub-second via a prefix-reusable KV cache, and (3) **evaluate** it with a benchmark whose metric rewards answering *inside the correct temporal window* and penalizes always-on chatter. The benchmark's construction pipeline is the crux figure below.

![[streaming-video-wild.png]]
> **Crux (Figure 5).** The three-stage Streaming-Eval construction pipeline: 138 curated public videos across 15 everyday categories with a balanced short/medium/long duration split → a task-formulation template that crosses interaction mode (proactive/punctual/contextual) with temporal position (backward/present/forward) → agent-harness QA generation (GPT 5.4 + Claude Opus 4.6 as annotators), then VLM checking at 1 FPS plus human verification on clarity/correctness/response-time. *Yao et al. (2026), arXiv:2606.08615. Embedded for personal research reference.*

![[streaming-video-wild-system.png]]
> **Crux (Figure 3).** StreamingHarness deployment system: a three-tier memory scheme compresses the stream into short/mid/long buffers (200 s / 1,000 s / 42,000 s ≈ 12 h of context); at each step the online VLM emits `</silence>` or `</response>` with sub-second latency via a vLLM-friendly prefix-cache design. *Yao et al. (2026), arXiv:2606.08615. Embedded for personal research reference.*

## Method + math

### Streaming as multi-turn online interaction (the formulation)
A video is a sequence of temporal segments $\mathcal{V}=\{v_1,\dots,v_T\}$, each segment $v_t$ a fixed short interval (one second) anchored by a timestamp token $\tau_t$ of the form `<ts-(t+1)s>`. The user may issue a textual query $q_t$ at sparse moments ($q_t=\varnothing$ when none). At each step the model emits action $a_t$ = either a `</response>` utterance (when accumulated evidence suffices to answer a pending query or narrate a salient event) or a `</silence>` token otherwise. An interaction unit is $u_t=(\tau_t,v_t,q_t,a_t)$ with history $\mathcal{H}_{t-1}=(u_1,\dots,u_{t-1})$, and the response is generated **causally** as
$$a_t = f_\theta(\mathcal{H}_{t-1},\,\tau_t,\,v_t,\,q_t),$$
with **no access to future segments** $\{v_{t+1},\dots,v_T\}$. The model decides whether to speak at every step.

### Streaming-native VLM: the trigger loss (Eq. 1–2)
Under naive supervised fine-tuning the assistant-token distribution is dominated by `</silence>` (silence runs vastly outnumber responses), so the gradient over-reinforces staying silent and dilutes the silence→response transition. Fix: assign role-dependent per-token weights. Let $\mathcal{A}$ be the supervised assistant-token positions and $\mathcal{C}=\{c_i\}_{i=1}^M\subseteq\mathcal{A}$ the ordered control-token positions where $y_{c_i}\in\{$`</silence>`, `</response>`$\}$. A *silence run* is a maximal consecutive subsequence of `</silence>` in $\mathcal{C}$. For $j\in\mathcal{A}$, with convention $y_{c_0}=\varnothing$:

$$
w_j=\begin{cases}
w_{\text{silence}}^{\text{first}}, & y_{c_i}=\text{</silence>},\ y_{c_{i-1}}\neq\text{</silence>}\\[2pt]
w_{\text{silence}}^{\text{repeated}}, & y_{c_i}=\text{</silence>},\ y_{c_{i-1}}=\text{</silence>}\\[2pt]
w_{\text{response}}, & y_{c_i}=\text{</response>}\\[2pt]
1, & \text{otherwise}
\end{cases}
$$

with $w_{\text{silence}}^{\text{first}}=1$, $w_{\text{silence}}^{\text{repeated}}<1$ to down-weight silence *continuations*, and $w_{\text{response}}>1$ to up-weight response *onsets*. The objective normalizes by the number of supervised positions:
$$
\mathcal{L}(\theta) = -\frac{1}{|\mathcal{A}|}\sum_{j\in\mathcal{A}} w_j \log p_\theta\!\left(y_j \mid y_{<j}\right).
$$
Chosen values in experiments: $w_{\text{silence}}^{\text{first}}=1$, $w_{\text{silence}}^{\text{repeated}}=0.8$, $w_{\text{response}}=1.5$.

### StreamingHarness deployment (bounded memory + sub-second latency)
Three components:
- **Three-tier memory management.** Context is partitioned into (i) **short-term** = the most recent $T_s$ seconds as raw vision tokens; (ii) **mid-term** = up to $M$ textual summaries of past short-term chunks, covering $T_m=M T_s$ seconds; (iii) **long-term** = up to $L$ aggressively compressed blocks, each consolidated from $M$ mid-term summaries, spanning $T_l=LMT_s$. When the short-term buffer fills, a *mid-term agent* condenses the chunk (per-frame semantics, salient details, temporal object/scene evolution) into a compact textual summary; once $M$ accumulate, a *long-term agent* merges them (removing redundancy, organizing causally related events on a timeline), with FIFO eviction at $L$ blocks. Both agents run asynchronously ahead of chunk boundaries, hiding consolidation behind mainline inference. Total temporal coverage:
$$
T_s+T_m+T_l = T_s\,(1 + M + LM).
$$
Default $T_s=200\text{ s},\,M=5,\,L=42$ gives $200(1+5+210)=43{,}200\text{ s}\approx 12$ hours.
- **Prefix-aware KV cache.** Recomputing the full KV cache each step would break the latency budget. The harness does a one-time prefill of the Memory Text at each chunk start into KV; at every subsequent step only the new frame and previous response are appended, and all previously cached KV states (Memory Text + earlier turns in the chunk) are **reused** — compatible with modern engines' prefix caching (vLLM), collapsing per-step TTFT.
- **Event-driven response triggering.** A *training-free* adaptation that lets even a base model without native proactive ability follow the protocol: at each step it judges the evidence (current frame + current-chunk dialogue + past video/QA history) and stays silent if no new informative content (key event not yet occurred / scene in transition), else emits a concise grounded response. Loop = query triggering → frame-by-frame judgment → timely response → monitoring resumption.

### Streaming-Eval benchmark protocol (the eval "math")
**Construction (three stages).** (i) *Video collection*: 138 videos aggregated from PhoStream, Video-MMMU, Ego4D, THUMOS, HiREST, MLVU, RTV-Bench, OVO-Bench, spanning 15 everyday categories, balanced across short (0–4 min, 29.41%), medium (4–8 min, 34.79%), long (8 min+, 35.80%). (ii) *Task formulation*: same taxonomy as training — each task = one *interaction mode* (proactive / punctual / contextual) × one *understanding position* (backward / present / forward). (iii) *QA generation*: an agent harness (guided by a structured Skill) with GPT 5.4 and Claude Opus 4.6 as annotators generates multiple-choice QA; each pair is filtered by a VLM checker at 1 FPS and human-verified on **question clarity, answer correctness, response time**.

**Task taxonomy.** Two families: **Narration** (continuously monitor, proactively describe salient events) and **Question Answering**, split into *Answer-invariant* (one-to-one question→answer) and *Answer-variant* — the latter into *Objective-fact Changing* (OFC, one-to-many) and *Question Changing* (many-to-many), which further splits into with/without *Inter-question Dependency* (IQD). Evidence frames can be **backward / present / forward** relative to the query.

**QA metric — Streaming Weighted F1 (SW-F1), Eq. 3.** Each question has a ground-truth answer window $W$. A prediction is a **TP** if emitted *within* $W$ and exactly matching the ground truth; any non-silence prediction *outside* $W$ is a **FP**; a question with no valid prediction in $W$ is a **FN**. Then
$$
\text{SW-F1} = \frac{w_{TP}\,TP}{w_{TP}\,TP + w_{FP}\,FP + w_{FN}\,FN},\qquad w_{TP}=w_{FN}=2.0,\ w_{FP}=0.2.
$$
The FP term (unlike accuracy-style metrics that count only TP) penalizes "always-respond" behavior that spams answers every step to inflate TP; the low $w_{FP}$ vs. vanilla F1 keeps answer correctness primary and prevents over-penalization.

**Narration metric.** 100 instances sampled from LiveSports3K-CC; each candidate narration is converted to a timestamped transcript `[t s] utterance` and Claude Sonnet 4.6 judges it pairwise against the ground-truth ASR transcript on two equal-weight criteria (narration style — pacing/tone; consistency with reference). To de-bias position, each pair is judged twice with candidate order swapped and the **win rate over the resulting $2N$ votes** is reported.

## Explicit design choices
- **Streaming = per-second multi-turn**: one interaction unit per 1 s segment, timestamp-anchored, strictly causal (no future frames).
- **Two special control tokens** `</response>` / `</silence>` as the decision surface; the model's core skill is *when*, not just *what*.
- **Role-weighted trigger loss** to counter silence dominance: down-weight repeated-silence tokens (0.8), up-weight response onsets (1.5), first-silence and other tokens at 1.0; normalized over supervised positions.
- **Three-tier compression** with two asynchronous summarization agents (mid-term, long-term), FIFO long-term eviction, default $T_s{=}200,\ M{=}5,\ L{=}42$ → ~12 h bounded context.
- **Prefix-cache-first system design** (Memory-Text prefill reused across steps) explicitly to be vLLM-compatible — a sliding window or token-pruning baseline breaks prefix reuse and gives no caching benefit.
- **Event-driven triggering is training-free** — a prompt-level adaptation so closed-source baselines can also run the protocol.
- **Streaming-native VLM**: continue-train from **Qwen3-VL-8B-Instruct** on **Streaming-Train-248K** (aggregated from EpicKitchens, YouCook2, Ego4D, EgoBlind, EgoExo4D, EgoLearn, DiDeMo, Charades, OmniStar, QVHighlights, ActivityNet, Shot2Story, Assembly101, HoloAssist, WTaG, MovieChat, QueryD, LiveWhisperX); per-second frame–text alignment, each segment tagged with a timestamp and a response/silence token; vision encoder **frozen**, connector + LLM updated; lr $1\times10^{-5}$, 1 FPS, max 131,072 pixels/frame, 48K max context; ~4,096 H200 GPU-hours.
- **Benchmark metric jointly scores correctness + timeliness** via a windowed TP/FP/FN with asymmetric weights — the central contribution for this sub-topic.
- **Inference config**: baselines at 1 FPS, ≤262,144 pixels/frame; open-source default $T_s{=}200,M{=}5,L{=}42$; closed-source reduced to $T_s{=}10,M{=}4,L{=}8$ due to API rate limits.

## Key results / what to remember
Verified against the paper's Table 1 (narration win rates), Table 2 (QA SW-F1), and Figure 6 (latency). All model names are the paper's frontier-labeled baselines. † = offline model adapted to the streaming harness.

**Narration (Table 1, pairwise win rate %, higher better; averaged over 4 reference judges):**
- **StreamingHarness-8B: 61.4 avg** — vs Claude Opus 4.6 Thinking 65.0, vs GPT 5.4 (High) 65.0, vs Gemini 3.1 Pro (High) 44.0, vs Doubao Seed 2.0 Pro 71.5.
- Best baseline **Gemini 3.1 Pro (High)† 59.1 avg**; GPT 5.4 (High)† 31.8; Claude Opus 4.6 Thinking† 15.3; Doubao Seed 2.0 Pro† 12.0; open-source Qwen3-VL-8B† 3.6, Qwen3-VL-32B† 2.1, Qwen3.5-9B (Thinking)† 0.5.
- Takeaway: the 8B streaming-native model **beats the strongest closed model** on narration; baselines over-narrate in dense-captioning style and hallucinate.

**QA (Table 2, SW-F1, higher better) — StreamingHarness-8B vs baselines:**
- **StreamingHarness-8B avg 45.8**; per split: Backward 54.2, Present 21.0, Forward 48.9, Objective-fact Changing (OFC) 32.5, Question-Changing w/ IQD 61.9, Question-Changing w/o IQD 56.3.
- Baseline averages: GPT 5.4 (High)† **18.3** (best baseline; Backward 26.0, Present 17.4, Forward 15.1, OFC 9.9, w/IQD 20.2, w/o IQD 21.0); Gemini 3.1 Pro (High)† 13.8; Doubao Seed 2.0 Pro† 13.3; Claude Opus 4.6 Thinking† 9.5; Qwen3-VL-32B† 11.9; Qwen3-VL-8B† 10.8; Qwen3.5-VL-9B (Thinking)† 0.1.
- Takeaway: **~2.5× the best baseline** on average SW-F1 (45.8 vs 18.3); largest gains on Backward and Inter-question-Dependency (harness memory retrieval) and on Forward/OFC (proactive deferral until future evidence arrives).

**Efficiency (Figure 6, single NVIDIA H200, 1 FPS, 2-hour sports broadcast, real-time threshold = 1 s):**
- Full-attention (w/ or w/o prefix cache) **saturates max context within a few hundred seconds**.
- Sliding window avoids overflow but latency **stabilizes ~4 s** (exceeds real-time); consecutive steps share no common prefix so prefix caching gives no benefit.
- **Full StreamingHarness keeps TTFT and end-to-end latency < 1 s throughout the entire 2-hour video**; enabling prefix cache drops latency from ~4 s to <1 s vs. the no-prefix-cache harness variant.

**Ablation (Figure 7):** overall average score rises as $w_{\text{silence}}^{\text{repeated}}$ grows over $\{0.2,0.5,0.8\}$ (larger value → fewer mistimed responses), and removing StreamingHarness components degrades QA/narration.

No Zotero highlights present.

Takeaways to remember: (1) the reframing — streaming = per-step "speak or stay silent" — is the unifying abstraction; (2) SW-F1's windowed TP/FP/FN with asymmetric weights is a reusable **timeliness-aware** metric other streaming benchmarks can adopt; (3) an 8B streaming-native model + a prefix-cache-friendly memory harness beats far larger closed models on this in-the-wild protocol while holding sub-second latency for 2 h.

## How it connects (evolution)
- [[streaming-benchmarks]] — this note is a member; Streaming-Eval sits alongside other streaming QA benchmarks here.
- [[streamingbench]], [[ovo-bench]], [[svbench]] — prior streaming benchmarks it critiques for grading correctness only / single-turn offline framing; Streaming-Eval adds joint timeliness scoring and forward (future-evidence) tasks.
- [[proactivevideoqa]], [[river-bench]] — kindred proactive/real-time streaming evaluations; shares the "when to respond" axis.
- [[videollm-online]], [[mmduet]] — earlier streaming-native VLMs that introduced per-frame speak/silence decisions; this paper's trigger loss is a direct descendant refinement.
- [[streaming-memory]] — the three-tier compression + KV-reuse harness connects to the memory-for-streaming line.
- [[streaming-video-understanding]] — the topic hub.

## Open questions / limitations
- **Reproducibility of the eval rests on frontier LLM judges** (GPT 5.4 / Claude Opus 4.6 as QA annotators, Claude Sonnet 4.6 as narration judge) — results and the benchmark's ground truth may drift as those judges change; only 138 videos and 100 narration samples.
- **Closed-source baselines are hobbled** by API rate limits (reduced memory config $T_s{=}10$ vs 200; $T_s$ set to 10 for narration) — the head-to-head "beats closed models" claim is partly a deployment-budget artifact, not purely capability.
- **12-hour coverage is a bound from a formula**, validated on up to ~2 h in the latency plot; behavior of the asynchronous summarization agents at true multi-hour scale (summary drift, error accumulation) is not stress-tested here.
- **Event-driven triggering is prompt-level and heuristic** ("is there new informative content?"); no learned threshold, so it may be brittle across domains outside the curated 15 categories.

*Verification: SW-F1 (Eq. 3, weights 2.0/0.2), the trigger loss (Eq. 1–2, weights 1/0.8/1.5), the three-tier config ($T_s{=}200,M{=}5,L{=}42\Rightarrow{\approx}12$ h), Table 1 narration win rates (StreamingHarness-8B 61.4 avg vs Gemini 59.1), Table 2 QA SW-F1 (45.8 avg vs GPT 5.4 18.3), and Figure 6 latency (<1 s over 2 h) were all read directly off the rendered PDF pages of arXiv:2606.08615v1. No project page/GitHub was consulted; arXiv HTML abstract used only for author/affiliation and dataset-source names.*

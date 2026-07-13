---
zotero_key: null
authors: Zichen Wen et al. (Shanghai Jiao Tong University / EPIC Lab; with collaborators)
year: 2026
arxiv: 2605.10343
pdf: https://arxiv.org/pdf/2605.10343
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# EvoStreaming: Your Offline Video Model Is a Natively Streaming Assistant

**Lineage role:** Self-evolved offline→streaming adaptation — the base VideoLLM acts as its own data generator, annotator, and roll-out policy to bootstrap ~1,000 streaming trajectories (139× less data than prior work) that teach *when to speak*, with no architecture change; paired with a new frame-level **RealStreamEval** protocol that scores correctness AND timing under a verbosity penalty.

## Problem — what was limited before this paper
Two coupled bottlenecks held back streaming (online) video assistants. (1) **Evaluation misalignment**: popular benchmarks quietly leak the timing decision to the evaluator. OVO-Bench trims each clip up to the question timestamp and feeds the whole prefix in one shot; StreamingBench polls the model every second ("is it time to respond?"). Both are reproducible but sidestep the *core* streaming challenge — deciding, causally, when to act under partial observation. (2) **Costly adaptation**: turning an offline VideoLLM into a streaming one usually meant new architecture (memory modules, decision heads) and large human-annotated timing datasets (e.g. TimeChat-Online's ~139K samples), which also erodes the base model's offline ability.

## Key idea — the core insight
A strong offline VideoLLM already contains the knowledge needed to be a good streaming assistant; what it lacks is a learned *timing policy* (silent vs. respond). EvoStreaming elicits that policy by **self-evolution**: the same frozen-architecture base model $\mathcal{M}$ is reused in three roles — a **data generator** (invents task-consistent questions), a **relevance annotator** (labels which video segments carry the evidence), and a **roll-out policy** (turns those labels into SILENT/RESPOND trajectories under a causal sliding window). Fine-tuning (LoRA) on ~1,000 of its own trajectories, optionally looped for a few iterations, installs the timing behavior with 139× less data and almost no loss of offline accuracy — because supervision lives in the model's own representation space, no external annotator is required.

![[evostreaming.png]]
> **Crux (Figure 3 + Algorithm 1).** The five-stage self-evolution pipeline — one base model in three roles (data generator, relevance annotator, roll-out policy) converts raw video into SILENT/RESPOND streaming trajectories with no external labels: (1) taxonomy balancing over 5 task types, (2) type-consistent question construction, (3) segment relevance annotation, (4) sliding-window causal roll-out, (5) conversation standardization → fine-tune, repeat for $I$ iterations. *Wen et al. (2026), arXiv:2605.10343. Embedded for personal research reference.*

![[evostreaming-realstreameval.png]]
> **Crux (Figure 2).** RealStreamEval's three-phase protocol — DISCRETIZE (align questions to sampled frames at ~0.5–2 fps), STREAM (frames arrive one-by-one; at each step the model chooses SILENT or RESPOND with no peeking at future frames), SCORE (a judge reads the trajectory and scores timed correctness while a verbosity meter penalizes over-talking). *Wen et al. (2026), arXiv:2605.10343.*

## Method + math

### Streaming as a causal sequential decision (the eval reformulation)
Frames arrive at a fixed rate (0.5 fps in the main setup). A user query $q_t$ may be issued at any timestep $t \in \{1,\dots,T\}$ (with $q_t=\varnothing$ when no new query). At each step the model picks an action from
$$a_t \in \mathcal{A} = \{\textsc{Silent}\} \cup \mathcal{R},$$
where $\mathcal{R}$ is the space of natural-language responses. The decision must be **causal** — it depends only on the history available up to $t$:
$$\mathcal{H}_t = (v_{1:t},\, q_{1:t},\, a_{1:t-1}), \qquad a_t = \pi(\mathcal{H}_t),$$
so future frames $v_{t+1},\dots,v_T$ are inaccessible at time $t$. This contrasts with standard offline inference $a^\star = f_{\text{offline}}(v_{1:T}, q)$, which conditions on the entire clip and therefore skips the streaming challenge of acting under partial information.

### RealStreamEval scoring
The overall score combines **timed answer matching** with a **multiplicative verbosity penalty** and a premature-response penalty. For a set of ground-truth answer times $\mathcal{T}^\ast$ with a temporal tolerance $\delta$:
$$S = \left[\frac{1}{|\mathcal{T}^\ast|}\sum_{t^\ast\in\mathcal{T}^\ast}\ \max_{t\in\mathcal{T},\ |t-t^\ast|\le \delta}\ \mathcal{J}(a_t, a^\ast_{t^\ast})\right]\cdot \mathcal{M}(R_{\text{ans}}) \;-\; P_{\text{early}}.$$
- **Response quality** $\mathcal{J}(\cdot,\cdot)\in[0,1]$: an LLM-as-judge (Qwen3-VL-235B-A22B-Instruct) scores semantic correctness rather than string match.
- **Answer rate** $R_{\text{ans}} = |\mathcal{T}|/T$ (fraction of timesteps the model chose to speak).
- **Verbosity multiplier** $\mathcal{M}(R_{\text{ans}})$, task-regime specific:
  - Forward Active Responding (FAR): $\mathcal{M}_{\text{FAR}}(R_{\text{ans}}) = 1.0$ if $R_{\text{ans}} < 0.4$, else $\max\!\big(1 - 0.2\lfloor 5R_{\text{ans}}\rfloor,\ 0.2\big)$.
  - Real-Time Visual Perception (RTVP) / Backward Tracing (BT): repetition penalty $\mathcal{M}_{\text{rep}} = 0.5$ for repeated responses, $1.0$ otherwise.
- **Premature penalty** $P_{\text{early}}$: $0.1$ for FAR (punishes responses issued before the evidence appears), $0$ otherwise. $\delta = 0$ for RTVP/BT.

### The five-stage self-evolution pipeline (Algorithm 1)
Given base model $\mathcal{M}^{(0)}$, a video pool $\mathbb{V}$, and $I$ outer iterations, for each iteration $i$ and video $\mathcal{V}$:
1. **Stage 1 — task-aware taxonomy.** $c \leftarrow \mathcal{M}^{(i)}_{\text{cls}}(\mathcal{V})$: classify the video into one of five self-defined categories — Immediate Visual (IV), Memory-Dependent (MD), Temporal Aggregation (TA), Anticipatory Monitoring (AM), Dynamic Event Description (DED). Categories are balanced across the pool.
2. **Stage 2 — self-questioning.** $\mathcal{Q} \leftarrow \mathcal{M}^{(i)}_{\text{gen}}(\mathcal{V}, c)$: generate $K$ type-consistent questions.
3. **Stage 3 — temporal grounding / relevance.** $\mathbf{A} \leftarrow [\mathcal{M}^{(i)}_{\text{rel}}(s_t, q_k)]_{t,k}$: a binary relevance matrix $\mathbf{A}\in\{I,R\}^{S\times K}$ over $S$ segments marking when evidence for each question becomes available (Relevant = keep, Irrelevant = discard).
4. **Stage 4 — causal roll-out.** Slide $t=1,\dots,S$ under partial observation: if $\mathbf{A}[t,k]=I$ (irrelevant) for **all** $k$, emit $a_t \leftarrow \textsc{Silent}$; else pick $k^\star$ with $\mathbf{A}[t,k^\star]=R$ and generate $a_t \leftarrow \mathcal{M}^{(i)}(\mathcal{V}_{1:t}, q_{k^\star})\in\mathcal{R}$. Append trajectory $(\mathcal{V}_{1:t}, q_{1:t}, a_t)$ to dataset $\mathcal{D}$.
5. **Stage 5 — self-evolution.** $\mathcal{M}^{(i+1)} \leftarrow \textsc{Finetune}(\mathcal{M}^{(i)}, \mathcal{D})$. Repeat.

Every stage reuses $\mathcal{M}$ in a different role, so supervision is aligned with the model's own representation space — no external annotator. In experiments $I\in\{1,2,3\}$ and $|\mathcal{D}|=1{,}000$ trajectories suffice; $I=1$ is a single fine-tuning round on self-generated data (no iterative refinement), $I\ge 2$ engages the self-evolution loop. Training uses **LoRA (rank 8)**; encoders stay frozen and the base architecture is untouched.

## Explicit design choices
- **No architectural modification** — no memory bank, no decision head; adaptation is a LoRA (rank 8) fine-tune on the base VideoLLM, so offline capability is preserved.
- **Self-supervision, three roles** — the same model generates questions, annotates segment relevance, and rolls out trajectories; the "labels" are its own judgments, aligning supervision with its representation space and needing zero human annotation.
- **Five-category task taxonomy** (IV / MD / TA / AM / DED) with class balancing to force trajectory diversity; questions are generated type-consistent with the sampled class.
- **Causal sliding-window roll-out** — the roll-out policy only ever sees $\mathcal{V}_{1:t}$, enforcing that SILENT/RESPOND decisions are made under partial observation (no future-frame leakage).
- **Tiny data budget** — ~1,000 self-generated trajectories vs. ~139K for TimeChat-Online (139×), with 1–3 self-evolution iterations.
- **Dual verbosity regimes in the eval** — FAR uses a piecewise frequency multiplier + a premature-response penalty; RTVP/BT use a binary repetition multiplier.
- **Judge-based scoring** — semantic LLM-as-judge (Qwen3-VL-235B) instead of exact match, with a robustness test across five judges (Appendix).
- **Backbone-agnostic** — validated on 5 different VideoLLM backbones (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, InternVL-3.5, MiniCPM-V4.5), all ~7–8B.
- **Resolution robustness** — trained at ~128 visual tokens/frame; the timing policy survives a switch to 768 tokens/frame at inference with <1.4-point shift.

## Key results / what to remember
Setting: **RealStreamEval on OVO-Bench**, accuracy (%) over three regimes — RTVP (real-time visual perception), BT (backward tracing), FAR (forward active responding = proactive timing) — plus **Overall**. Numbers verified against the arXiv HTML Table 2 (main results) and Table 3 (offline preservation).

No Zotero highlights present.

- **Consistent overall gains across all 5 backbones** (vanilla → EvoStreaming, Overall): Qwen2-VL 43.8 → **54.6 (+10.8)**; Qwen2.5-VL 43.0 → **47.4 (+4.4)**; Qwen3-VL 38.5 → **46.8 (+8.3)**; InternVL-3.5 40.5 → **43.7 (+3.2)**; MiniCPM-V4.5 34.4 → **39.7 (+5.3)**.
- **The big wins are on FAR (proactive timing)** — exactly the "when to speak" skill: Qwen2-VL 12.6 → **30.6**; Qwen3-VL 15.8 → **37.6**; InternVL-3.5 14.6 → **39.4**; MiniCPM-V4.5 14.9 → **39.3**. Roughly 2–2.6× improvement in proactive responding.
- **Beats streaming specialists** by a wide margin on RealStreamEval Overall: EvoStreaming-Qwen2 54.6 vs. Dispider-7B 33.3 (+21.3), TimeChat-Online 27.5 (+27.1), Flash-VStream-7B 24.0 (+30.6). Human ceiling ≈ 92.8 Overall.
- **Offline ability preserved** — average across VideoMME, LVBench, LongVideoBench, EgoSchema, MLVU drops <2 points: Qwen2-VL 55.0 → 53.6 (−1.4); Qwen2.5-VL 56.5 → 55.8 (−0.7); InternVL-3.5 61.0 → 59.2 (−1.8).
- **Data efficiency:** ~1,000 self-generated trajectories, **139×** fewer than TimeChat-Online's ~139K human-annotated samples.

Takeaways: (1) the timing policy is *cheap to install* and largely orthogonal to the base model's knowledge — a small LoRA on self-generated data flips a competent offline model into a competent streaming one; (2) the largest gains land precisely on proactive responding (FAR), which is the crux of a streaming assistant; (3) RealStreamEval's causal + verbosity-penalized protocol is a stricter, more deployment-faithful measure than OVO-Bench/StreamingBench, and it re-ranks methods.

## How it connects (evolution)
- [[timechat-online]] — the ~139K-sample prior-best that EvoStreaming beats with 139× less data; the explicit data-efficiency baseline.
- [[dispider]] — architecture-heavy streaming specialist (decoupled perception/decision) that EvoStreaming outperforms without any architecture change.
- [[flash-vstream]] — memory-augmented streaming model used as a specialist baseline on RealStreamEval.
- [[ovo-bench]] — the source task suite that RealStreamEval re-scores under a strict causal protocol; the "evaluation misalignment" it critiques.
- [[streamingbench]] — the polling-based streaming benchmark whose "ask every second" design RealStreamEval argues leaks the timing decision.
- [[mmduet]] — a fine-tuned proactive/dense-caption streaming approach; contrast in adaptation cost vs. EvoStreaming's self-evolution.
- [[proactivevideoqa]] — proactive-response evaluation sibling; complementary protocol for "when to speak".

## Open questions / limitations
- **Self-generated supervision is capped by the model's own errors** — the binary relevance labels inherit the base encoder's segment-level mistakes; a manual audit (n=50 trajectories, Appendix) is needed to gauge label noise, and quality bounds the ceiling.
- **Iteration count and model collapse** — only $I\in\{1,2,3\}$ tested; recursively training on self-generated data risks degeneration (paper cites the model-collapse literature), so the self-evolution loop may not scale to many rounds.
- **Task/taxonomy coverage** — the five categories and OVO-Bench-derived tasks may not span real streaming deployments; generalization beyond this benchmark is unexplored.
- **Judge dependence** — semantic scoring hinges on one large VLM judge (Qwen3-VL-235B); despite a five-judge robustness check, judge bias and the hand-tuned piecewise verbosity/premature-penalty constants may not transfer across domains. A small but consistent offline accuracy trade-off (0.7–1.8 pts) remains.

*Verification: equations (causal decision Eqs. 1–2, RealStreamEval score, Algorithm 1 stages) and all headline numbers checked against the arXiv HTML full text and Tables 2–3 (arxiv.org/html/2605.10343); figures cropped from the arXiv PDF (Fig. 3 + Algorithm 1, p.5; Fig. 2, p.3). Zotero unavailable this session. Overall/gain arithmetic is internally consistent (e.g. 43.8→54.6 = +10.8).*

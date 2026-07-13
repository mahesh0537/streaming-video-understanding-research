---
zotero_key: null
authors: Siwei Wen, Zhangcheng Wang, Xingjian Zhang, Lei Huang, Wenjun Wu (Beihang University; Paradigm Inc.)
year: 2026
arxiv: 2602.15329
pdf: https://arxiv.org/pdf/2602.15329
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# EventMemAgent: Hierarchical Event-Centric Memory for Online Video Understanding with Adaptive Tool Use

**Lineage role:** turns the streaming-memory buffer from a passive frame store into an *active agent* — an event-segmented hierarchical memory that an RL-trained ReAct agent queries and augments with OCR/detection tools on demand.

## Problem — what was limited before this paper (short)
Online video LLMs must answer at an arbitrary timestamp $t$ using only content seen so far, from a potentially infinite stream, inside a finite context window. Prior streaming approaches passively compress frames into a flat, fixed-length buffer (KV cache or uniform frame memory). Two failures follow: (1) **information decay** — old evidence is overwritten as the stream grows; (2) **semantic fragmentation** — a fixed-length window slices a single coherent event across buffer boundaries, so a query about that event sees only a fragment. They are also passive: they cannot go back and re-inspect fine detail (read on-screen text, localize a small object) once a frame is compressed.

## Key idea — the core insight, 2-4 sentences
Structure memory around **events**, not fixed frame counts, and make the model an **active agent** rather than a passive summarizer. A short-term buffer is segmented online into semantically coherent events (histogram-correlation boundaries) and kept unbiased within long events via reservoir sampling; completed events are distilled into a structured long-term memory (visual anchor + caption + embedding + change log). A ReAct agent then decides *when* to retrieve memory (temporal/semantic search) and *when* to invoke fine-grained perception tools (OCR, object detection), and this decision policy is trained end-to-end with Agentic RL (GRPO). The payoff: it matches or beats 1-fps online baselines while touching only ~32 frames.

![[eventmemagent.png]]
> **Crux (Figure 1).** The full pipeline: incoming frames are event-segmented into short-term memory (with reservoir sampling), distilled by an MLLM into structured long-term memory (anchor image, caption, vector, change log), and an agent — trained by Agentic RL — searches memory and selects perception tools to answer a question at timestamp $t$. *Wen et al. (2026), arXiv:2602.15329. Embedded for personal research reference.*

## Method + math

**Short-term memory (STM) as events.** The STM is a fixed-capacity buffer of at most $K$ frames, but organized as a set of $m$ semantically coherent events rather than a flat sequence:
$$\mathcal{E}_{st} = \{E_1, E_2, \dots, E_m\}, \qquad \sum_{i=1}^{m} n_i \le K,$$
where $\{E_1,\dots,E_{m-1}\}$ are completed segments and $E_m$ is the active event accumulating incoming frames; event $E_i = \{f_{i,j}\}_{j=1}^{n_i}$ holds $n_i$ frames of one segment. Frames are sampled at **1 FPS** from the stream $V=\{f_1,f_2,\dots\}$.

**Online event segmentation.** For each new frame $f_t$, decide whether it continues the active event $E_m$ or starts a new one, by comparing the normalized grayscale histogram $\mathbf{h}_t$ of $f_t$ against the average histogram $\bar{\mathbf{h}}_m$ of frames currently in $E_m$, using the Pearson correlation coefficient:
$$\rho(\bar{\mathbf{h}}_m, \mathbf{h}_t) = \frac{\operatorname{cov}(\bar{\mathbf{h}}_m, \mathbf{h}_t)}{\sigma_{\bar{\mathbf{h}}_m}\,\sigma_{\mathbf{h}_t}}.$$
A boundary is triggered when $\rho < \delta$ (threshold $\delta = 0.2$): $E_m$ is archived to long-term memory and a fresh event begins.

**Intra-event reservoir sampling.** Within a long event, once the number of processed frames $n$ exceeds capacity $K$, the $n$-th frame is admitted with probability $P = K/n$ and, if admitted, replaces a uniformly random buffered frame. This gives every frame of the event an equal inclusion probability, so the retained set is an unbiased summary rather than a recency-biased one.

**Structured long-term memory (LTM).** Each archived event is stored as a tuple
$$E'_i = \{\, I_i^{\text{first}},\; c_i,\; e_i,\; \Delta_i \,\},$$
with $I_i^{\text{first}}$ the first frame (visual anchor for later tool calls), $c_i$ an MLLM-generated natural-language caption, $e_i$ the caption's semantic embedding (for retrieval), and $\Delta_i$ a **change log** recording state transitions relative to the previous event.

**Multi-granular perception toolkit $T$.** The agent actively acquires evidence via:
- *Memory search* — **temporal retrieval** (filter events whose span intersects a queried window $[t_{\text{start}}, t_{\text{end}}]$) and **semantic retrieval** (max cosine similarity between query embedding $e_q$ and event embeddings $e_i$).
- *Specialized perception* — **OCR** (DeepSeek-OCR) for on-screen text and **object detection** (Grounding DINO) for entity localization, applied to LTM anchors or STM frames for fine-grained re-inspection.

**Agentic RL (the decision policy).** The agent runs ReAct-style trajectories $\tau = \{(t_1,a_1,o_1),\dots,(t_n,a_n,o_n)\}$ of Thought $t$ (task decomposition), Action $a$ (tool call or final answer), Observation $o$ (tool feedback). The tool/reasoning policy is optimized with **GRPO**: for query $q$, sample a group of trajectories $\{\tau_1,\dots,\tau_G\}$ from $\pi_{\theta_{old}}$ and maximize
$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^{G}\min\!\left(\frac{\pi_\theta(\tau_i\mid q)}{\pi_{\theta_{old}}(\tau_i\mid q)}A_i,\; \operatorname{clip}\!\left(\frac{\pi_\theta(\tau_i\mid q)}{\pi_{\theta_{old}}(\tau_i\mid q)}, 1-\epsilon, 1+\epsilon\right)A_i\right)\right],$$
with group-normalized advantage $A_i = (r_i - \operatorname{mean}(\mathbf{r}))/\operatorname{std}(\mathbf{r})$. The reward is a bare outcome signal on final-answer correctness:
$$r_i = \begin{cases} 1, & \text{if the predicted answer is correct} \\ 0, & \text{otherwise.} \end{cases}$$

## Explicit design choices
- **Backbone:** Qwen3-VL-8B-Instruct as the MLLM; detection = Grounding DINO; OCR = DeepSeek-OCR.
- **Sampling / budget:** 1 FPS stream sampling; STM max capacity = 32 frames; the whole system answers using only $\le 32$ frames at inference.
- **Event boundary metric:** Pearson correlation of normalized grayscale histograms; threshold $\delta = 0.2$.
- **Unbiased event summary:** reservoir sampling *within* an event (accept $n$-th frame w.p. $K/n$) rather than truncation/recency eviction.
- **LTM record = 4 fields:** visual anchor image + caption + embedding + change log — captions/embeddings enable semantic search; the anchor image enables later OCR/detection re-inspection; the change log encodes cross-event state deltas.
- **Frame metadata:** frame numbers and timestamps are drawn onto the images to improve spatiotemporal perception.
- **Agent paradigm:** ReAct (Thought/Action/Observation) with dynamic tool selection, not a fixed pipeline.
- **Training:** end-to-end GRPO with an outcome-only (0/1 correctness) reward — no separate critic, no process reward.
- **RL data:** 10K MovieChat samples labeled by VideoMarathon; 8 × 80G A100 GPUs.

## Key results / what to remember
Verified against the paper's Table 1 (OVO-Bench) and Table 2 (StreamingBench real-time part). EventMemAgent = 8B, $\le 32$ frames.

**OVO-Bench (Table 1), Overall accuracy:**
- EventMemAgent **60.75** (Real-Time Visual Perception 68.29 / Backward Tracing 58.03 / Forward Active Responding 55.92).
- Beats GPT-4o (59.54, 64 frames) and StreamForest-7B (55.57), and its own backbone Qwen3-VL-8B (55.81); StreamAgent-7B 49.40. Below Gemini 1.5 Pro (63.00, 1 fps) and far below Human (92.81).
- Reported gains over best open-source: +4.27% Real-Time Visual Perception, +1.08% Backward Tracing, +1.1% Forward Active Responding.

**StreamingBench real-time (Table 2), ALL:**
- EventMemAgent **77.00** (OP 83.92 / CR 76.56 / CS 87.70 / ATP 85.29 / EU 77.02 / TR 79.44 / PR 77.78 / SU 68.29 / ACP 72.24 / CT 48.70).
- Essentially ties StreamForest-7B (77.26), above TimeChat-Online (75.36), StreamAgent-7B (74.28), GPT-4o (n/r overall in this table — per-task; overall not directly comparable), and backbone Qwen3-VL-8B (70.20). Human 91.46.

**Ablations:**
- *Memory (Table 3):* hierarchical event memory vs fixed-length — OVO 60.75 → 60.16 (−0.59), StreamingBench 77.00 → 76.80 (−0.20). Small but consistent gain.
- *Perception tools (Table 4, OVO-Bench Overall):* removing detection 60.75 → 56.87 (−3.88); removing OCR → 57.88 (−2.87). Tools matter more than the memory-structure choice on this benchmark.

No Zotero highlights present.

Takeaways: (1) event-structured memory + adaptive tools let an 8B model reach ~GPT-4o-level OVO-Bench with ~32 frames instead of dense 1-fps ingestion; (2) the biggest lever is *active fine-grained perception* (OCR/detection), not the memory partitioning per se; (3) outcome-only GRPO is enough to teach *when* to call which tool.

## How it connects (evolution)
- [[streamforest]] — the strongest streaming baseline it ties on StreamingBench and beats on OVO-Bench; contrasts a memory-tree store against this event-agent design.
- [[streamagent]] — the closest sibling "online video agent" it consistently outperforms; both frame streaming QA as agentic tool use.
- [[dispider]] — active/disentangled perception-decision streaming model, an earlier "when to act" precursor to this ReAct-style agent.
- [[timechat-online]] — online model in the StreamingBench comparison; shares the timestamped-frame, answer-at-$t$ setting.
- [[flash-vstream]] / [[rekv]] — memory-compression streaming baselines that this paper's event memory + reservoir sampling is positioned against (passive vs active memory).
- [[ovo-bench]] / [[streamingbench]] — the two evaluation protocols; the task taxonomy (real-time / backward / forward) frames what "online" means here.

## Open questions / limitations
- **Memory structure gives a tiny margin** (−0.2 to −0.6 vs fixed-length); most of the reported edge comes from perception tools, so how much of the "event-centric" thesis is truly load-bearing is unclear.
- **Grayscale-histogram boundaries** are a cheap, appearance-only cue — brittle to gradual scene changes, camera motion, or lighting shifts that don't cross $\delta=0.2$ (and vice versa).
- **Latency/cost of tool loops** (OCR, detection, MLLM captioning per event, multi-turn ReAct) is not quantified against the real-time constraint the benchmarks target.
- **Outcome-only 0/1 reward** gives no credit assignment across a long tool trajectory; efficiency of tool use is only shaped indirectly.

*Verification: equations (1),(2),(4),(5), the reservoir rule, and the LTM tuple checked against the rendered page 3 (Methodology) and page 5 of arXiv:2602.15329; all headline numbers cross-checked against the rendered Table 1 (OVO-Bench, p.4) and Table 2 (StreamingBench, p.5). No GitHub/project page used.*

---
zotero_key: null
authors: Yulin Zhang, Cheng Shi, Sibei Yang (ShanghaiTech / Sun Yat-sen Univ / HKU)
year: 2026
arxiv: 2602.22142
pdf: https://arxiv.org/pdf/2602.22142
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---

# WeaveTime: Stream from Earlier Frames into Emergent Memory in VideoLLMs

**Lineage role:** Diagnoses "Time-Agnosticism" in streaming Video-LLMs and fixes it two ways — a training-time *Temporal Reconstruction* objective that makes representations order-aware, plus an inference-time *Past–Current Dynamic Focus Cache* that does uncertainty-triggered coarse-to-fine recall — turning an unordered "bag of frames" cache into an emergent, causally ordered memory (CVPR 2026).

## Problem — what was limited before this paper (short)
Video-LLMs are trained offline on whole clips with bidirectional attention, so at streaming time they treat the frame history as an *unordered bag of evidence* rather than a causally ordered sequence. The paper calls this **Time-Agnosticism** and shows it empirically: shuffling frame order barely dents (and sometimes even raises) accuracy, whereas humans collapse on the same temporal questions once timestamps are removed. Two concrete failures follow: **temporal order ambiguity** (the model can't reason over "before/after" among semantically similar segments) and **past–current focus blindness** (it can't tell a present observation from accumulated history, so it answers with stale past content or ignores relevant past).

## Key idea — the core insight
First *teach* order, then *use* order. WeaveTime is a lightweight, model-agnostic add-on with no architectural change. **Teach:** an auxiliary Temporal Reconstruction task — feed shuffled segments with explicit timestamps and ask the model to first restore the true chronological order, then answer — instilling order-aware representations with minimal LoRA finetuning and *no streaming-specific data*. **Use:** at inference the model answers from a short local window first, measures its own uncertainty (predictive entropy), and only when uncertain expands into history via a cheap coarse-to-fine retrieval over the KV cache. The net effect is an *emergent* structured memory rather than a flat, position-agnostic cache.

![[weavetime.png]]
> **Crux (Figure 1).** The two failure modes WeaveTime targets: *Temporal Order Ambiguity* (top — two histories that are near-identical in content but differ in order give opposite answers to "where are the orange flowers?") and *Past–Current Focus Blindness* (bottom — the model attends to the wrong point in time, answering about a past frame instead of the current one for "what color is the flower now?" / "where is the mirror?"). This diagnostic figure *is* the paper's argument that streaming needs causal order, not a bag of frames. *Zhang, Shi, Yang (2026), arXiv:2602.22142. Embedded for personal research reference.*

## Method + math — the mechanism, then the central objective/equations

**1) SOPE — Streaming Order Perception Enhancement (training).**
Given a video split into segments, WeaveTime prepends an explicit timestamp token before each segment's frame content, then *shuffles* the segment order while keeping the timestamps visible. The QA prompt is augmented with an instruction of the form "these segments are shuffled — list each segment's true time range," so the model must first emit the recovered chronological order and only then answer the original question. This is trained as ordinary next-token prediction — no new module, no custom loss — leveraging the LLM's native ability to reorder text. Formally, for shuffled segments $\{s_{\pi(1)},\dots,s_{\pi(M)}\}$ with permutation $\pi$ and target order labels $o = (o_1,\dots,o_M)$, the auxiliary term is a cross-entropy over the reconstruction tokens folded into the standard QA objective:

$$
\mathcal{L} \;=\; \underbrace{-\sum_t \log p_\theta\!\left(y_t \mid y_{<t}, \text{video}, q\right)}_{\text{QA}}
\;+\; \underbrace{-\sum_m \log p_\theta\!\left(o_m \mid o_{<m}, \text{shuffled video}, q\right)}_{\text{Temporal Reconstruction}}
$$

The reconstruction task encourages a "consistent, causal temporal manifold," converting the unordered cache into a structured chain.

**2) PCDF-Cache — Past–Current Dynamic Focus Cache (inference).**
*Look now, recall if needed.* When a query $q$ arrives at time $t$, the model first answers from only the local window of the last $C$ frames of memory $\mathcal{M}_{t-1}$:

$$
a_t^{(0)} \;=\; \mathrm{Answer}\!\big(\mathcal{M}_{t-1}[-C:],\, q\big)
$$

It then computes the predictive entropy of that draft answer and gates on it:

$$
H_t = \mathrm{Entropy}\!\big(a_t^{(0)}\big),\qquad
a_t =
\begin{cases}
a_t^{(0)}, & H_t < \delta \\[4pt]
\mathrm{Answer}\!\big(\mathrm{Load}_{C2F}(\mathcal{M}_t, q),\, q\big), & H_t \ge \delta
\end{cases}
$$

with entropy threshold $\delta = 0.6$. Low entropy ⇒ trust the present; high entropy ⇒ pull relevant history. This makes long-range recall *selective*, avoiding a history load on every query.

**Coarse-to-Fine (C2F) recall.** When triggered, retrieval is hierarchical to avoid exhaustive token-level matching over the whole cache. A **coarse** pass does frame-level similarity filtering to build a small candidate set $\mathcal{M}_{\text{coarse}}$; a **fine** pass then scores those candidates with a max-similarity late-interaction (ColBERT-style) score between query tokens $\{f_j^q\}$ and each candidate frame's visual tokens $\{f_{i,k}^v\}$:

$$
\mathrm{maxSim}\big(\{f_{i,k}^v\}, \{f_j^q\}\big) \;=\; \sum_{j=1}^{N_q} \max_{1 \le k \le N_i} \big\langle f_j^q,\, f_{i,k}^v \big\rangle
$$

Top-$K$ frames within the coarse set (by this score) are loaded for the final answer — token-level precision at frame-level cost. (A fine-only variant over the full cache runs out of memory, motivating the coarse gate.)

## Explicit design choices
- **Two-stage philosophy** "teach order, then use order": SOPE at training time, PCDF-Cache at inference time; the two are complementary, not alternatives.
- **No architectural change** — plug-and-play over existing Video-LLMs (evaluated on LLaVA-OV-7B and Qwen2-VL-7B backbones) and over an existing retrieval-cache baseline (ReKV).
- **Temporal Reconstruction as next-token prediction** — order recovery is folded into the QA conversation, so no separate optimization stage, custom loss, or extra module.
- **Data efficiency**: only ~30K *offline* video samples from LLaVA-Video-178K; zero streaming-specific data.
- **Cheap training**: single-epoch LoRA finetuning, learning rate $1\times10^{-5}$, 8 GPUs.
- **Uncertainty gating** via predictive entropy with a single threshold $\delta = 0.6$ — recall history only when the local answer is uncertain.
- **Coarse-to-fine retrieval**: frame-level coarse filter → late-interaction (maxSim) fine scoring on the candidate set → top-$K$ frames; keeps late-interaction precision without full-cache cost.

## Key results / what to remember
Numbers below are from the arXiv HTML rendering and the project page (see verification note — I could **not** cross-check them against the paper's own rendered PDF tables because the PDF served at this arXiv id was a different paper).

- **OVO-Bench (real-time), LLaVA-OV-7B over ReKV baseline:** 66.15% → **72.13%** (≈ +6.0 pts).
- **StreamingBench (real-time), LLaVA-OV-7B:** baseline 53.56% → **57.57%**.
- **Per-task gains reported:** Action Recognition +11.09%, Action Perception +7.56%, Event Understanding +9.04% (n/r — subsets not individually re-verified).
- **Ablation (LLaVA-OV-7B + ReKV):** naive timestamp finetuning alone *hurts* (−3.68%, distribution mismatch); **SOPE / Temporal Reconstruction +5.82%**; adding **PCDF-Cache +1.87%** further → 57.57% combined.
- **C2F recall (QAEgo4D / MLVU / EventHALL):** ReKV recall 23.9% / acc 54.3% → **+C2F 25.2% / 55.2%**; fine-only over full cache = out-of-memory.
- **Efficiency vs StreamForest:** WeaveTime uses 30K offline samples + 8 GPUs and matches gains that StreamForest gets with ~121K stream-tailored samples + 32 GPUs (~4× resources).
- **Entropy threshold:** $\delta = 0.6$ is the accuracy/latency sweet spot; latency drops monotonically as $\delta$ rises (fewer history loads); the paper claims WeaveTime improves accuracy *while reducing latency*.

No Zotero highlights present.

Takeaways: (1) streaming failure is largely a *representation* problem — models never learned causal order, so a tiny order-reconstruction task recovers most of it cheaply; (2) history retrieval should be *conditional* on the model's own uncertainty, not run every step; (3) coarse-to-fine + late-interaction is the practical way to get token-level recall precision without blowing memory on a growing KV cache.

## How it connects (evolution)
- [[rekv]] — the retrieval-augmented KV-cache baseline WeaveTime builds directly on and improves.
- [[livevlm]] — sibling streaming KV-cache + query-conditioned retrieval line (efficient online video understanding); same "cache + retrieve from history" family.
- [[streamforest]] — the training-heavy streaming baseline WeaveTime undercuts on data/compute (30K+8GPU vs 121K+32GPU).
- [[ovo-bench]] — primary real-time streaming benchmark used for the headline gains.
- [[streamingbench]] — second streaming benchmark (real-time visual understanding) reported.
- [[streaming-memory]] — sub-topic hub: emergent, order-aware memory over earlier frames.

## Open questions / limitations
- The Temporal Reconstruction task assumes coherent clips with preserved temporal anchors; on clips with weak visual anchors or heavy repetition, order recovery (and thus the taught prior) may degrade.
- Entropy gating uses a single fixed threshold ($\delta=0.6$); it is unclear how sensitive this is across backbones/benchmarks or how it behaves under distribution shift in the answer's token entropy.
- Gains are largest on order- and event-centric tasks; net benefit on tasks already solvable from the local window (where the "bag of frames" shortcut works) is smaller.
- Coarse-to-fine still relies on frame-level similarity to seed candidates — if the coarse filter drops the truly relevant frame, the fine pass cannot recover it (recall ceiling ~25% in the reported table).

*Verification: equations and mechanism checked against the arXiv HTML rendering + the official project page (zhangyl4.github.io/publications/weavetime) and the abstract on arXiv:2602.22142; headline numbers taken from the HTML summary and flagged (n/r) where not directly cross-checked — the PDF served at arxiv.org/pdf/2602.22142 was mis-served as a different paper (LiveVLM, arXiv:2505.15269), so table numbers could not be confirmed against a rendered PDF. Crux figure is the paper's real Figure 1 downloaded from the project page (WeaveTime_Fig1/teaser asset).*

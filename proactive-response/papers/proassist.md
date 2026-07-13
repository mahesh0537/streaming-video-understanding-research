---
zotero_key: null
authors: Yichi Zhang et al. (Meta + University of Michigan)
year: 2025
arxiv: 2506.05904
pdf: https://arxiv.org/pdf/2506.05904
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# Proactive Assistant Dialogue Generation from Streaming Egocentric Videos

**Lineage role:** Meta/UMich EMNLP-2025 work that builds PROASSIST — a large synthetic proactive task-guidance dialogue dataset from egocentric video streams — plus a matching evaluation protocol and a VideoLLM-Online-based assistant that must decide *when* and *what* to say frame-by-frame; a data+benchmark+model triple for [[proactive-response]] guidance.

## Problem — what was limited before this paper
A useful AR/wearable assistant has to watch a first-person video stream and volunteer help *at the right moment* without being asked — deciding both *when* to speak and *what* to say, while also answering user questions when they come. Existing multimodal LLMs are built for offline settings (the whole video is available up front), so they cannot judge response timing online. Proprietary streaming-capable models are too slow/expensive for per-frame decisions (deciding what to say at 2 FPS on a 30-minute video needs ~6000 API calls) and still struggle with timing. Crucially, there was **no training data** pairing egocentric task videos with proactive assistant dialogue, and **no metric** that jointly scores timing and content for streaming video-to-text.

## Key idea
Manufacture the missing supervision synthetically: run a strong text LLM (LLaMA-3.1-70B) over the dense human step/action annotations already present in six egocentric datasets to *hallucinate a plausible proactive assistant conversation* for each video, with controllable user talkativeness, then quality-filter it. This yields PROASSIST — 30,135 dialogues over 479 hours. Pair the data with (a) a *pairwise timing-aware precision/recall/F1* metric that matches predicted utterances to reference utterances under a combined semantic+temporal cost, and (b) an *LLM-as-judge* end-to-end score. Then adapt a lightweight streaming model (VideoLLM-Online) with two tricks — negative-frame sub-sampling to survive the speak/silent class imbalance, and iterative progress summarization to break the context-length wall on long videos.

![[proassist.png]]
> **Crux (Figure 3).** Left: the streaming assistant encodes each live frame into a few visual tokens, and at each "speaking decision point" (yellow star) emits either `[EOS]` (stay silent) or an autoregressive assistant utterance; positive speaking frames are rare, creating the imbalance NFS targets. Right: iterative progress summarization — when the context limit nears, the model writes a compact task-progress summary and restarts generation with that summary as the new system prompt. *Zhang et al. (2025), arXiv:2506.05904. Embedded for personal research reference.*

## Method + math

### Data construction pipeline (the "math" of this benchmark)
Source: six annotated egocentric datasets — **Ego4D-Goalstep, EpicKitchens, HoloAssist, Assembly101, EgoExoLearn, WTaG** (cooking, object manipulation, assembly, lab work). A five-stage LLM pipeline turns their step/action timestamps into dialogues:
1. **Task goal & recipe generation** — LLM summarizes the video's goal; generates 10 candidate "recipes" (ordered step lists) then distills to one.
2. **Video pre-filtering** — LLM classifies each video as (0) wrong domain, (1) follows a single procedure, (2) multitasking; only category-1 procedural videos are kept.
3. **Multi-round dialogue generation** — for each video, generate 10 dialogues conditioned on a **user behavior profile**: `no_talk` (silent except stating the goal), `talk_some` (task questions on ~20% of steps), `talk_more` (conversation on ~40% of steps), sampled roughly 2:4:4. Long videos are generated in chunks to respect context limits, then a refinement pass cleans them.
4. **Dialogue annotation** — LLM tags each assistant turn with **intent** (instruction / mistake-correction / feedback) and **response type** (responsive vs. proactive), and writes rolling **progress summaries**.
5. **Automatic QC & post-filtering** — an LLM judge scores timing precision, step coverage, and responsiveness; ~25% of dialogues (~41 hours) are dropped.

### Pairwise timing-aware metric
Predictions and references are two sets of timestamped utterances; score them via **bipartite (one-to-one) matching** that minimizes a combined cost, then compute Precision / Recall / F1 from the matched pairs. For predicted utterance $\hat{s}_j$ at frame $j$ and reference $s_i$ at frame $i$:

$$
\begin{aligned}
\text{cost}_{\text{sem}}(i,j) &= 1 - \cos\!\big(\mathrm{emb}(s_i),\,\mathrm{emb}(\hat{s}_j)\big) \\[4pt]
\text{cost}_{\text{time}}(i,j) &= \begin{cases} |i-j|^{p} & \text{if } i-j \in [-L,\,R] \\ \infty & \text{otherwise} \end{cases}
\end{aligned}
$$

The temporal window forbids matches outside $[-L, R]$ (setting $R < L$ rewards *early/prompt* predictions over late ones); $p$ tunes how sharply lateness is penalized. Only pairs finite under both terms can match. P/R/F1 then follow from #matched vs. #predicted (precision) and #matched vs. #reference (recall). The metric is dataset-agnostic — it applies to any streaming video-to-text task with timestamped ground truth (e.g. online action narration).

### End-to-end LLM-as-judge
Claude scores a full generated dialogue on a **5-point Likert scale** across four axes — **correctness** of guidance, **promptness/appropriateness of timing**, **efficiency** of information delivery, and **overall helpfulness** — averaged over three independent runs.

### Assistant model
Base: **VideoLLM-Online** with a **LLaMA-3.1-8B-Instruct** backbone and a **frozen SigLIP-SO400M** frame encoder + trainable projector; each frame becomes only $I \in \{1,5,10\}$ visual tokens for real-time throughput. At the last visual token of each frame the model does a binary decision: emit `[EOS]` (silent) or start generating an assistant turn. Two additions:
- **Negative Frame Sub-sampling (NFS).** Speaking frames are sparse, so cross-entropy on every frame biases the model toward silence. NFS computes the loss only on positive (speaking) frames plus a uniformly sampled fraction $\rho$ of negative frames; best $\rho = 0.1$.
- **Iterative Progress Summarization (IPS).** When context fills up, the model generates a **progress summary** — (1) time elapsed, (2) task goal, (3) completed steps, (4) other discussed topics, (5) current activity — then continues generation with that summary injected into a fresh system prompt. Requires no special training and handles arbitrarily long videos.

At inference a per-subset **speaking threshold $\theta$** on the response probability trades precision vs. recall (swept over 0.1–0.9).

## Explicit design choices
- **Synthetic-from-annotations data engine**: no new video collection; reuse dense human step/action labels and let a 70B LLM write the dialogue — cheap and scalable, at the cost of the LLM never seeing the pixels.
- **Controllable user talkativeness** (`no_talk`/`talk_some`/`talk_more`, ~2:4:4) so the model learns both purely proactive and mixed responsive+proactive behavior.
- **Intent + response-type annotations** (instruction / mistake-correction / feedback; proactive / responsive) baked into every turn for fine-grained analysis.
- **Aggressive QC filter** (~25% of dialogues, 41h dropped) using an LLM judge on timing/coverage/responsiveness.
- **Two-task evaluation**: simpler *action narration* (narrate what's happening) vs. harder *dialogue generation* (full proactive conversation), both scored with the same pairwise metric.
- **Compact frame budget** (1–10 visual tokens/frame) to keep per-frame decisions real-time — far fewer than mainstream MLLMs.
- **Timing baked into the loss and the metric**, not just content — the model is graded on *when* as well as *what*.
- **Progress-summary system prompt** as the long-horizon mechanism instead of retraining for longer context.

## Key results / what to remember
No Zotero highlights present.

- **Dataset scale**: 30,135 dialogues, 479 hours, six egocentric domains (Table 1).
- **Data quality (human eval, 4-point scale, Table 2)**: correctness **3.27±0.79**, helpfulness **3.46±0.77**, alignment **2.91±1.00**, naturalness **3.54±0.70** — the synthetic dialogues scored *higher* than real human HoloAssist dialogues (correctness 2.88, helpfulness 2.62).
- **Metric–human agreement**: pairwise F1 matched human choice of speaking threshold at **0.80** for action narration and **0.67** for dialogue generation (Table 4).
- **Model performance (I=5, with task knowledge, Table 5)**: Action Narration P/R/F1 = **62.81 / 61.12 / 61.96**; the harder Dialogue Generation P/R/F1 = **44.24 / 31.52 / 36.25**, with LLM-judge scores correctness **2.50**, promptness **2.78**, efficiency **2.31**, overall **2.41** — dialogue is markedly harder than narration.
- **NFS ablation (Table 6, ρ=0.1)**: action-narration F1 jumps **30.1 → 58.7**; dialogue F1 **32.9 → 34.4** — the imbalance fix is essential for narration timing.
- **IPS vs. drop-middle context handling (Table 7)**: IPS P/R/F1 = **49.6 / 25.9 / 32.9** vs. drop-middle F1 **25.7** — progress summaries beat naively truncating context.
- **Per-domain (Table 8)**: best overall on **WTaG (2.67)**, worst on **EgoExoLearn (1.93)** — performance tracks how procedural/clean the source annotations are.
- **Takeaway**: proactive dialogue is genuinely harder than narration; the gains come from *timing-aware* supervision (NFS) and *long-horizon* handling (IPS), and a 70B LLM writing over dense annotations is a viable data engine when no proactive-dialogue ground truth exists.

## How it connects (evolution)
- [[videollm-online]] — the exact streaming base model ProAssist adapts (per-frame speak/`[EOS]` decision, compact frame tokens).
- [[proactive-response]] — the sub-topic hub: ProAssist is a data+benchmark+model contribution to when-to-speak proactive assistance.
- [[proactivevideoqa]] — sibling benchmark specifically for proactive/streaming QA behavior.
- [[mmduet]] — related streaming dialogue model that also learns response timing on video streams.
- [[egospeak]] — egocentric "when to speak" prediction, the timing decision in a narrower framing.
- [[streaming-benchmarks]] — the benchmark family this evaluation protocol belongs to.

## Open questions / limitations
- **Synthetic-only supervision**: the 70B LLM writes dialogue from *text annotations*, never the pixels — so hallucinated guidance can be visually ungrounded, and the ceiling is set by annotation quality (EgoExoLearn worst).
- **Judge circularity**: both QC filtering and end-to-end scoring use LLM judges; absolute Likert scores (~2.4/5 overall on dialogue) are hard to calibrate and may inherit judge biases.
- **Timing metric hyperparameters** ($p, L, R$) and per-subset thresholds $\theta$ are hand-tuned; agreement with humans is only moderate (0.67) for full dialogue.
- **Small backbone / short reasoning**: an 8B model with 1–10 tokens/frame limits fine-grained visual verification of user actions (e.g. mistake detection), and IPS summaries can lose detail across restarts.

*Verification: metric formulas, dataset scale (30,135 dialogues / 479h / 6 domains), NFS ρ=0.1, IPS vs drop-middle, and Table 2/4/5/6/7/8 headline numbers checked against the arXiv HTML (arxiv.org/html/2506.05904) and the rendered PDF Figure 3 (page 5); author/affiliation list from the PDF front matter.*

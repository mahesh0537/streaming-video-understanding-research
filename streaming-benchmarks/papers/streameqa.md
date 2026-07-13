---
zotero_key: null
authors: Yifei Wang, Zhenkai Li, Tianwen Qian, Huanran Zheng et al. (East China Normal University; Zhejiang Univ. of Technology; INSAIT / Sofia University)
year: 2025
arxiv: 2512.04451
pdf: https://arxiv.org/pdf/2512.04451
tier: deep
subtopics: [streaming-benchmarks]
tags: [streaming-video-understanding, streaming-benchmarks]
---
# StreamEQA: Towards Streaming Video Understanding for Embodied Scenarios

**Lineage role:** The first benchmark to cross the **embodied cognition axis** (Perception → Interaction → Planning) with the **streaming/temporal axis** (Backward → Real-time → Forward), yielding ~21K timestamped, query-time-anchored QA pairs over egocentric video that expose how badly current MLLMs — including online streaming ones — fail at situated, forward-looking reasoning.

## Problem — what was limited before this paper (short)
Existing streaming-video QA benchmarks (e.g. [[ovo-bench]], [[streamingbench]], [[svbench]]) test *general* online understanding — third-person or web video, "what is happening now" style questions — and treat the video as a passive stream to describe. They do not test **embodied** competence: an egocentric agent that must perceive its own hands/objects, track its own manipulations, and *plan next actions*. Conversely, embodied/egocentric QA sets are typically **offline** (the model sees the whole clip). No benchmark had jointly stressed (a) egocentric embodied reasoning and (b) the streaming constraint that a question is asked *at a moment in time* and can point backward, at the present, or into the future.

## Key idea — the core insight, 2-4 sentences
Situated intelligence is defined by *when* you ask relative to *what the agent is doing*. StreamEQA builds a 3×3 task matrix: three **embodied levels** (Perception, Interaction, Planning) × three **temporal scopes** (Backward = recall what happened, Real-time = read the present state, Forward = anticipate future events/actions/risks). Each question is anchored to a query timestamp $t_q$ so the model only receives the video **up to that instant**, faithfully simulating a streaming embodied assistant. Evaluating 13 MLLMs shows a steep difficulty gradient — Perception > Interaction > Planning, and Backward/Real-time > Forward — with the best model far below human-usable accuracy, revealing that "online" and "egocentric" training do not yet buy embodied foresight.

![[streameqa.png]]
> **Crux (Figure 3).** The full StreamEQA task taxonomy: three embodied levels (Perception 31.3%, Interaction 38.6%, Planning 30.1%) each split into Backward/Real-time/Forward, giving 42 named sub-tasks (e.g. Action Retrieval, Risk Prediction, Next Action Prediction), with per-scope QA-count distributions and representative four-choice examples — it *is* the benchmark's design. *Wang et al. (2025), arXiv:2512.04451. Embedded for personal research reference.*

## Method + math — eval protocol, taxonomy, metric, data pipeline
**Streaming input construction.** For an event beginning at timestamp $t_e$ and a question $Q_i$ posed at query time $t_q$, the model is given only the clip
$$V_i = \text{Video}[\,t_e : t_q\,],$$
i.e. from the event onset up to the moment of asking — never the future. This simulates online/streaming inference while trimming irrelevant early context. The query timestamp is deliberately set **after** the relevant event so that Backward tasks genuinely require "backward tracing" rather than trivially reading the last frame. Offline video-LLMs sample 64 frames from $V_i$; proprietary models (GPT-5, Gemini-2.5-Pro) are fed at 1 fps. Frame/fps budgets are held constant across a model class for fairness.

**Task taxonomy (the "math" of a benchmark).** The label space is the Cartesian product
$$\{\text{Perception},\ \text{Interaction},\ \text{Planning}\}\times\{\text{Backward},\ \text{Real-time},\ \text{Forward}\}$$
= 9 major categories, refined into **42 sub-tasks**. Perception = recognize entities/attributes/spatial relations; Interaction = track actions and ongoing manipulations (retrieval, recognition, intention, result/next-action prediction, repetition counting); Planning = reflect on / optimize / revise procedures and predict future steps and risks (e.g. *Risk Prediction*, *Forward Plan Adjustment*, *Steps Merging*).

**Metric.** Every item is a single-round, zero-shot, four-choice (A/B/C/D) MCQ. The only metric is **standard accuracy**,
$$\text{Acc} = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}\!\left[\hat{a}_i = a_i^{\star}\right],$$
the fraction of questions whose predicted option $\hat a_i$ equals the gold option $a_i^\star$. Random-guess baseline is 25%. Category/scope scores are the same accuracy restricted to that slice; "Avg" is the mean over the level's Backward/Real-time/Forward accuracies.

**Data-construction pipeline (three stages).**
1. **Meta-information extraction.** Source video is **HD-EPIC** egocentric footage with dense narrations, event time ranges, eye-gaze priming, and object spatial tracks. GPT-5 fuses these into structured meta-elements — *objects, actions, object episodes, object relationships, action motivations, event durations* — each aligned to precise timestamps (what appears, why it happens, how it works).
2. **QA construction.** A hand-designed schema of per-sub-task templates has placeholders filled with the extracted meta-info; the correct answer is composed from the grounded action template, and 3 distractors are generated by randomly swapping actions/objects from the meta pool (same context, wrong details), forming the four-choice item. Guarantees each QA is grounded in real visual+linguistic content.
3. **Quality control.** GPT-5 refines the distractors to be *logically plausible yet ultimately contradicted by the video* (leaving the correct option untouched), then human annotators verify answer correctness and check for question clarity / option ambiguity. Output = validated (question, options, correct answer) triplets with precise temporal boundaries.

## Explicit design choices
- **Two orthogonal axes**, not a flat task list: embodied level × temporal scope — makes the difficulty gradient measurable and diagnosable.
- **Query-timestamp anchoring** with input clip $V[t_e:t_q]$ — the streaming constraint is baked into the data, not just the inference harness; query time set *after* the event to force real temporal reasoning.
- **Egocentric HD-EPIC source** (first-person hands/objects, gaze, spatial tracks) so questions are genuinely embodied, not observer-view.
- **Template + LLM-refined distractors** grounded in extracted meta-info → hard, plausible negatives instead of random wrong answers; human-verified.
- **Four-choice MCQ + plain accuracy** — cheap, deterministic, model-agnostic scoring; avoids LLM-judge noise.
- **Fixed frame/fps budget per model class** for fair cross-model comparison; 64 frames (open) / 1 fps (proprietary).
- **Broad evaluated set spanning three paradigms**: proprietary (GPT-5, Gemini-2.5-Pro), offline video-MLLMs (Qwen3-VL, LongVA, MiniCPM-V-4.5, InternVL3, VideoLLaMA3), online/streaming MLLMs ([[flash-vstream]], [[videollm-online]], [[dispider]], [[timechat-online]]), and egocentric MLLMs (EgoGPT, EgoVLPv2).
- **Scale/balance:** 156 videos, 20,731 QA, 42 sub-tasks, roughly balanced across the three embodied levels (31.3% / 38.6% / 30.1%).

## Key results / what to remember
No Zotero highlights present.

Verified against Tables 1–3 and Figure 3 of the paper:
- **Scale:** 156 videos, **20,731 QA** (~21K), **42 sub-tasks**. Perception 6,500 (31.3%), Interaction 7,996 (38.6%), Planning 6,235 (30.1%); Backward 5,736 / Real-time 5,996 / Forward 8,499. Avg clip duration 67.55 s (Table 1).
- **Overall difficulty (Table 2, average accuracy):** best model **GPT-5 = 61.3%**; EgoGPT 60.9%, Gemini-2.5-Pro 60.0%, Qwen3-VL 51.7%; lowest **EgoVLPv2 = 23.7%** (below 25% chance). Most mainstream models cluster near 40%, i.e. little margin over the 25% random baseline.
- **Embodied gradient (Table 2 / text):** Perception > Interaction > Planning. Qwen3-VL: **Perception 63.0% → Interaction 49.9% → Planning 42.0%**. EgoGPT (embodied-trained) is strongest on Interaction (60.5%) and Planning (55.8%), nearly matching GPT-5.
- **Online models underperform on embodiment:** the four online streaming MLLMs are relatively weak overall (VideollmOnline 34.1%, TimeChatOnline 36.9%, Dispider 42.8%, FlashVStream 48.0% avg), showing online-video training ≠ embodied competence. Curiously, FlashVStream does *better* on Planning (45.2%) than Interaction (40.7%).
- **Temporal gradient (Fig. 4):** Backward and Real-time scores are higher and more stable; **Forward is hardest and most variable** — models struggle to anticipate future actions/risks.
- **Embodiment penalty (Fig. 5, Qwen3-VL, OVO-Bench → StreamEQA):** Real-time drops **74% → 54.26%**; Forward drops **51.51% → 41.18%** — a large, consistent penalty for making tasks embodied (~1.6× harder overall vs general streaming QA).
- **Streaming-input penalty (Table 3, EgoGPT offline vs online):** offline (full event clip) Avg **62.03%** → online (streaming clip) Avg **60.93%**, a ~1.10% reduction (Backward 65.7→65.29, Real-time 65.83→63.54, Forward 57.22→56.40) — the streaming constraint adds a modest but consistent extra cost on top of the embodiment gap.

## How it connects (evolution)
- [[ovo-bench]] — the general online/streaming QA benchmark StreamEQA directly compares against to quantify the embodiment penalty (74%→54.26% real-time).
- [[streamingbench]] — earlier general streaming-video benchmark; StreamEQA is the embodied/egocentric specialization of this line.
- [[svbench]] — streaming video benchmark with temporal linkage; sibling in the streaming-eval family, non-embodied.
- [[egopro-bench]] — egocentric/embodied benchmark; shares the first-person source and embodied-reasoning framing but not the streaming query-time axis.
- [[omnimmi]] — streaming multimodal interaction benchmark; related on the real-time/interaction axis.
- [[streaming-benchmarks]] — the sub-topic hub mapping how streaming-video eval evolved toward embodied, timestamp-anchored tasks.

## Open questions / limitations
- **Single source dataset (HD-EPIC).** All videos are kitchen-centric egocentric footage; generality to other embodied domains (navigation, assembly, outdoor) is untested.
- **LLM-in-the-loop construction.** Meta-info extraction and distractor refinement both rely on GPT-5, so the benchmark's "ground truth" and hard negatives inherit GPT-5's biases despite human verification.
- **MCQ ceiling.** Four-choice accuracy rewards elimination and can mask whether a model *actually* reasons; open-ended or action-execution eval would probe embodied planning more directly.
- **Modest streaming penalty (~1.10%)** vs large embodiment penalty (~1.6×) suggests most difficulty is embodied semantics, not the online constraint per se — the "streaming" framing may be under-stressed by the current clip-cropping protocol.

*Verification: all statistics and headline numbers cross-checked against the rendered PDF — Table 1 (scale/durations), Table 2 (per-model per-level accuracies, GPT-5 61.3 / EgoVLPv2 23.7), Table 3 (EgoGPT offline 62.03 vs online 60.93), Figures 3–5 (taxonomy, temporal variance, OVO-Bench 74→54.26 / 51.51→41.18); metric formula and $V[t_e:t_q]$ protocol from Sec. 3.4/4.1 text.*

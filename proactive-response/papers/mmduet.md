---
zotero_key: null
authors: Yueqian Wang et al. (Wangxuan Institute, Peking University; Huawei Noah's Ark Lab; BIGAI)
year: 2024
arxiv: 2411.17991
pdf: https://arxiv.org/pdf/2411.17991
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# VideoLLM Knows When to Speak: Enhancing Time-Sensitive Video Comprehension with Video-Text Duet Interaction Format

**Lineage role:** Recasts streaming video-LLM interaction as a *video-text duet* (video plays continuously as a turn-taking participant) and adds two dedicated per-frame classification heads (informative + relevance) so the model decides *when to speak* frame-by-frame — trained on the purpose-built MMDuetIT data, including the novel MAGQA multi-answer grounded QA task.

## Problem — what was limited before this paper (short)
Most VideoLLMs use a *whole-video interaction format*: the entire clip (all sampled frames) is fed in, then the user asks, then the assistant answers once. This blocks real-time/live use (the video must "end" before answering), so it fails at time-sensitive tasks (temporal grounding, highlight detection, dense captioning, streaming QA) where a response must be tied to a specific moment and emitted *as the video plays*. Prior proactive approaches (e.g. VideoLLM-Online) decide when to talk by predicting a single special token from the LM head, which lacks a principled per-frame training label and is hard to control.

## Key idea — the core insight, 2-4 sentences
Treat the video stream as a conversation participant that alternates turns with the user and the assistant: frames are streamed in continuously, and after *every* sampled frame the assistant may seize the floor and respond. To make "when to speak" a well-posed, trainable decision, the model computes two per-frame binary scores — an **informative score** (does this frame add new information worth captioning?) and a **relevance score** (is this frame related to the user query?) — from two small heads bolted onto the backbone, and a task-specific `need_response` rule thresholds these scores to trigger generation. Because responses can be inserted where the video content is most relevant, the model also learns to *ground* answers to a smaller, fine-grained slice of preceding video.

![[mmduet.png]]
> **Crux (Figure 1).** Contrasts the standard *Whole Video Interaction Format* (user must supply all text+video before the assistant responds once) with the proposed *Video-Text Duet Interaction Format* (video keeps playing; after each frame the assistant decides whether to start a response, so replies are emitted in real time and grounded to specific moments). *Wang et al. (2024), arXiv:2411.17991. Embedded for personal research reference.*

## Method + math — mechanism, then the objective
**Backbone + heads.** MMDuet is initialized from LLaVA-OneVision (7B). Two linear classification heads are added alongside the LM head, each a linear layer of weight shape $h \times 2$ (with $h$ the LLM hidden size), reading the hidden state of each sampled video frame's token(s):
- **Informative head** — predicts $\Pr(\text{TRUE})$ = how much *new* information the current frame carries. Its TRUE probability is the **informative score** $I_t \in [0,1]$.
- **Relevance head** — predicts $\Pr(\text{TRUE})$ = whether the current frame relates to the user query. Its TRUE probability is the **relevance score** $R_t \in [0,1]$.

Contrast with VideoLLM-Online, which decides via one special token from the LM head. The authors argue their two-head design gives (1) better ground-truth labels derived from the video itself rather than ad-hoc response positions, (2) flexible multi-criterion triggering by combining two scores, and (3) a relevance head reusable for temporal grounding / highlight detection.

**Inference / turn-taking (`need_response`).** Streaming loop, one sampled frame at a time: first inject any pending user turn; then feed the frame, update the KV cache, and compute $I_t, R_t$; then a task-specific `need_response` decides whether the assistant generates now:

$$
\begin{aligned}
\textbf{Dense captioning:} \quad & \text{respond when } \sum_{\tau} I_\tau \ge s \ \ (s=2\text{ default; robust for } s\in[1,3])\\
\textbf{MAGQA (streaming QA):} \quad & \text{respond when } I_t + R_t > t \ \ (t\in\{0.3,0.4,0.5,0.6\})\\
\textbf{Temporal grounding / highlight:} \quad & \text{use } R_t \text{ (smoothed over window } w), \ w=2\ (\text{QVHighlights}),\ w=6\ (\text{Charades-STA})
\end{aligned}
$$

For dense captioning the running informative-score sum resets after each emitted response. Everything runs incrementally via KV-cache updates on each new frame/text.

**"rm. prev. resp." trick.** VideoLLMs badly over-repeat prior captions. Because MMDuet's outputs are split across turns, they simply *remove previously generated assistant turns from the context* (keeping their attention K/V already in cache) — this cuts repetition and lifts CIDEr/SODA_c substantially. The trick is not applicable to whole-video baselines (removing latest words leaves the cache unchanged, so they regenerate the same text).

**Training.** Only the projector, the two heads, and LoRA weights are trained; the rest is frozen. One epoch, 8× Tesla V100, ~1 day; LR 2e-5, batch 1, grad-accum 8, up to 120 frames per video. (The paper does not write out explicit loss equations; heads are binary classification over per-frame TRUE/FALSE labels, jointly with LM cross-entropy on the inserted response text.)

**MMDuetIT data construction (109k examples).** Annotations of existing datasets are *reformatted* into duet turns; the informative/relevance labels come from the reformatting:
- **Dense captioning** — Shot2Story (43k human-annotated) + COIN (2–4 min videos). For each captioned segment, randomly sample an insertion position between 50% and 75% of the segment's duration and place the caption there as a model turn; set informative-head label = TRUE for frames between 50% of the segment and the insertion point, FALSE elsewhere (see Figure 2).
- **MAGQA (Multi-Answer Grounded Video QA)** — *novel task.* One question maps to *multiple* time-localized answer turns from different segments (vs conventional single-answer VideoQA). Dataset **Shot2Story-MAGQA-39k** (36,834 train + 2,000 test), questions generated by GPT-4o-2024-08-06 over one-or-more captions; same insertion method as dense captioning; >95% of a manually checked sample judged high quality.
- **Temporal video grounding** — DiDeMo, HiREST_grounding, QuerYD, used *only* to train the relevance head (TRUE for annotated relevant frames).

**In-span score (MAGQA metric).** Both correctness *and* timeliness must be scored. For each gold answer $q$ with time span $[\text{start}_q, \text{end}_q]$, gather the set $P_q$ of predicted answers whose emission time falls inside that span; an LLM scores each prediction–gold similarity $s_{p,q}\in[1,5]$; average within the span then over all gold answers:

$$
\text{score}_q = \frac{1}{|P_q|}\sum_{p\in P_q} s_{p,q}, \qquad
\text{in\_span\_score} = \frac{1}{|Q|}\sum_{q\in Q}\text{score}_q
$$

The paper reports the score with both GPT-4o-2024-08-06 and LLaMA-3.1-70B-Instruct as the judge (to guard against API drift). Whole-video baselines are forced to emit (start, end, text) triples after seeing the entire video, then the mid-span time is used as their "response time".

## Explicit design choices
- **Duet chat template**: three roles (user, assistant, *video*); when a user/assistant turn ends, the video role may take the floor and stream frames until interrupted.
- **Two classification heads** (informative, relevance) instead of one LM special token — trained on video-derived labels, not ad-hoc response positions.
- **Task-specific `need_response` rules**: sum-of-informative threshold $s$ for captioning; $I_t+R_t>t$ for MAGQA; smoothed relevance for grounding/highlight (window $w$).
- **Reset informative sum after each response** in dense captioning.
- **"rm. prev. resp."** — drop past assistant turns from context to kill repetition (keeps their KV).
- **Reformat existing datasets** into duet turns; insertion position randomized in 50–75% of segment (ablation shows this matters).
- **Multi-informative labeling**: mark a *range* of frames (50%→insertion) as informative, not just the single frame before the response.
- **Novel MAGQA task + Shot2Story-MAGQA-39k** to simulate streaming multi-answer QA; **in-span score** metric for correctness+timeliness.
- **Cheap training**: LoRA + heads + projector only, 109k examples, ~1 day on 8×V100; low-cost recipe intentional.

## Key results / what to remember
Backbone LLaVA-OneVision-7B; baselines include TimeChat, VTG-LLM, and two LLaVA-OV whole-video variants (LLaVA-OV-TC / LLaVA-OV-VT). All zero-shot unless noted.

- **Highlight detection — QVHighlights (mAP / HIT@1):** MMDuet **31.3 / 49.6** vs LLaVA-OV-VT 19.0 / 40.0, VTG-LLM 16.5 / 33.5, TimeChat 14.5 / 23.9. (Table 1)
- **Temporal grounding — Charades-STA (R@IoU=0.5 / R@0.7):** MMDuet **42.4 / 18.0** vs LLaVA-OV-VT 36.5 / 12.3, VTG-LLM 33.8 / 15.7. (Table 1)
- **Dense captioning — YouCook2 (SODA_c / CIDEr / F1):** MMDuet **2.4 / 5.7 / 19.2**; higher CIDEr/SODA_c than baselines, F1 not a clear win (start/end times derived crudely from responses). (Table 1)
- **MAGQA — Shot2Story-MAGQA-39k, in-span score (LLaMA / GPT judge):** MMDuet $t{=}0.3$ **3.13 / 2.93** (27.0 turns) vs LLaVA-OV-TC 2.77 / 2.64, LLaVA-OV-VT 2.54 / 2.42, VideoLLM-Online 1.33 / 1.26. Lower $t$ → higher score but more (duplicate) turns and ~2.9× inference time. (Table 2)
- **MAGQA on 5×-prolonged videos:** MMDuet $t{=}0.3$ **2.63 / 2.45** — the gap over baselines *widens* on long videos (baselines struggle to localize answer times). (Table 2)
- **StreamingBench Proactive Output (Acc; reply must land within 2s of the answering scene):** MMDuet **29.44** vs Dispider 25.34, VideoLLM-Online 3.92, Flash-VStream 1.96. (Table 3)
- **Ablation — YouCook2 (SODA_c / CIDEr / F1):** full 2.9 / 8.8 / 21.7; w/o random response position 2.1 / 7.3 / 19.0; w/o multi-informative 2.9 / 8.0 / **16.5** — both labeling choices matter, especially for F1. (Table 4)

No Zotero highlights present.

Takeaways: (1) the win is *interaction format*, not scale — a duet chat template plus two cheap per-frame heads unlocks real-time, moment-grounded replies from a standard 7B VideoLLM; (2) making "when to speak" a per-frame classification problem with video-derived labels beats a single LM special token; (3) the "rm. prev. resp." context trick is a simple, format-specific fix for the chronic caption-repetition problem; (4) MAGQA + in-span score give a streaming-native way to measure timeliness, and the advantage grows on longer videos.

## How it connects (evolution)
- [[videollm-online]] — the prior proactive baseline that decides when to speak via one LM special token; MMDuet directly critiques and replaces this with two labeled heads.
- [[dispider]] — contemporaneous streaming/proactive VideoLLM; the strongest StreamingBench Proactive-Output baseline MMDuet beats (29.44 vs 25.34).
- [[proactive-response]] — sub-topic hub; MMDuet is a core "when to speak" datapoint here.
- [[streamingbench]] — the benchmark whose Proactive Output task MMDuet reports on.
- [[streammind]] — related streaming proactive-response line (when-to-respond gating).

## Open questions / limitations
- **Single question assumption in MAGQA eval**: the user asks one question at the video start; multi-question / interleaved-query streaming is left as future work.
- **Duplicate replies at low $t$**: real-time gains come with many redundant turns and higher inference cost; no learned stopping, only a hand-set threshold.
- **Crude temporal spans for captioning**: start/end times are inferred from response positions, hurting F1-style metrics that need accurate boundaries.
- **Small data / frozen backbone**: 109k reformatted examples and LoRA-only training were chosen for cost; scaling behavior and out-of-distribution robustness are untested.

*Verification: equations and all numbers checked against the paper's own rendered pages — Figure 1 (page 2), Figure 2 + method text (page 4), Section 6 / in-span-score text (page 7), Table 2 & Table 3 & Table 4 (page 8); Table 1 values cross-read from the arXiv HTML. No external project page used.*

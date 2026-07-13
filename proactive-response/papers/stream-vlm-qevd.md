---
zotero_key: null
authors: Sunny Panchal, Apratim Bhattacharyya et al. (Qualcomm AI Research)
year: 2024
arxiv: 2407.08101
pdf: https://arxiv.org/pdf/2407.08101
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# What to Say and When to Say It: Live Fitness Coaching as a Testbed for Situated Interaction

**Lineage role:** The earliest work to explicitly split proactive video assistance into *what to say* (content fluency) **and** *when to say it* (temporal alignment), backing it with the QEVD live-fitness-coaching benchmark + the streaming action-token baseline Stream-VLM (NeurIPS 2024 Datasets & Benchmarks).

## Problem — what was limited before this paper (short)
Video-language models were **turn-based**: they wait for an explicit user prompt, then answer about a (mostly finished) clip. Real situated assistance — a coach watching you exercise — is **asynchronous and proactive**: the assistant must decide *on its own initiative* both the content of feedback and the exact moment to deliver it, often staying silent for seconds. No benchmark isolated the "when to speak" decision as a scored quantity, and no model architecture could opt to stay silent while continuously observing a live stream. Existing datasets (Ego-4D, HowTo100M, HoloAssist, WTAG, etc.) lacked the combination of fine-grained human actions, a coach-participant interactive dyad, corrective feedback, and explicit user mistakes.

## Key idea — the core insight, 2-4 sentences
Situated interaction has two separable subproblems — **content** ("what") and **timing** ("when") — and both must be measured. The paper contributes (1) **QEVD**, a fitness-coaching benchmark with temporally-anchored live feedback (initiation, correction, encouragement, counting, closure) and a **Temporal F-Score** that scores whether feedback fires in the right ±3 s window, and (2) **Stream-VLM**, a streaming baseline whose LLaMA-2 backbone emits special **action tokens** — `<next>` to request the next video frame (stay silent) and `<feedback>` to start speaking — so the model learns *when* to talk end-to-end rather than being prompted at fixed intervals.

![[stream-vlm-qevd.png]]
> **Crux (Figure 3).** Stream-VLM's architecture: a causal 3D CNN encodes the live visual stream frame-by-frame, features are fused into a LLaMA-2-7B via interleaved cross-attention layers, and the LM autoregressively emits either `<next>` (observe another frame, say nothing) or `<feedback>` (begin generating the spoken coaching response, e.g. "Smooth on the way down…"). This single decode loop is what unifies *when* and *what*. *Panchal, Bhattacharyya et al. (2024), arXiv:2407.08101. Embedded for personal research reference.*

## Method + math — mechanism, then the eval protocol in full

### Stream-VLM model (the baseline)
- **Vision backbone.** A publicly-available **3D CNN** (mixed 2D/3D conv layers) chosen over CLIP/ViT encoders because coaching needs *motion* cues, not just scene content. Its convolutions are **causal** (only past frames), which is what makes streaming inference possible. Pre-trained on ImageNet, then fine-tuned on the QEVD-FIT-300K fine-grained exercise labels.
- **Fusion.** Vision features are injected into the LM at several layers via **cross-attention adapters** (methodology following the Flamingo/gated-xattn lineage). Self-attention layers alternate with cross-attention layers inside the language stack.
- **Language backbone.** LLaMA-2-7B, adapted with **LoRA (dim = 32)** in stage 3.
- **Action tokens.** Two special tokens make proactivity end-to-end trainable:
  - `<next>` — "I have nothing to say yet; give me the next frame." (observation mode)
  - `<feedback>` — "Start speaking now," after which the LM decodes the natural-language feedback string.
  This lets one autoregressive loop switch between *observe* and *respond* with no external prompting heuristic; the model learns *when* to speak conditioned on *what* it has observed.

**Three-stage training.**
1. **Vision pre-training** — 3D CNN on ImageNet → fine-tune on QEVD-FIT-300K fine-grained action classification (cross-entropy).
2. **Alignment** — end-to-end on QEVD-FIT-300K questions + feedbacks (short clips, no long-range videos); **only the cross-attention adapter is updated**, 3D CNN frozen, to align the LM with the recognizer's action features.
3. **Interactive fine-tuning** — LoRA on LLaMA-2 over long-range QEVD-FIT-COACH videos with feedbacks interleaved into the token stream; training is limited to **30-second single-exercise segments** (multi-exercise workouts left as future work). 3D CNN and adapter kept frozen.

### QEVD benchmark (the "math" of a benchmark = its protocol)
**Data.**
- **QEVD-FIT-300K** (short clips, training data): ~281,660 train / ~16,429 test videos, avg 5.6 ± 1.1 s, 148 exercises (~10 variations each → 1,842 fine-grained classes), 1,800+ crowd participants, 460+ hours. Annotations: fine-grained activity labels; **fitness questions** (high-level "what exercise?" ~535K and fine-grained "is the squat shallow?" ~378K QA pairs, semi-auto-generated with Mixtral-8B-Instruct); **fitness feedbacks** (~2.0 per short clip, end-of-clip).
- **QEVD-FIT-COACH** (long-range benchmark): 149 train / 74 test sessions, ~3.5 min each (~213 s), 23 exercises, ~5.0 feedbacks per exercise with an average **silence period of 5.2 ± 1.4 s** between them (avg inter-feedback interval 5.3 ± 1.2 s). Feedback follows a naturalistic protocol: **initiation** (acknowledge exercise start), **correction** (immediate on visible mistake), **encouragement** (after a fix), **counting** (rep tracking), **closure** (end-of-exercise summary).

**Metrics.** Feedback is scored on both axes:

*Fluency — "what to say."* Predicted responses are first **temporally matched** to ground-truth: each GT feedback is paired with the closest prediction within a **3-second window**, preserving temporal order; then compute on matched pairs only:
- **METEOR** and **ROUGE-L** (lexical / LCS overlap — reward correct domain terms like "arm", "not moving").
- **BERTScore** (semantic-level match via contextual embeddings).
- **LLM-Accuracy** — a holistic score by **LLaMA-3-70B-Instruct** given GT + prediction, scored on an integer **1–5** scale (LLMs correlate better with human preference than n-gram metrics).

*Temporal accuracy — "when to say it."* Using the same ±3 s matching, count TP = temporally matched feedback, FP = unmatched predictions, FN = missed GT, and report the **Temporal F-Score**:
$$
\text{Precision}=\frac{TP}{TP+FP},\qquad
\text{Recall}=\frac{TP}{TP+FN},\qquad
\text{T-F-Score}=\frac{2\,\cdot\,\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}.
$$
Turn-based baselines are evaluated by force-emitting at **regular 5-second intervals** (marked † in the tables) — which caps their T-F-Score near chance because they cannot *choose* timing.

## Explicit design choices
- **Action-token proactivity** (`<next>` / `<feedback>`) instead of a separate "should I speak now?" classifier — timing is decoded in-band with content, end-to-end.
- **Motion-first encoder**: a causal 3D CNN, not CLIP/ViT, because form correction depends on movement dynamics; causal convs enable frame-by-frame streaming.
- **Cross-attention fusion** (Flamingo-style gated adapters) rather than prefix-token concatenation of visual features.
- **Two-axis evaluation**: fluency metrics *and* a dedicated Temporal F-Score — the benchmark refuses to let "when" collapse into "what".
- **±3 s matching window** with order-preserving assignment before any fluency metric is computed.
- **LLM-as-judge** (LLaMA-3-70B-Instruct, 1–5) as the primary holistic content metric.
- **Staged training** that first teaches the recognizer, then aligns the LM to recognizer features (adapter-only), then learns interactive timing via LoRA on long videos.
- **Scoped to 30 s single-exercise segments** to keep the interactive fine-tuning tractable.
- **Privacy/QC in data construction**: audio + metadata stripped (PII), background person detection + manual review, expert-designed exercise list, second-annotator verification of feedbacks.

## Key results / what to remember
All numbers on **QEVD-FIT-COACH**; † = non-interactive model force-evaluated at 5 s intervals.

- **Stream-VLM (full):** METEOR **0.127**, ROUGE-L **0.112**, BERTScore **0.863**, LLM-Acc **2.45** / 5, Temporal F-Score **0.56** (Table 4) — best on every axis among fine-tuned models.
- **Fine-tuned turn-based baselines** (Table 4): Video-ChatGPT† 0.108 / 0.093 / 0.863 / 2.33 / **0.50**; LLaMA-VID† 0.106 / 0.090 / 0.860 / 2.30 / **0.50**; Socratic-LLaMA-2-7B (text-only) 0.094 / 0.071 / 0.860 / 2.17 / 0.50. Their T-F-Score sits at ~0.50 — they cannot choose *when*.
- **Zero-shot VLMs** (Table 3) are essentially unusable here: LLM-Acc ranges only 1.29–2.27 (best LLaVA-NeXT 2.27, METEOR 0.104) — repetitive, domain-blind, turn-based.
- **Ablations (Table 4):** remove 3D CNN → 0.090 / 0.083 / 0.857 / 2.11 / 0.51; remove pre-training → 0.095 / 0.087 / 0.858 / 2.08 / 0.52; remove action tokens → 0.125 / 0.110 / 0.861 / 2.41 / **0.50**. Dropping the action tokens keeps fluency almost unchanged but **collapses the Temporal F-Score to the turn-based 0.50** — direct evidence that the action-token mechanism is what buys "when to say it".
- Even a **text-only Socratic baseline** (LLM narrating 3D-CNN activations at 1 s intervals) beats all zero-shot VLMs on LLM-Acc (2.17), underscoring how much the motion-aware visual encoder matters.

No Zotero highlights present.

Takeaways: (1) proactivity is a *decodable* action, not a post-hoc scheduler; (2) you must score timing separately (T-F-Score) or a fluent-but-mistimed model looks fine; (3) motion encoders + domain pre-training are load-bearing for form feedback; (4) absolute scores are low (LLM-Acc 2.45/5) — the benchmark is far from solved.

## How it connects (evolution)
- [[proactive-response]] — this note anchors the sub-topic's "what + when" framing.
- [[videollm-online]] — contemporaneous streaming VLM; also learns *when* to speak via a per-frame decision, but on ego/instructional video rather than a coaching testbed.
- [[mmduet]] — "video as a multi-turn duet"; densely-timed proactive replies, a direct descendant of the timing-as-decoding idea.
- [[dispider]] — disentangles perception / decision / reaction for streaming proactivity; a structural take on the same "when to respond" problem.
- [[proactivevideoqa]] / [[streamingbench]] — later benchmarks that generalize proactive/streaming evaluation beyond fitness, inheriting the two-axis (content + timing) scoring logic.
- [[streaming-benchmarks]] — hub for the benchmark lineage QEVD helped start.

## Open questions / limitations
- **Single-exercise, 30 s scope**: no multi-exercise workout sequences or long-horizon session state; real coaching spans minutes and transitions.
- **Domain specificity**: fitness-only; whether the action-token + motion-encoder recipe transfers to other situated domains (cooking, assembly, physio) is untested.
- **Low absolute performance & hallucination**: best LLM-Acc is 2.45/5; the LM can still emit incorrect fitness advice despite visual grounding.
- **Fairness/bias**: age/gender bias in the crowd-sourced participants is acknowledged but not mitigated; single fixed camera framing.

*Verification: METEOR/ROUGE-L/BERT/LLM-Acc/T-F-Score numbers and ablations verified against the rendered Table 3 and Table 4 (PDF p.7); Temporal F-Score formula, ±3 s matching, action-token mechanism, and 3-stage training verified against the paper's Method/Experiments text (PDF p.6–8) and Figure 3; dataset stats from the paper's Table 1/Table 2 summary. Data/code: qualcomm.com/developer/software/qevd-dataset, github.com/Qualcomm-AI-research/FitCoach.*

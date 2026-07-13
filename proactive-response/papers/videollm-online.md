---
zotero_key: null
authors: Joya Chen et al. (Show Lab, National University of Singapore; Meta / Reality Labs)
year: 2024
arxiv: 2406.11816
pdf: https://arxiv.org/pdf/2406.11816
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# VideoLLM-online: Online Video Large Language Model for Streaming Video

**Lineage role:** Origin of the EOS/streaming-token paradigm — the LIVE framework (CVPR 2024) that first made a video LLM decide *when to speak vs. stay silent* frame-by-frame, seeding nearly all later proactive/streaming dialogue models.

## Problem — what was limited before this paper (short)
Video LLMs before this treated a video as a fixed, pre-recorded clip: the whole clip is encoded, then a single answer is produced offline. That design is blind to *time* — it cannot follow a live stream, cannot tell you the instant an event happens, and its context blows up as frames accumulate. For a genuinely online setting (an assistant watching an egocentric stream and narrating/reminding in real time) three things were unsolved together: (1) temporal alignment — responding at the right moment, not late; (2) bounded long-context over a continuous stream; and (3) real-time throughput without dropping frames. Naively querying the model every frame is both semantically wrong (it forces speech even when silence is correct) and computationally wasteful.

## Key idea — the core insight, 2-4 sentences
Interleave the assistant/user dialogue turns and the incoming video frames into one temporally-ordered token sequence, and train the LLM with **two** objectives: the usual language-modeling loss on the text it should produce, *plus* a novel **streaming EOS (end-of-sequence) loss** applied on every frame where the model should remain silent. The EOS target teaches the model to emit "nothing to say" on quiet frames and to break its silence only at the frames that warrant a response — so *when to talk* becomes a learned per-frame decision rather than a heuristic. Because silent frames only cost one forward step (no generated tokens) and EOS is never written back into the context, the method is simultaneously temporally aligned, context-lean, and fast (>10 FPS on a 5-min stream).

![[videollm-online.png]]
> **Crux (Figure 4).** The LIVE training layout: dialogue turns and 2-FPS video-frame tokens are laid out in temporal order; a *streaming loss* supervises `[EOS]` on frames where the model should stay silent, and the standard *LM loss* supervises the actual reply tokens (e.g. "appeared", "disappeared") only at the frames that should trigger speech. This dual target is the whole idea — it turns "when to respond" into a learned signal. *Chen et al. (2024), arXiv:2406.11816. Embedded for personal research reference.*

## Method + math — the mechanism and central objective

**Architecture (LLaVA-style, three parts).**
- **Image encoder (frozen):** CLIP ViT-L, pretrained on DataComp-1B, extracts per-frame embeddings at **2 FPS**. Each frame is a `(1 + h_p × w_p) × c` tensor: a CLS token plus average-pooled spatial tokens. The most efficient setting uses `h_p = w_p = 0`, i.e. **1 CLS token per frame** (handles ~30-min video in a 4096 context); demo variants use `1 + 3×3 = 10` tokens/frame for more spatial detail at the cost of shorter max length.
- **MLP projector (trainable):** a 2-layer MLP maps frame embeddings into the LLM's token space, exactly as in LLaVA-1.5.
- **Language model (frozen + LoRA):** Llama-2-7B-Chat or Llama-3-8B-Instruct, adapted with **LoRA (rank 128, scaling 256)** on every linear layer. Frame tokens are interleaved with text tokens as LLM inputs.

**The two training objectives.** Organize context, frames and dialogue in temporal order. For a response that must be produced at time $t_2$:

$$\max\; P\big([\text{Txt}^{t_2}_{i+1}] \,\big|\, [\text{Ctx}^{<t_2}],\, [\text{Frame}^{t_2}],\, [\text{Txt}^{t_2}_{\le i}]\big)$$

i.e. ordinary autoregressive LM over the reply tokens. And for every frame at time $t$ in a silent interval $t_1 \le t < t_2$ (between the query at $t_1$ and the next response at $t_2$):

$$\max\; P\big(\text{EOS} \,\big|\, [\text{Ctx}^{<t}],\, [\text{Frame}^{t}]\big)$$

i.e. predict EOS on the frame embedding, telling the model to keep silent. The EOS token is a *supervision target only* — it is never inserted into the input/output sequence, so it costs nothing in later context.

**Combined loss (Eq. 5), cross-entropy on both terms:**

$$L = \frac{1}{N}\sum_{j=1}^{N}\Big(\underbrace{-\log l_{j+1}\, P_j^{[\text{Txt}_{j+1}]}}_{\text{LM Loss}} \;-\; \underbrace{w \,\log f_j\, P_j^{[\text{EOS}]}}_{\text{Streaming Loss}}\Big)$$

where the condition indicators gate which loss applies at token $j$: $l_j = 1$ iff token $j$ is a language-response token (else 0), and $f_j = 1$ iff token $j$ is the **last** token of a frame that precedes a response — i.e. the streaming EOS loss is applied to frames right before responding. When a frame has multiple patch tokens, EOS loss is put only on its last token. $P_j^{[\text{Txt}_{j+1}]}$ is the LM head's probability of the correct next text token; $P_j^{[\text{EOS}]}$ is the head's probability of EOS. $w$ is a balance weight, default $w = 1$.

**Inference.** Frames stream in one at a time; at each frame the model does a single forward pass. If EOS is predicted, it stays silent and moves to the next frame (silent frames never enter the KV history as generated text, keeping context bounded). If a non-EOS token wins, it generates the full reply then resumes streaming. A probability threshold $\theta \approx 0.5$–$0.8$ on $P[\text{EOS}]$ is used to suppress over-eager talking.

**Data pipeline — offline annotations → online dialogue.** Two sources:
- **Ego4D narration stream:** annotators already narrate a 5-min egocentric clip *in real time*, so these timestamped narrations are used directly as temporally-grounded assistant turns — a naturally streaming supervision signal.
- **COIN offline-to-online conversion (Figure 3):** a template library of **150 questions** (50 each for past / present / future tense) is combined with COIN's timestamped step annotations; an LLM (Llama-2-13B-Chat or Llama-3-8B-Instruct) writes assistant responses anchored at state-change boundaries. Queries are randomly inserted, responses before the query timestamp are filtered out, and at most 3 queries per sample are kept.

**Evaluation protocol (streaming metrics).** Because responses are time-stamped, offline QA accuracy is insufficient; the paper defines:
- **LM-PPL** — language-modeling perplexity of the intended responses (lower better).
- **LG-Match / LM-Correctness** — for a turn, the ratio of the position of the first *error* token to the total number of tokens (measured autoregressively; higher = longer correct prefix).
- **TimeDiff** — mean discrepancy (in seconds) between the timestamp the model actually responds and the ground-truth expected response time (lower = better temporal alignment).
- **Fluency** — the proportion of *consecutive* successfully-predicted tokens within a dialogue turn; a single combined score reflecting both language and temporal quality (higher better).

## Explicit design choices
- **Streaming EOS loss on silent frames** is the load-bearing choice — it converts "should I talk now?" into a learned per-frame classification, and reduces redundant dialogue history.
- **EOS is a target, never a stored token** — silent frames leave no text in the context, so long streams stay context-bounded and cheap.
- **1 CLS token per frame** (efficient setting) vs. 10 tokens/frame (demo): a deliberate detail-vs-length knob; even 10-token models run >10 FPS on 5-min streams.
- **Frozen CLIP + frozen LLM + LoRA + trainable MLP projector** — cheap adaptation, LLaVA-style.
- **2 FPS frame sampling** during training.
- **Cross-entropy (standard CE) for both losses** — ablation shows OHEM / Focal loss are *not* needed to fight EOS class imbalance (CE actually best).
- **Balance weight $w$ (a.k.a. $\tau$)**: default $w=1$; ablation finds a slightly higher weight (~$2.0$) gives the best TimeDiff trade-off.
- **EOS probability threshold $\theta \in [0.5, 0.8]$** at inference to prevent over-suppression.
- **Data built from real streaming annotations (Ego4D) + template-driven LLM conversion (COIN)** rather than hand-labeling online dialogues.

## Key results / what to remember
No Zotero highlights present.

Verified against the paper's Tables 1–3 (Ego4D Narration Stream validation; COIN & Ego4D LTA test).

- **Streaming vs. alternatives (Table 1a, Ego4D Narration Stream val):** proposed **Streaming Dialogue** reaches **LM-PPL 2.43, TimeDiff 2.32 s, Fluency 42.6%** — beating *Interleaved Dialogue* (2.45 / 6.47 s / 11.1%) decisively on temporal alignment & fluency, and beating *Per-frame Streaming* (3.34 / 2.52 s / 37.7%) on all three, while using far fewer training tokens (1694 vs 6737) and 12h vs 22h training cost. (No-training baseline: LM-PPL 498.5, Fluency 0.1%.)
- **Efficiency (Table 1d):** Streaming runs at **13.5 FPS using 18.2 GB**, vs Per-frame 7.5 FPS / 24.9 GB and Interleaved 1.5 FPS / 34.4 GB — supports streaming dialogue over a 5-min clip at **>10 FPS on a single A100** with **<20 GB** memory.
- **Loss-function ablation (Table 1b):** plain **CE (2.43 / 2.32 / 42.6%)** beats OHEM (2.53 / 2.39 / 41.0%) and Focal (2.59 / 2.44 / 39.4%) — no special class-imbalance handling for EOS needed.
- **Weight ablation (Table 1c):** default $\tau=1.0$ gives 2.43 / 2.32 / 42.6%; $\tau=2.0$ gives best TimeDiff **2.31 s**.
- **COIN benchmarks (Table 2a, Top-1 Acc, VideoLLM-online-8B-v1+):** Step Recognition **63.1%**, Task Recognition **92.7%**, Next-step Forecasting **49.1%**, Procedure Forecasting **49.8%**, Procedure Forecasting w/ goal **54.1%** — new SOTA over prior end-to-end methods (e.g. ProcedureVRL step 56.9 / next 46.8; VideoTaskGraph task 90.5).
- **Ego4D LTA (Table 2b, ED@Z=20, lower better, 8B-v1+, end-to-end):** Verb **0.689**, Noun **0.671**, Action **0.884** — best among end-to-end models (AntGPT's 0.650 verb uses heavy cascaded pipelines + egocentric pretrained features).
- **Model variants (Table 3, Ego4D Narration val, LG-Match / TimeDiff / Fluency):** 7B-v1 42.3% / 2.25 / 42.6%; 8B-v1 48.3% / 2.05 / 45.2%; **8B-v1+ 49.0% / 2.05 / 45.3%** — stronger LLM (Llama-3-8B) helps most; extra tokens/frame ("v1+") help vision-language but give limited online gain.

## How it connects (evolution)
- [[proactive-response]] — this note's home sub-topic; VideoLLM-online is the seed of the proactive-response line.
- [[streaming-video-understanding]] — the topic hub; this is the paradigm origin.
- [[mmduet]] — extends the per-frame "informative/response" decision into a dense video-language dialogue head, directly building on the streaming-response idea here.
- [[dispider]] — disentangles perception, decision and reaction for streaming interaction, a successor tackling the same "when to respond" problem.
- [[streammind]] — pushes the proactive-speaking-vs-silent decision to higher FPS / lower latency, extending the EOS-style gating.
- [[vispeak]] — visual-instruction-driven proactive streaming dialogue, a later member of the same lineage.

## Open questions / limitations
- The "when to speak" decision is trained on *converted/annotated* triggers (Ego4D narrations, template-based COIN dialogues) — real open-world conversational cues are richer, so learned timing may not transfer to arbitrary user intents.
- Frame representation is aggressively compressed (often 1 CLS token/frame at 2 FPS) — fine detail and sub-second events can be missed; there is a genuine detail-vs-context tradeoff (v1 vs v1+).
- No explicit long-term memory: context stays bounded because silent frames leave no text, but there is no mechanism to *recall* far-past events beyond the raw frame-token window.
- Evaluated largely on egocentric/instructional domains (Ego4D, COIN); robustness to third-person, multi-speaker, or noisy streams is untested.

*Verification: equations (Eq. 5 dual LM+EOS loss, the two max-probability objectives) and all headline numbers checked directly against the rendered paper PDF — Figure 4 and the method text on page 6, and Tables 1a/1b/1c/1d, 2a, 2b, 3 on pages 8–9 of arXiv:2406.11816; architecture/data details cross-checked against the arXiv HTML full text.*

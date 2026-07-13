---
zotero_key: null
authors: Shenghao Fu et al. (Sun Yat-sen University; Tongyi Lab, Alibaba Group; Peng Cheng Laboratory)
year: 2025
arxiv: 2503.12769
pdf: https://arxiv.org/pdf/2503.12769
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# ViSpeak: Visual Instruction Feedback in Streaming Videos

**Lineage role:** Defines the *Visual Instruction Feedback* task family (respond proactively to purely visual cues — gestures, salutations, anomalies — with no text/audio prompt), ships the ViSpeak-Bench evaluation + ViSpeak-Instruct training set, and an omni-modal streaming LMM (ViSpeak) as the baseline. ICCV 2025.

## Problem — what was limited before this paper
Streaming video LMMs had been pushed toward *time-sensitive* QA and *proactive* text output (e.g. [[videollm-online]], [[mmduet]], [[dispider]]), but the interaction was still driven by a **user's textual/spoken query**. Real assistants must also react to what they *see*: a user waving hello, pointing at an object, making a "stop" gesture, or an accident unfolding — none of which arrive as a language prompt. No prior streaming benchmark isolated this "the visual scene itself is the instruction" setting, and no model was trained to fire a response at the right moment purely from a visual trigger while continuously ingesting video and audio.

## Key idea
Frame **Visual Instruction Feedback**: the agent must recognize a visual action/event in a live stream and emit an appropriate, in-time response *without any explicit text/audio query*. The paper contributes (1) **ViSpeak-Bench**, a 7-subtask benchmark (1,000 videos / 1,000 QA) covering wake-up, anomaly warning, gesture understanding, visual reference, visual interruption, humor reaction, visual termination; (2) **ViSpeak-Instruct**, ~34k training samples; and (3) **ViSpeak**, an omni-modal (video + audio + text) 7B LMM with a **two-stream chat template** (separate user-input and agent-output streams merged before the LLM, enabling interruption à la Moshi) and a dedicated binary **informative head** that decides *when to speak* from visual segments. It is trained with a three-stage recipe and reaches GPT-4o-level scores on general streaming benchmarks.

![[vispeak.png]]
> **Crux (Figure 2).** ViSpeak's architecture: multi-modal encoders feed a two-stream layout — a *User Input* stream (image/audio/text segments delimited by `seg` tokens) and a self-generated *Agent Input* stream — that is fused into one *Combined Input* for the LLM; a separate **Informative Head** reads visual segment tokens and predicts, at each `seg` boundary, whether to proactively speak. *Fu et al. (2025), arXiv:2503.12769. Embedded for personal research reference.*

## Method + math

### The task and its scoring (this is the "math" — an eval protocol)
Each sample has a ground-truth trigger interval and a tolerance $T$; a valid response must land in $[t_1, t_2+T]$.

**Time Accuracy** per subtask $s$ (binary hit if the response timestamp $T_{\text{res}}$ falls in the window):
$$\mathcal{T}_{\text{acc}}^{s} = \frac{1}{N^{s}}\sum_{i=1}^{N^{s}} \mathbb{I}\!\left(T_{\text{res}}^{(i)} \in [t_1, t_2+T]\right)$$
where $N^s$ is the number of questions in subtask $s$ and $\mathbb{I}$ is the indicator. For **Visual Reference** (multiple-choice) time accuracy is fixed to $1$.

**Text Score** $\mathcal{S}^s \in [0,5]$: GPT-4o judges each open-ended response for correctness / consistency with the reference / contextual appropriateness. Subtask-specific rubrics, e.g. Anomaly Warning = description consistency (3 pts) + advice rationality (2 pts); Gesture Understanding = recognition (3 pts) + contextual appropriateness (2 pts).

**Overall Score** — the product of timing and content, averaged over the $N=7$ subtasks:
$$\mathcal{O} = \frac{1}{N}\sum_{s=1}^{N}\mathcal{T}_{\text{acc}}^{s}\times \mathcal{S}^{s}$$
So a response that is textually perfect but mistimed (or vice-versa) is penalized — timing and content must both be right.

### Data construction
Videos are a mix of **self-collected** recordings (≈1,185 visual-interruption, ≈4,689 visual-reference, ≈1,188 wake-up/termination, ≈1,507 gesture clips; 610 participants, ages 10–70, 5 provinces, homes/offices/factories/warehouses/supermarkets/wilderness) and **open-source** sources repurposed per subtask (Holmes-VAU → anomaly warning, OOPS → unintentional actions, FunQA → humor, Jester → gestures). Human annotators mark action timestamp boundaries and write scripted responses, with spot-check QC. **ViSpeak-Instruct** (~34k samples) augments this with offline understanding data (SMILE humor, IntentQA, Social-IQ body-language; 678 Social-IQ videos re-annotated with >1 s gestures → 1,861 annotations) to inject reasoning ability the streaming clips alone don't teach.

### Model mechanism
Encoders: **InternViT-300M-448px** (visual, from VITA 1.5), a 341M **audio encoder** (VITA), and **Qwen2.5-7B** LLM. Streaming input is segmented at **1 fps** for video and **1-second** audio snippets; each segment ends with a special `<<<seg>>>` token, and **the model may only begin speaking at a `seg` boundary**.

- **Two-stream chat template** (inspired by Moshi): one stream carries *user inputs*, a parallel stream carries the *agent's self-generated outputs*, so the model keeps consuming incoming video while it is talking (enables interruption). The two time-aligned streams are combined by a **weighted sum** before the LLM; the per-token weights are produced by a **linear layer** by default ("Add" — equal — worked about as well in ablation). Response type is signaled by a prefix token: `⇐` text, `⇒` audio, `⇓` visual; a Visual Interruption is handled by simply emitting `⇓ Stop!` at a `seg`.
- **When to speak** — two heads. The **original LM head** (next-token) is enough for *text* answers (it emits `⇐` at the end of a text segment). But proactive *visual* output is a harder turn-taking problem, so a separate **informative head** — a binary "speak / don't-speak" classifier following [[mmduet]] — reads the visual segment representation and fires when its score exceeds a **fixed threshold of 0.35**. It is trained *jointly* with the LLM (ablation shows joint training on the visual token is decisively better). Audio turn-taking is left out of scope for simplicity.

### Three-stage finetuning
1. **Template alignment** (~2.0–2.7M mixed samples: text/image/video/audio/cross-modal): adapt a pretrained offline omni-model to the streaming two-stream template while preserving general ability. MLP-adapter pretrain (lr 5e-4, bs 256) then LLM LoRA (lr 1e-4, bs 128); TTS augmentation via CosyVoice2 over 5,962 VoxCeleb2 speakers.
2. **Streaming finetuning** (~657k: 81k MMDuet, 42k ET-Instruct, 42k EgoTimeQA, 500k offline): learn streaming QA + proactive output; train the informative head jointly with the LLM.
3. **Visual instruction feedback** (34k ViSpeak-Instruct): learn to recognize the 7 visual actions and respond in time.

Training on 32× NVIDIA L20, max context 6,200 tokens; image tokens 256 (stage 1) → 64 (stages 2–3, 2× downsample); ≤16 images/video (stage 1) → ≤64 (stages 2–3); 1 fps sampling.

## Explicit design choices
- **Task restricted to conversational scenarios** and to **7 concrete visual triggers** — wake-up, anomaly warning, gesture understanding, visual reference, visual interruption, humor reaction, visual termination — giving a closed, scorable taxonomy.
- **Product scoring** $\mathcal{T}_{\text{acc}}\times\mathcal{S}$ forces *both* correct timing and correct content; Visual Reference uses multiple-choice with $\mathcal{T}_{\text{acc}}\equiv1$ to isolate grounding.
- **Two input streams (user vs. agent) fused before the LLM** rather than a single role-tagged transcript — role-tagged templates can't interrupt; the two-stream design can.
- **Linear/weighted-sum stream combination** chosen as default (ablation: adaptive-sum, linear, add all comparable — simplicity wins).
- **Dedicated informative head** (binary, threshold 0.35, trained jointly, on the *visual* token) for proactive triggering; LM head reused for text turn-taking.
- **Prefix tokens** `⇐ / ⇒ / ⇓` disambiguate text vs. audio vs. visual responses (follows VITA); `⇓ Stop!` implements interruption.
- **Segment granularity**: 1 fps video, 1 s audio; speech only allowed at `<<<seg>>>` boundaries.
- **Omni-modal encoders reused from VITA 1.5** (InternViT + audio) + Qwen2.5-7B; audio turn-taking deliberately excluded.
- **Offline reasoning data mixed into stage 3** to teach humor/gesture understanding the raw streaming clips lack.

## Key results / what to remember
No Zotero highlights present. Numbers verified against the paper's Tables 4–10.

- **ViSpeak-Bench (Table 6, overall = mean of $\mathcal{T}_{\text{acc}}\times\mathcal{S}$):** ViSpeak (s3) **2.76** overall (Time Acc 80.42%, Text Score 3.25). Full ranking: **Human avg 3.69** (max 4.01) > **GPT-4o 2.99** (TAcc 87.50, TS 3.27) > **ViSpeak s3 2.76** > **Qwen2.5-VL-72B 2.62** > **Dispider 1.63** > **VITA 1.5 1.54**. So ViSpeak is the strongest *open streaming* baseline but still below GPT-4o and humans.
- **Per-subtask (ViSpeak s3, Table 6):** Gesture Understanding TAcc **99.00%**, Visual Wake-Up TS **4.95** / TAcc 93.00%, Visual Interruption TAcc 83.00% / TS 3.84, Visual Termination TAcc 79.00% / TS 3.15, Anomaly Warning TAcc 56.50% / TS 3.75, Visual Reference TAcc 72.00% / TS 2.63; **Humor Reaction TS 1.07** is the hardest (weakest subtask).
- **StreamingBench (Table 4, overall):** ViSpeak **s3 62.58** / **s2 62.00** — beats VITA 1.5 (54.27) and Dispider (53.12), near GPT-4o (64.31); below Gemini-1.5-Pro (70.26). s3 Omni-Source 61.60, Contextual 43.90.
- **OVO-Bench (Table 5, overall):** ViSpeak **s2 61.08** (Real-Time 66.28, Backward 57.52, Forward 54.25) — above GPT-4o 58.58 and VITA 1.5 55.49.
- **Proactive-head ablation (Table 8, StreamingBench overall):** joint-trained informative head on the *visual* token = **62.00**, vs. non-joint informative head **34.80–36.00**, vs. LM-head turn-taking 60.91 — joint training of the informative head is the key.
- **Offline data (Table 9):** removing offline reasoning data drops overall 2.76 → 2.70 (Humor 1.07 → 1.02, Gesture 3.36 → 3.17).
- **Progressive training (Table 10):** general ability is largely preserved across stages — MME 2237 (s1) → 2051 (s2) → 2182 (s3); Video-MME 55 → 58 → 60.

Takeaways: purely-visual proactive triggering is a *distinct, unsolved* axis of streaming understanding; a small binary head trained jointly on visual tokens handles the "when to speak" turn-taking that a next-token LM head cannot; product scoring cleanly separates timing from content; humor/subjective reactions remain the hard case.

## How it connects (evolution)
- [[mmduet]] — direct ancestor of the **informative (binary speak-or-not) head**; ViSpeak adapts it for *visual* proactive output.
- [[dispider]] — prior streaming LMM compared head-to-head on StreamingBench/OVO-Bench/ViSpeak-Bench (ViSpeak's baseline to beat).
- [[videollm-online]] — foundational proactive/streaming-dialogue formulation this task extends beyond text-triggered turns.
- [[proactive-response]] — the sub-topic hub: ViSpeak is a canonical "respond without being asked" system.
- [[proactivevideoqa]] — sibling benchmark specifically for proactive streaming QA; complementary evaluation of the same capability.
- [[streamingbench]] — general streaming benchmark ViSpeak reports strong (GPT-4o-level) results on.

## Open questions / limitations
- **Audio turn-taking is out of scope** — the two-stream interruption mechanism only handles the visual/text path, so full duplex voice interaction is unaddressed.
- **Humor Reaction is barely solved** (TS 1.07): subjective/affective visual reactions remain far below the other subtasks, and offline data barely helps.
- **Still below GPT-4o and humans on ViSpeak-Bench** (2.76 vs 2.99 vs 3.69) — the baseline is a starting point, not a solved task.
- **Fixed threshold (0.35) and 1 fps / seg-boundary-only speaking** cap temporal precision; how sensitive timing accuracy is to these hyperparameters is not deeply explored.

*Verification: equations (Time Acc, Overall Score) and all headline numbers checked against the paper's Tables 4–10 and Eq. (2) via the arXiv HTML full text and the rendered PDF (page 4 architecture, Figure 2). ViSpeak-Bench 2.76/GPT-4o 2.99/Human 3.69 ranking confirmed from Table 6.*

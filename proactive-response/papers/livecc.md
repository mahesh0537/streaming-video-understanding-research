---
zotero_key: null
authors: Joya Chen, Ziyun Zeng, Yiqi Lin et al. (Show Lab, NUS + ByteDance)
year: 2025
arxiv: 2504.16030
pdf: https://arxiv.org/pdf/2504.16030
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale

**Lineage role:** the streaming-ASR-pretraining-at-scale line — turns raw YouTube closed captions into a temporally-dense frame↔word supervision signal so a video LLM learns to *speak while it watches* (real-time commentary), rather than caption a clip after the fact.

## Problem — what was limited before this paper (short)
Video-LLM training data is bottlenecked: high-quality instruction data comes from expensive human annotation or proprietary API distillation (GPT-4o), which caps scale. The one signal that *is* cheap and abundant — ASR / closed-caption transcripts on web video — was previously used badly: treated as a **global caption** for the whole clip, which throws away the fine-grained temporal alignment between *which words were spoken* and *what was on screen at that instant*. That post-hoc, whole-clip framing also can't produce **proactive, low-latency** output: the model waits for the full clip before responding, so it can't emulate a live commentator who narrates as events unfold.

## Key idea — the core insight, 2-4 sentences
Interleave ASR **words with their corresponding frames densely along the time axis**, using the transcript timestamps, and train the LLM to autoregressively predict the next words given the frames seen so far. Because supervision is per-frame-interval rather than per-clip, the model learns tight vision↔language temporal correlations and, at inference, emits commentary token-by-token as each new frame arrives (<0.5 s/frame). The whole thing is bootstrapped from a scalable automatic pipeline (Live-CC-5M pretraining, Live-WhisperX-526K SFT) with **no human captioning and no GPT-4o distillation of the commentary itself**.

![[livecc.png]]
> **Crux (Figure 5).** The modeling overview: streaming frames go through a frozen visual encoder to 2-FPS visual tokens; the timestamped ASR words for each interval become the text tokens; the LLM is trained with a causal LM loss to predict those words in the densely interleaved `[Con]<F><W><F><W>…` sequence, with prior ASR / title as optional context. *Chen, Zeng, Lin et al. (2025), arXiv:2504.16030. Embedded for personal research reference.*

## Method + math — the mechanism, then the objective in full

**Interleaving format.** The training sequence densely alternates frame-token blocks and word-token blocks along time:
$$
[\text{Con}]\;\langle F_{t:t+k}\rangle\langle W_{t:t+k}\rangle\;\langle F_{t+k:t+2k}\rangle\langle W_{t+k:t+2k}\rangle\;\cdots\;\langle F_{t+nk:t+(n+1)k}\rangle\langle W_{t+nk:t+(n+1)k}\rangle,
$$
where $[\text{Con}]$ is context (video title, previous ASR), $\langle F\rangle$ is the visual tokens for a frame interval, $\langle W\rangle$ the ASR words spoken in that interval, $t$ the time index and $k$ the interval (default $k=1$ at **2 FPS**). A newline concatenates title and previous ASR when both are present.

**Objective.** Standard causal (autoregressive) LM loss, but computed **only over the word tokens** — visual tokens are non-predictive inputs. With the interleaved sequence flattened to tokens $x_{1..N}$ and $\mathcal{W}$ the set of positions that are ASR-word tokens:
$$
\mathcal{L} \;=\; -\sum_{i\in\mathcal{W}} \log p_\theta\big(x_i \mid x_{<i}\big).
$$
So each word is predicted conditioned on all preceding frames *and* words — the model must use the current on-screen content to continue the narration. To disambiguate a true end-of-utterance from a mere silent pause in streaming, silent frames (no subtitle) are trained to predict an **ellipsis / EOS token** ("…", written `</e>` in the figure); this is what lets the model *stay quiet* until something worth narrating happens.

**Word-level timestamp alignment.** Raw YouTube ASR only has coarse ~2–3 s segment timestamps; the method uniformly distributes each segment's duration across its constituent words to approximate per-word timing for pretraining. For SFT it upgrades to **WhisperX (large-v3-turbo)** which gives precise word-level timestamps.

**Two-stage training.**
- **Pre-training** on Live-CC-5M with pure dense-interleaving sequences: align frame features with temporally-synchronized ASR words to learn temporal correlation. Context = video title ∥ previous ASR (hybrid: title only used as fallback when no prior ASR, to avoid information leakage).
- **SFT** on **Live-WhisperX-526K** (streaming commentary, in streaming mode) **jointly with LLaVA-Video-178K** (general caption/QA), using the Qwen2-VL conversation template so streaming training is chat-compatible. Here the context part is reduced to just the user query, to match real deployment.

**Inference / streaming.** Frames are fed sequentially; KV pairs for prior prompts, visual tokens, and generated text are cached to accelerate decoding. For long streams, **visual tokens are discarded every 240 s** (the SFT max duration) while text tokens are retained to re-prefill — bounding memory while keeping the narrative thread.

**Data pipeline (why it scales).**
- *Live-CC-5M (pretraining):* ~5.7M YouTube videos from HD-VILA / YT-Temporal-1B / VidChapters / HowTo100M, filtered by resolution ≥480p, 30 s–10 min, English (XLM-RoBERTa conf 0.9), ≥30 distinct caption words; segmented at ASR word-gaps >3 s into 30–240 s clips at 1–4 words/s; LM-perplexity filter (Qwen2-1.5B, 1.5–6.5); talking-head removal via Qwen2-VL-2B. Ranked into 1M/2.5M/5M/10M subsets.
- *Live-WhisperX-526K (SFT):* restricted to 7 YouTube categories, WhisperX word-level timestamps, sentence-boundary clips ≤240 s, tighter perplexity 1.5–5, Active-Speaker-Detection (Light-ASD, ~250× sped up) to drop talking-head footage, and GPT-4o only to synthesize **user prompts** that match the speech style.

**Benchmark math (LiveSports-3K).** Two tracks over 416 sports videos / 49 categories:
- *LiveSports-3K-CC* (1,702 events): given title + preceding CCs, generate real-time commentary; scored by **pairwise LLM-as-judge win rate** — GPT-4o compares model vs reference on semantic + stylistic consistency, GPT-4o-08-06 predictions as the fixed **baseline anchor**. Win rate = fraction of pairwise judgments the model wins.
- *LiveSports-3K-QA* (1,174 MCQs): each event decomposed into **When / What / Who** (query any element given the other two), plus an OCR bucket; metric = multiple-choice **accuracy**.

## Explicit design choices — concrete decisions
- **Base model:** Qwen2-VL-7B-**Base** (deliberately the base, not Instruct — less video-text exposure, so streaming pretraining is the main teacher). Frozen visual encoder.
- **Frame rate 2 FPS**, interval $k=1$. Min spatial res 100×28×28.
- **Token/context budget:** pretraining ablations 120 frames / 16K visual tokens; formal runs 480 frames (240 s) / 24K visual tokens.
- **Loss on words only**; visual tokens are inputs, not prediction targets.
- **Ellipsis/EOS token for silent frames** — separates real end-of-turn from pauses, enabling "stay silent until worth speaking."
- **Hybrid context** "Title ∥ Prev. ASR" (title only when no prior ASR) — best trade-off; title-always slightly hurts VideoMME (leakage), prev-ASR-always breaks at clip starts.
- **KV-cache streaming inference** + **240 s visual-token eviction** (retain text) for unbounded streams.
- **Joint SFT** with a general video-instruction set (LLaVA-Video-178K) to keep general QA ability alongside commentary.
- **Training infra:** batch 512 on 128 GPUs; pretrain LR 2e-5, SFT LR 1e-5.
- **Data scale sweet spot:** 5M pretraining clips optimal; 10M *hurts* general QA (single-source streaming data over-specializes).

## Key results / what to remember — with settings, verified against the paper's tables

**Commentary (LiveSports-3K-CC, win rate; GPT-4o-08-06 = baseline anchor 72.2):**
- **LiveCC-7B-Instruct 41.5**, LiveCC-7B-Base **43.2** — both **beat every open 72B model**: LLaVA-Video-72B 35.0, Qwen2.5-VL-72B-Instruct 30.4, Qwen2-VL-72B 17.0 (Table 4). Among 7B: LLaVA-Video-7B 27.1, Qwen2-VL-7B-Instruct 9.3.

**Streaming/general QA (LiveSports-3K-QA overall accuracy, Table 4):** LiveCC-7B-Instruct **66.8** (Who 71.4 / When 56.1 / What 70.8) — behind the 72B models (LLaVA-Video-72B 71.1, Qwen2.5-VL-72B 73.7) and GPT-4o 74.0 on QA, but it is the only *streaming* (Live? ✓) model and dominates on the commentary track.

**General benchmarks (Table 3, 7B/8B scale):** LiveCC-7B-Instruct **VideoMME 64.1** (w/o sub) / **70.3** (w sub), **MVBench 62.8**, **OVOBench avg 59.8** — beats Qwen2-VL-7B-Instruct (63.3 / 69.0 / OVOBench 50.4) and LLaVA-Video-7B (63.3 / OVOBench 52.9). Note the large **OVOBench** jump (59.8 vs ~50–53), the online/streaming-QA benchmark.

**Pretraining ablation (Table 1):** streaming-interleaving pretraining gives **32.9** CC win rate vs **14.0** for caption-style pretraining at 5M, at essentially equal VideoMME (~61) — direct evidence the interleaving, not just the data, drives commentary skill.

**Latency (Table 5):** LiveCC-7B-Instruct **~0.17 s/frame** streaming vs LLaVA-Video-7B 5.62 s and LLaVA-Video-72B 20.51 s (clip-wise) — i.e. genuinely real-time (<0.5 s/frame at 2 FPS).

No Zotero highlights present.

Takeaways: (1) timestamp-aligned dense frame↔word interleaving converts free web ASR into a *proactive, temporally-grounded* training signal — the key unlock for live commentary. (2) A 7B streaming model can out-narrate 72B clip-models on their own commentary metric while staying real-time, because the *training objective*, not scale, is what teaches "narrate as you watch." (3) The ellipsis/silence token is the small mechanism that turns a captioner into something that decides *when* to speak.

## How it connects (evolution)
- [[videollm-online]] — the earlier "learning-in-video-stream" / streaming dialogue framing LiveCC builds past by learning *when to speak* from ASR at scale rather than from constructed dialogue.
- [[ovo-bench]] and [[proactivevideoqa]] — LiveCC reports a strong **OVOBench** jump; these benchmarks measure exactly the proactive/streaming-QA ability its pretraining targets.
- [[egospeak]] — the "when to speak" / turn-taking problem in streaming, solved here via the silence/ellipsis token.
- [[streammind]] and [[dispider]] — sibling proactive-response systems that decide response timing on a video stream; LiveCC differs by learning it directly from web-ASR supervision.
- [[proactive-response]] — the sub-topic hub this note anchors.

## Open questions / limitations
- **Domain skew to sports/commentary:** SFT and the flagship benchmark are sports-heavy; general QA (LiveSports-3K-QA) still trails 72B and GPT-4o, so the win is narrow-but-deep on narration rather than broad video understanding.
- **ASR = ground truth:** commentary quality is bounded by (and evaluated against) noisy human ASR; visually-ungrounded or speaker-opinion speech that survives filtering could teach hallucinated narration.
- **LLM-as-judge win rate** is relative to a GPT-4o anchor and stylistic-match-based — it rewards sounding like the reference commentator, not factual correctness of the play-by-play.
- **240 s visual eviction** bounds memory but likely limits truly long-horizon reference ("earlier in the match…") — an explicit long-term memory is out of scope here.

*Verification: equations (interleaving format, word-only causal LM loss, ellipsis-token EOS) and all numbers checked against the paper's own Figure 5 and Tables 1/3/4/5 rendered from the arXiv PDF (2504.16030); CC win rates, VideoMME/MVBench/OVOBench, and latency read directly from Tables 3–5.*

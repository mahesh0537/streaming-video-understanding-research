---
zotero_key: null
authors: Haibo Wang et al. (Apple; Fudan University)
year: 2025
arxiv: 2505.05467
pdf: https://arxiv.org/pdf/2505.05467
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# StreamBridge: Turning Your Offline Video Large Language Model into a Proactive Streaming Assistant

**Lineage role:** the offline-to-streaming *adaptation* thesis — rather than train a streaming model from scratch, wrap any existing offline Video-LLM in a modular framework (memory buffer + compression + a small decoupled activation model) plus a purpose-built instruction dataset (Stream-IT) to grant multi-turn real-time understanding and proactive response.

## Problem — what was limited before this paper (short)
Mainstream Video-LLMs assume the whole clip is available offline: they sample frames from a complete, pre-recorded video and answer one question in a single turn. Streaming settings (assistive agents, robotics, driving) break both assumptions — frames arrive continuously and unbounded, users interleave multiple queries over time, and the model must sometimes speak *on its own* (proactively) at the right moment rather than only when prompted. Two concrete gaps: (1) a single-turn offline model has no mechanism to preserve and reuse the running history of interleaved video/text across turns; (2) it has no notion of *when* to respond without an explicit trigger. Retraining a bespoke streaming model per base model is expensive and tends to erode the base model's general video-understanding ability.

## Key idea — the core insight, 2-4 sentences
Keep the powerful offline Video-LLM frozen-in-spirit as the generation engine and bolt on three lightweight, model-agnostic pieces: a **memory buffer** that stores encoded frame + query + answer embeddings under a producer–consumer paradigm, a **round-decayed compression** that keeps the sequence under a token budget for unbounded streams, and a **compact, decoupled activation model** (a 0.5B MLLM) that watches incoming frames and emits a per-frame binary "answer now?" signal. Because proactive-timing decisions are offloaded to the separate activation model, the main LLM never has to interleave timing tokens with language, preserving its generation quality. A dedicated instruction dataset, **Stream-IT**, supplies the interleaved multi-turn video–text supervision these capabilities need.

![[streambridge.png]]
> **Crux (Figure 2).** The StreamBridge pipeline: incoming frames are encoded and appended one-by-one into the memory buffer (①③), interleaved with a query $\mathcal{Q}$ posed mid-stream (②); a small activation model monitors frames and returns a binary signal $\mathcal{D}$ (④) that tells the LLM when to start answering, with optional round-decayed compression before the flattened buffer enters the LLM. *Wang et al. (2025), arXiv:2505.05467. Embedded for personal research reference.*

## Method + math — the mechanism, then the central objective/equations

**Overall framework (Algorithm 1).** In addition to the frame encoder $\mathcal{I}(\cdot)$ and the LLM $\mathcal{LLM}(\cdot)$, StreamBridge adds a memory buffer $\mathcal{MB}$, a compression operator $\mathcal{COM}(\cdot)$, and an activation model $\mathcal{ACT}(\cdot)$. At each time step $t$ (videos sampled at 1 FPS), the incoming frame $F_t$ is encoded and its embeddings appended to $\mathcal{MB}$. When a user query $\mathcal{Q}$ arrives it is stored too, with a recorded timestamp. Once an outstanding query exists, the activation model produces a decision $\mathcal{D}$; if $\mathcal{D}$ is positive, the buffer is flattened into a single sequence of input embeddings (visual + textual), compressed if it exceeds the budget, and fed to $\mathcal{LLM}(\cdot)$ to generate response $\mathcal{R}$. The response $\mathcal{R}$ is appended back into $\mathcal{MB}$ so temporal continuity and full multi-turn history are preserved.

**Memory buffer (producer–consumer).** The encoder is the *producer* — continuously turning frames into features — and the LLM is the *consumer* — retrieving the accumulated embeddings only when it must answer. The flattened input is an interleaved sequence conceptually of the form
$$\text{InputEmbeds} = [\,V_1, V_2, V_3, \otimes\, \mathcal{Q}\, \otimes, V_4, V_5, V_6, \otimes\, \texttt{ASSISTANT:}\,]$$
where $V_i$ are per-frame token blocks and $\otimes$ denotes concatenation. This keeps historical visual and dialogue context in place, which is exactly what single-turn offline evaluation discards.

**Round-decayed compression $\mathcal{COM}(\cdot)$.** Streaming can be arbitrarily long, so a maximum embedding length $\mathrm{MaxLen}$ is fixed (default $16{,}384$ tokens). Before each generation, if the current input exceeds $\mathrm{MaxLen}$, tokens are merged **starting from the earliest dialogue rounds**, frame-by-frame, via average pooling of adjacent frame tokens, until
$$\text{Len}(\text{InputEmbeds}) \le \mathrm{MaxLen}.$$
The decay is *round-ordered*: oldest rounds lose fidelity first, the most recent context is preserved at full resolution. This beats plain truncation (drops old context entirely) and uniform compression (degrades everything equally).

**Decoupled activation model $\mathcal{ACT}(\cdot)$ (Figure 3, §3.2.3) — the proactive-response core.** A compact MLLM (LLaVA-OV-0.5B) runs in parallel with the main LLM. Design details:
- The standard language head is replaced by a binary **score head** that outputs a per-frame probability of "should respond now."
- A learnable **`<ACT>` token** is appended to each frame's embeddings; its hidden state feeds the score head, letting the model read out a decision at every frame.
- Visual tokens are aggressively pooled to **16 tokens per frame** so the monitor is cheap enough to run every frame.
- Input is formatted as the interleaved history $\langle Q\rangle\,\langle V_1\rangle\,\langle A_1\rangle\,\langle V_2\rangle\,\langle A_2\rangle\dots$ so timing decisions are conditioned on prior turns, not just the current frame.
- **Training label scheme:** for a segment whose ground-truth response falls at its end, only the **last $P\%$ of frames** are labeled positive, with $P$ **dynamically sampled in $[0, 50]\%$** per example. This teaches a soft, temporally-tolerant trigger rather than a single hard frame, and the sampled $P$ regularizes against latching onto a fixed position.
- At inference, a frame fires a response when its score exceeds a threshold $\alpha$ (default $\approx 0.35$).

Because timing is a separate binary classifier, the generation LLM is "free from the burden of proactive decision-making," which the paper credits for its higher text-similarity (context-aware) scores on generation-based streaming tasks.

**Stream-IT dataset (the supervision).** Long-form training videos are synthesized by concatenating semantically similar short clips: ~1.28M clips filtered from WebVid-10M / Panda-70M / InternVid-10M are ordered by pairwise middle-frame similarity (Algorithm 2) into sequences of ~10 clips (~150+ s). GPT-4o then writes diverse interleaved multi-turn QA across 8 task types. Two augmentations fight position overfitting: **Random QA Drop** ($P_\text{drop}=0.55$) removes some QA pairs, and **QA Interval Shift** ($P_\text{shift}=0.1$) shifts visual content between a question and its response. Stream-IT combines a synthetic multi-turn split (StreamingQA-120K) with dense video captioning, sequential step recognition, and grounded VideoQA; ~600K offline samples (LLaVA-178K, VCG-Plus, ShareGPT4Video) are mixed in to preserve general ability.

**Eval protocol / metrics.** Real-time multi-turn understanding is scored as accuracy on the real-time subtasks of OVO-Bench and Streaming-Bench, but under an *(Offline → Streaming) Multi-Turn* setting: the model processes the streaming video at 1 FPS while keeping history within the token budget, rather than the original single-turn setting that discards history. Proactive/timing ability is measured on ET-Bench generation tasks — Dense Video Captioning (DVC) and Step Localization & Captioning (SLC) — reporting an $F_1$ for event localization and a **Sim** (text similarity) score for description quality, plus Temporal Video Grounding ($\text{TVG}_{F1}$) and Temporal Action Localization ($\text{TAL}_{F1}$).

## Explicit design choices — concrete decisions
- **Adaptation over retraining:** frozen offline Video-LLM as generator; all streaming logic lives in add-on modules → same recipe works for LLaVA-OV-7B, Qwen2-VL-7B, and Oryx-1.5-7B.
- **Producer–consumer memory buffer** storing interleaved visual + textual embeddings; response appended back to preserve full multi-turn history.
- **Round-decayed (oldest-first) token merging** via average pooling of adjacent frame tokens, budget $\mathrm{MaxLen}=16{,}384$ — chosen over truncation and uniform compression.
- **Timing decoupled into a separate 0.5B model** with a binary score head + learnable `<ACT>` token, 16 pooled visual tokens/frame, so the main LLM's language head is untouched.
- **Soft temporal labels:** only the last $P\%$ frames positive, $P\sim\text{Uniform}(0,50)\%$; decision threshold $\alpha\approx0.35$.
- **Data-side:** similarity-ordered clip concatenation to fabricate long streams; GPT-4o interleaved multi-turn QA over 8 task types; QA-drop / interval-shift augmentation; ~600K offline samples mixed in to avoid degrading general video understanding.
- **1 FPS** streaming throughout; default base model for ablations is Qwen2-VL-7B.

## Key results / what to remember
Base models evaluated: **LLaVA-OV-7B, Qwen2-VL-7B, Oryx-1.5-7B**. († = under StreamBridge; ‡ = StreamBridge + fine-tuned on Stream-IT.)

**Multi-turn real-time (OVO-Bench AVG / Streaming-Bench AVG, accuracy %):**
- **Qwen2-VL-7B**: baseline (single-turn) 55.98 / 69.04 → StreamBridge† **63.35 / 72.01** → +Stream-IT‡ **71.30 / 77.04** (verified, Table 1).
- **LLaVA-OV-7B**: 64.02 / 71.12 → StreamBridge† 61.64 / 68.39 (slight drop) → +Stream-IT‡ **69.93 / 70.92**.
- **Oryx-1.5-7B**: StreamBridge† 59.25 / 74.79 → +Stream-IT‡ **71.17 / 74.79** (Oryx gains +11.92 on OVO from Stream-IT, per text).
- Qwen2-VL‡'s **71.30** OVO-Bench average **exceeds GPT-4o (64.46) and Gemini 1.5 Pro (69.32)**; human ceiling is 93.20 OVO / 91.46 Streaming (Table 1, verified).

**Proactive response — ET-Bench (Table 3, verified).** StreamBridge (Qwen2-VL‡): $\text{DVC}_{F1}$ **38.3**, $\text{DVC}_{Sim}$ **25.1**, $\text{SLC}_{F1}$ 22.6, $\text{SLC}_{Sim}$ 17.1; $\text{TVG}_{F1}$ 34.3, $\text{TAL}_{F1}$ 24.3 (identical across StreamBridge variants — shared activation model). Baselines: **Dispider** $\text{DVC}_{F1}$ 33.8 / $\text{DVC}_{Sim}$ 18.9 / $\text{SLC}_{F1}$ 18.8, but higher $\text{TVG}_{F1}$ 36.1 / $\text{TAL}_{F1}$ 27.3; **VideoLLM-Online** far lower ($\text{DVC}_{F1}$ 24.0, $\text{SLC}_{F1}$ 9.9). Takeaway: StreamBridge wins on the *similarity/description* metrics (context-aware captions) — attributed to decoupling timing from generation — while Dispider's dedicated grounding still edges the pure temporal-grounding $F1$s.

**General video understanding preserved (Table 2, Qwen2-VL-7B‡, verified):** MVBench 64.4 (↓2.6), PerceptionTest 69.9 (↑7.6), TempCompass 71.1 (↑3.2), EgoSchema 66.9 (↑0.2), MLVU 69.6, VideoMME(w/o subs) 64.4 (↑1.1); LongVideoBench 59.1 (the one dip). Oryx-1.5-7B‡ reaches VideoMME 65.5 (↑6.7). Streaming adaptation is roughly ability-neutral or positive on offline tasks.

**Ablations:** round-decayed compression 71.30 OVO vs truncation 68.88 vs uniform 69.91; removing StreamingQA-120K costs ~5–6% across benchmarks; $\mathrm{MaxLen}$ sweet spot ~16k tokens.

No Zotero highlights present.

Takeaways to remember: (1) you can *retrofit* streaming onto strong offline Video-LLMs without retraining the generator and without hurting offline ability; (2) **decoupling the "when to speak" decision into a tiny per-frame classifier** is the clean way to get proactive response while keeping generation quality; (3) synthetic long-stream interleaved instruction data (Stream-IT) is what actually unlocks the multi-turn gains — the framework alone can even slightly regress a base model (LLaVA-OV) until Stream-IT fine-tuning.

## How it connects (evolution)
- [[videollm-online]] — the streaming-dialogue predecessor whose interleaved "always-on" LM must decide when to talk; StreamBridge instead *offloads* that timing to a separate model. Also an ET-Bench baseline it beats.
- [[dispider]] — the disentangled perception/decision/reaction streaming model; the strongest ET-Bench proactive baseline here (higher $\text{TVG}/\text{TAL}$ $F1$, lower similarity scores).
- [[mmduet]] — proactive/duet-style streaming response with a response head; shares the "separate response-timing signal" motif.
- [[dispider]], [[streammind]] — contemporaneous proactive-timing designs; StreamBridge's activation model is a minimal, model-agnostic take on the same "response-head" idea.
- [[proactive-response]] — the sub-topic hub: StreamBridge is the offline-to-streaming adaptation entry.
- [[streaming-benchmarks]] — it is evaluated on OVO-Bench and Streaming-Bench real-time tasks, plus ET-Bench.

## Open questions / limitations
- The activation model is trained on soft last-$P\%$ labels with a fixed threshold $\alpha$; the paper doesn't deeply probe latency vs. precision of triggering (early vs. exact-moment firing) or false-trigger rates in the wild.
- Round-decayed compression is lossy average-pooling of old rounds — for very long streams needing precise *old* detail (long-horizon grounding), oldest-first decay could silently erase the evidence.
- Stream-IT is GPT-4o-synthesized from concatenated *unrelated* clips ordered by similarity; how well this proxy matches genuinely continuous real-world streams (robotics/driving, the motivating use cases) is untested there.
- Gains are uneven across base models (LLaVA-OV even regresses under the framework alone), so "any offline model" adaptation is not uniformly free — it leans on Stream-IT fine-tuning.

*Verification: OVO-Bench / Streaming-Bench AVG numbers cross-checked against the rendered Table 1; ET-Bench $F1$/Sim against rendered Table 3; general-video benchmarks against rendered Table 2; method/dataset/algorithm details from the arXiv HTML (§3.2–§3.2.3, Algorithms 1–2) and the paper's own prose. Activation threshold $\alpha\approx0.35$ is from the HTML summary and not independently re-verified against a table.*

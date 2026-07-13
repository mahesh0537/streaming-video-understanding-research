---
zotero_key: null
authors: Rui Qian, Shuangrui Ding et al. (CUHK + Shanghai AI Laboratory)
year: 2025
arxiv: 2501.03218
pdf: https://arxiv.org/pdf/2501.03218
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# Dispider: Enabling Video LLMs with Active Real-Time Interaction via Disentangled Perception, Decision, and Reaction

**Lineage role:** The trigger-head archetype for proactive streaming video LLMs — an always-on *lightweight* perception stream feeds a *separate* binary decision head that fires a trigger, and only then does a *heavier asynchronous* reaction LLM generate the answer; the three roles run on distinct modules so response decoding never blocks perception (CVPR 2025).

## Problem — what was limited before this paper (short)
Prior online video LLMs (notably [[videollm-online]]) fold perception, the decide-when-to-speak signal, and answer generation into a *single* autoregressive LLM. That creates two conflicts: (1) perception and decision want a *fine, fast* per-frame scale while a full response wants a *coarse, slow* generative scale, and (2) autoregressive decoding of a response **blocks** the model from continuing to watch the stream — during a long generation the model goes "blind." So a single-LLM design cannot simultaneously keep monitoring and answer well; proactive-output timing on streaming benchmarks was near-zero.

## Key idea — the core insight, 2-4 sentences
Disentangle the three capabilities into three modules that operate at their own scales and run asynchronously. A lightweight scene-based perception module continuously encodes the stream into non-uniform clips; a compact decision LLM reads memory + clip + question and, on a special `⟨TODO⟩` token, does a binary classify — *wait* or *respond*; and a separate larger reaction LLM generates the detailed answer only when triggered, retrieving relevant history, while the perception+decision loop keeps running uninterrupted. The trigger is a switch, not a takeover, so perception is never blocked by decoding.

![[dispider.png]]
> **Crux (Figure 1).** Dispider's disentangled paradigm (top) — perception → decision → *trigger* → asynchronous reaction — versus VideoLLM-online (bottom), where perception and a *blocking* reaction share one LLM and are mutually exclusive (perception must switch off while responding). *Qian, Ding et al. (2025), arXiv:2501.03218. Embedded for personal research reference.*

## Method + math — the mechanism, then the central objective

**1. Scene-based Perception (streaming, non-uniform clips).**
Rather than fixed-length frame windows, the video is segmented into scene-coherent clips. Frames are sampled at intervals, encoded with a pretrained SigLIP encoder into L2-normalized embeddings, and a scene boundary is placed where consecutive-frame cosine similarity drops below a threshold; an exclusion window after each boundary prevents degenerate ultra-short clips. Each clip $i$ is compressed to a clip feature block $F_i$ plus a special clip-indicator token. This is the "light" perception scale — cheap, always running.

**2. Real-time Response Decision (compact LLM + binary head).**
Let $V_{[0,t]}$ be the stream up to time $t$ and $C_t$ the running context (question text + memory). The decision policy is a binary action on the final-layer embedding of a `⟨TODO⟩` token:
$$a_t = \pi\big(C_t,\, V_{[0,t]}\big) \in \{\text{wait},\ \text{respond}\}.$$
The interleaved input the decision LLM consumes is built incrementally. At the moment a user question $Q$ arrives at timestamp $q_{\text{pos}}$:
$$F[0:q_{\text{pos}}] \oplus Q \oplus \langle\text{TODO}\rangle,$$
where $\oplus$ is concatenation. Features up to $q_{\text{pos}}$ are then pooled into a **global memory** $M$, and the sequence is extended with later clips so decisions keep being made as new clips stream in:
$$M[0:q_{\text{pos}}] \oplus Q \oplus F[q_{\text{pos}}:] \oplus \langle\text{TODO}\rangle.$$
When the model has already emitted $k$ answers (at timestamps $a^1_{\text{pos}}, \dots, a^k_{\text{pos}}$), each answer is marked with a `⟨ANS⟩` token and folded back into memory, so the ultimate multi-answer format is:
$$
\begin{aligned}
& M[0:q_{\text{pos}}] \oplus Q \oplus \\
& M[q_{\text{pos}}:a^1_{\text{pos}}] \oplus \langle\text{ANS}\rangle \oplus \\
& M[a^1_{\text{pos}}:a^2_{\text{pos}}] \oplus \langle\text{ANS}\rangle \oplus \cdots \\
& F[a^k_{\text{pos}}:] \oplus \langle\text{TODO}\rangle.
\end{aligned}
$$
A binary classification head sits on the final-layer `⟨TODO⟩` embedding and is trained with standard **binary cross-entropy** to predict *respond* vs *wait* at each timestamp. Crucially, **none of the tokens the decision LLM consumes come from the reaction module's generated text** — that is what keeps the decision loop unblocked and continuously monitoring.

**3. Asynchronous Interaction / Reaction (larger LLM).**
Once *respond* fires, a dedicated larger LLM generates the fine-grained answer. For the $k$-th answer it retrieves the relevant historical clips (by similarity to the `⟨TODO⟩`/query embedding) to support multi-hop temporal reasoning, and can emit a `⟨SILENT⟩` token as a secondary filter — a way to "rethink" and stay quiet if the decision module's trigger was a false alarm. Because reaction runs asynchronously on a separate module, perception + decision keep advancing over the incoming stream while the answer is being decoded.

**Special tokens (the glue):** `⟨TODO⟩` = decision probe (should I speak now?); `⟨ANS⟩` = marks/segments an emitted answer so memory stays consistent; `⟨SILENT⟩` = final-LLM veto to suppress a spurious response.

## Explicit design choices
- **Three separate modules, three scales:** light perception (SigLIP + compression), a *compact* decision LLM, and a *larger* reaction LLM — instead of one shared LLM.
- **Non-uniform, scene-based clipping** (SigLIP cosine-similarity boundaries + exclusion window) rather than fixed 16-frame / uniform windows — ablation shows this helps both QA and streaming metrics.
- **Binary decision head on `⟨TODO⟩`** trained with BCE — proactive timing is a per-timestamp classification, decoupled from answer generation.
- **Global memory pooling** of pre-question features ($M[0:q_{\text{pos}}]$) so long history is compressed but retained; answered segments re-pooled behind `⟨ANS⟩`.
- **Decision input excludes reaction's generated tokens** → non-blocking, continuous monitoring (the core anti-"blindness" invariant).
- **`⟨SILENT⟩` as a second-stage filter** in the reaction LLM to cut false-positive triggers.
- **Retrieval-augmented reaction:** relevant historical clips fetched for multi-hop temporal reasoning at answer time.
- **Runs at 1 fps input**, 224×224 frames; two-stage training (streaming processor + decision first, then freeze and train the interaction module).

## Key results / what to remember
Numbers below are verified against the paper's Tables 1–4 (Dispider = 7B, 1 fps).

- **StreamingBench (Table 1).** Dispider **Overall 53.12**, vs VideoLLM-online **32.48** and Flash-VStream **24.04**. Real-Time Visual Understanding (All) **67.63**; Omni-Source (All) **35.66**; Contextual (All) **33.61**. Proprietary refs: Gemini 1.5 Pro 67.07, GPT-4o 60.15.
- **Proactive Output (PO) subtask on StreamingBench:** Dispider **25.34**, versus VideoLLM-online **3.92** and Flash-VStream **1.96** — roughly a 6× jump, the headline proactive-timing result.
- **Long/offline video QA (Table 2):** EgoSchema **55.6** (> VideoChat2 54.4), MLVU **61.7** (VideoXL leads at 64.9), VideoMME **57.2** (> Kangaroo 56.0, VideoXL 55.5) — conventional QA is *not* sacrificed for streaming ability.
- **ETBench streaming setting (Table 3):** Dispider (1 fps) TVG$_{F1}$ **36.1**, DVC$_{F1}$ **33.8**, SLC$_{F1}$ **18.8**, VHD$_{F1}$ **54.2** — vs VideoLLM-Online (2 fps) TVG$_{F1}$ 13.2, DVC$_{F1}$ 24.0.
- **ETBench conventional setting (Table 3):** Dispider TVG$_{F1}$ **43.6**, EPM$_{F1}$ **17.2**, *without* specialized time tokens — beating time-token models like ETChat (TVG 38.6).
- **Ablations:** scene-based vs uniform clipping (Table 4) — MLVU 61.7 vs 59.8, VideoMME 57.2 vs 55.4, TVG 36.1 vs 34.5. Removing `⟨TODO⟩` collapses streaming TVG$_{F1}$ (36.1 → 26.3), confirming the decision probe is load-bearing; removing `⟨ANS⟩`/`⟨SILENT⟩` gives smaller drops.

No Zotero highlights present.

Takeaways: the paper's contribution is *architectural disentanglement* — separating "when to speak" (a cheap binary head) from "what to say" (a heavy async decoder) is what unlocks both proactive timing (6× PO gain) and preserved offline QA. The `⟨TODO⟩`/`⟨ANS⟩`/`⟨SILENT⟩` token protocol is the reusable interface; the non-blocking decision loop (never feeding reaction's tokens back into decision) is the reusable invariant.

## How it connects (evolution)
- [[videollm-online]] — the single-LLM predecessor Dispider explicitly critiques (blocking reaction, mutually-exclusive perception/response); Dispider's whole design is a fix for it.
- [[proactive-response]] — sub-topic hub; Dispider is a canonical trigger-head instance of proactive output.
- [[mmduet]] — contemporaneous proactive/duet-style streaming interaction; alternative approach to deciding when to respond.
- [[streammind]] — another decision-gated proactive streaming design; compare trigger mechanisms.
- [[streamingbench]] — the benchmark Dispider headlines on (Proactive Output subtask).
- [[proactivevideoqa]] — proactive video QA evaluation lineage that Dispider's PO results speak to.

## Open questions / limitations
- **Extra parameters/compute:** two LLMs (compact decision + larger reaction) plus a perception encoder — heavier footprint than a single-LLM online model; real-world latency of the async handoff isn't stress-tested at scale here.
- **Trigger reliability:** proactive timing still tops out at ~25 PO on StreamingBench — far from human; `⟨SILENT⟩` is a patch over false positives rather than a calibrated policy.
- **Scene-boundary sensitivity:** clip quality hinges on the SigLIP cosine-similarity threshold and exclusion window; robustness to gradual scene changes / static long takes is unclear.
- **Retrieval scope:** multi-hop reasoning depends on clip retrieval by similarity — no explicit test of failure when the relevant evidence is many clips back or visually similar to distractors.

*Verification: architecture and equations checked against the rendered PDF (Figure 1 schematic page 1, Figure 2 + §3.2/§3.3 sequence formulas page 4); all headline numbers cross-checked against the rendered Tables 1–4 (StreamingBench, long-video QA, ETBench, clip-segmentation ablation) on pages 7–8 of arXiv:2501.03218v1.*

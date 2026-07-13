---
zotero_key: null
authors: Rui Qian et al. (CUHK + Shanghai AI Laboratory)
year: 2024
arxiv: 2405.16009
pdf: https://arxiv.org/pdf/2405.16009
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# Streaming Long Video Understanding with Large Language Models

**Lineage role:** VideoStreaming (NeurIPS 2024) reframes long-video LLMs from "stuff all frame tokens into the LLM" to a *memory-propagated streaming encoder* that condenses each clip into a fixed-length memory conditioned on the previous clip's memory, then answers a question by selecting only a constant number V of the most question-relevant clip memories — decoupling video length from the LLM's token budget.

## Problem — what was limited before this paper (short)
Video LLMs feed frame tokens directly into the LLM, so cost grows with video length and long videos are intractable. Prior fixes each break down: sparse sampling / temporal pooling loses temporal detail; token-merging (LLaMA-VID's 2 tokens/frame) ignores temporal relations across frames; memory-bank methods (MovieChat) accumulate history but have no explicit time indicators, so they fail on *moment-specific* ("breakpoint") questions; and caption-bridge methods (divide→caption→aggregate with a text LLM) cannot be trained end-to-end and are bottlenecked by caption quality. What was missing: a length-agnostic, end-to-end trainable encoder that keeps temporal structure and can retrieve the right moments for a given question.

## Key idea — the core insight, 2-4 sentences
Insert a small trainable language model (Phi-2, 16 layers, 1.3B) *between* the vision encoder and the answering LLM (Vicuna-7B) to iteratively compress the video clip-by-clip. Each clip is encoded into a fixed number of "summarization" memory tokens, and crucially the encoding of clip $k$ is conditioned on the memory $H_{k-1}$ propagated from the previous clip plus an explicit time prompt — so a constant-length memory carries the whole history while retaining temporal dynamics. At question time, an adaptive (Gumbel-Topk) selector picks the V clip memories most relevant to the question, so the answering LLM always sees a **constant** number of tokens regardless of video length.

![[videostreaming.png]]
> **Crux (Figure 1).** (a) Overview: a long video is split into clips, each streaming-encoded into a compact memory $H_k$ + clip indicator $\hat H_k$; a question then selects a constant subset of memories (green ✓ / red ✗) fed to the LLM. (b) The streaming encoder for the k-th clip fuses historical memory $H_{k-1}$, a time prompt, frame features $F_k$ and summarization tokens $S_k$ inside a small LM to emit the condensed representation. *Qian et al. (2024), arXiv:2405.16009. Embedded for personal research reference.*

## Method + math — the mechanism, then the central objective/equations
**Setup.** A long video is segmented into $K$ short clips. A frozen CLIP ViT-L extracts frame features; adjacent spatial tokens are concatenated to compress spatial tokens by 75%, giving per-clip frame features $F \in \mathbb{R}^{TN\times C}$ ($T$ frames, $N$ tokens/frame). A small language model $g(\cdot)$ (Phi-2) with a **modified attention mask** does the condensing.

**Single-clip condensing (Sec 3.1).** Learnable *summarization tokens* $S \in \mathbb{R}^{TP\times C}$ ($P$ tokens per frame) are appended to the frame features. The modified causal mask $M$ forces information to flow only from frame tokens into the summarization tokens (summarization tokens attend to frames + preceding summarization tokens; frame tokens do not attend to summarization tokens), so all clip content is consolidated into $S$'s output positions:
$$
H = g\big([\,F \circ S\,]\big) \in \mathbb{R}^{TP\times D}
$$
where $\circ$ is concatenation along the token axis and $H$ is the condensed clip memory ($TP$ tokens, far fewer than $TN$).

**Memory-propagated streaming encoding (Sec 3.2).** Clips are encoded sequentially. For the $k$-th clip the encoder additionally ingests the previous memory $H_{k-1}$ (historical context), the frame features $F_k$, the summarization tokens $S_k$, and an extra set of *clip-indicator* summarization tokens $\hat S_k$:
$$
H_k,\ \hat H_k = g\big([\,H_{k-1} \circ F_k \circ S_k \circ \hat S_k\,]\big)
$$
$H_k$ is the propagated compact memory (fixed length, carried to clip $k{+}1$); $\hat H_k$ is a short *clip indicator* used later for retrieval. A **time prompt** ("this contains a history of 0 to $t_{k-1}$ seconds, and a clip sampled in $t_{k-1}$ to $t_k$ seconds") is tokenized and prepended, injecting explicit absolute timestamps so the model can answer moment-specific questions. Because $H_{k-1}$ has fixed size, the memory stays constant-length for arbitrarily long video.

**Adaptive memory selection (Sec 3.3).** Given a question, the final global memory $H_K$ + question text is passed through the encoder to produce an *instruction indicator*. Cosine similarity $s$ is computed between this indicator and all historical clip indicators $\{\hat H_1,\dots,\hat H_K\}$. A differentiable **Gumbel-Topk** picks the $V$ most relevant clips:
$$
I = \text{Gumbel-Topk}(s, V) \in \{0,1\}^K, \qquad \tilde H = \{\, H_k \mid I_k = 1 \,\}
$$
Only the $V$ selected memories (plus, in the best config, the clip-level and global-memory prompts) are projected and fed to the LLM — a **constant** token count. Gumbel noise makes the top-$V$ selection differentiable so the selector trains end-to-end.

**Progressive training (Sec 3.4).**
- *Stage 1 — single-clip:* pretrain on ~790K image/video caption pairs (CC3M + WebVid-2.5M subset) then instruction-tune on ~763K QA pairs; the Phi-2 encoder is frozen then unfrozen in sequence.
- *Stage 2 — streaming long video:* train on 25K movie QA + 300K multi-round QA (from a Panda-70M subset) + 20K synthesized long videos (concatenated short clips). A **warm-up on ~30K pairs uses temporal supervision** (a KL-divergence loss on the selection distribution vs. ground-truth relevant segments); the remaining ~315K pairs use only **weak supervision** (answer loss propagates through Gumbel-Topk). AdamW optimizer.

## Explicit design choices
- **Two-model sandwich:** frozen CLIP ViT-L vision encoder → trainable small LM (Phi-2, 16 layers, 1.3B) as *streaming encoder* → Vicuna-7B answering LLM. Encoder is trainable end-to-end with the task (unlike caption-bridge baselines).
- **Modified attention mask** so frame tokens write into summarization tokens but not vice-versa — a learned pooling that consolidates a clip into $P$ tokens/frame.
- **75% spatial token reduction** by concatenating adjacent CLIP tokens before the encoder.
- **Explicit time prompt** with per-clip absolute timestamps — the fix for moment/breakpoint questions that memory-bank methods miss.
- **Clip indicators $\hat H_k$** as a lightweight retrieval key, separate from the memory content $H_k$.
- **Differentiable Gumbel-Topk** selection of a constant $V$ memories → constant LLM input length decoupled from video length.
- **Hyperparameters:** $P = 4$ summarization tokens/frame and $V = 4$ selected clips → **256 total video tokens** to the LLM (the headline "constant video tokens").
- **Temporal warm-up (KL) then weak supervision** two-phase Stage-2 schedule.

## Key results / what to remember
Exact headline numbers with setting (verified against the paper's Table numbers via arXiv HTML):
- **EgoSchema (zero-shot, fullset):** 44.1% accuracy.
- **NExT-QA (zero-shot val, accuracy):** All 66.2 (Causal 65.1, Temporal 62.2, Descriptive 78.1).
- **NExT-GQA (grounded QA):** Acc@GQA 17.8, mIoP 32.2, IoP@0.5 31.0, mIoU 19.3 — i.e. it can localize *and* answer.
- **MovieChat-1K:** Global accuracy 90.4 / score 4.42; Breakpoint accuracy 54.9 / score 2.80.
- **VideoChatGPT benchmark:** Correctness 3.33, Detail 3.27, Context 3.73, Temporal 2.74, Consistency 3.15.
- **MovieNet-QA (100 hour-long movies, ~108 min avg):** vision-only, **256 tokens**, 5.32 s latency; Overview 2.65, Plot 3.13, Temporal 1.88.
- **Ablations (EgoSchema fullset unless noted):** no memory propagation 37.3 → full 44.1 (memory matters for global understanding); selecting last-4 clips gives 39.1 breakpoint acc vs 54.9 with adaptive selection (selection matters for moment questions); encoder depth 16 layers (44.1) beats 24 (43.8) and 32 (41.3) — bigger encoder overfits; temporal warm-up lifts NExT-GQA Acc@GQA from 9.8 → 17.8; time prompt: clip+memory prompts 44.1 vs memory-only 42.1 vs clip-only 40.5; best config $P{=}4, V{=}4$.

No Zotero highlights present.

Takeaways: the durable ideas are (1) a *trainable* recurrent compression LM that propagates fixed-length memory, so context length is the encoder's problem not the LLM's; (2) explicit timestamp prompts as the cure for moment-specific/breakpoint questions; (3) differentiable top-V memory retrieval to hold the LLM's video-token budget constant (256) at any video length — this is the "constant video tokens" property that later streaming-memory work builds on.

## How it connects (evolution)
- [[flash-vstream]] — contemporaneous constant-memory streaming architecture; both fix a memory budget independent of length (via STAR memory vs. propagated encoder memory).
- [[videollamb]] — recurrent memory bridge for long video that likewise propagates compressed memory clip-by-clip; a close design cousin.
- [[rekv]] / [[hermes-kv]] — later KV-cache-side approaches to the same "constant footprint over unbounded video" goal, but at the cache level rather than a learned encoder.
- [[streammem]] / [[streamkv]] — streaming-memory successors that inherit the fixed-length-memory + question-conditioned retrieval pattern.
- [[streaming-memory]] — the sub-topic hub this note anchors (memory-propagated encoding + question-relevant selection).

## Open questions / limitations
- **Sequential encoding is inherently serial** — clip $k$ needs $H_{k-1}$, so ingesting a long video cannot be parallelized across clips; latency grows linearly (reported 5.32 s on hour-long movies).
- **Fixed $V{=}4$ / 256 tokens** caps how much can reach the LLM; questions needing many dispersed moments may be starved despite good retrieval.
- **Retrieval depends on the clip indicators** — a wrong Gumbel-Topk selection is unrecoverable since unselected memories never reach the LLM; robustness of selection under distractor-heavy long video is untested at scale.
- **Not truly "online"** in the interactive sense — it streams the *encoding* but answers after the whole video is seen (needs $H_K$ for the instruction indicator), so it is not yet a live/proactive per-frame responder.

*Verification: equations ($H=g([F\circ S])$, memory-propagation recurrence, Gumbel-Topk selection) and all numbers cross-checked against the arXiv HTML (Tables 2–10) of 2405.16009; crux figure cropped from the paper PDF page 2 (Figure 1). Title corrected to the paper's actual title "Streaming Long Video Understanding with Large Language Models" (task-prompt alias "…with Constant Video Tokens").*

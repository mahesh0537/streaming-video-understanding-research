---
zotero_key: null
authors: Zhenyu Yang et al. (Institute of Automation, CAS · UCAS · ShanghaiTech · Kuaishou Technology · Peng Cheng Laboratory)
year: 2025
arxiv: 2511.05299
pdf: https://arxiv.org/pdf/2511.05299
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# LiveStar: Live Streaming Assistant for Real-World Online Video Understanding

**Lineage role:** Replaces the EOS/"silence-token" gate that prior online Video-LLMs use for response timing with a training-free, single-pass *perplexity-verification* criterion (SVeD) — deciding *when* to speak by how much a new frame inflates the perplexity of the current caption — and ships OmniStar, a 5-task streaming benchmark whose RNG split becomes a reference proactive-narration testbed.

## Problem — what was limited before this paper (short)
Online/streaming Video-LLMs must decide *when* to emit a response as frames arrive, not just *what* to say. The dominant approach ([[videollm-online]], VideoLLM-MoD, and relatives) trains the model to emit an EOS/silence token whenever it should stay quiet, then speaks only on non-EOS steps. The authors argue this "silence token" paradigm has four structural flaws: (1) a severe response-silence imbalance (~1:35 speak-to-silence steps in 1-minute videos), which biases training toward silence; (2) conflicting supervision across visually near-identical adjacent frames (one labeled speak, the next silent); (3) misalignment with the LLM's pretraining objective, since a special decision token is grafted onto a next-token model; and (4) vocabulary confusion between the control token and content tokens. The net effect is poor timing and degraded narration quality on genuinely streaming tasks.

## Key idea — the core insight, 2-4 sentences
Instead of learning an explicit silence classifier, LiveStar keeps the model as a pure streaming captioner and reframes *response timing as a verification problem*: it maintains the current decoded caption $[\mathrm{Dec}]$ and, for each new frame, checks whether that caption still "fits" — measured by perplexity. When a fresh frame makes the held caption much more perplexing (perplexity jumps above a scaled threshold), the scene has changed enough to warrant speaking, so the decoding gate fires and a new caption is generated; otherwise the model stays silent and just carries the caption forward. Training uses interleaved frame-caption sequences with Streaming Causal Attention Masks (SCAM) so the model learns incremental, temporally-aware alignment without EOS supervision, and a Peak-End memory-compression scheme keeps inference cheap on 10-minute-plus streams.

![[livestar.png]]
> **Crux (Figure 2).** The SVeD inference loop: for each new frame the VLM does one forward pass, computes the perplexity of the current caption $[\mathrm{Dec}^k]$, compares it against $\alpha\cdot$(previous perplexity); if greater it appends a new caption ($\mathrm{Dec}^{k+1}$, a "speak" step), if less it swaps the frame into context and stays silent — yielding the Cap/Silent trace at the bottom without any EOS token. *Yang et al. (2025), arXiv:2511.05299. Embedded for personal research reference.*

## Method + math — mechanism, then the central objectives in full

### Streaming video-language alignment (training objective)
The static image-text objective is reformulated as predicting a clip's text conditioned on the accumulated streaming context. For a semantic clip $C_k=\{t_i\}_{i=m}^{n}$ (a run of consecutive frames sharing one caption $[\mathrm{Txt}^k]$):

$$\max\; P\big([\mathrm{Txt}^k]\;\big|\;[\mathrm{Ctx}^{<t_i}],\,[\mathrm{Frm}^{t_i}]\big),\quad \forall\, t_i\in C_k$$

where $[\mathrm{Frm}^{t_i}]$ is the frame at timestamp $t_i$ and $[\mathrm{Ctx}^{<t_i}]$ is the accumulated multimodal context (prior frames + prior captions). Training data is built as **interleaved frame-caption sequences** in a chat format: each turn is a frame $[\mathrm{Frm}^{t_i}]$ followed by its caption $[\mathrm{Cap}^k]$; consecutive frames of the same semantic clip reuse the caption, and to avoid overfitting the caption is randomly drawn from a pool of $M$ paraphrases $[\mathrm{Cap}^k_j]$.

### Streaming Causal Attention Masks (SCAM)
A plain causal mask fails here for three reasons the authors call out: (1) **leakage** — because clip frames share an identical caption, the model can trivially copy the already-visible caption instead of grounding in vision; (2) **token-specific context** — within the caption currently being generated, tokens must still see earlier tokens of that same caption for coherence; (3) **scene-transition signaling** — the *last* caption of each clip must persist to downstream frames to mark clip boundaries. SCAM replaces the standard causal mask with a matrix $\mathrm{Mask}^{\le t_i}$ that blocks attention to all **non-terminal** caption tokens of clips $\{C_1,\dots,C_k\}$ prior to $t_i$ (keep terminal captions as boundary markers, hide the intermediate repeats). The final training objective:

$$\max\; P\big([\mathrm{Cap}^k_j]\;\big|\;[\mathrm{Ctx}^{<t_i}\{\mathrm{Mask}^{\le t_i}\}],\,[\mathrm{Frm}^{t_i}]\big),\quad \forall\, t_i\in C_k$$

The loss is standard autoregressive cross-entropy computed over **language tokens only**; inter-frame visual segments are excluded from the loss via SCAM. There is no EOS/silence-token objective anywhere.

### Streaming Verification Decoding (SVeD) — the response-timing mechanism
At inference the model holds the current decoded caption $[\mathrm{Dec}]$ (of $N$ tokens). Its perplexity given the context and a frame is

$$\mathrm{PPL}^{t_i}([\mathrm{Dec}]) = \sqrt[N]{\dfrac{1}{P\big([\mathrm{Dec}]\mid[\mathrm{Ctx}^{<t_i}],[\mathrm{Frm}^{t_i}]\big)}}$$

i.e. the $N$-th root of the inverse autoregressive probability (equivalently $\exp$ of the mean token NLL). For each incoming frame $[\mathrm{Frm}^{t_j}]$, SVeD does **one forward pass** to recompute the perplexity of the *same* held caption. The **decoding gate** fires — generate a new/updated caption — iff

$$\mathrm{PPL}^{t_j}([\mathrm{Dec}]) \;>\; \alpha\cdot\mathrm{PPL}^{t_i}([\mathrm{Dec}]),\qquad \alpha\ge 1$$

Intuition: if the new frame makes the current caption much less likely (perplexity spikes), the scene has drifted and the caption is stale → speak. Otherwise the model stays **silent**: it swaps the new frame into context, moves $[\mathrm{Dec}]$ to the end of the context to preserve coherence, and waits. $\alpha$ (default $1.03$) is the sole timing hyperparameter — a training-free knob trading responsiveness for redundancy. On a "speak" step it appends the new caption $\mathrm{Dec}^{k+1}$. This is the whole "no silence token" claim: timing is a verification test on perplexity, not a learned control token.

### Peak-End memory compression + streaming KV cache
For long streams, SVeD probabilistically prunes frames older than a window $W$ (default $40$), with deletion likelihood proportional to a frame's relative perplexity within its semantic clip and its elapsed time — inspired by the cognitive **Peak-End rule**: keep low-perplexity keyframes (treated as more informative "peaks") and each clip's final caption (the "end"), drop the rest. A dual-level **streaming KV cache** (intra-dialogue caches for frame-level processing, inter-dialogue caches for long-context preservation) gives the reported ~1.53x inference speedup.

## Explicit design choices
- **No EOS/silence token at all** — timing is a perplexity-verification gate (SVeD), fully training-free at inference; only $\alpha\ge1$ is tuned.
- **Perplexity gate:** speak iff $\mathrm{PPL}^{t_j}>\alpha\cdot\mathrm{PPL}^{t_i}$ of the *held* caption; $\alpha=1.03$ default. Single forward pass per frame (cheap verification).
- **SCAM mask** replaces causal masking during training: hide non-terminal intra-clip caption tokens (anti-leakage), keep intra-caption visibility (coherence), persist terminal captions (scene boundaries).
- **Interleaved frame-caption instruction tuning** with a per-clip pool of $M$ paraphrased captions; default $M=1$ (larger $M$ helps semantics slightly but hurts timing).
- **Peak-End memory compression:** prune frames beyond window $W=40$ by relative-PPL × elapsed-time; keep low-PPL keyframes + final clip captions.
- **Dual-level streaming KV cache** (intra- + inter-dialogue) for the ~1.53x speedup.
- **Base model:** InternVideo2.5 (InternViT vision encoder + InternLM2.5-7B), 16 tokens/frame at 448×448, 1-4 FPS, 8K context.
- **Training:** two phases, ~83K samples (63K Phase-I from ActivityNet/Shot2Story/Ego4D/MVBench + 20K Phase-II from OmniStar), 8× A800, lr $4\times10^{-5}$, batch 32, 1 epoch.
- **OmniStar benchmark:** 20,137 expert-annotated streams (19,137 train / 1,000 test) across 15 real-world scenario families (46 fine-grained categories); 5 tasks — RNG, OTG, FDQ, COQ, MIQ.

### OmniStar task taxonomy + metrics (the benchmark "math")
- **RNG** (Real-time Narration Generation): emit temporally-coherent narration at semantic transitions.
- **OTG** (Online Temporal Grounding): localize the segment matching a query using only past frames.
- **FDQ** (Frame-level Dense QA): a standing query whose correct answer changes over time.
- **COQ** (Contextual Online QA): multi-turn, semantically-linked question chains distributed over time.
- **MIQ** (Multi-turn Interactive QA): multi-turn dialogue that must stay temporally aware.

Timing/quality metrics:
- **TimDiff** ↓: mean absolute time gap between model responses and ground-truth response times; a missed response is penalized with the full scene duration.
- **TimRedun** ↓: average number of unnecessary/redundant responses per scene.
- **TimCover** ↑: fraction of scenes covered ($1$ if the scene got $\ge1$ valid response, else $0$).
- **SemCor** ↑: GPT-4o semantic score, mean of Semantic Accuracy, Language Quality, Information Completeness (each 0-10).
- **SumFluen** ↑: GPT-4o holistic fluency of the concatenated responses (Writing Logicality, Language Fluency, Conciseness, Semantic Consistency, Narrative Completeness).
- **PPL** (reference-only, not cross-model comparable due to vocab differences) and **TokAcc** for offline/fixed-decoding evaluation.

## Key results / what to remember
Headline claim: on OmniStar's five tasks, LiveStar averages **+19.5% SemCor** and **−18.1% TimDiff** vs. prior online Video-LLMs while improving **FPS by +12.0%**.

- **OmniStar-RNG, online eval** (Table 1): LiveStar SemCor **3.19**, SumFluen **4.25**, TimDiff **1.91**, TimRedun **0.95**, TimCover **0.71** — vs [[mmduet]] SemCor 1.93 / TimDiff 2.32, [[videollm-online]] SemCor 1.68 / TimDiff 2.67, VideoLLM-MoD SemCor 1.66 / TimDiff 2.54. Human reference: SemCor 6.09, TimDiff 1.08.
- **OmniStar-RNG, offline (fixed-decoding) eval** (Table 1): LiveStar PPL **5.14**, TokAcc **0.62**, SemCor **4.62**, SumFluen **4.55** — best of the online-assistant / open-source group; MMDuet next at SemCor 4.29.
- **OmniStar multi-task, online** (Table 2): FDQ SemCor **6.44** (vs MMDuet 4.78), COQ **5.85** (vs 5.71), MIQ **5.78** (vs 5.62); OTG TimDiff **3.57** (vs MMDuet 4.42, VideoLLM-online 9.69). LiveStar FPS **3.82** vs MMDuet 0.91 on 5-min videos (+12.0% over next-best that is not the slow MMDuet — see paper note).
- **Ego4D narration stream, offline** (Table 3): LiveStar PPL **1.97**, TimeDiff **1.76**, TokAcc **61.1%** — TokAcc is +8.7% over next-best [[lion-fs]] (52.4%); MMDuet TokAcc 39.3%.
- **SVBench** (Table 6): LiveStar zero-shot Dialogue OS **51.37** / Streaming OS **48.15**; fine-tuned **58.95 / 55.87** (~+15.37% avg), surpassing open-source Qwen2.5-VL (57.57/52.84) and MiniCPM-V 2.6, approaching GPT-4o (62.57/59.97).
- **Ablations:** Peak-End + full KV cache gives SemCor 3.19 / TimDiff 1.91 / FPS 3.82 (vs Peak-End with no KV cache: FPS 2.50) → the ~1.53x speedup (Table 4). Caption pool: $M=1$ best for timing (TimDiff 1.91) though $M=3$ nudges SemCor up to 3.24 (Table 5).

No Zotero highlights present.

Takeaways: (1) response timing can be a *verification* signal (perplexity of the held caption under a new frame) rather than a learned control token — cheap, training-free, one knob $\alpha$; (2) the silence-token imbalance/leakage problems are addressed at the data+masking level via SCAM, not by reweighting a classifier; (3) OmniStar-RNG is a useful proactive-narration reference split with explicit timing metrics (TimDiff/TimRedun/TimCover) separate from semantic quality.

## How it connects (evolution)
- [[videollm-online]] — the EOS/silence-token streaming-dialogue paradigm LiveStar is explicitly reacting against (VideoLLM-MoD is its efficient variant, also a baseline).
- [[mmduet]] — the strongest prior proactive/dense-narration baseline on OmniStar; also uses a learned "informative-head" response trigger rather than perplexity verification.
- [[lion-fs]] — Ego4D narration baseline LiveStar beats on TokAcc; a frame-selection streaming approach.
- [[dispider]] — a decoupled perception/decision/reaction streaming design; contrasting architecture for the same "when to respond" problem.
- [[streammind]] / [[vispeak]] — other proactive "when to speak" streaming assistants using different gating signals.
- [[proactive-response]] — the sub-topic hub this note anchors (perplexity-gated timing as a distinct design axis).

## Open questions / limitations
- The perplexity gate rests on a single scalar $\alpha$ tuned globally; robustness across scene types / motion speeds (fast action vs. slow tutorials) and to caption length/normalization is not deeply stressed.
- Timing quality still trails humans (TimDiff 1.91 vs 1.08 on RNG) and SemCor 3.19 vs 6.09 — the method narrows but does not close the proactive-narration gap.
- SVeD verifies the *held* caption's perplexity; it can be slow to react to a genuinely new event whose caption would be low-perplexity, or over-fire on visually noisy but semantically stable stretches — TimRedun (0.95) remains non-trivial.
- Evaluation leans on GPT-4o judging (SemCor/SumFluen), inheriting LLM-judge biases; PPL is explicitly not cross-model comparable.

*Verification: SVeD/SCAM equations and all headline numbers checked against the arXiv HTML of 2511.05299 (Tables 1-6, method Sections 3.1-3.2) and the rendered PDF page 4 (Figure 2); figure cropped from the PDF page 4.*

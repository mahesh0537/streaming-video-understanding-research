---
zotero_key: null
authors: Guangzhi Sun, Yixuan Li, Xiaodong Wu, Yudong Yang, Wei Li, Zejun Ma, Chao Zhang (Tsinghua University + ByteDance)
year: 2025
arxiv: 2510.11129
pdf: https://arxiv.org/pdf/2510.11129
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# video-SALMONN S: Memory-Enhanced Streaming Audio-Visual LLM

**Lineage role:** Puts the long-term memory of a streaming audio-visual LLM into a *test-time-trained* MLP whose fast weights are updated online (reconstruction + long-span prediction loss) plus a prompt-dependent memory reader — letting a fixed-size memory carry 3-hour, 1 FPS audio-visual streams without a growing KV-cache.

## Problem — what was limited before this paper
Streaming/long-video LLMs must watch hours of frames but keep GPU footprint bounded, so they either (a) drop or merge tokens (similarity merging as in MovieChat, or pruning/compression like [[streamforest]]/PEMF, [[flash-vstream]], [[rekv]]), which loses fine detail and over-smooths in very long videos, or (b) keep everything and blow up the KV-cache. Explicit memory banks and KV compression trade recall for footprint; recurrent state-space alternatives (Mamba-2) compress but are trained once and can't specialise their compression to what a query actually needs. Existing benchmarks also rarely test *episodic* recall — answering from knowledge seen tens of minutes earlier — so the memory failure mode is under-measured.

## Key idea — the core insight
Instead of *storing* a fixed set of past tokens, store the long-term context inside the **weights of a small MLP that is trained on the fly** as the stream arrives (test-time training, TTT). The TTTMEM layer treats consecutive chunks of video encodings as mini-batches and takes an online gradient step on each, minimising a reconstruction loss (compress the current chunk) plus a **long-span prediction loss** that forces the fast weights to stay predictive of a chunk seen `T` steps earlier — a temporal-consistency regulariser that fights catastrophic forgetting. Its output tokens then pass a cosine-similarity token-discarding step to hold memory at a fixed `N` tokens, and an optional **modality-aware, prompt-dependent memory reader** selects which cached KV pairs each Transformer block attends to based on the actual question. Net effect: near-constant GPU memory, and streaming beats equal-budget non-streaming models on long-video and episodic tasks.

![[video-salmonn-s.png]]
> **Crux (Figure 1).** Video encodings $\mathbf{X}_t$ flow through the TTTMEM layer (fast weights $\mathbf{W}_t\!\to\!\mathbf{W}_{t+1}$), then similarity-based discarding keeps the memory tokens $\tilde{\mathbf{Z}}_t$ at a fixed size; text/timestamp tokens are added back and the LLM reads the fixed memory with an optional prompt-dependent reading mechanism. *Sun et al. (2025), arXiv:2510.11129. Embedded for personal research reference.*

## Method + math

**Overall flow.** Frames are sampled at up to 1 FPS / 360p, run through a frozen visual encoder + modality aligner (Qwen3-VL 8B backbone; audio via Whisper-Large-v3 + a Q-Former aligner over 0.5 s windows). The visual token stream is chopped into fixed-size chunks $\mathbf{X}_1,\mathbf{X}_2,\dots$ (~12–16 frames each) and fed *in order* to the TTTMEM layer, which behaves like an RNN over chunks. Audio tokens (~1/75 the count of visual tokens) bypass TTTMEM and go straight to the discarding stage.

**TTTMEM layer.** For a chunk $\mathbf{X}_t\in\mathbb{R}^{K\times d}$ the layer emits memory tokens and updates its own fast weights:
$$\mathbf{Z}_t,\ \mathbf{W}_t = \mathrm{TTTMEM}(\mathbf{X}_t,\ \mathbf{X}_{t-T},\ \mathbf{W}_{t-1}) \tag{1}$$
where $\mathbf{W}_{t-1}$ carries the history. The fast weights sit in an MLP $f(\cdot;\mathbf{W})$ (2 FC layers + GeLU, residual + LayerNorm, following the TTT formulation of Dalal et al. 2025). Two self-supervised losses drive the online update, using trainable projections $\theta_K,\theta_V,\theta'_K,\theta'_V,\theta_Q$:
$$l_{\text{recon}}(t) = \big\| f(\theta_K \mathbf{X}_t;\ \mathbf{W}_{t-1}) - \theta_V \mathbf{X}_t \big\|_2 \tag{2}$$
$$l_{\text{long-span}}(t) = \big\| f(\theta'_K \mathbf{X}_t;\ \mathbf{W}_{t-1}) - \theta'_V \mathbf{X}_{t-T} \big\|_2 \tag{3}$$
$$l(\mathbf{X}_t,\mathbf{X}_{t-T};\ \mathbf{W}_{t-1}) = l_{\text{recon}}(t) + l_{\text{long-span}}(t) \tag{4}$$
The reconstruction term makes the MLP compress the *current* chunk; the long-span term makes the same weights still predict a chunk from $T$ steps back, so distant context is not overwritten as the stream grows. One gradient step gives the fast-weight update
$$\mathbf{W}_t = \mathbf{W}_{t-1} - \eta\,\nabla l(\mathbf{X}_t,\mathbf{X}_{t-T};\ \mathbf{W}_{t-1}),$$
and the output (memory) tokens are read out with the query projection:
$$\mathbf{Z}_t = f(\theta_Q \mathbf{X}_t;\ \mathbf{W}_t). \tag{5}$$
A gating mechanism (à la Dalal et al.) modulates the update. Default prediction span $T=2$.

**Fixed-size memory by discarding, not merging.** The new tokens $\mathbf{Z}_t$ are concatenated with previous memory tokens: $\mathbf{Z}'_t=\mathrm{Concat}(\tilde{\mathbf{Z}}_{t-1},\mathbf{Z}_t)\in\mathbb{R}^{(N+K)\times d}$. They then **discard the $K$ tokens whose cosine similarity to their next token is highest**, i.e. drop $\arg\max \cos(\mathbf{Z}'_{t,n},\mathbf{Z}'_{t,n+1})$, leaving exactly $N$ tokens $\tilde{\mathbf{Z}}_t$. The authors deliberately *discard* rather than *merge* (MovieChat-style) — merging over-smooths in extremely long videos, whereas the fast-weight state already summarises the past, so the discrete tokens can stay sharp.

**Modality-aware prompt-dependent memory reading (Sec 3.3).** To let the memory be larger than the LLM's context while holding GPU footprint fixed, each Transformer block compresses its KV-cache *conditioned on the prompt*. Memory tokens are chunked (chunk size $m$); attention scores from the prompt tokens to each chunk give an importance score, and the reader keeps at most $m$ KV pairs with the highest importance for the multimodal input positions (Eqs. 6–10). Text/timestamp tokens are re-inserted at their original relative positions after TTTMEM so ordering is preserved.

**Two-stage training.** Stage 1: TTTMEM projections randomly initialised and jointly trained with the LLM (LoRA rank 128) on 1024 frames @ 4 FPS, 3 epochs (~48 h on 32×H800). Stage 2: freeze the TTTMEM projections $\theta$ but *keep the fast-weight update rule live*, train on 2048 frames, 1 epoch (~16 h). Visual encoder + aligner are frozen after initial training; only TTTMEM params and LoRA are trainable.

## Explicit design choices
- **Memory = fast weights of a 2-layer MLP** (8 heads, 512-dim each; 16 fast-weight blocks of 512×512), updated by online gradient descent — not a token bank.
- **Long-span prediction loss** with span $T$ (default $T=2$) as an explicit temporal-consistency regulariser against forgetting; ablation sweeps $T$ (Table 10).
- **Cosine-similarity token discarding**, chosen over merging (MovieChat) and K-means clustering, to avoid over-smoothing.
- **Audio bypasses TTTMEM** (only ~1/75 of visual token count) and enters at the discarding stage — keeps TTTMEM visual-only and cheap.
- **Text/timestamp tokens re-added at original relative positions** after memory processing.
- **Prompt-dependent modality-aware KV reading** per Transformer block to decouple memory size from context length.
- **Backbone:** Qwen3-VL 8B + Whisper-Large-v3 audio + Q-Former (0.5 s windows); default budget 16k visual memory tokens + 1k real-time perception tokens; 1 FPS / 360p to reach >3 h streams.
- **Two-stage schedule:** joint train, then freeze projections but keep the TTT update rule active.
- **New benchmark ELViM** (Episodic Learning in Videos / Memory): 1,849 questions over 1,021 target videos (up to ~2 h), with the queried knowledge appearing ≥30 min before the question, to isolate long-horizon episodic recall.

## Key results / what to remember
All numbers % accuracy, 16k memory-token setting (Table 1), verified against the paper's tables.

- **Video-MME (overall / Short / Medium / Long):** video-SALMONN S w/ reading **76.9 (82.5 / 76.8 / 71.3)** vs non-streaming video-SALMONN 2+ 75.6 (81.6/75.7/69.6) and Qwen3-VL 8B 69.7. On the **Long** split, +1.7 over the non-streaming baseline (69.6→71.3).
- **LVBench:** **55.6** (w/ reading) vs 52.7 non-streaming, +2.9.
- **VideoEvalPro:** **58.9** (w/ reading) vs 53.5 non-streaming, +5.4.
- **ELViM:** **46.7** (w/ reading) vs 32.5 non-streaming video-SALMONN 2+ and 28.1 Qwen3-VL — the headline ~15% absolute episodic-memory gain (28.1→46.7). Even w/o reading, 43.9.
- **StreamingBench (Real-time / Omni / Contextual):** 67.1 (78.9/57.5/41.5) w/ reading — on par with PEMF (67.1) and Similarity Merging (66.8).
- **Ablation (Table 2, Long-video / ELViM):** Similarity merging 69.4/35.4 → +TTTMEM (w/o Stage 2) 71.3/43.6 → +Stage 2 71.3/43.9 → +Reading 71.3/**46.7**. TTTMEM beats Mamba-2 (70.3/39.6) and vanilla TTT-video (70.0/42.3) as the memory mechanism.
- **Efficiency:** near-flat GPU memory (~22.5 avg / 24.4 GB peak w/ reading; Table 3); TTTMEM adds only ~0.1 s and reading ~0.4 s per sample. Figure 6: matches merging baselines' accuracy using **≤25% of the memory tokens**.
- **ELViM difficulty (Table 6):** even Gemini-2.5-Pro / Qwen3-VL score ~23% on the *incomplete-context* (episodic) version vs ~99% when given the full relevant clip — confirming the benchmark isolates memory, not perception.

No Zotero highlights present.

Takeaways: (1) online test-time training of an MLP is a viable *parametric* long-term memory that beats token-merging/pruning at equal or smaller budget; (2) a long-span prediction loss is the key ingredient for retaining distant context; (3) discarding beats merging for sharpness in very long streams; (4) a prompt-dependent reader gives most of the ELViM/VideoEvalPro gain by choosing *what* to recall per question; (5) episodic recall (ELViM) is where streaming memory decisively beats equal-budget non-streaming models.

## How it connects (evolution)
- [[streamforest]] — PEMF token-compression baseline it directly outcompetes on episodic recall; same fixed-memory streaming setting.
- [[flash-vstream]] / [[rekv]] — prior fixed-footprint memory/KV-compression streaming VLMs; TTTMEM is the parametric-memory alternative to their token banks.
- [[streammem]] / [[streamkv]] / [[hermes-kv]] — streaming KV-cache memory/compression neighbours; the modality-aware prompt-dependent reader is the analogous idea here.
- [[streaming-model-remember]] — episodic/long-horizon memory evaluation, the same failure mode ELViM targets.
- [[streamingbench]] — one of the streaming benchmarks it reports on.
- [[streaming-memory]] — sub-topic hub.

## Open questions / limitations
- The online gradient step and $T=2$ span are small; how the long-span loss scales to *multi-hour* dependencies (not just $T$ chunks back) and whether a single fast-weight MLP saturates is not fully characterised.
- Memory is a black-box parametric state — unlike a token bank it is hard to inspect or edit, and grounding a retrieved fact back to a timestamp is unclear.
- Gains concentrate on episodic/long benchmarks; on Short Video-MME and StreamingBench it is roughly at parity with cheaper merging/pruning, so the extra TTT machinery pays off mainly under long-horizon recall.
- ELViM is authored by the same team; independent validation of the episodic-recall claims on external benchmarks is limited.

*Verification: equations (1)–(5), the two-stage training config, and all Table 1/2/3 and Table 6 numbers checked against the arXiv:2510.11129 full text (HTML) and the cropped Figure 1 from the PDF; affiliation (Tsinghua + ByteDance) inferred from the video-SALMONN author line, not printed on the submission-format PDF.*

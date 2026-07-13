---
zotero_key: null
authors: Xingyi Zhou, Anurag Arnab, et al. (Google)
year: 2024
arxiv: 2404.01297
pdf: https://arxiv.org/pdf/2404.01297
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# Streaming Dense Video Captioning

**Lineage role:** The pre-LLM ancestor of streaming video understanding (CVPR 2024) — it introduces the two ideas the later video-LLM literature reuses: a *fixed-size clustering memory* that compresses an unbounded frame stream into K cluster-center tokens, and *streaming decoding* at intermediate decoding points so outputs emerge before the whole video is seen. It predates the "video LLM online" wave and uses task-specific captioning heads (GIT / Vid2Seq) rather than a chat LLM, but sets the template later works ([[flash-vstream]], [[videollm-online]], [[mmduet]]) build on.

## Problem — what was limited before this paper (short)
Dense video captioning must jointly localize events on the timeline and caption each one. Prior SOTA (Vid2Seq, PDVC) is **non-causal and non-streaming**: it encodes the *entire* video at once, so to fit a fixed compute budget it aggressively downsamples (a handful of frames, or one pooled token per frame), losing the fine detail needed for localization on long videos, and it emits a single prediction only *after* seeing everything — high latency, unusable for live streams. Token-based memories that keep all past activations (e.g. MemViT) grow unboundedly and cannot handle arbitrarily long video.

## Key idea — the core insight, 2-4 sentences
Process frames **one at a time** and keep a constant-size memory of $K$ tokens that are the **cluster centers** of all frame tokens seen so far, updated by a differentiable K-means step. Because memory size is fixed, videos of arbitrary length cost the same at decode time. Separately, train the decoder to emit, at uniformly spaced **decoding points**, every event that finished before that point — feeding earlier captions back as a text **prefix** so the model does not repeat events and retains early information through language even after the bounded memory has forgotten the pixels.

![[streamingdvc.png]]
> **Crux (Figure 2).** The full framework: each frame is encoded independently, a clustering memory compresses features into fixed-size "streaming features," and at flagged decoding points the language decoder turns the current memory (plus past decoded captions as prefix) into new timestamped captions — enabling arbitrary-length, low-latency output. *Zhou, Arnab et al. (2024), arXiv:2404.01297. Embedded for personal research reference.*

## Method + math — the mechanism, then the central objective/equations IN FULL

### Setup
Each frame $t$ is passed through a frozen per-frame image encoder (CLIP ViT-L), giving $N_f = 257$ tokens $\mathbf{f}_t \in \mathbb{R}^{N_f \times D}$. A sparsely-sampled (≈1 FPS) few-minute video has $T > 64$ frames, so feeding all $T\cdot N_f$ tokens to a text decoder is prohibitive (quadratic self-attention). Instead a memory $\mathbf{M}_t \in \mathbb{R}^{K \times D}$ of fixed size $K$ summarizes everything seen up to frame $t$. $K$ is set as a multiple of $N_f$; memory is initialized from the first $K/N_f$ frames: $\mathbf{M}_{K/N_f} = [\mathbf{f}_1,\dots,\mathbf{f}_{K/N_f}]$.

### Clustering memory update (Algorithm 1)
At each incoming frame, concatenate the old $K$ memory tokens with the new $N_f$ frame tokens and run $\tau$ K-means iterations, so the memory tokens become the cluster centers of the concatenated set. To stop cluster centers from being dragged toward fresh incoming tokens, each center carries a **momentum weight** $\mathbf{W}_t \in \mathbb{R}^K$ = the running count of tokens merged into it, so heavily-populated (older) centers move slower. One update:

$$
\begin{aligned}
\mathbf{X} &\leftarrow [\mathbf{M}_{t-1},\, \mathbf{f}_t] \in \mathbb{R}^{(K+N_f)\times D}, \qquad \mathbf{W} \leftarrow [\mathbf{W}_{t-1},\, \mathbf{1}] \\
\text{for } i=1..\tau:\quad
\mathbf{d} &= \text{pairwise\_l2}(\mathbf{X}, \mathbf{M}_t) \in \mathbb{R}^{(K+N_f)\times K} \\
\boldsymbol{\delta} &= \text{one\_hot}\big(\arg\min_{\text{axis}=1}\mathbf{d}\big) \in \{0,1\}^{(K+N_f)\times K} \quad\text{(hard assignment)} \\
\mathbf{W}_t &= \boldsymbol{\delta}^{\top}\mathbf{W} \qquad\text{(new cluster sizes)} \\
\mathbf{A} &= \boldsymbol{\delta}^{\top} / \mathbf{W}_t \qquad\text{(row-normalized weight matrix)} \\
\mathbf{M}_t &= \mathbf{A}\,\mathbf{X} \qquad\text{(centers = weighted mean of assigned tokens)}
\end{aligned}
$$

**Why it is trainable.** The assignment $\boldsymbol{\delta}$ (the $\arg\min$) is non-differentiable, but conditioned on $\boldsymbol{\delta}$ the output is a *linear map* $\mathbf{M}_t = \mathbf{A}\mathbf{X}$. So $\partial \mathbf{M}_t / \partial \mathbf{X}$ exists (gradient flows through $\mathbf{X}$, not through $\mathbf{A}$), and the memory can sit anywhere in the network with learnable layers before it.

### Streaming decoding with decoding points (Sec. 3.3)
Define **decoding points** $d_i$ = a set of intermediate timestamps (uniformly sampled frames) at which the model decodes. At $d_i$ the supervision target is **all events that ended before $d_i$**:

$$
\mathcal{Y}_i = \{(w^s_j,\, w^e_j,\, c_j)\ \mid\ w^e_j \le d_i\}
$$

where $w^s_j, w^e_j, c_j$ are the start time, end time and caption of event $j$. Events are ordered by end time and serialized as timestamp-tagged text (Vid2Seq-style). To avoid re-predicting events already emitted, a random split index picks a prefix / target boundary during training:

$$
p_i = [c'_1, c'_2, \dots, c'_{j-1}] \qquad\text{(prefix: earlier captions)}
$$
$$
y_i = [c'_j, c'_{j+1}, \dots, c'_{|\mathcal{Y}_i|}] \qquad\text{(target to generate)}
$$

The prefix $p_i$ is fed to the decoder as given context; only $y_i$ is supervised with label-smoothed cross-entropy. The randomized split (a form of prefix augmentation / random masking) makes the model robust to imperfect earlier predictions at inference. The prefix is caption-text only (no timestamps — see ablations). At inference the model steps through decoding points at a fixed stride, carrying forward its own past captions.

## Explicit design choices
- **Fixed memory size** $K = 514$ = 2 frames' worth of tokens ($2\times257$); $K$ constrained to be a multiple of $N_f$ for clean initialization.
- **K-means iterations** $\tau = 2$ per frame update (τ=1 and τ=4 both slightly worse).
- **Momentum weight** on cluster centers (running merged-token count) so old, populous clusters change slowly — a hand-built forgetting bias, not learned.
- **Hard assignment** K-means (not soft/GMM), exploiting the "differentiable-given-assignment" trick so no straight-through estimator is needed.
- **Frozen CLIP ViT-L** per-frame encoder; only the decoder/temporal parts are trained.
- **Two decoder backbones** to show generality: **GIT** (single 6-layer multimodal decoder, memory fed straight to it) and **Vid2Seq** (spatial-pool to 1 token/frame → 12-layer temporal transformer → 12-layer T5-Base decoder, memory inserted before the temporal transformer).
- **16 decoding points** during training (diminishing returns beyond ~8–16); inference decoding **stride 32** (for 64 input frames).
- **Prefix = captions only**, with random-split augmentation; beam search (beam 4) at decode.
- Streaming output serves double duty: it *lowers latency* (predict before video ends) **and** *improves accuracy* (language prefix recovers early info the bounded memory has forgotten).

## Key results / what to remember
All numbers from the paper's own tables (ActivityNet Captions / YouCook2 / ViTT; CIDEr, SODA_c, METEOR, F1). "Streaming" = this paper's memory + decoding added on top of the named backbone.

**ActivityNet Captions (dense captioning):**
- Streaming GIT: **CIDEr 41.2, SODA_c 6.6, METEOR 9.0, F1 50.9** vs GIT baseline 29.8 / 5.7 / 7.8 / 50.6 — **+11.4 CIDEr**.
- Streaming Vid2Seq: CIDEr 37.8, SODA_c 6.2, METEOR 10.0, F1 52.9 vs Vid2Seq† 30.2 / 5.9 / 8.5 / 51.8 — **+7.6 CIDEr**.

**YouCook2:** Streaming Vid2Seq CIDEr **32.9** (SODA_c 6.0, F1 24.1) vs Vid2Seq† 25.3; Streaming GIT 15.4 vs 12.1.

**ViTT:** Streaming Vid2Seq CIDEr 25.2, SODA_c 10.0 vs Vid2Seq† 23.0 / 9.8; Streaming GIT 18.5 / 8.3 vs 15.1 / 7.1.

**Paragraph captioning (no timestamps):** Streaming GIT CIDEr 33.4 (ActivityNet) / 33.9 (YouCook2) vs GIT baseline 32.5 / 28.4 — the memory alone helps 1–5 CIDEr.

**Ablations that carry the argument:**
- *Memory module* (Table 1, ActivityNet GIT CIDEr at increasing frame count T): clustering holds up as T grows — T=64 **30.6**, T=128 **30.4** — while EMA (TeSTra) collapses (T=64 → 22.0, T=128 → 16.3) and temporal pooling decays (T=128 → 25.2). MovieChat-style memory stays flat ~29 but never gains. So the win is *robustness to long inputs*, not a small constant edge.
- *Streaming decoding* (Table 3, ActivityNet GIT CIDEr): 1 decoding point 30.6 → 4 points 38.8 → **16 points 40.6** (20 points 40.4, saturating). ~**+10 CIDEr (≈33% relative)** from streaming output alone.
- *Prefix is essential:* no prefix **23.1** vs captions-as-prefix **40.6**; captions+timestamps as prefix slightly worse (39.3). Random-split augmentation: 37.6 → 40.6.
- *Hyperparameters:* K=514 best (29.4 at 257, 29.8 at 771); momentum on 30.6 vs off 29.7.

No Zotero highlights present.

**Takeaways to remember:** (1) A *constant-size cluster-center memory* is a cheap, trainable way to ingest arbitrarily long video without token blow-up, and it degrades far more gracefully than EMA/pooling as videos lengthen. (2) *Decoding-point supervision with a caption prefix* is where most of the accuracy comes from — the language prefix acts as an "explicit memory," letting a small bounded visual memory suffice. (3) The method is backbone-agnostic (helps both GIT and Vid2Seq), and helps even non-timestamped paragraph captioning.

## How it connects (evolution)
- [[flash-vstream]] — later fixed/compressed streaming visual memory for long-video LLMs; direct descendant of the "constant-size memory" idea here.
- [[videollm-online]] — brings streaming decoding into an LLM chat setting; the "predict before the video ends" premise generalized to free-form dialog.
- [[mmduet]] — proactive/streaming response timing; the "decoding point" question ("decode now? ✔/✗") becomes a learned response-trigger.
- [[rekv]] / [[hermes-kv]] — streaming KV-cache memories that solve the same unbounded-growth problem with a different (retrieval/eviction) mechanism.
- [[streaming-memory]] — sub-topic hub tying together fixed-size vs retrieval memory designs.
- [[streamingbench]] — benchmark for streaming/online understanding; evaluates the low-latency "output before the end" property this paper first operationalized.

## Open questions / limitations
- **Hard forgetting is heuristic, not learned:** momentum-weighted K-means is a fixed inductive bias for *what* to keep; there is no content- or query-aware selection of which past events matter, so genuinely long videos still lose fine detail once the pixel memory is overwritten (mitigated only via the language prefix).
- **Error accumulation via the prefix:** because past captions are fed forward as context, an early wrong caption can propagate; random-split augmentation dampens but does not eliminate this.
- **Task-specific decoders, not a general LLM:** it captions with GIT/Vid2Seq heads, so it cannot answer arbitrary questions or hold dialog — the gap that the later video-LLM streaming works close.
- **Decoding-point schedule is fixed/uniform:** decoding at uniform strides is not event-adaptive; a truly reactive system would decide when to emit based on content (the direction [[mmduet]] and proactive works take).

*Verification: All equations (Algorithm 1 memory update, Eqs. 1–3 for decoding-point supervision) and all numbers (Tables 1–5: ActivityNet/YouCook2/ViTT CIDEr·SODA_c·METEOR·F1, memory and streaming ablations) checked against the arXiv HTML (arxiv.org/html/2404.01297) and the rendered PDF pages (Figs. 2–3, Alg. 1). Figure 2 cropped directly from the PDF.*

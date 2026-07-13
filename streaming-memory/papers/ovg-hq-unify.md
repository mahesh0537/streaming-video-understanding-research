---
zotero_key: null
authors: Runhao Zeng, Jiaqi Mao, Minghao Lai et al. (Shenzhen MSU-BIT University; University of Adelaide)
year: 2025
arxiv: 2508.11903
pdf: https://arxiv.org/pdf/2508.11903
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# OVG-HQ: Online Video Grounding with Hybrid-modal Queries

**Lineage role:** Recasts video temporal grounding as an *online* task with *hybrid-modal* queries (text / image / video segment), and — most relevant to streaming-memory — replaces an explicit memory bank or LSTM hidden state with a **parametric memory** whose network weights are updated at test time by self-supervised reconstruction to carry history forward under a strict streaming constraint.

## Problem — what was limited before this paper (short)
Classical temporal video grounding is *offline*: it assumes the whole video is available and localizes a moment from a *text-only* query. Two real constraints break this. (1) **Streaming**: surveillance / live feeds arrive frame-by-frame and a prediction must be emitted immediately, without ever seeing the future. (2) **Query flexibility**: a user may not have a clean text description — the most natural query is often an *example image* or a *reference video clip* of the target behavior, alone or combined with text. Existing metrics also ignore *timeliness* (they reward a late-but-correct localization the same as a prompt one). Training a single model over mixed query modalities additionally suffers **modality imbalance**: the dominant modality (text) crowds out weaker visual queries.

## Key idea — the core insight
Unify the setting into **Online Video Grounding with Hybrid-modal Queries (OVG-HQ)** and solve it with one streaming model, `OVG-HQ-Unify`. The engine is a **Parametric Memory Block (PMB)**: instead of storing past features in a bank, the *parameters* of a small "parameter-as-memory layer" ($f_{\text{PML}}$) are updated online by a self-supervised reconstruction loss so that history is compressed into weights (higher capacity than a fixed LSTM state). To beat modality imbalance, a **hybrid distillation** scheme trains a teacher on the strong complementary text+segment query and distills it into the unified student across every modality combination. Finally, new **decay-weighted online metrics** penalize prediction delay so timeliness is scored, not just accuracy.

![[ovg-hq-unify.png]]
> **Crux (Figure 2).** The streaming pipeline: hybrid text/image/segment queries and the current video window are fused (memory-guided multi-modal fusion, with a PMB inside the residual path), then anchor queries are decoded and predictions are refined by a second PMB; the inset shows the PMB's two steps — Step 1 memorizes the current input by updating $f_{\text{PML}}$'s weights via a reconstruction loss (backprop at test time), Step 2 produces the memory-augmented output. *Zeng, Mao, Lai et al. (2025), arXiv:2508.11903. Embedded for personal research reference.*

## Method + math
**Feature extraction.** Queries are encoded with CLIP: text → $F_t$ (text encoder), image → $F_i$ (image encoder, temporally duplicated to a sequence), video segment → $F_s$ (frame-level CLIP features sampled every $M$ seconds). Video arrives as a stream processed by a **sliding window**, so only the current window plus memory is ever available.

**Parametric Memory Block (PMB).** The core is the *parameter-as-memory layer* $f_{\text{PML}}(\cdot; W^m)$, whose weights $W^m$ act as the memory. It runs in two steps per timestep on input $r_t$.

*Step 1 — memorize the current input* via a self-supervised reconstruction loss (analogous to masked-prediction pretraining, but done online):
$$
\mathcal{L}_{\text{PML}}(r_t; W^m) = \big\lVert\, f_{\text{PML}}(W_K r_t;\, W^m) - W_V r_t \,\big\rVert^2,
$$
with learnable projections $W_K, W_V$. The memory weights are then updated by one gradient step
$$
W^m \leftarrow W^m - \eta_{\text{PML}}\cdot \nabla \mathcal{L}_{\text{PML}}(r_t; W^m),
\qquad \eta_{\text{PML}} = \sigma(W_{lr}\cdot r_t),
$$
where the learning rate $\eta_{\text{PML}}$ is *input-adaptive* (sigmoid of a learned projection). After this update $W^m$ encodes both prior and current information.

*Step 2 — produce the memory-augmented output* by passing $r_t$ through the freshly updated function:
$$
\hat r_t = f_{\text{PML}}(r_t; W) = W_O \cdot \text{LN}\big(f_{\text{PML}}(W_Q r_t;\, W^m)\big),
$$
with query projection $W_Q$, LayerNorm, and output projection $W_O$. **Two parameter sets are distinguished:** the *memory* weights $W^m$ (updated by Eqn. 1–2 at test time via reconstruction) and the *remaining* weights $W^r = \{W_Q, W_K, W_V, W_O\}$ (trained by the downstream grounding loss, fixed at inference). Order per step: fix $W^r$, update $W^m$ (Eqn. 1→2), then generate $\hat r_t$ (Eqn. 3); $W^r$ is learned offline by the grounding task.

**Memory-guided multi-modal fusion.** A 2-layer Transformer decoder fuses the video features (as queries) against the hybrid-query features (as keys/values) to give query-aware features $F_{qv}$, which flow through residual blocks: `LayerNorm → PMB → (+) → LayerNorm → MLP → (+)`. The PMB inside the residual path is what injects streaming history into fusion.

**Moment prediction (anchor-based).** Predefined anchors ending at the current time $t$ with lengths $L_n = L_q / 2^{\,n-1}$ generate multi-scale proposals. A decoder over *anchor queries* yields anchor features $F_a$, split to a **classification head** (foreground/background) and a **regression head** (length/offset). Boundaries:
$$
s_n = e_n - L_n\cdot \exp(\Delta l_n), \qquad e_n = t + L_n\cdot \Delta o_n .
$$

**Prediction refinement.** Current predictions are concatenated with anchor features into $F_c$ and passed through a *second* PMB instance, so historical prediction context refines the scores/offsets (the "classification′ / regression′" outputs in Fig. 2).

**Hybrid distillation (against modality imbalance).** A teacher trained on the strong text+segment combination guides the unified student across all query types:
$$
\mathcal{L}_d = \frac{1}{N}\sum_i \Big[\, \mathcal{L}_{\text{KL}}(F_{a,i}^{s}, F_{a,i}^{t}) + \mathcal{L}_2(r_i^{s}, r_i^{t}) + \mathcal{L}_2(c_i^{s}, c_i^{t}) \,\Big],
$$
matching student ($s$) to teacher ($t$) at anchor features, intermediate reps, and classification logits. Total objective with focal classification and L1 regression:
$$
\mathcal{L} = \mathcal{L}_d + \lambda\, \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{reg}}, \qquad \lambda = 10 .
$$
Query types are cycled in alternating batches during training.

**Online (timeliness) metrics.** A decay factor $\beta\in(0,1)$ weights each prediction by how promptly it is emitted: $\beta=1$ at the ground-truth end time, decaying linearly to $0$ once the delay exceeds threshold $t_s\in\{1,3,5\}$ s, then averaged over thresholds.
$$
\text{oR}@n,\text{IoU}@m = \frac{1}{N_q}\sum_i \beta_i \cdot r(n,m,q_i),
$$
where $r(n,m,q_i)=1$ if a top-$n$ result has $\text{IoU}>m$. The timeliness-weighted mAP:
$$
\text{omAP}_m = \frac{1}{N_q}\sum_i \text{oAP}_m^{(i)}, \quad
\text{oAP}_m^{(i)} = \sum_j \big(\beta_{i,j}R_{i,j} - \beta_{i,j-1}R_{i,j-1}\big)\,\beta_{i,j}P_{i,j}.
$$

## Explicit design choices
- **Memory = weights, not a bank.** History is stored in $f_{\text{PML}}$'s parameters $W^m$, updated by one gradient step per timestep — no growing KV cache, no explicit memory tokens, fixed footprint.
- **Test-time self-supervised update.** The reconstruction loss (Eqn. 1) drives the online weight update; this is genuine test-time training of the memory sub-layer, decoupled from the task-trained $W^r$.
- **Input-adaptive learning rate** $\eta_{\text{PML}} = \sigma(W_{lr} r_t)$ — the model decides per-input how strongly to write to memory.
- **Two PMBs**, one in fusion, one in prediction refinement — memory guides both "where to look" and "how to correct predictions."
- **Anchor-based, causal decoding**: anchors end at the current time $t$; multi-scale lengths $L_q/2^{n-1}$; no access to future frames.
- **Sliding-window streaming** with CLIP features (segment frames sampled every $M$ s).
- **Hybrid distillation from a text+segment teacher** with a 3-term loss (KL on anchors + L2 on reps + L2 on logits); alternating-batch multi-modal training; $\lambda=10$.
- **New dataset QVHighlights-Unify**: augments QVHighlights with Image-R (InternVL-ranked retrieved images), Image-G (Stable Diffusion, 4 styles), Segment-G (GPT-4o enriched text → CogVideoX-5B 6-s clips), and complementary Text-C+Image-C pairs → 19.0K text / 26.3K image / 8.8K segment queries.
- **Timeliness-aware evaluation**: decay factor $\beta$ over delay thresholds $\{1,3,5\}$ s baked into oR and omAP.

## Key results / what to remember
All numbers verified against the paper's tables (as reported via the arXiv HTML).
- **QVHighlights-Unify, text query (Table 1):** OVG-HQ-Unify **oR@1, IoU=0.5 = 23.26%**, **omAP@0.5 = 23.09%**, vs TwinNet baseline oR@1_0.5 = 20.78%, omAP_0.5 = 19.73%.
- **Hybrid distillation effect (multi-modal, unified model):** Text oR@1_0.5 = 23.37%; Segment-G = 20.13%; Segment-G+Text = 22.40%. Without distillation the visual-only query collapses (Segment-G = 14.07%) — distillation recovers ~6 pts there and up to ~9% for Image-R (Fig. 3), the concrete evidence for the modality-imbalance claim.
- **Text-query grounding on standard benchmarks (Table 2):** ANet-Captions R@1_0.5 = 26.57%, R@1_0.7 = 14.36% (TwinNet 25.48 / 12.56); TACoS R@1_0.5 = 30.98%, R@1_0.7 = 21.17% (TwinNet 29.74 / 19.07); MAD R@5_0.3 = 6.32%, R@5_0.5 = 3.27% (TwinNet 4.71 / 2.00). Consistent gains over the baseline across all three.
- **Efficiency (RTX 4090, Table 3):** full model ~45.95 FPS, 21.76 ms latency; PMB overhead only ~2.20 ms (454.5 FPS), dynamic weight update ~0.30 ms (3333 FPS) — the memory update is nearly free, key for streaming.

No Zotero highlights present.

Takeaways: (1) a *parametric* memory updated online by self-supervision is a viable, near-free alternative to KV caches / feature banks for streaming grounding; (2) modality imbalance is real and distillation from the strongest combined modality fixes it; (3) online tasks need timeliness-decayed metrics, not offline recall.

## How it connects (evolution)
- [[streaming-memory]] — this note's home sub-topic hub; PMB is a *parameter-as-memory* point in the memory design space.
- [[flash-vstream]], [[rekv]], [[infinipot-v]] — contrast: explicit compressed / KV-bank memories for streaming video, vs OVG-HQ's weights-as-memory.
- [[streammem]], [[streamkv]], [[hermes-kv]] — KV / token memory management for streaming VLMs; OVG-HQ sidesteps a cache entirely.
- [[streaming-model-remember]] — the broader question of how streaming models retain history, which PMB answers via test-time weight updates.
- [[dispider]] — another streaming perception/decision architecture under a strict no-future constraint.

## Open questions / limitations
- The online metrics still sit in the low-20s% oR@1 range — absolute streaming grounding accuracy remains hard; how much is the task vs the model is unclear.
- One-gradient-step per timestep on $W^m$ raises stability/drift questions over very long streams (no reported long-horizon forgetting analysis).
- Synthetic visual queries (Stable Diffusion / CogVideoX generations) may not match the distribution of real user example clips/images — generalization to in-the-wild visual queries is untested.
- Hybrid distillation needs a teacher trained on the privileged text+segment combination; its benefit outside this specific dataset construction is unverified.

*Verification: equations (PMB Eqns. 1–3, boundary regression, distillation/total loss, oR/omAP definitions) and all headline numbers checked against the arXiv HTML of 2508.11903v1 (Tables 1–3, Figs. 2–3); figure cropped from the downloaded PDF page 4 (Figure 2).*

---
zotero_key: null
authors: Wei Li, Bing Hu, Rui Shao, Leyang Shen, Liqiang Nie (Harbin Institute of Technology, Shenzhen)
year: 2025
arxiv: 2503.03663
pdf: https://arxiv.org/pdf/2503.03663
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# LION-FS: Fast & Slow Video-Language Thinker as Online Video Assistant

**Lineage role:** A dual-system ("Fast & Slow", CVPR 2025) online video assistant that decouples *when to respond* (a cheap routing-based Fast Path over high-frame-rate streams) from *what to respond* (an expensive multi-granularity keyframe augmentation Slow Path), directly upgrading the [[videollm-online]] streaming-dialogue recipe.

## Problem — what was limited before this paper (short)
Online video assistants for wearables (smart glasses, head-mounted cameras) must, at every incoming frame, decide *whether* to speak and only then decide *what* to say — all in real time. The prior streaming-dialogue framework [[videollm-online]] (LIVE) traded efficacy for efficiency: it encoded low-frame-rate video (2 FPS) with coarse fixed-token features. This causes three failures — (1) low accuracy in the *response-determination* decision because low frame rate misses fast egocentric events; (2) imprecise responses because a fixed small token budget per frame discards fine spatial and hand-object detail; (3) training/inference inefficiency because any attempt to add tokens expands them indiscriminately across *all* frames, most of which need no response.

## Key idea — the core insight, 2-4 sentences
Mirror human dual-process cognition: split online dialogue into a **Fast Path** that runs on *every* frame at high frame rate to cheaply determine whether a response is warranted, and a **Slow Path** that runs *only on keyframes* (frames that trigger a response) to enrich them with fine-grained visual detail before generation. The Fast Path fuses two complementary encoders (SigLIP for general spatial features at 2 FPS, EgoVLPv2 for first-person temporal features at 8 FPS) via a **Token Aggregation Router** *without* increasing token count, then prunes redundant tokens layer-by-layer with a **Token Dropping Router** — this is what buys a 4× frame-rate increase over LIVE at lower FLOPs. Only the rare keyframes pay the cost of multi-granularity augmentation (global Grid Tokens + local hand/object Box Tokens), injected into a **Multimodal Thinking Template** so the LLM generates precise, egocentrically-grounded answers.

![[lion-fs.png]]
> **Crux (Figure 2).** The whole LION-FS framework: the Fast Path (bottom) dual-encodes every frame with $E_{gen}$ (SigLIP, 2 FPS) + $E_{ego}$ (EgoVLPv2, 8 FPS), fuses them in the Token Aggregation Router and sparsifies via the Token Dropping Router (right inset) to decide `[EOS]` vs `Assistant` per frame; the Slow Path (top) fires only on a keyframe, adding Grid Tokens (4-grid AvgPool) and Box Tokens (detect→crop→AvgPool of hands/objects) into the Multimodal Thinking Template. *Li et al. (2025), arXiv:2503.03663. Embedded for personal research reference.*

## Method + math — mechanism then objective in full

**Online dialogue setup (following LIVE).** The video arrives as a stream; for each frame the assistant first *determines* whether an immediate response is appropriate, and if so *generates* an answer autoregressively conditioned on the preceding video context. Determination is framed as predicting an `[EOS]` (or any sentinel) token — if the model would emit `[EOS]` it stays silent, otherwise it speaks.

**Training objective.** LION-FS reuses the LIVE two-part loss — a *streaming loss* for response determination plus a standard *language-modeling loss* for response generation:
$$
\mathrm{Loss} = \frac{1}{N}\sum_{j=1}^{N}\Big[\underbrace{-w\,s_j \log P_j^{[\mathrm{EOS}]}}_{\text{Streaming Loss}}\;\;\underbrace{-\,l_{j+1}\log P_j^{[\mathrm{Txt}]_{j+1}}}_{\text{LM Loss}}\Big]
$$
where $P_j^{[\mathrm{EOS}]}$ is the probability the LLM assigns to the EOS/sentinel token at position $j$ (the "should I respond now?" signal), $P_j^{[\mathrm{Txt}]_{j+1}}$ is the autoregressive probability of the next text token, and $s_j,\,l_{j+1}\in\{0,1\}$ are binary masks selecting the determination vs. generation term for that position. The balancing weight $w$ defaults to $1$.

**Fast Path — dual encoding.** Each frame is encoded twice: $E_{gen}$ = SigLIP on 2-FPS frames giving general spatial features, and $E_{ego}$ = EgoVLPv2 on 8-FPS frames (grouped in 4-frame windows) giving first-person temporal features. Each stream yields **10 tokens/frame** (1 CLS + a $3\times3$ pooled grid).

**Token Aggregation Router (dual fusion, no token growth).** Rather than concatenating (which doubles tokens) or naively adding, a lightweight MLP+SoftMax router produces per-frame adaptive weights from a **Visual Guidance** cue $[\mathrm{VG}]$ (the SigLIP CLS token), then convex-combines the two encoders' frame tokens:
$$
\begin{aligned}
G_f([\mathrm{VG}]) &= \mathrm{SoftMax}\big(W_2\,\sigma(W_1[\mathrm{VG}] + b_1) + b_2\big)\\
[\mathrm{Frm}]_i &= G_f([\mathrm{VG}])_0 \cdot [\mathrm{Frm}_s]_i + G_f([\mathrm{VG}])_1 \cdot [\mathrm{Frm}_t]_i
\end{aligned}
$$
so 10 (SigLIP) + 10 (EgoVLPv2) tokens fuse back to **10** output tokens per frame — richer features, same budget.

**Token Dropping Router (sparse decoding).** Inside the LLM, a per-layer router scores each token and keeps/updates only the high-scoring ones, mimicking Mixture-of-Depths. With score $r(i,n)^l = w_\theta^\top [\mathrm{Frm}]^l_{(i,n)}$ and a $\beta$-th percentile threshold $P_\beta^l$:
$$
[\mathrm{Frm}]^{l+1}_{(i,n)} =
\begin{cases}
r(i,n)^l\, f_i(\tilde{X}^l) + [\mathrm{Frm}]^l_{(i,n)}, & r(i,n)^l > P_\beta^l\\[4pt]
[\mathrm{Frm}]^l_{(i,n)}, & r(i,n)^l < P_\beta^l
\end{cases}
$$
i.e. only tokens above the percentile pass through the layer's attention/FFN update $f_i$; the rest skip the compute. Applied on *interleaved* layers at drop ratio $\beta=0.5$ this cuts FLOPs while preserving accuracy — the source of the "4× frame rate at lower cost" claim.

**Slow Path — multi-granularity keyframe augmentation** (only on keyframes, so cost is amortized):
- *Global uniform (Grid Tokens):* split the keyframe into 4 uniform grids and $3\times3$-pool each, reshaping the $1\times6\times6$ token map into a $4\times3\times3$ pattern — fine-grained global detail, **training-free**.
- *Local adaptive (Box Tokens):* a Faster R-CNN detects hands and interacting objects; unpooled patch tokens inside each box are globally pooled to one **Box Token** per region (up to 3: two hands + one object).
- *Multimodal Thinking Template:* the augmented tokens are slotted into a structured prompt — roughly `"Stream: [Frame Tokens] [Grid Tokens] User: Please focus on [Box Tokens]. Assistant:"` — steering generation toward the hand-object interaction that matters egocentrically.

## Explicit design choices
- **Two-system decoupling** of *response determination* (Fast Path, every frame) from *response generation* (Slow Path, keyframes only) — the central architectural bet, aligned to human fast/slow cognition.
- **Dual encoders**: SigLIP ($E_{gen}$, 2 FPS, general spatial) + EgoVLPv2 ($E_{ego}$, 8 FPS, egocentric temporal); EgoVLPv2 processes 4-frame groups.
- **Fixed 10-token/frame budget** (1 CLS + $3\times3$) held constant through adaptive fusion — richer without token inflation.
- **Visual Guidance = SigLIP CLS token** drives the aggregation router (ablation confirms it beats other guidance choices).
- **Token Dropping on interleaved layers at $\beta=0.5$** as the accuracy/efficiency sweet spot.
- **Slow-Path augmentation only on keyframes**, so the expensive Grid+Box detail never touches the ~majority of non-response frames.
- **Grid Tokens are training-free**; **Box Tokens** come from an off-the-shelf Faster R-CNN detecting hands + objects (≤3 boxes/frame).
- **Multimodal Thinking Template** as the injection interface for augmented tokens.
- **Backbone**: Llama-3-8B-Instruct with LoRA; trained 10 epochs on Ego-Exo4D, 2 on Ego4D; AdamW lr $2\times10^{-4}$, cosine, 5% warmup; 8× A800 (80G).
- **Loss** = streaming (EOS) + LM (text), binary-masked per token, weight $w=1$ — inherited from LIVE.

## Key results / what to remember
No Zotero highlights present.

Metrics: **LM-PPL** (language-model perplexity, ↓), **TimeDiff** (response-vs-expected timestamp gap, ↓), **Fluency** (fraction of correctly predicted tokens across turns, ↑), **LM-Correctness** (token-level match to reference, ↑).

**Ego-Exo4D narration (val)** — LION-FS **LM-PPL 2.04** / **TimeDiff 0.74** / **Fluency 36.5%** / **LM-Correctness 48.2%**, vs [[videollm-online]] (VideoLLM-online) 2.24 / 0.78 / 33.7% / 44.8% and VideoLLM-MoD 2.12 / 0.82 / 33.8% / 45.3%. LION-FS wins all four.

**Ego4D narration (val)** — LION-FS **LM-PPL 2.09** / **TimeDiff 2.15** / **Fluency 46.1%** / **LM-Correctness 52.4%**, vs VideoLLM-online 2.40 / 2.04 / 45.3% / 49.0% and VideoLLM-MoD 2.41 / 2.04 / 45.2% / 48.9%. LION-FS wins on PPL, Fluency, Correctness; is slightly *worse* on TimeDiff (2.15 vs 2.04) — honest to note.

**Efficiency (Table 6, Ego-Exo4D same-sample FLOPs):** LION-FS at 8 FPS = **12.40T** FLOPs (no augmentation) / **21.53T** (keyframe augmentation), vs LIVE **59.49T** at 8 FPS (no aug) and 55.65T at 2 FPS (all-frame aug). So LION-FS runs 4× the frame rate at a fraction of the cost.

**Ablations:** Token Aggregation adaptive routing (10+10→10) ≈ 2.25 LM-PPL, matching/beating concat and add strategies (Table 2); Token Dropping interleaved @ $\beta=0.5$ = 2.16 LM-PPL with ~1.12× speedup, 51.40T vs 61.44T FLOPs (Table 3); SigLIP-CLS as $[\mathrm{VG}]$ best at 2.25 (Table 4); Grid+Box together = best 2.04 LM-PPL / 48.2% correctness (Table 5). (Exact ablation cells transcribed from the arXiv HTML tables; treat single-decimal ablation numbers as HTML-parsed unless re-checked against the typeset PDF.)

Takeaways: (1) *decouple determination from generation* to spend compute only where it matters; (2) *fuse encoders without growing tokens* via a guidance-driven router; (3) *MoD-style layerwise token dropping* is what makes high frame rate affordable; (4) *hand/object Box Tokens + Grid Tokens* are the egocentric-precision lever, applied sparsely.

## How it connects (evolution)
- [[videollm-online]] — the LIVE framework LION-FS directly extends: same streaming-dialogue task, same EOS-streaming + LM loss, the explicit efficacy/efficiency baseline it beats.
- [[mmduet]] — contemporaneous proactive/dense per-frame response head for streaming dialogue; a sibling take on *when to respond*.
- [[dispider]] — decoupled perception/decision/reaction streaming assistant; parallel "split the pipeline into cheap-vs-expensive stages" philosophy.
- [[streammind]] — proactive streaming with a decoupled "think vs speak" gating, another fast/slow-style response-timing design.
- [[proactive-response]] — the sub-topic hub for models that decide *when* to speak on a live stream.
- [[streaming-video-understanding]] — the umbrella topic (online/streaming video LLMs).

## Open questions / limitations
- **TimeDiff regressions on Ego4D** (2.15 vs LIVE's 2.04) suggest the aggressive frame-rate/token-dropping trade can slightly delay *when* it responds even as content quality improves — timing vs. content is not uniformly won.
- **Slow Path depends on an external Faster R-CNN** for hand/object boxes; detector errors or non-egocentric/non-hand-object domains could blunt the Box-Token benefit, and it adds a non-end-to-end component.
- **Evaluation is narration-centric (Ego4D / Ego-Exo4D)** with perplexity/fluency/correctness proxies — no report here on downstream QA or long-horizon streaming benchmarks, so real-world proactive-assistant efficacy is inferred, not shown.
- **Keyframe definition = frames that trigger a response**; if determination mis-fires, the expensive augmentation is spent on the wrong frame (or skipped on a needed one) — the two paths' errors compound.

*Verification: architecture, equations (loss Eq.1, aggregation Eqs.4–5, dropping router), and all headline numbers checked against the arXiv:2503.03663 HTML full text (Tables 1–6) and cross-read against the typeset PDF Figure 2 / Section 3; ablation single-decimal cells parsed from HTML tables and flagged as such. GitHub: github.com/JiuTian-VL/LION-FS.*

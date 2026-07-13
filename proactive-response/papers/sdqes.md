---
zotero_key: null
authors: Cristóbal Eyzaguirre, Eric Tang, Shyamal Buch, Adrien Gaidon, Jiajun Wu, Juan Carlos Niebles (Stanford University)
year: 2024
arxiv: 2412.03567
pdf: https://arxiv.org/pdf/2412.03567
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# Streaming Detection of Queried Event Start (SDQES)

**Lineage role:** formalizes "when to react" as language-queried *event-onset* detection over a streaming egocentric video — a task + benchmark (EgoSDQES on Ego4D) with latency-aware metrics (Streaming Recall, Streaming Minimum Distance) that measure *how early and how precisely* a proactive system fires, rather than whether an offline answer is correct. NeurIPS 2024 Datasets & Benchmarks track.

## Problem — what was limited before this paper
Embodied applications (robotics, driving, AR assistants) need to *react the instant* a user-specified event begins, but the existing task landscape did not cover this. Offline temporal-localization-with-language tasks require observing the whole event (often the whole video) before emitting an answer — useless when time-sensitivity is the point. Online Detection of Action Start (ODAS) captures the low-latency urgency but only over a *fixed, closed vocabulary* of atomic action classes, and its benchmarks are third-person/allocentric and short (<180 s). Action anticipation likewise predicts closed-set classes and is mostly offline. No prior task combined **open-vocabulary natural-language event specification** with **online/streaming prediction** on long egocentric video. There was also no dataset and no metric suited to it: frame-level FPR/accuracy misrepresent task performance, and p-mAP (the ODAS metric) ignores temporal order, so it is not strictly online.

## Key idea — the core insight
Define a new task at the intersection of multimodal event understanding and online video: given a streaming video and a natural-language query, emit the *start time* of the described event with high accuracy and low latency, allowed to output a set of early guesses so precision (few false-positive early fires) also matters. Build a benchmark (EgoSDQES) by repurposing Ego4D's temporally-grounded annotations through an LLM (GPT-4) pipeline that rewrites events into disambiguated "remind-me-when…" streaming queries, and introduce *latency-aware* metrics — Streaming Recall (SR@k) with an asymmetric anticipation/latency tolerance window and Streaming Minimum Distance (SMD@k). Provide efficient baselines by inserting streaming temporal **adapters** into frozen dual-encoder vision-language backbones so each new frame is processed at constant cost.

![[sdqes.png]]
> **Crux (Figure 1).** The SDQES task: a streaming video feed (left) plus user natural-language queries (right, e.g. "Alert me if a child is crossing the street") drive an online model that must fire the moment "Now" the queried event *starts* — the diagram is the paper's whole thesis: proactive reaction is event-onset detection under latency. *Eyzaguirre et al. (2024), arXiv:2412.03567. Embedded for personal research reference.*

## Method + math

### Task formulation
Let $V^{(i)}_{\text{stream}} = \{f_1, f_2, \dots, f_i\}$ be the streaming frame sequence up to the current frame $f_i$ at time $i$, $q$ a natural-language query, and $t_s$ the ground-truth start time of the queried event. A model is
$$M(V_{\text{stream}}, q) \mapsto t_{\text{out}},$$
with the goal $t_{\text{out}} = t_s$ at low latency. Because $t_s$ is unknown in advance and frames arrive sequentially, the model is **not restricted to a single prediction**; it may emit a set of early guesses $t_{\text{out}} < t_s$. Hence a second goal: high precision, i.e. minimal false-positive early outputs. Total latency is decomposed into **computation latency** (wall-clock per frame) and **observation latency** (how many frames *into* the event the model must see before firing); the new metrics target observation latency.

### Metric 1 — Streaming Recall (SR@k), the accuracy metric
Extends p-mAP to the ambiguity of open-vocabulary starts by scoring the first (up to) $k$ predictions. Let $P^{(k)}_M = \{t_{\text{out}_1}, \dots, t_{\text{out}_k}\}$ be the first $k$ predictions. The output set is **correct** iff
$$\exists\, t'_{\text{out}} \in P^{(k)}_M : \; -\text{anticipation} \le t_s - t'_{\text{out}} \le \text{latency}.$$
The window is **asymmetric**: a prediction may lead the true start by up to `anticipation` seconds or lag it by up to `latency` seconds and still count. Because only the *first* $k$ predictions are eligible, a model that fires spuriously early exhausts its $k$ guesses before the event and is penalized — so SR@k jointly rewards recall and precision. Sweeping $k$ and the tolerances gives fine-grained curves (experiments use anticipation = 5 s, latency = 10 s).

### Metric 2 — Streaming Minimum Distance (SMD@k), the timeliness metric
For the closest of the first $k$ predictions, define the minimum distance
$$d_{\min} = \min_{t_{\text{out}} \in P_M} |t_s - t_{\text{out}}|,$$
and report **SMD@k** = the average of $d_{\min}$ over all queries (using each model's first $k$ predictions). Complementary to SR: it measures how *temporally close* predictions land, in seconds (lower is better).

### Benchmark construction (EgoSDQES)
Source: Ego4D's Moments (action localization), NLQ (natural language queries), and dense Narrations; also extensible to EgoExoLearn via the same pipeline. Per-annotation LLM (GPT-4) pipeline (Fig. 3): (1) **extract** the core event from the annotation (e.g. "Where did I last leave the box?" → "leave box"); (2) **check groundedness** — confirm the event appears in the narrations (queries must refer to visible events); (3) **check for prior occurrences** of the event earlier in the video; (4) **generate a streaming query** in a "set a reminder to do X when the event begins" template, disambiguated so it cannot refer to an earlier occurrence; (5) **confirm specificity** — verify the query uniquely identifies the intended instance among repeats. Rule-based + LLM filters discard: annotations that don't fit the 8k-token context window (to avoid truncating the prior-occurrence check), duplicates, ungrounded events; a specificity filter is applied only to events that recurred. Resulting stats: 12,767 annotations over ~740 hours of video (1,331 train / 442 val videos, following Ego4D's split); average video duration 1,553 s — far longer than NLQ (492 s), Moments (472 s), EgoSchema (180 s), or ODAS datasets (<180 s). Annotations skew to the first ~30 min of each video (a consequence of the context-window filter plus Ego4D biases).

### Baseline models — Streaming-Adapters
Given a pretrained image model $F$, adapt it into a spatio-temporal streaming model $F^*$ reusing as many frozen parameters as possible. Temporal-aggregation adapter layers are inserted into a frozen ViT (Fig. 5): at the start of each block and before the MLP, initialized near identity. Adapters operate in a **reduced dimension** $d' < d$ for efficiency. The key requirement — constant per-frame cost — is met by choosing streaming-friendly temporal aggregators that update incrementally as frames arrive; convolutional variants **left-pad** (zero-pad on the left) so no future information leaks into the past. Four adapter variants: a non-temporal MLP **Adapter**; **ST-Adapter** (1D convolution, à la ST-Adapter); **QR-Adapter** (Quasi-Recurrent NN, gated, with a CUDA kernel); **RN-Adapter** (RetNet, linear-attention Transformer analog for low-cost inference). Backbones: CLIP, EgoVLP, LaViLa, and SOTA egocentric dual-encoder EgoVideo — all **dual-encoder** (so the video is not reprocessed per query). Each multi-frame backbone is modified to ingest one frame at a time.

### Training objective
Data sampling addresses the extreme positive/negative imbalance (events are rare in long videos) by a **dense-labeling** reformulation: a window of $w_s$ frames $f_{i-w_s+1},\dots,f_i$ is paired with per-frame labels $y_{i-w_s+1},\dots,y_i$ where $y=1$ iff the frame lies in the query's event region. Predictions come from thresholding the cosine similarity between frame embedding $e_{f_i}$ and query embedding $e_q$:
$$s_i = \frac{e_{f_i}\cdot e_q}{\lVert e_{f_i}\rVert\,\lVert e_q\rVert}, \qquad p_i = \sigma(s_i).$$
A (class-weighted) binary cross-entropy loss is optimized:
$$\mathcal{L} = -\sum_{i=1}^{N} y_i \log(p_i),$$
with a weighting scheme to counter the negative-frame majority.

## Explicit design choices
- **Task lets the model emit multiple early predictions** (a set, not one label) — precision is baked into the metric via the first-$k$ rule, matching real proactive systems that may guess before firing.
- **Asymmetric tolerance window** (anticipation vs latency, 5 s / 10 s in experiments) — encodes that firing slightly early differs from firing late; a principled fix to p-mAP's order-insensitivity.
- **Two complementary metrics**: SR@k (did it catch the start within tolerance, precision-aware) + SMD@k (how many seconds off, in absolute terms).
- **LLM-authored, disambiguated queries** with explicit groundedness / prior-occurrence / specificity filters — ensures each query is answerable from *past-only* stream context, not future frames.
- **Long untrimmed egocentric videos** (avg 1,553 s, up to ~2 h) as the eval regime — stresses constant-per-frame streaming, unlike <180 s ODAS clips.
- **Frozen dual-encoder backbones + lightweight temporal adapters** at reduced dimension — image-to-video transfer at parameter- and compute-efficient cost; dual encoders avoid re-encoding video per query.
- **Left-padding / recurrent (QRNN, RetNet) aggregators** for strictly causal, constant-cost frame updates (vs. a sliding window that recomputes).
- **Dense-labeling supervision + class-weighted BCE** to fight positive/negative frame imbalance.
- Trained on 60-frame windows at 1 FPS (RetNet and EgoVideo limited to 30 frames for stability / memory).

## Key results / what to remember
No Zotero highlights present.

Setting: EgoSDQES val, 1 FPS, SR anticipation = 5 s / latency = 10 s. From Table 2 (1-minute and 5-minute clip evaluations):
- **Zero-shot CLIP** (single-frame, no training): SR@1 = 16.9 (1 min); 7.9 (5 min) — every trained adapter beats it, confirming the dataset's value.
- **Best 1-minute SR@1: LaViLa + QR-Adapter = 29.1** (SMD@1 = 18.1 s), closely followed by EgoVLP + QR-Adapter = 28.8 (SMD@1 = 17.7 s) and EgoVideo + Adapter = 27.1.
- **Temporal QR-Adapter consistently beats the non-temporal Adapter** across backbones (e.g. EgoVLP: 28.8 vs 18.1 SR@1 at 1 min; CLIP: 23.7 vs 19.5) — temporal modeling helps SDQES.
- **RN-Adapter (RetNet)** is competitive with QR-Adapter (EgoVLP+RN-Adapter SR@1 = 25.7), while the **1D-conv ST-Adapter underperforms** (EgoVLP+ST-Adapter SR@1 = 17.4) — richer temporal modeling than plain convolution is needed.
- **5-minute setting is much harder**: best SR@1 = 16.0 (EgoVideo + Adapter); SR@3 rises to 26.4 for the same model (more allowed guesses → higher recall). SMD@k in the 5-min setting is large (~100–175 s), reflecting the difficulty of long-video onset localization.
- **Efficiency (Table 3, single V100, per-frame over a ~5:50 video):** EgoVLP backbone = 180.92 M params, 7.85 TMACs, 15.7 TFLOPs, 1.68 s latency. Adapters add only ~13% operations (+7.5–7.9% params). QRNN-Adapter: +12.0% MACs, +21.5% latency; RetNet-Adapter: +99.5% latency (no optimized CUDA kernel). A **sliding-window** baseline costs +298.5% MACs / +260.2% latency for only 4 s of context — the constant-cost streaming design is ~4× cheaper.
- Dataset headline: **12,767 annotations, ~740 h, avg video 1,553 s** (Table 1) — larger and much longer than comparable Ego4D-derived sets.

Takeaways: (1) proactive reaction can be operationalized as language-queried onset detection with explicit latency tolerance — SR@k/SMD@k are reusable metrics for any "when to speak/act" system; (2) even the best baselines are weak (SR@1 ≈ 29% at 1 min, ~16% at 5 min), so the benchmark is far from solved; (3) constant-per-frame temporal adapters (QRNN, RetNet) are the efficient sweet spot vs sliding windows.

## How it connects (evolution)
- [[proactive-response]] — parent sub-topic; SDQES is the onset-detection framing of "when to react."
- [[streaming-benchmarks]] — SDQES is a streaming eval/benchmark; SR@k and SMD@k are latency-aware metrics for that family.
- [[egospeak]] — egocentric "when to speak" proactive turn-taking; SDQES is the retrieval/detection analog of the same react-at-the-right-moment problem.
- [[proactivevideoqa]] — benchmark for proactive/timed video QA; shares the "fire at the correct moment, penalize early/late" evaluation philosophy.
- [[stream-vlm-qevd]] — queried-event streaming detection with a VLM; directly adjacent task on language-queried streaming events.
- [[videollm-online]] — streaming online video-LLM whose free-form real-time responses need exactly this kind of onset-timing evaluation.

## Open questions / limitations
- **Annotation skew**: because ungrounded/over-long scripts are filtered out, most queried events fall in the first ~30 min of each video, so very-late-onset detection is under-tested despite long videos.
- **Baselines are simple and weak**: dual-encoder + cosine-threshold adapters (no explicit reasoning, no generative language head); best SR@1 is ~29%/16% (1/5 min), leaving large headroom — the paper positions these as starting points, not solutions.
- **LLM-generated queries (GPT-4)** may carry template bias and hallucinated groundedness despite filters; query realism/diversity is bounded by the "remind-me-when…" template and Ego4D's narration coverage.
- **Metric tolerances are hand-set** (anticipation 5 s, latency 10 s); the "right" asymmetry is application-dependent and unvalidated against human judgments of acceptable reaction lag.

*Verification: equations (Eqs. 1–2, SMD, cosine-similarity BCE loss) transcribed from PDF pages 4–8; all numbers (SR@k, SMD@k, efficiency deltas) checked against Table 2 (p.8), Table 3 (p.10), and Table 1 (p.6) of the arXiv v1 PDF. arXiv HTML was unavailable (404); worked from the PDF text + rendered Figure 1.*

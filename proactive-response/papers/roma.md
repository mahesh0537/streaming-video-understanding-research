---
zotero_key: null
authors: Xueyun Tian et al. (ICT, CAS Key Lab of AI Safety; UCAS; Tsinghua)
year: 2026
arxiv: 2601.10323
pdf: https://arxiv.org/pdf/2601.10323
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# ROMA: Real-time Omni-Multimodal Assistant with Interactive Streaming Understanding

**Lineage role:** Omni-modal (audio+video) streaming assistant that unifies reactive QA and proactive triggering by bolting a **decoupled lightweight "speak head"** (binary timing decision) onto a frozen-encoder omni backbone, trained with a **two-stage proactive curriculum**.

## Problem — what was limited before this paper (short)
Prior streaming/online video LLMs split into two camps with disjointed capabilities. Reactive systems answer queries on demand but cannot autonomously decide *when* to speak; proactive systems (VideoLLM-online, MMDuet, Dispider) do time-sensitive triggering but are **video-centric and ignore streaming audio**, and they entangle the "should I respond now?" decision with the text-generation head, so generative biases contaminate timing. No single model combined full omni-modality (text+visual+audio) with autonomous proactive monitoring (alerts + real-time narration) *and* reactive QA over a continuous stream.

## Key idea — the core insight
Separate *timing* from *content*. ROMA keeps a standard omni LLM backbone but adds a **speak head** — a small two-layer MLP running in parallel to the LM head — that, at each one-second multimodal unit, emits a probability of "respond now." Only when that probability crosses a threshold does the LM head generate text. Because the timing decision is decoupled from generation, the LM head is never asked to encode silence/timing in its token distribution, removing the interference that degrades entangled proactive models. Audio and video are packed into fixed one-second synchronized units with a chunked temporal RoPE so the growing stream stays temporally aligned.

![[roma.png]]
> **Crux (Figure 2).** The omni streaming backbone consumes per-second multimodal units (aligned video + audio + optional text); at each step the **speak head** outputs a probability and, only when it exceeds the threshold (e.g. 0.95 or 0.89), fires the **LM head** to speak — otherwise (0.25, 0.31) the model stays silent. This is exactly the decoupled timing-vs-content design. *Tian et al. (2026), arXiv:2601.10323. Embedded for personal research reference.*

## Method + math

**Multimodal units (temporal alignment).** Following Qwen2.5-Omni's tokenization, each one-second interval becomes a *unit*: video frames sampled in that interval plus the co-occurring audio, wrapped with special tokens as
`<|vision_bos|> <|audio_bos|> [video tokens] [audio tokens] <|audio_eos|> <|vision_eos|>`.
Units are fed sequentially as the stream unfolds; a persistent KV cache accumulates aligned cross-modal context so only the current unit is freshly encoded.

**Chunked TMRoPE (position encoding).** ROMA adapts Qwen2.5-Omni's Time-aligned Multimodal RoPE to *incremental* streaming. Within a unit, video tokens share a constant temporal position ID, while audio tokens keep fine-grained IDs at **40 ms** resolution to preserve auditory temporal fidelity. Each new unit extends the global timeline cumulatively from the previous unit's maximum position ID, giving continuous, boundary-aligned encoding as units arrive.

**Speak head (decoupled decision module).** A lightweight two-layer MLP parallel to the LM head predicts, per unit, the binary probability $p_t$ that a response is required. Its input is a **learnable weighted combination of the hidden states from the last $K=4$ layers** (multi-layer aggregation gives more robust timing signals than the last layer alone). If $p_t > \tau$ the LM head generates; otherwise the model stays silent. Generation is capped at ~25 tokens (~1 s) per segment to stay synchronized with the stream.

**Stage 1 — Streaming template alignment.** Reactive QA data $\mathcal{D}_{\text{QA}}$ is restructured into sequential multimodal units (with an appended audio query + text response) to close the offline→streaming distribution gap. Encoders are frozen; remaining parameters $\theta$ are fine-tuned with standard autoregressive LM loss:
$$\mathcal{L}_{\mathrm{LM}}=-\mathbb{E}_{(X,Y)\sim\mathcal{D}_{\text{QA}}}\left[\sum_{i=1}^{L}\log P(y_i \mid y_{<i}, X; \theta)\right]$$

**Stage 2 — Time-aware decision making.** Response timing is cast as per-unit binary classification. Positive labels are task-dependent: event windows for alerts, segment boundaries for narration. Because triggers are sparse, a weighted BCE is used with positive weight $w_{\mathrm{pos}}=N_{\mathrm{neg}}/N_{\mathrm{pos}}$ (best value $w_{\mathrm{pos}}=3$):
$$\mathcal{L}_{\mathrm{time}}=-\mathbb{E}_{X\sim\mathcal{D}_{\text{stream}}}\left[\frac{1}{T}\sum_{t=1}^{T}\Big(w_{\mathrm{pos}}\,z_t\log p_t + (1-z_t)\log(1-p_t)\Big)\right]$$
where $z_t$ is the ground-truth trigger label. Timing and generation are jointly optimized, with Stage-1 reactive QA mixed in to preserve linguistic competence:
$$\mathcal{L}_{\text{total}}=\mathcal{L}_{\mathrm{time}}+\lambda\cdot\mathcal{L}_{\mathrm{LM}}$$

**Streaming dataset.** ~676K samples in three families: **Online Proactive** (27K; DiDeMo, OOPS, Charades-STA — event-driven alerts), **Online Narration** (109K; MMDuetIT, COIN, YouCook2, ActivityNet — captions only at segment transitions for concise real-time updates), **Reactive QA** (540K; InternVid, CogStream, AVQA, EgoSchema). Text queries are synthesized into speech for the audio channel.

## Explicit design choices
- **Backbone:** Qwen2.5-Omni; encoders **frozen**, only backbone + heads tuned.
- **Speak head = 2-layer MLP**, parallel to LM head; input = learnable weighted sum of last **K=4** hidden layers.
- **One-second multimodal units**; video sampled at **2 fps**; audio position IDs at **40 ms** granularity.
- **Chunked TMRoPE** extends the global timeline cumulatively across units (persistent KV cache; only current unit encoded — ~0.37 s avg encode/unit).
- **Two-stage curriculum:** (1) streaming template alignment, then (2) time-aware decision; single-stage is markedly worse.
- **Weighted BCE** for sparse triggers, $w_{\mathrm{pos}}=3$; joint loss $\mathcal{L}_{\text{time}}+\lambda\mathcal{L}_{\text{LM}}$ with QA mixing.
- **Generation capped ~25 tokens/segment** to stay in lockstep with the stream.
- Task labels: **event windows** for alerts, **segment boundaries** for narration.

## Key results / what to remember
Verified against the paper's tables (WebFetch of arXiv HTML).
- **Static temporal grounding (Table 2):** QVHighlights **53.7 mAP / 53.0 HIT@1** (MMDuet 31.3/49.6); Charades-STA **44.3 R@0.5 / 19.9 R@0.7** (MMDuet 42.4/18.0). Ablations: K=1 → 46.4/47.4; mixed (single-stage) train → 50.3/44.7.
- **Dynamic streaming decision / alerts (Table 3, OmniMMI etc.):** ROMA **PA 37.50, PO 53.60, CRR 35.42, REC 33.81**; VideoLLM-Online PA 0.50 / PO 4.13; Dispider PO 25.34 / CRR 48.75 / REC 18.05. **Removing the speak head collapses proactive ability** (PA 12.50, PO 12.00, CRR 0.00, REC 6.46).
- **Real-time narration (Table 4):** YouCook2 **F1 35.21 / GPT 0.39**, OVO-Bench **F1 14.54 / GPT 0.42**; VideoLLM-Online 18.82/0.17 and 10.24/0.18. Without speak head: F1 collapses to 9.25 (YouCook2).
- **Full-modality QA (Table 5):** Video-MME (no subtitles) **33.30** vs Qwen2.5-Omni 20.50, VITA-1.5 28.56; EgoSchema 55.40 (Qwen2.5-Omni 58.40).
- **Reactive streaming QA:** OVO-Bench (Table 6) ROMA e.g. OCR 68.10, ATR 69.31, STU 58.15, ~59 avg (Dispider ~55); StreamingBench (Table 7) ROMA ATP 82.05, PR 82.41, TR 72.90, ~76 avg (Dispider ~72).
- **Ablations:** two-stage vs single-stage — REC drops 33.81 → 13.13 on OmniMMI single-stage; K=4 > K=1; $w_{\mathrm{pos}}=3$ optimal (reactive QA insensitive to it).

No Zotero highlights present.

Takeaways: the decoupled speak head is the load-bearing idea — it is what lets one omni model do proactive alerts, narration, *and* reactive QA without the timing decision poisoning generation; ablating it wrecks every proactive metric while barely moving QA. Multi-layer (K=4) aggregation and the two-stage curriculum are the two secondary levers that make the timing head well-calibrated.

## How it connects (evolution)
- [[videollm-online]] — the online-format proactive predecessor; ROMA's main baseline, beaten decisively on alerts/narration.
- [[mmduet]] — dense per-frame response-decision proactive model; the design ROMA contrasts (over-response) with its thresholded speak head.
- [[dispider]] — decoupled perception/decision streaming model; closest architectural cousin (explicit decision module), a key reactive-QA baseline.
- [[proactive-response]] — the sub-topic hub this note anchors: proactive "when to speak" timing.
- [[omnimmi]] — omni-modal proactive benchmark ROMA reports PA/PO/CRR/REC on.
- [[streamingbench]] / [[ovo-bench]] — streaming reactive-QA benchmarks used for the QA tables.

## Open questions / limitations
- Robustness: still susceptible to signal degradation and **audio-video asynchrony** distortions in real streams.
- Long horizon: hours-long dependencies remain bounded by the finite context window / KV cache.
- Efficiency vs quality: the ~25-token/segment cap and per-second unit granularity trade response richness for synchronization; the tuning of threshold/window is task-sensitive.
- Timing supervision is heuristic (event windows / segment boundaries as positives); calibration of $w_{\mathrm{pos}}$ and threshold does not obviously transfer across task types.

*Verification: figures/numbers checked against the paper's own Tables 2–7 and Eqs. for $\mathcal{L}_{\mathrm{LM}}$, $\mathcal{L}_{\mathrm{time}}$, $\mathcal{L}_{\text{total}}$ via WebFetch of the arXiv:2601.10323 HTML; crux figure cropped from the downloaded PDF (Figure 2, p.3).*

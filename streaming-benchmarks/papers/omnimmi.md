---
zotero_key: null
authors: Yuxuan Wang et al. (BIGAI / Peking University WICT / SJTU X-LANCE)
year: 2025
arxiv: 2503.22952
pdf: https://arxiv.org/pdf/2503.22952
tier: deep
subtopics: [streaming-benchmarks]
tags: [streaming-video-understanding, streaming-benchmarks]
---
# OmniMMI: A Comprehensive Multi-modal Interaction Benchmark in Streaming Video Contexts

**Lineage role:** The first benchmark to jointly probe *both* streaming temporal-state awareness *and* proactive turn-taking / duplex behavior for omni (video + audio) LLMs — pushing streaming evaluation beyond "answer when asked" into "know *when* and *whether* to speak." (CVPR 2025.)

## Problem — what was limited before this paper (short)
Prior video-LLM benchmarks are *offline*: the model sees the whole clip, then answers a question posed after the fact. That misses two things real interactive agents need. (1) **Temporal state awareness** — in a live stream the correct answer to the *same* question changes as the video unfolds (e.g. "how many cars have shown up?"), and the model can only use history up to *now*, not the future. (2) **Proactive behavior** — an interactive agent must decide *when* to emit a warning, *whether* an utterance even warrants a reply, and *who* is speaking, rather than passively responding to every prompt. Earlier streaming benchmarks tested reactive QA on visual-only input; none combined streaming state, multi-turn dependency, and proactive/duplex control with a synchronized audio channel.

## Key idea — the core insight, 2-4 sentences
OmniMMI reframes video understanding as **online multi-modal interaction**: queries arrive at specific timestamps (as text *and* synthetic audio), and the model must respond using only the video seen so far, sometimes across dependent multi-turn chains, and sometimes by *proactively* initiating (or correctly *withholding*) a response. It packages six tasks into two families — **streaming video understanding** and **proactive reasoning** — over 1,121 videos and 2,290 questions. Alongside the benchmark the authors ship **M4** (Multi-modal Multiplexing Modeling), a lightweight streaming baseline that adds proactive triggering, interruption detection, and parallel decoding on top of an omni LLM.

![[omnimmi.png]]
> **Crux (Figure 1).** The six-task protocol: streaming video understanding (Action Prediction, Dynamic State Grounding, Multi-Turn Dependency) where answers evolve with the timeline and later turns depend on earlier ones; and proactive reasoning (Proactive Alerting, Proactive Turn-Taking, Speaker Identification) where the model must fire at the right moment, stay silent on noise, or attribute speakers. Each query enters as both text and synthetic audio. *Wang et al. (2025), arXiv:2503.22952. Embedded for personal research reference.*

## Method + math — the eval protocol (the "math" of a benchmark) + the M4 baseline

### Task taxonomy (6 subtasks, 2 families)
**Streaming Video Understanding** — the model may only use frames up to the query timestamp:
- **Action Prediction (AP)** — given a goal in natural language plus the streamed history, pick the correct *next* action from a predefined candidate set.
- **Dynamic State Grounding (SG)** — the *same* query is posed at multiple timestamps; the answer must reflect the video state up to each point (e.g. running counts). Scored cumulatively at the 1st/2nd/3rd probe.
- **Multi-Turn Dependency (MD)** — sequential questions where a later question's referent is resolved by an earlier answer (coreference chains); *all* turns must be correct.

**Proactive Reasoning** — the model must control *when/whether* to speak:
- **Proactive Alerting (PA)** — user asks to be warned when an event occurs; the model must emit the alert within the correct time window.
- **Proactive Turn-Taking (PT)** — distinguish a legitimate query from noise/self-talk and respond *only* when appropriate (correct behavior on noise = staying silent).
- **Speaker Identification (SI)** — attribute an utterance to the right named speaker from multi-party dialogue introductions.

### Metrics
- **SG / MD / AP / SI — accuracy.** For SG and MD the score is reported at the 1st/2nd/3rd cumulative stage plus an **avg** across all data points; MD demands correctness across *every* dependent step. SI accuracy is judged by GPT-4o against the ground-truth name.
- **PA — precision, IoU, and accuracy** of the alert timestamp against the ground-truth event window (temporal-localization style: an alert is credited only if it lands inside the designated interval; IoU measures overlap of predicted vs. reference response span).
- **PT — accuracy** measured as the fraction of *noise* instances on which the model correctly produces **no response** (silence is the correct action). A model that always replies scores ✗ here.

Models that structurally cannot withhold a reply or emit a time-stamped alert are marked **✗** on PT/PA — this is the axis on which conventional (reactive) video-LLMs fail categorically.

### Data construction pipeline
- **Sources:** Ego4D, COIN, Shot2Story-20K, QVHighlights, MLVU (val/test splits), plus fresh YouTube collection via 425 keywords (≤50 videos/keyword, <1 h each); 78 newly curated videos survive quality filtering.
- **Scale:** 1,121 videos, avg. duration **324 s**; **2,290** questions; multi-turn chains of **1–3 turns**.
- **Annotation:** annotators watch chronologically and mark action/state time-spans (VIA tool), then generate questions from templated prompts covering actions/attributes/interactions; a second annotator verifies consistency and inter-annotator agreement is computed. Every query is rendered to **synthetic audio** so the benchmark exercises the audio channel, not just text.

### M4 — Multi-modal Multiplexing Modeling (the accompanying baseline)
Three mechanisms bolted onto an omni LLM (Qwen2-7B backbone) to make it duplex/streaming:

**1. Proactive generation — "Highlight Spot" (Algorithm 1).** As frames stream, the model keeps a rolling KV cache and, using the current query as $Q$ and frames as $K$, computes attention
$$s = \mathrm{softmax}\!\left(\frac{qK^\top}{\sqrt{d_k}}\right).$$
Frames whose score exceeds $\mu + \alpha\sigma$ (mean + $\alpha$ std-devs) are pushed to a max-heap as candidate "highlight" spots; when a frame index recurs above a frequency threshold $\gamma$, the model *triggers* a response at that moment. This is the mechanism behind proactive alerting.

**2. Interruption detection (noise vs. stop).** Both decisions are made from a token-probability threshold set by the running perplexity/entropy. For the "$\langle\text{bos}\rangle$" (start) token used to judge whether a new query is *noise*:
$$
p(x_{n+k}\mid x_1,\dots,x_{n+k-1}) \;>\; \beta\cdot\exp\!\big(-S(\,p(\cdot\mid x_1,\dots,x_{n+k-1})\,)\big),
$$
where $\beta$ is a scaling factor and $S(\cdot)$ is the entropy of the predictive distribution. Higher perplexity lowers the threshold, so a low-probability "$\langle\text{bos}\rangle$" flags the query as noise not warranting a reply. **Stopping detection** (when to end generation, key to duplex behavior) uses the identical threshold on the "$\langle\text{eos}\rangle$" token.

**3. Parallel decoding.** To keep latency real-time, when a new query arrives mid-generation the model decodes the next token of the ongoing response *and* ingests the new query in a single forward pass, via a composite attention mask: **causal** mask for language tokens, **prefix** mask for the video context, **block** mask to separate concurrent queries. If a just-decoded token for the new query is judged noise, it is removed from the KV cache so the prior generation continues undisturbed.

**M4-IT** — a **video-free**, GPT-4o-synthesized instruction-tuning set ($4.91 total) with four parts: (i) original instruction replay (LLaVA-NeXT), (ii) interleaved image-text instructions, (iii) noise instructions (statements needing no reply), (iv) stop instructions (stop-phrase collections). It teaches the noise/stop behaviors cheaply without new video data.

## Explicit design choices
- **Timestamped, dual-modality queries:** every query is delivered as text *and* synthetic audio at a specific point in the stream, forcing history-only (causal-in-time) reasoning.
- **Two orthogonal capability axes:** streaming *comprehension* (AP/SG/MD) vs. proactive *control* (PA/PT/SI) — separating "did you understand" from "did you act at the right time."
- **Cumulative SG/MD scoring at 1st/2nd/3rd turn** to directly expose the collapse of models past a single reasoning/state step.
- **Silence-as-correct for PT** and **windowed-timestamp scoring for PA** — proactive tasks are scored on *restraint* and *timing*, not just content; reactive models get ✗.
- **GPT-4o as judge** for SI (and used in data/instruction synthesis).
- **M4 as a minimal reference implementation** of duplex behavior (highlight-spot trigger + perplexity-thresholded noise/stop + masked parallel decoding), trainable from a cheap video-free instruction set on a 7B omni backbone at ~1 fps streaming.

## Key results / what to remember
All numbers from the paper's Table 3 (accuracy %, "avg" = mean over all data points for SG/MD):

- **Best streaming-comprehension scores are low across the board.** Best **SG avg = 16.33** (Gemini-1.5-Pro); GPT-4o **SG avg = 15.00**. Best **AP = 43.00** (Gemini-1.5-Pro), GPT-4o **AP = 39.50**. Best **MD avg = 12.33** (GPT-4o), Gemini **12.00**. Open-source video-LLMs sit far lower (most SG avg 3–4, MD avg 2–3).
- **Multi-turn collapse:** on MD the 3rd-turn accuracy is ~0–0.5 for nearly every open-source model (e.g. many report **MD 3rd ≈ 0.00–0.51**); most fail to satisfy all steps of a 3-turn chain. GPT-4o MD 3rd = 7.65.
- **Proactive tasks are out of reach for reactive models:** almost all commercial + open-source video-LLMs score **✗ on PT and PA** (structurally cannot withhold/alert). VideoLLM-online manages only **PA = 0.50**.
- **Proactive-capable models:** best **PT = 68.5** (M4-a), then **VITA = 67.00**, **M4 = 62.00**. On **PA**, **M4 = 25.50** is the strongest (precision/IoU/acc in the ablation table), vs. essentially ✗/0.50 elsewhere.
- **M4 (Qwen2-7B) headline:** SG avg 5.67, AP 33.5, MD avg 1.67, SI 9.00, **PA 25.50, PT 62.00** — weaker on pure comprehension than the giant commercial models but the only compact model with real proactive ability.
- **General-video sanity check (Table 5):** M4-Qwen2 = **51.74** on VideoMME (vs. LongVA-style baseline ~52.4) — adding streaming/duplex machinery barely dents general performance.
- **Three stated findings:** (1) models decline sharply on multi-turn/streaming vs. single-step static QA; (2) audio+visual models do *not* beat visual-only ones → audio-visual feature misalignment; (3) bigger models don't help, but longer-context models do → input-length vs. memory-efficiency is the real lever.

No Zotero highlights present.

Takeaways: OmniMMI turns "video QA" into "duplex interaction," and the headline story is a *floor*: even Gemini/GPT-4o barely clear ~15% on streaming state, and standard models can't do proactive turn-taking *at all*. Proactivity (PT/PA) is a distinct capability that needs an explicit trigger/withhold mechanism (M4's highlight-spot + perplexity thresholding), not just a stronger backbone.

## How it connects (evolution)
- [[streaming-benchmarks]] — this is a flagship entry in the sub-topic hub.
- [[streamingbench]], [[ovo-bench]], [[svbench]] — sibling streaming-video benchmarks; OmniMMI is distinguished by its proactive/duplex + audio axis.
- [[proactivevideoqa]], [[omni-duplexeval]] — benchmarks that likewise center *proactive*/duplex response timing; closest in spirit.
- [[videollm-online]] — the online video-LLM whose PA≈0.50 result here motivates proactive triggering; conceptual predecessor to M4's streaming design.
- VITA-1.5 — omni model that scores well on PT (67.00), used as an OmniLLM baseline.

## Open questions / limitations
- **Very low ceilings** on comprehension tasks make fine-grained model ranking noisy; near-floor SG/MD numbers may under-resolve genuine differences between systems.
- **PT metric rewards silence**, so a pathologically quiet model could inflate PT while being useless — the benchmark needs paired "legitimate query" recall to balance the noise-rejection precision (partly addressed but worth scrutiny).
- **M4 is a modest baseline**, not a strong solver: it trades comprehension for proactivity, so OmniMMI leaves open how to get *both* in one compact streaming model.
- **Synthetic audio** (TTS-rendered queries) may not reflect real acoustic conditions (overlapping speech, noise) that a deployed duplex agent faces.

*Verification: Task taxonomy, metrics, pipeline, and M4 mechanism/equations verified against the arXiv HTML (2503.22952) and PDF pages 5 (method, Eq. 1) and 6 (Table 3). All quoted accuracies (SG/AP/MD/SI/PA/PT, M4 rows, VideoMME 51.74) read directly from the rendered Table 3 and Section 5; project page https://omnimmi.github.io.*

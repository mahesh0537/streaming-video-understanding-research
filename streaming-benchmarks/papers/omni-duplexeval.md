---
zotero_key: null
authors: Chaoqun He, Mingyang Xiang et al. (Tsinghua University · Shanghai Qi Zhi Institute · ModelBest Inc.)
year: 2026
arxiv: 2605.17360
pdf: https://arxiv.org/pdf/2605.17360
tier: deep
subtopics: [streaming-benchmarks]
tags: [streaming-video-understanding, streaming-benchmarks]
---
# Omni-DuplexEval: Evaluating Real-time Duplex Omni-modal Interaction

**Lineage role:** The first benchmark + automatic-judge framework that grades *both what a duplex model says and when it says it* over streaming omni-modal input; exposes that the best streaming MLLM manages only ~20% on proactive event detection versus ~93% for humans.

## Problem — what was limited before this paper (short)
Today's MLLMs are fundamentally offline: the whole clip is ingested before a single word comes out, so perception and response are decoupled. Real interaction is duplex — the model must keep watching, keep talking, and decide *when* to speak. Prior video/audio-visual benchmarks (Video-MME, LVBench, StreamingBench, OVO-Bench) score only the final answer, usually via multiple choice, and never measure temporal alignment or continuous adaptation. OmniMMI gives open-ended answers but they are sparse and simple; ProactiveVideoQA / PhoStream target proactive detection but lack fine-grained temporal + response-quality scoring. No benchmark jointly evaluated real-time duplex behavior, and no automatic method existed to judge it.

## Key idea — the core insight
Split real-time duplex ability into two complementary scenarios and build an LLM-as-a-Judge pipeline that scores content *and* timing together. **Real-Time Description** asks the model to emit continuous, sentence-level, time-stamped commentary that tracks evolving video+audio. **Proactive Reminder** asks the model to stay silent until a salient event occurs, then fire the right response within a tolerance window. The judge is *timestamp-aware*: it slides multiple candidate windows around each emitted sentence to tolerate realistic ~2 s perception-to-generation latency while still penalizing genuinely mistimed or hallucinated output.

![[omni-duplexeval.png]]
> **Crux (Figure 5).** The automatic evaluation pipeline for Real-Time Description: every emitted sentence $s_i$ carries a timespan; a Content-Consistency judge scores global correctness against a GT answer while a four-step Temporal-Sensitivity judge (relevance filter → multi-window sampling → multimodal context extraction → per-window scoring) grades streaming alignment, and the two are averaged. *He, Xiang et al. (2026), arXiv:2605.17360. Embedded for personal research reference.*

## Method + math — the eval protocol in full
The benchmark = 660 videos (all < 60 s, mean ≈34 s; domains: education, entertainment, sports, daily life), each with an open-ended question and human annotations, spanning **9 sub-tasks**. Models are run under their *native duplex inference protocol* — outputs are recorded with the timestamp at which they are emitted, exactly as a live system would produce them.

**Task taxonomy.**
- *Real-Time Description* (6 tasks): Counting (CT), Interaction Relation (IR), Omni (fuse vision+audio), World Knowledge (WK), OCR (evolving on-screen text), Fine-grained Movement (FM).
- *Proactive Reminder* (3 tasks): Event Reminder (ER, fire when a future event happens), Post-Event Reminder (PER, detect recurrence of a past event), Correction (CR, revise a wrong user description).

### Real-Time Description scoring
A model's stream is $S=\{s_1,\dots,s_n\}$ with each sentence carrying an interval $[\,t_i^{\text{start}}, t_i^{\text{end}}\,]$. Two dimensions, each on a 0–3 scale later mapped linearly to 0–100.

**Content Consistency** $S_{\text{content}}$ — a deduction-based judge starting from 3.00 and subtracting per-error-type penalties (Table 5), e.g. hallucination $-1.50$, critical factual error $-1.00$, key omission $-0.75$. Floored at 0.01, or 0.00 if empty/irrelevant. Judged by an LLM against the GT annotation.

**Temporal Sensitivity** $S_{\text{temporal}}$ — a four-step pipeline:
1. *Semantic relevance filtering*: each $s_i$ is labeled relevant/irrelevant (filler, polite phrases, thinking pauses are irrelevant). Let $S_{\text{irr}}\subseteq S$ and irrelevance ratio
$$ r = \frac{|S_{\text{irr}}|}{|S|}. $$
2. *Multi-window sampling*: to absorb latency, four candidate windows are formed around each sentence's interval by shifting the endpoints by 1–2 s:
$$ w_1=[t_i^{\text{start}}\!-\!1,\,t_i^{\text{end}}\!-\!1],\ w_2=[t_i^{\text{start}}\!-\!1,\,t_i^{\text{end}}],\ w_3=[t_i^{\text{start}}\!-\!2,\,t_i^{\text{end}}\!-\!2],\ w_4=[t_i^{\text{start}}\!-\!2,\,t_i^{\text{end}}\!-\!1]. $$
3. *Multimodal context extraction*: for each window pull the video frames (2 FPS) + audio.
4. *Scoring*: the judge scores $s_i$ against each window's context and takes the best-aligned window:
$$ \text{score}(s_i) = \max_{k\in\{1,2,3,4\}} \text{LLM}\big(q,\, s_i,\, \text{video}_{w_k},\, \text{audio}_{w_k}\big). $$
The final temporal score averages over the *relevant* sentences $S_{\text{rel}} = S\setminus S_{\text{irr}}$ and attenuates by the irrelevance ratio (with $\lambda=1$):
$$ S_{\text{temporal}} = \Bigg(\frac{1}{|S_{\text{rel}}|}\sum_{s_i\in S_{\text{rel}}}\text{score}(s_i)\Bigg)\,\big(1-\lambda\,r\big). $$

**Overall RTD** score equally weights the two dimensions:
$$ S_{\text{overall}} = 0.5\,S_{\text{content}} + 0.5\,S_{\text{temporal}}. $$

### Proactive Reminder scoring
For an annotated event at time $t_{\text{event}}$, only responses landing in the tolerance window $t_{\text{event}} \le \text{time}(s_i) \le t_{\text{event}} + \Delta$ (with $\Delta = 10$ s) are considered. Each of the $N$ events in a sample gets a binary judge score $\text{score}_j\in\{0,1\}$ using task-specific prompts. Scoring is **strict** — the sample passes only if *every* event is handled correctly:
$$ S_{\text{sample}} = \mathbb{1}\Big[\textstyle\sum_{j=1}^{N}\text{score}_j = N\Big]. $$

**Judge calibration.** The LLM judge was iteratively calibrated against re-annotated human scores on a small set (63 responses / 7 instances), reaching Spearman $\rho > 0.9$ for Content Consistency and ~0.8 for Temporal Sensitivity.

## Explicit design choices
- **Two-scenario split** cleanly separates "what to say continuously" (RTD) from "when to speak up" (Proactive Reminder) — covering both timing failure modes.
- **Sentence-level timestamped output** as the unit of evaluation — the property Table 1 argues prior benchmarks lack (no temporal alignment).
- **Multi-window ±1–2 s tolerance** encodes the empirical ~2 s perception-to-generation latency so reasonable delay isn't punished, but egregious mistiming is.
- **Irrelevance filter + $(1-\lambda r)$ penalty** stops models gaming the temporal score with non-substantive filler.
- **Deduction-from-3.00 content rubric** with explicit per-error penalties (Table 5) rather than a single holistic score.
- **Strict all-or-nothing** proactive scoring (Eq. 7) — reflecting that a real reminder system can't miss events.
- **All open-ended questions** (no multiple choice) to force genuine generation.
- **Two human baselines** — Human-Offline (no time pressure) vs Human-Duplex (live constraint) — bounding the completeness/timeliness trade-off.
- **Native duplex inference**: outputs recorded as emitted over time; short clips (<60 s) keep the streaming setting tractable and event-dense.

## Key results / what to remember
No Zotero highlights present.

Overall benchmark score (equal-weighted across both scenarios), verified against the paper's tables:
- **Best model MiniCPM-o 4.5: 39.6 overall** — vs Human-Duplex **81.8** and Human-Offline **91.5**. Other models: MMDuet2 35.2, LiveCC-Inst 23.8, StreamingVLM 19.0, LiveCC-Base 18.4.
- **Real-Time Description (avg, 0–100):** MiniCPM-o 4.5 **59.1**, MMDuet2 58.4, LiveCC-Inst 42.9, StreamingVLM 36.2, LiveCC-Base 34.8; Human-Offline 83.0, Human-Duplex 70.8.
- **Proactive Reminder (avg):** best model MiniCPM-o 4.5 only **20.0** (ER 18.8 / PER 11.1 / CR 27.8); MMDuet2 11.9, LiveCC-Inst 4.7, LiveCC-Base 1.9, StreamingVLM 1.7; Human-Duplex **92.8**, Human-Offline 100.0. This is the headline gap — SOTA ≈20% vs human ≈93% on *when to respond*.
- **Failure mode (Table 4, Proactive Reminder):** streaming-commentary models over-fire (LiveCC-Base 91.1% wrong, StreamingVLM 96.7% wrong), while the strongest models over-stay-silent (MMDuet2 75.8% "no answer", MiniCPM-o 4.5 49.2% "no answer"). Neither reliably times a response.
- In Real-Time Description, models stay silent for roughly 50–60% of video duration — outputs are temporally sparse and fragmented.

Takeaways: (1) Duplex timing, not raw perception, is the bottleneck — content scores are middling but proactive timing collapses. (2) The two dominant failure shapes are "won't stop talking" vs "won't start talking." (3) A large human-vs-model gap remains open, especially for proactive event detection.

## How it connects (evolution)
- [[streamingbench]], [[ovo-bench]], [[omnimmi]] — prior streaming/omni benchmarks this one argues score only final answers, no temporal alignment (Table 1 comparison).
- [[proactivevideoqa]] — proactive-detection benchmark; Omni-DuplexEval adds fine-grained temporal + content scoring on top of "when to respond."
- [[livecc]] and [[streamingvlm]] — evaluated models here (LiveCC over-fires; StreamingVLM near-zero on proactive), motivating the benchmark.
- [[mmduet2]] — a top model on this benchmark, illustrating the over-silence failure mode.
- [[streaming-benchmarks]] — the sub-topic hub aggregating these streaming evals.

## Open questions / limitations
- All clips are < 60 s; long-horizon duplex behavior (minutes of streaming, memory retention) is untested here.
- The whole protocol rests on an LLM judge; calibration is strong ($\rho>0.9$ content) but only on a 63-response set, and temporal alignment (~0.8) is noisier.
- The ±1–2 s window and $\Delta=10$ s tolerance are hand-set constants — results may be sensitive to these latency thresholds.
- Strict all-or-nothing proactive scoring (Eq. 7) may under-credit models that get most-but-not-all events right, compressing the visible signal.

*Verification: metric formulas (Eqs. 1–3, 7; relevance ratio, four windows) read from PDF page 6/7 rendering and cross-checked with the arXiv HTML; all result numbers (overall 39.6, Proactive 20.0, RTD 59.1, Table 4 error split, human baselines) taken from Tables 2 and 4 of arXiv:2605.17360 via the arXiv HTML and are internally consistent (0.5·59.1 + 0.5·20.0 ≈ 39.6). No project page consulted.*

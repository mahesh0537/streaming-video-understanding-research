---
zotero_key: null
authors: Yueqian Wang et al. (Wangxuan Institute, Peking University; Huawei Noah's Ark Lab)
year: 2025
arxiv: 2507.09313
pdf: https://arxiv.org/pdf/2507.09313
tier: deep
subtopics: [streaming-benchmarks]
tags: [streaming-video-understanding, streaming-benchmarks]
---
# ProactiveVideoQA: A Comprehensive Benchmark Evaluating Proactive Interactions in Video Large Language Models

**Lineage role:** The first benchmark whose scoring is *timing-aware* — it defines PAUC, a proactive area-under-curve metric that jointly rewards response correctness AND how early it arrives, evaluated over open-ended, multi-turn proactive QA across four domains (web / ego / TV / anomaly).

## Problem — what was limited before this paper (short)
Prior video-LLM evaluation splits into two paradigms: **offline** (whole video seen, then answer) and **online** (a query is posed at a moment, answer immediately). Neither captures **proactive interaction**, where the model itself must decide *when* to speak as the video streams. The few proactive setups that exist have two flaws: (1) their targets are overly simplistic ("tell me when event X happens"), so they grade *timing only* and ignore answer content; and (2) they support only a single response round, not multi-turn evolving output. More fundamentally, there was **no metric** that scores a *sequence* of responses emitted at varying timestamps — evaluation collapsed the temporal trajectory into a static snapshot.

## Key idea — the core insight
Treat a proactive answer not as one string but as a **time-indexed trajectory of responses**, and score the *area under the correctness-vs-time curve*. The benchmark presents one question at the very start of the video; the model must autonomously detect when answer-relevant segments appear and reply then, possibly multiple times. The paper's central contribution is **PAUC (Proactive Area Under Curve)** — inspired by the HCI "user journey map" — which plots how the LLM-judged correctness of accumulated responses rises over the ground-truth reply span, normalizes it against the ideal (always-perfect-immediately) area, and thereby rewards responses that are **both early and accurate** in a single number. A tunable weight $\omega$ dials how much timeliness matters.

![[proactivevideoqa.png]]
> **Crux (Figure 3).** A worked PAUC example: over a ground-truth reply span [311s, 360s] the model emits four responses whose LLM-judged scores (0,1,2,2) trace a step curve; PAUC = (area under that curve) / (ideal area = span x max-score) = 48.5/98 = 0.49. Earlier, more-correct replies enlarge the area, so the metric fuses timing and correctness. *Wang et al. (2025), arXiv:2507.09313. Embedded for personal research reference.*

## Method + math — the evaluation protocol
**Setup.** A test example = a video + one question posed at $t=0$ + a set of $G$ ground-truth reply turns. Each turn $g$ has textual content $gold_g$ and a timespan $(t_g^{start}, t_g^{end})$ — the interval during which the user expects to receive the information in $gold_g$. The model streams and produces responses at chosen timestamps; each response has content $pred_p$ and timestamp $\tau_p$.

**PAUC is computed per ground-truth reply turn**, then averaged over turns for the video-level score. Dropping the turn subscript $g$: for a span $(t^{start}, t^{end})$ suppose $P$ model responses fall inside it at $t^{start} < \tau_1 < \tau_2 < \cdots < \tau_P < t^{end}$.

**Judging (the score sequence).** To get the correctness at each timestamp, an LLM judge (**GPT-4.1**) receives the question, the ground-truth answer $gold$, and the **accumulated** set of model responses up to that point $\{pred_1,\dots,pred_p\}$, and returns a discrete score $s_p$:
$$s_p \in \{0,1,2\} \quad\text{(0 = completely incorrect, 1 = partially correct, 2 = mostly correct)}, \qquad S = 2 = \max\text{ score}.$$
Judging on *accumulated* responses means an early wrong reply lingers in the context and depresses all later scores — so hallucinated/contradictory replies are penalized. A coarse 3-point scale ($S=2$) was chosen over a finer $\{0..4\}$ scale because human interviews found evaluators insensitive to subtle quality differences under proactive settings, and coarse scoring gives better consistency.

**The polyline.** Build a step polyline in (time, score) space using $\tau_p$ as x and $s_p$ as y. Two endpoints make it continuous over the span: the initial point $(t^{start}, 0.5)$ — a 0.5 "no-response" floor encoding that *saying nothing is better than being fully wrong* (which would score 0) — and the final point $(t^{end}, s_P)$.

**The metric.** PAUC is the area under this polyline divided by the maximum possible area $(t^{end}-t^{start})\times S$:
$$
\text{PAUC} = \frac{ (\tau_1 - t^{start})\times 0.5 \;+\; \sum_{p=1}^{P-1}(\tau_{p+1}-\tau_p)\times s_p \;+\; (t^{end}-\tau_P)\times s_P }{ (t^{end}-t^{start})\times S }.
$$
(The paper's Eq. 1 prints the denominator as $(q^{end}-q^{start})\times S$, a typo for $t$.) Each segment contributes width x held-score; because a constant-correctness reply delivered earlier leaves its high score in force for longer, earlier delivery yields a larger area and higher PAUC.

**Timeliness weight $\omega \in [0,1]$.** Before computing, timestamps are contracted toward the span start:
$$
\tau_p \;\to\; \tau_p' = t^{start} + (1-\omega)\,(\tau_p - t^{start}).
$$
- $\omega = 0$: full timeliness pressure (late responses inside the span are strongly penalized).
- $\omega = 0.5$: default balanced setting.
- $\omega = 1$: timing collapsed out — PAUC reduces to a pure correctness metric.

**Data construction pipeline.** Four domains, each repurposed from an existing source into proactive QA with timespanned ground-truth turns:
- **[WEB]** general web video — from Shot2Story-MAGQA-39k.
- **[EGO]** egocentric — from Ego4D Goalstep, with QA generated from dense captions.
- **[TV]** TV series (dialogue / social relations) — from TVQA.
- **[VAD]** video anomaly detection (surveillance) — from UCF-Crime with manually written anomaly descriptions.
Post-processing merges consecutive ground-truth turns whose content is similar and whose gap is < 3 s, to avoid artificially fragmented targets.

**Human-preference validation.** To check PAUC tracks human judgment, they sample 100 reply turns per task (50 for [VAD]), generate two model predictions per sample (Incremental-Chunks method with GPT-4.1-mini and Gemini-2.0-Flash), and have annotators pick the preferred (more timely + accurate) prediction; agreement is Cohen's kappa (no-weighting / linear-weighting) between PAUC ranking and human preference.

## Explicit design choices
- **Fully open-ended, multi-turn answers** — not multiple-choice — so evaluation reflects free-form interactive competence (harder to grade, closer to real use).
- **Question fixed at $t=0$**; the model must proactively decide response timing during playback (autonomous timing is the tested capability).
- **Per-turn PAUC averaged to a video score** — the atomic unit is a single ground-truth reply turn.
- **Accumulated-context LLM judging** — the judge always sees all prior model replies, so early errors have lasting cost.
- **Coarse 3-point correctness scale ($S=2$)** deliberately chosen over 5-point for annotator/judge consistency.
- **0.5 floor initial point** encodes "silence beats a wrong answer."
- **$\omega$ knob** cleanly separates the timing-aware metric ($\omega{=}0.5$) from a timing-blind correctness baseline ($\omega{=}1$), enabling the ablation that proves timing awareness helps.
- **Four heterogeneous domains** (web / ego / TV series / anomaly) spanning text, video and speech modalities; 1,377 videos / 1,427 examples total.

## Key results / what to remember
No Zotero highlights present.

Scores below are PAUC x100 at the default $\omega=0.5$ (verified against Table 3, page 7). Order per row: [WEB] / [EGO] / [TV] / [VAD].
- **Human baseline:** 38.6 / 38.2 / 47.0 / 53.6. Notably *not* the ceiling — humans are hampered by having to pause the video and type at the exact relevant moment (cumbersome, unnatural), so several models beat human on [WEB]/[EGO].
- **GPT-4.1 (proprietary offline):** 51.7 / 58.8 / 56.8 / 46.2. **GPT-4.1-mini:** 47.8 / **65.8** / 59.4 / 47.7 — strongest overall, notably top on [EGO].
- **Best open-source, LLaVA-OV 7B:** 55.0 / 61.6 / 45.1 / 25.6; **Qwen2.5-VL 7B:** 45.7 / 42.8 / 32.7 / 27.9. Open-source is competitive on web/ego but weak on [TV] and [VAD].
- **Proactive-trained models are worst:** [[mmduet|MMDuet]] 38.9 / 46.0 / 21.1 / 27.4; [[videollm-online|VideoLLM-Online]] 25.9 / 25.0 / 18.3 / 25.0 — despite being *built* for timing they underperform offline models prompted into proactive use, because of limited training resources and a tendency to repeat content.
- **PAUC's $\omega=0.5$ beats the timing-blind $\omega=1$ at matching humans** (Table 4, Cohen's kappa, no-weight/linear): e.g. [VAD] 0.31/0.36 ($\omega{=}1$) → **0.45/0.49** ($\omega{=}0.5$); [WEB] 0.23/0.30 → **0.37/0.40**. Timing-aware scoring aligns with human preference across all four tasks (though absolute kappa stays modest, reflecting inherent subjectivity).
- **Proactive models repeat themselves** (Table 5, proportion of duplicate predicted turns): MMDuet 81.3 / 99.4 / 92.8 / 99.2 % — near-total duplication on [EGO]/[VAD], the mechanism behind their low PAUC.

Takeaways: (1) proactive interaction is genuinely unsolved — even the best model tops out around 50-66 PAUC; (2) timing-aware evaluation ($\omega{=}0.5$) is empirically justified by better human alignment than correctness-only; (3) current proactive-specialist models lose to strong generalist LLMs driven proactively, largely due to degenerate repetition.

## How it connects (evolution)
- [[streaming-benchmarks]] — this note is a member; ProactiveVideoQA is a defining timing-aware entry in the streaming-benchmark family.
- [[mmduet]] and [[videollm-online]] — the two proactive-response models used as evaluated systems here; their weaknesses (repetition, low PAUC) are quantified.
- [[proactive-response]] — this benchmark operationalizes exactly the proactive-response capability that thread studies.
- [[streamingbench]] and [[ovo-bench]] — sibling streaming benchmarks; ProactiveVideoQA differs by scoring *response timing* via PAUC rather than fixed-timestamp MCQ.
- [[omniinteract]] and [[omni-duplexeval]] — related benchmarks probing proactive / duplex interaction timing.

## Open questions / limitations
- **LLM-judge dependence:** all correctness scores come from GPT-4.1 on accumulated responses; judge bias/variance directly shapes PAUC, and the reported human-agreement kappas remain modest (0.3-0.5).
- **Human baseline is confounded** by the pause-and-type interface, so "models beat humans" reflects tooling friction as much as model skill.
- **Coarse 3-point scale** trades resolution for consistency — it may blur genuinely different-quality answers that both round to "mostly correct".
- **Repurposed source datasets** (Shot2Story, Ego4D, TVQA, UCF-Crime) inherit their domain and annotation biases; the 3-second turn-merge heuristic and 0.5 floor are design choices not independently validated.

*Verification: PAUC formula, $\omega$ transform, and the 0/1/2 (S=2) judge scale transcribed from page 3 text + Eq. 1 and the Figure 3 worked example (0.49); all headline numbers ([WEB]/[EGO]/[TV]/[VAD] at ω=0.5, human vs GPT-4.1/-mini vs LLaVA-OV/Qwen vs MMDuet/VideoLLM-Online) read directly off Table 3, kappa alignment off Table 4, duplication off Table 5, page 7 of the arXiv PDF; benchmark scale/sources cross-checked against the arXiv HTML.*

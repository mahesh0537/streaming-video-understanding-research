---
zotero_key: null
authors: Zhiwei Yang, Chen Gao, Jing Liu, Peng Wu, Guansong Pang, Mike Zheng Shou (Xidian University · Show Lab, NUS)
year: 2025
arxiv: 2503.21904
pdf: https://arxiv.org/pdf/2503.21904
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# AssistPDA: An Online Video Surveillance Assistant for Video Anomaly Prediction, Detection, and Analysis

**Lineage role:** the first video-LLM assistant that unifies proactive anomaly *prediction* (flag danger before it fully unfolds) with online detection and open-ended analysis in a single streaming model, introducing the VAPDA-127K instruction benchmark that grounds the "proactive response" idea in surveillance.

## Problem — what was limited before this paper
Prior video anomaly detection (VAD) systems are offline: they ingest a trimmed clip, then output a frame-level anomaly score, so they can neither anticipate an anomaly *before* it happens nor converse about it. Explanation-oriented VAD-LLMs (e.g. Holmes-VAD) still operate on pre-segmented clips and answer post-hoc. None of these run frame-by-frame on an unbounded stream, and none proactively speak up at the right moment. There was also no instruction dataset combining *prediction*, *detection*, and *open-ended analysis* over untrimmed surveillance video with timestamps.

## Key idea — the core insight
Cast surveillance as an **online, proactively-responding assistant** that watches a live stream frame-by-frame and decides *both when to speak and what to say*. It unifies three tasks — Video Anomaly **P**rediction (anticipate an anomaly before onset), **D**etection (flag it as it occurs, with timestamps), and **A**nalysis (answer free-form user questions about the ongoing event) — in one autoregressive model. Two enablers make this work: (1) a **SpatioTemporal Relation Distillation (STRD)** module that transfers an offline VLM's long-range video reasoning into cheap streaming per-frame tokens, and (2) a **streaming EOS objective** that teaches the model to stay silent until the informative moment and then respond. The whole pipeline is trained on the purpose-built VAPDA-127K instruction dataset.

![[assistpda.png]]
> **Crux (Figure 3).** The streaming pipeline: frozen Qwen2-VL vision encoder turns each incoming frame-pair into visual tokens, the STRD module (with KV-cache) refines them into video-aware tokens, and a frozen QwenLM + LoRA interleaves visual and text tokens to emit assistant turns (prediction / detection / analysis) exactly when warranted. *Yang et al. (2025), arXiv:2503.21904. Embedded for personal research reference.*

## Method + math — the mechanism

**Streaming vision front-end.** Frames are sampled at 2 FPS and consumed frame-by-frame. To cut redundancy the frozen Qwen2-VL ViT encoder $\varphi_v$ processes every two consecutive frames together; for the $(i{-}1)$-th and $i$-th frames,
$$V_{i-1,i} = \{v^1_{i-1,i}, v^2_{i-1,i}, \dots, v^N_{i-1,i}\} = \varphi_v(\nu_{i-1}, \nu_i),$$
with $N$ patch tokens per two-frame "frame". These per-frame tokens are what stream into the LLM.

**SpatioTemporal Relation Distillation (STRD).** The problem: an offline VLM sees the *whole* clip at once and captures long-range spatiotemporal relations, but a streaming model sees frames one at a time and loses this. STRD (a lightweight 2-layer Multi-Head Self-Attention network $\phi$) is trained to make the streaming per-frame tokens mimic the offline "global video" tokens. Let the offline global video representation (whole clip fed at once) be
$$\overset{\frown}{V}_{video} = \{v^1_i, v^2_i, \dots, v^M_i\} = \varphi_v(\nu),$$
and the concatenated streaming image tokens be
$$V_{images} = \mathrm{concat}(V^1_{i-1,i}, V^2_{i+1,i+2}, \dots, V^{T/2}_{T-1,T}).$$
STRD maps the streaming tokens $\overset{\frown}{V}_{images} = \phi(V_{images})$ and is pre-trained with an MSE feature-consistency loss to the offline target:
$$L_{distill} = \frac{1}{M}\big\| \overset{\frown}{V}_{video} - \overset{\frown}{V}_{image} \big\|_2^2.$$
So STRD *distills* offline spatiotemporal reasoning into the online path without ever running the expensive whole-clip pass at inference. A KV-cache stores STRD outputs, giving a temporal receptive field of ~20 minutes.

**Interleaved streaming dialogue + EOS gating.** Visual and text tokens are interleaved on the QwenLM timeline (see figure): at each step the model either predicts the next text token (an assistant response) or predicts a streaming **EOS** token meaning "nothing to say yet, keep watching." The joint objective mixes standard autoregressive LM loss with an EOS-prediction loss:
$$L = \frac{1}{N}\sum_{i=1}^{N}\Big( -\, l_{i+1}\,\log P_i^{[Txt_{i+1}]} \;-\; w\, f_i\,\log P_i^{[EOS]} \Big),$$
where $l_i=1$ if the $i$-th position is a language-response token (else 0), $f_i=1$ if the token is the last of a frame *and* the next token is not a language token (i.e. the model should stay silent / emit EOS), and $w$ (default 1) balances the two terms. At inference an EOS **threshold** $\gamma$ decides whether to respond at a frame ($\gamma=0.96$ for prediction, $\gamma=0.7$ for detection) — this is the knob that controls proactivity.

**Two-stage training.** Stage 1: pre-train STRD with $L_{distill}$ (AdamW, lr $1\times10^{-4}$, cosine decay, 10 epochs). Stage 2: instruction fine-tuning of the LLM with LoRA ($r{=}32,\ \alpha{=}64$, 2 epochs) using the combined $L$ above; the vision encoder and LLM backbone stay frozen. Backbone is Qwen2-VL-2B-Instruct.

**VAPDA-127K construction (the benchmark "protocol").** 2,415 untrimmed videos from UCF-Crime + XD-Violence, 15 anomaly categories, 127,451 timestamped samples. Pipeline (Fig. 2): sample at 1 FPS, generate dense frame captions with an ensemble of five BLIP-2 variants, then prompt an LLM to synthesize each task's supervision —
- **VAP:** feed captions up to the anomaly onset; the LLM decides the *earliest predictable frame* and writes anomaly type + a causal prediction.
- **VAD:** reuse HIVAU-70K timestamp annotations; feed sequential captions during the anomaly window with history; LLM judges ongoing-anomaly and writes a description at each timestamp.
- **VAA:** extract key detection captions and generate open-ended 5W2H questions (Who/What/When/Where/Why/How/How-much) plus factually-grounded answers.
All LLM-generated data is manually reviewed (5 annotators, ~10 h each).

**Metrics.** Training/ablation uses **LM-PPL** (language perplexity, ↓), **TimeDiff** (offset between predicted and reference response timestamp, ↓), and **Fluency** (fraction of correctly-emitted continuous tokens, ↑). Response quality uses **MoverScore (MS)**, **BLEURT**, and **UniEval**. Task-specific: **weighted F1** for anomaly-type classification (VAP & VAD), and **Average Advance Time (AAT)** in seconds — how far *before* the true onset a correct prediction fires (VAP only).

## Explicit design choices
- **One streaming autoregressive model for three tasks** (predict / detect / analyze) rather than separate pipelines.
- **Frozen Qwen2-VL ViT + frozen QwenLM, only STRD + LoRA + output layer trained** — cheap to fit (2B backbone).
- **Two-frame joint encoding** at 2 FPS to halve visual-token load while keeping short-term motion.
- **Distill offline→online** (STRD MSE to whole-clip tokens) instead of adding a recurrent memory unit — recovers long-range context on a per-frame budget.
- **Streaming EOS token + per-task threshold $\gamma$** as the explicit proactivity/silence controller (0.96 for prediction is conservative; 0.7 for detection is more eager).
- **KV-cache** for a ~20-min receptive field; 2-layer MHSA for STRD (ablated as the sweet spot vs 1 or 3 layers).
- **Data built by caption-ensemble + LLM synthesis + human review**, reusing HIVAU-70K timestamps for detection grounding.

## Key results / what to remember
Verified against the paper's tables (VAPDA-127K test split).

**Main results (Table 1):**
- **VAP:** weighted F1 **64.69%**, AAT **29.19 s**, MS 61.89%, BLEURT 51.63%, UniEval 76.69%.
- **VAD:** weighted F1 **65.45%**, MS 63.83%, BLEURT 72.46%, UniEval 62.87%.
- **VAA:** MS **61.12%**, BLEURT **88.32%**, UniEval (n/r).

**Baselines on VAP (sliding-window, 5 s / 2 FPS), F1 / AAT:** Video-LLaMA2-7B 28.26% / 10.32 s; Video-LLaVA-7B 38.63% / 12.34 s; Video-ChatGPT-7B 18.94% / 7.25 s; InternVL2-2B 16.16% / 6.32 s; Qwen2-VL-2B 30.71% / 11.64 s; Holmes-VAD-7B 47.91% / 15.68 s; VideoLLM-online-8B 0% / (n/r). AssistPDA (2B) beats the best baseline (Holmes-VAD-7B) by ~+16.8 F1 and nearly doubles the advance time, with far fewer params.

**Ablations:** STRD full config gives LM-PPL 1.68 / TimeDiff 1.07 / Fluency 53.81% vs no-STRD baseline 1.76 / 1.52 / 53.02% (Table 2); 2-layer MHSA optimal (Table 3); LoRA $r{=}32/\alpha{=}64$ best (Table 4). Runs at **15–20 FPS on an A6000**, ~20-min temporal window via KV-cache.

No Zotero highlights present.

Takeaways: (1) proactive *prediction* (AAT ~29 s lead) is achievable by a small streaming VLM when the training signal explicitly rewards early, causally-grounded flags; (2) distilling offline video tokens into the online path (STRD) is a cheaper substitute for an explicit memory module; (3) a streaming EOS objective with a task-specific threshold is the concrete lever that turns a reactive QA model into a *proactive* one.

## How it connects (evolution)
- [[proactive-response]] — hub for the proactive/when-to-speak sub-topic; AssistPDA is a domain-specialized (surveillance) instance of it.
- [[videollm-online]] — the streaming-dialogue + EOS-gating formulation AssistPDA builds on (and reports as a 0%-F1 baseline on this harder anomaly task).
- [[mmduet]] — proactive "dense video commenting" with an explicit response-decision head; same when-to-respond problem, general domain.
- [[streammind]] / [[vispeak]] — other proactive streaming assistants that decide the moment to speak; parallel mechanisms to the EOS threshold here.
- [[proactivevideoqa]] — benchmark for proactive streaming response; AssistPDA's VAPDA-127K is a task-specific counterpart focused on anomalies.
- [[dispider]] — disentangles perception/decision/reaction for streaming interaction; complementary architecture for the same proactive goal.

## Open questions / limitations
- Anomaly-type F1 (~65%) and prediction quality depend heavily on **LLM-synthesized captions/labels** (BLIP-2 ensemble + LLM), so error and bias from that pipeline are baked into both training and evaluation.
- **Prediction ground truth is subjective** — the "earliest predictable frame" is LLM-decided, so AAT (29 s lead) measures agreement with a synthetic oracle, not real-world actionable warning.
- Only **15 categories** from UCF-Crime/XD-Violence; generalization to unseen anomaly types or non-CCTV domains is untested.
- No head-to-head against a strong offline VAD-LLM on a shared *frame-level AUC* metric, so the classic VAD accuracy trade-off of going streaming is not quantified.

*Verification: equations, training config, VAPDA-127K construction, and all headline numbers (F1 64.69/65.45, AAT 29.19 s, VAA BLEURT 88.32, ablation LM-PPL/TimeDiff/Fluency, baseline F1/AAT, 15–20 FPS) cross-checked against the arXiv HTML (arxiv.org/html/2503.21904) Tables 1–4 and Figure 3, which was cropped directly from the PDF page 4. VAA UniEval not surfaced in the source I read → marked (n/r).*

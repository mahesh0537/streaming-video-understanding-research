---
zotero_key: null
authors: Ming Xie et al. (Fudan University, Nanjing University, Shanghai AI Laboratory, Shanghai Innovation Institute, HUST)
year: 2026
arxiv: 2605.25621
pdf: https://arxiv.org/pdf/2605.25621
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# StreamOV: Streaming Omni-Video Understanding via Evidence-Guided Memory and Response Triggering

**Lineage role:** couples an evidence-guided long-short memory to a hidden-state "MLLM-as-Trigger" that decides *when* to speak — an omni-modal (audio+visual) proactive-response system built on a frozen Qwen3-Omni backbone with only an ~19M-param trainable trigger.

## Problem — what was limited before this paper (short)
Omni-modal MLLMs are strong offline but break in true streaming settings on two fronts. (1) **Memory:** the multimodal context (visual + audio tokens) grows without bound, so naive caching is intractable and uniform down-sampling discards the informative moments. (2) **Timing:** a proactive assistant must decide *autonomously* when to respond versus stay silent, but prior streaming video LLMs either emit special "silence tokens" the LM must be retrained to produce, or bolt on an external router — both add latency/parameters and don't extend cleanly to synchronized audio-visual streams. Existing benchmarks also lack a task that jointly tests real-time answering, long-term recall, and proactive triggering over audio-visual dialogue.

## Key idea — the core insight, 2-4 sentences
StreamOV separates *what to remember* from *when to speak*. For memory, it constructs cheap **multimodal evidence** at each timestep — fusing query-agnostic saliency (visual change, audio peaks, audio-visual co-bursts) with query-aware CLIP/CLAP-style relevance — and routes it into three disentangled channels (visual-only, audio-only, audio-visual-aligned) that drive a fixed-budget long-short-term memory. For timing, instead of generating silence tokens, it reuses the **frozen MLLM's own prefill hidden states** as features for a tiny cross-attention trigger head that outputs a binary Wait/Respond decision, so proactivity costs almost no extra compute and no backbone retraining.

![[streamov.png]]
> **Crux (Figure 3).** The StreamOV pipeline: per-timestep query-agnostic + query-aware evidence is routed into $(\hat E_v,\hat E_a,E_{av})$, updates a long-short term memory bank $M_t=\mathcal M_t^S\cup\mathcal M_t^L$ that is fed to a frozen MLLM, whose prefill hidden states $\mathcal H$ feed a lightweight Trigger deciding Wait-or-Respond. *Ming Xie et al. (2026), arXiv:2605.25621. Embedded for personal research reference.*

## Method + math — mechanism then objective in full

**Streaming formulation.** At time $t$ the observation $x_t=(v_t,a_t)$ (visual + audio) updates a state memory and, given a query, produces an answer:
$$
M_t=\mathcal F_{\text{mem}}(M_{t-1},x_t),\qquad y_t=\phi(M_t,q_t).
$$

**Multimodal evidence construction.** Two families of scores are computed per observation.
- *Query-agnostic* (no question needed): visual change $S_v$ (consecutive-frame differencing), audio saliency $S_a$ (waveform peak detection), and audio-visual co-burst $S_{cob}$ (synchronized cross-modal events).
- *Query-aware* (depends on the current $q_t$): visual semantic relevance $S_{qv}$ and audio semantic relevance $S_{qa}$ from pretrained encoders (CLIP-style visual, CLAP-style audio).

Because the raw scores live on different scales, each is **rank-normalized locally** within the current observation window before aggregation:
$$
S=\frac{r}{r_{\max}},
$$
where $r$ is the zero-indexed rank of the observation's raw score (a local importance-sampling normalization, robust to absolute magnitudes).

**Audio gating.** Semantic audio matching is noisy, so query-aware audio is gated to fire only when the segment is physically salient or synchronized:
$$
\hat S_{qa}=S_{qa}\cdot\max(S_a,S_{cob}).
$$

**Evidence routing into three disentangled channels.** Preliminary per-modality evidence:
$$
E_v=\max(S_{qv},S_v),\qquad E_a=\max(\hat S_{qa},S_a).
$$
Audio-visual-aligned evidence combines semantic agreement and event-level synchronization:
$$
E_{av}=\max\!\big(\min(S_{qv},\hat S_{qa}),\ \min(E_v,E_a,S_{cob})\big).
$$
To avoid double-counting aligned evidence in the modality-specific branches, it is subtracted out (hinge at zero):
$$
\hat E_v=[E_v-E_{av}]_+,\qquad \hat E_a=[E_a-E_{av}]_+,\qquad [\cdot]_+=\max(\cdot,0).
$$
The final evidence of an observation is $\max(\hat E_v,\hat E_a,E_{av})$, and the timestep is routed to whichever channel is visual-only, audio-only, or audio-visual-aligned — a compact structured representation for streaming reasoning.

**Long-short term memory update.** Each observation gets a base score
$$
B(t)=\max\!\big(\hat E_v(t),\hat E_a(t),E_{av}(t)\big).
$$
- **Short-term** $\mathcal M_t^S$: the dense Top-$K_S$ observations from the current temporal window (preserves recency/detail).
- **Long-term** $\mathcal M_t^L$: a fixed Top-$K_L$ budget of the most informative historical observations, selected by comparing *cached* evidence $M_{t-1}$ against *current* evidence $B(t)$ (bounded memory).
- Current memory bank: $M_t=\mathcal M_t^S\cup\mathcal M_t^L$, fed to the frozen MLLM.

**Response trigger (MLLM-as-Trigger).** Rather than emit silence tokens, StreamOV probes the backbone's prefill hidden states. Let $\mathcal H_t=\{h_{t,0},h_{t,1},\dots,h_{t,k}\}$ be prefix hidden states from the last decoder layer and $Q_{tr}\in\mathbb R^{1\times D}$ a learnable query. A single cross-attention pools them and an MLP outputs binary logits:
$$
z_t=\mathrm{Softmax}\!\Big(\frac{Q_{tr}\mathcal H_t^{\top}}{\sqrt D}\Big)\mathcal H_t,\qquad p_t=\mathrm{MLP}(z_t;\theta_{tr}),\quad p_t\in\mathbb R^2\ (\text{Respond/Wait}).
$$
For efficiency only the final prefilling state $h_{t,0}$ is needed (no autoregressive decoding to make the timing call). The trigger is trained with standard binary cross-entropy while the backbone stays frozen — only $\theta_{tr}$ (~18.9M params) is learned.

**Training data for the trigger.** ~5,000 samples from FineVideo (~2,500 positive + ~2,500 negative), LLM-constructed (Gemini-2.5 prompting) with manual verification. *Positive* = queries whose answer information eventually appears in the stream (drawn from the later half of multi-turn dialogues, with feasibility filters that exclude continuous states, video-boundary references, and omnipresent topics). *Negative (silence)* = queries whose information is genuinely absent, passed through strict failure-mode filters to remove ambiguous cases (Appendix D prompts).

## Explicit design choices
- **Backbone:** Qwen3-Omni-30B-A3B (30B total, 3B activated / MoE), kept **frozen**; only the trigger head is trained.
- **Trigger is a cross-attention pooler over prefill hidden states**, not a generated silence token and not an external router — reuses features the MLLM already computes.
- **Only the final prefill state $h_{t,0}$** used at inference to avoid autoregressive overhead in the timing decision.
- **Three-way evidence disentanglement** (visual-only / audio-only / audio-visual-aligned) with hinge subtraction to prevent double-counting.
- **Audio gating** $\hat S_{qa}=S_{qa}\cdot\max(S_a,S_{cob})$ — trust audio semantics only when physically salient/synchronized.
- **Local rank normalization** $S=r/r_{\max}$ so heterogeneous score types compose.
- **Bounded long-short memory:** dense Top-$K_S$ short-term window + fixed Top-$K_L$ long-term budget; **64-frame** memory on SOVBench, **32-frame** on other benchmarks.
- **Optimizer:** AdamW, lr $3\times10^{-4}$, batch size 32.
- **New benchmark SOVBench:** SOVBench-O (172 sessions, 1,739 QA turns, 969 dialogue groups; 15 top-level / 86 fine-grained categories) tests Real-Time QA, Recall QA, Proactive QA; SOVBench-T (226 samples: 120 positive / 106 negative) tests the trigger as binary classification.

## Key results / what to remember
No Zotero highlights present.

- **SOVBench-O (multi-round audio-visual QA):** StreamOV avg **83.8%** (context+QA) and **81.6%** (context-only) vs Qwen3-Omni-30B baseline 79.9% / 73.7%. Per-task: Real-Time **86.9%** (vs 85.3), Recall **73.2%** (vs 77.3 — the one axis where the baseline is higher), Proactive **78.6%** (vs 64.8). Proactive-QA F1 **90.5**.
- **SOVBench-T (trigger classification):** Precision/Recall on positive class **86.1% / 98.3%**, on negative (silence) class **97.8% / 82.1%** — high recall on "should respond," strong precision on "stay silent."
- **StreamingBench:** Omni (audio-visual) subset avg **68.6%** (+7.6 vs Qwen3-Omni, +22.5 vs [[roma]]); Visual-Only avg **86.2%** (+8.3 vs baseline).
- **OVO-Bench (visual-only online):** **64.0%** overall vs StreamForest 55.6% (+8.4).
- **Video-Holmes (audio-visual):** avg **53.1** vs Qwen3-Omni 52.8 (small net gain; wins SR/TCI/TA, loses IMC/CTI).
- **Daily-Omni (audio-visual):** **69.3%** avg vs Qwen3-Omni 67.8%.
- **Video-MME (long video, no subtitles):** overall **73.5%**, Long subset **63.4%** (+5.0 vs Qwen3-Omni).
- **Ablation (SOVBench-O avg):** baseline 73.7 → query-agnostic only 78.0 → query-aware only 80.0 → both 80.4 → + long-memory **81.6**. Both evidence families and the long-term budget each contribute.
- **Takeaway:** proactive timing can be added to a frozen omni-MLLM almost for free (~19M params, single hidden state) and the biggest jump is on Proactive-QA (+13.8 pts), i.e. the trigger + evidence memory mainly help *when-to-speak* and long-horizon recall, not raw per-frame perception.

## How it connects (evolution)
- [[proactive-response]] — this is a proactive-response system; its core contribution is the *when-to-respond* trigger.
- [[roma]] — omni-modal streaming baseline StreamOV compares against (+22.5 on StreamingBench-Omni).
- [[streaming-memory]] — its long-short evidence-guided bounded memory is a streaming-memory design.
- [[vispeak]] — peer proactive audio-visual streaming assistant with learned response timing.
- [[mmduet]] — earlier proactive video-LLM that emits response/silence decisions per frame (contrast: token-based vs hidden-state trigger).
- [[dispider]] — disentangles perception/decision/reaction for streaming response, a structural cousin of StreamOV's evidence→memory→trigger split.

## Open questions / limitations
- **Recall regression:** StreamOV's Recall-QA (73.2%) is *below* the offline Qwen3-Omni baseline (77.3%) — bounded memory trades some long-term recall for efficiency; the fixed Top-$K_L$ budget may drop info needed for later recall queries.
- **Evidence relies on external encoders** (frame-diff, peak detection, CLIP/CLAP): quality/robustness of the trigger and memory hinges on these hand-picked, largely un-learned scorers rather than end-to-end training.
- **Trigger trained on ~5k LLM-generated FineVideo samples** with heavy filtering — how well the Wait/Respond boundary generalizes to open-world streams and adversarial "almost-answerable" moments is untested at scale.
- **Only $h_{t,0}$ used at inference** for the timing call — potentially discarding information a full prefix would give, and the ablation on this efficiency/accuracy trade-off is limited.

*Verification: equations (audio gating, routing $E_v/E_a/E_{av}/\hat E_v/\hat E_a$, base score $B(t)$, trigger $z_t$/$p_t$) checked against the rendered method page (p.6) of the arXiv PDF; headline numbers cross-checked against arXiv HTML Tables 1–4/7/8 (SOVBench, StreamingBench, Video-Holmes, Daily-Omni, OVO-Bench, Video-MME, ablation). Figure 3 cropped from PDF p.5.*

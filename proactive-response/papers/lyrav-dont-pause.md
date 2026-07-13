---
zotero_key: null
authors: Zhenyu Yang, Kairui Zhang, Shengsheng Qian, Weiming Dong, Changsheng Xu (CASIA / MAIS, Institute of Automation, Chinese Academy of Sciences)
year: 2026
arxiv: 2606.06991
pdf: https://arxiv.org/pdf/2606.06991
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# Don't Pause: Streaming Video-Language Synchrony for Online Video Understanding

**Lineage role:** A **training-free, verification-based finite-state machine** (continue / start-new / stay-silent) layered over a *frozen* online Video-LLM, paired with a lightweight learned **token pacer** that emits a sub-budget token chunk per frame so perception is never blocked for a full sentence — turning proactive "when to respond" gating into fine-grained "speak-while-watching" synchrony.

## Problem — what was limited before this paper (short)
Existing online Video-LLMs (VideoLLM-online's EOS gating, MMDuet's response-gating classifier, LiveStar's single-pass verification) all decide *whether* to respond at each frame, but they **pause video perception while decoding a full sentence** and resume only after generation finishes. This "pause-and-decode" pattern breaks real-time video-language synchrony: the model stops watching to speak, accumulating perceptual delay and stutters. Two unmet needs follow: (1) concurrent streaming perception and generation at frame granularity, and (2) a way to arbitrate, at each incoming frame, between *continuing* an in-progress utterance and *triggering* a new one as the visual scene evolves.

## Key idea — the core insight, 2-4 sentences
Wrap a frozen online Video-LLM in a **hierarchical, plug-and-play synchrony control layer** that decouples *when to speak* from *how fast to speak*. The **Frame-Driven Transition Controller (FDTC)** is a training-free finite-state machine over three states — **Triggered (T), Continuing (C), Silent (S)** — that at every frame recomputes the perplexity (PPL) of the current utterance prefix against the growing visual context and uses PPL drift to decide whether to keep extending the utterance, start a new one, or stay quiet. The **Streaming Token Pacer (SToP)** — the only trainable module — predicts a content-aware per-frame token budget so the model emits only a small chunk of tokens per frame interval, capped by a hard real-time latency cutoff. Together they interleave incoming frames with word tokens so perception is never paused for a full sentence.

![[lyrav-dont-pause.png]]
> **Crux (Fig. 3).** (a) The FDTC three-state machine (Triggered / Continuing / Silent) with the pacer $\Phi_{pacer}$ regulating emission, and (b) the SToP architecture — a light Transformer encoder + MLP predicting a per-frame token count $\hat m_i$, reconciled against the time-budget bar $T_{prec}+\sum T_{gen}=\Delta t$ before decoding. This one figure IS the paper: the FSM controller decides *when* to speak, the pacer decides *how many tokens* fit this frame. *Yang et al. (2026), arXiv:2606.06991. Embedded for personal research reference.*

## Method + math — the mechanism, then the central objective/equations in full

### The SVLS paradigm (what changes vs. prior online decoding)
For an offline VideoLLM $\psi$ over $K$ frames $\{[\mathrm{Frm}_{t_i}]\}_{i=0}^{K-1}$, dense captioning outputs a response only at designated decode points $P$:
$$\psi\big(V=\{[\mathrm{Frm}_{t_i}]\}_{i=0}^{K-1}\big)=\big\{\,\langle t,[\mathrm{Rsp}]_t\rangle \mid t\in P\,\big\}.$$
**Traditional online** decoding (single response time $t_p$) dumps the *entire* $N$-token sentence at frame $p$ and is silent elsewhere:
$$\psi([\mathrm{Frm}_{t_i}])=\begin{cases}\{[\mathrm{Rsp}]_j\}_{j=0}^{N-1}, & i=p\\ \varnothing, & \text{otherwise}\end{cases}$$
**Streaming Video-Language Synchrony (SVLS)** instead emits $m$ word tokens *per frame* after trigger time $t_p$, spreading the $N$-token utterance across successive frames:
$$\psi([\mathrm{Frm}_{t_i}])=\begin{cases}\{[\mathrm{Rsp}]_j\}_{j=j_{\text{start}}}^{j_{\text{end}}}, & p\le i<i_{\max}\\ \varnothing, & \text{otherwise}\end{cases}$$
$$\begin{aligned}
j_{\text{start}} &= (i-p)\cdot m,\\
j_{\text{end}} &= \min\{(i-p+1)\cdot m-1,\; N-1\},\\
i_{\max} &= \min\{\lceil N/m\rceil + p,\; K\}.
\end{aligned}$$
The constant $m$ is only for notational simplicity; in practice $m_i$ is predicted per frame by SToP and capped by a hard latency cutoff. The guarantee is that perception is **never blocked for a full sentence** — at most one sub-budget chunk decodes per frame.

### FDTC — verification-based three-state FSM (training-free)
FDTC extends LiveStar-style single-pass verification into an **incremental, stateful** check with a *prefix-level continuation* mechanism (the novel Continuing State). It builds a scene-local context $C_t=[\mathrm{Frm}_1,\dots,\mathrm{Frm}_t;\ \mathrm{Rsp}_{1:\ell}]$ concatenating the current scene's frames with tokens generated so far. At each triggered decode step $t_i$ it computes the utterance perplexity:
$$\mathrm{PPL}_{t_i}([\mathrm{Rsp}])=\sqrt[N]{\dfrac{1}{P([\mathrm{Rsp}]\mid[\mathrm{Ctx}_{<t_i}],[\mathrm{Frm}_{t_i}])}},$$
where $N$ is the token count of $[\mathrm{Rsp}]$ and $P(\cdot)$ is the autoregressive probability from token logits. For each new frame $t_j$ it re-verifies the latest caption by recomputing $\mathrm{PPL}_{t_j}([\mathrm{Rsp}])$ and applies a single **PPL-drift test** with tunable scaling factor $\alpha$:

- **Triggered (T)** — if $\mathrm{PPL}_{t_j}([\mathrm{Rsp}])>\alpha\cdot\mathrm{PPL}_{t_i}([\mathrm{Rsp}])$: sharp PPL rise = semantic drift → terminate the old utterance, generate the first $m_j$ tokens of a *new* one (context $[\mathrm{Ctx}_{<t_j};\mathrm{Rsp};\mathrm{Frm}_{t_j}]$), reset checkpoint $t_i\leftarrow t_j$.
- **Continuing (C)** — else, if the last token of $[\mathrm{Rsp}]$ is not `<EOS>`: stable/decreasing PPL means the frame belongs to the same clip → decode up to $m_j$ more tokens of the *same* utterance prefix (context $[\mathrm{Ctx}_{<t_j};\mathrm{Frm}_{t_j};\mathrm{Rsp}]$).
- **Silent (S)** — else (`<EOS>` already emitted): utterance complete → keep silent, await the next trigger.

Transition structure (Fig. 3a): all triggered utterances enter Continuing ($T\to C$); $C\to C$ extends, $C\to S$ on EOS, $C\to T$ on drift; $S\to S$ stays quiet on same-event frames, $S\to T$ on drift. **$T\to T$ is undefined** (and never observed empirically) — the first $m$ tokens carry too little information to distinguish a genuinely new event. FDTC adds **no new trainable parameters**; its only novel ingredient over the backbone's verification is the stateful Continuing State (prefix-level continuation).

### SToP — content-aware token pacer (the one trainable module)
A lightweight pacing head $\Phi_{pacer}$ (a **2-layer Transformer encoder**, $L=2$) predicts how many tokens to emit this frame. It runs over a short multimodal history $C_{hist}=\{v_{i-N},t_{i-N},\dots,v_{i-1},t_{i-1},v_i\}$ ($N=4$-frame window), where $v_j\in\mathbb{R}^D$ is a frame visual embedding and $t_j\in\mathbb{R}^D$ is a learnable embedding of the discrete token-count category at frame $j$. A prepended `[CLS]` aggregates global context; its output $z_{\text{CLS}}$ feeds a 2-layer MLP + Softmax over $K$ token-count categories:
$$p_i=\mathrm{Softmax}(\mathrm{MLP}(z_{\text{CLS}})),\qquad p_i\in\mathbb{R}^K.$$
At inference, $\hat m_i=\arg\max$ category (expected value also valid). Training is **cross-entropy classification** with **label smoothing** to preserve ordinal structure among adjacent count categories:
$$\mathcal{L}_{pacer}=\frac{1}{|\mathcal{D}|}\sum_{(V,\mathcal{T})\in\mathcal{D}}\sum_i \mathrm{CE}(p_i,\,y_{i,gt}).$$
Ground-truth counts $m_{i,gt}$ come from **word-level timestamps of human transcripts** (Live-WhisperX-526K): a human narrator's instantaneous speaking rate is treated as a cheap, abundant, behaviorally-grounded proxy for ideal narration tempo (commentators speed up on dense action, slow down on static scenes). It is acknowledged as only a proxy (speaker habits + ASR noise), used as a *soft prior*, not a hard constraint.

### Latency-constrained adaptive decoding
The pacer's target $\hat m_i$ is reconciled with a **hard real-time cutoff**: total frame processing time $T^{(i)}_{total}=T_{perc}+\sum_{j=1}^{k}T^{(j)}_{gen}$ must fit the frame interval $\Delta t=1/\mathrm{FPS}$. The emitted count is the more restrictive of the two:
$$m_i=\min\Bigg(\hat m_i,\ \max\bigg\{k \ \Big|\ T_{perc}+\sum_{j=1}^{k}T^{(j)}_{gen}\le\Delta t\bigg\}\Bigg).$$
Decoding also stops immediately on `<EOS>`. This dual constraint prioritizes staying in sync with playback over completing the content-aware target — so moderate label noise in the pacing proxy only affects *how many* tokens are emitted, not *whether* synchrony holds. (Full per-frame loop: Algorithm 1.)

## Explicit design choices
- **Frozen backbone, thin control layer.** Backbone is LiveStar's frozen InternViT + InternLM2.5-7B; LyraV is *not* a new end-to-end model — it is a plug-and-play synchrony wrapper. Offline (full-video) settings deactivate FDTC + SToP, preserving backbone integrity.
- **FDTC is 100% training-free** — reuses the verification/PPL signal; adds only the stateful three-state FSM and the novel prefix-level Continuing State. No classifier head, no EOS-token supervision.
- **Single PPL-drift threshold $\alpha$** governs all trigger decisions (T/C/S) — one interpretable hyperparameter instead of a learned gate.
- **Sub-budget per-frame emission** ($m_i$ tokens/frame, not full sentences) is the mechanism that keeps perception unblocked; a hard $\Delta t=1/\mathrm{FPS}$ latency cutoff enforces stutter-free playback.
- **Pacer = 2-layer Transformer encoder** (8.55M params), $N=4$-frame history, `[CLS]` pooling, $K$-way count classification with label-smoothed CE. Chosen over RNN/LSTM for accuracy + parallel low-latency inference.
- **Weak supervision from ASR timestamps** (Live-WhisperX-526K, ~526K time-aligned video-transcript pairs) — no manual rate labels; speaking rate as visually-driven tempo proxy.
- **Scene-local context** $C_t$ groups current-scene frames + ongoing response tokens for verification, rather than the full video history.
- Trained with AdamW, lr $1\times10^{-4}$, batch 4, frames 448×448, single NVIDIA H100.

## Key results / what to remember
Backbone throughout is **LiveStar** (InternViT+InternLM2.5-7B). Headline claims verified against the paper's tables.

- **Synchrony (Table V, OmniStar synchrony benchmark).** *Un-truncated (full decode, 2fps in / real fps out):* LyraV **Sync Rate 98.29%** vs LiveStar 78.93%, LiveCC 92.41%, VideoLLM-online 88.26%, Dispider 69.05%, MMDuet 56.81% — with the best content quality among fast models (SS 3.62, NF 4.07). *Truncated (zero-latency, all models forced to 100% SR):* LyraV best SS 3.63 / NF 4.03, improving NF ~2.0% and SS ~7.4% over LiveStar — the "fair" comparison isolating FDTC+SToP from mere truncation.
- **Real-time throughput (Table I, OmniStar-RNG @1fps):** LyraV **rFPS 3.89** (vs LiveStar 3.82); SS 3.37 (vs 3.19, +5.64%), RL 1.82 (vs 1.91, −4.95% latency), NF 4.19 (slightly below LiveStar's 4.25 — FDTC-induced incomplete responses).
- **Ego4D Narration Stream (Table I):** PPL 1.94 (LiveStar 1.97), TimeDiff 1.69 (1.76), TokAcc 0.62 (0.61) — all marginally ahead of backbone.
- **Streaming QA (Table I):** StreamingBench Real-Time 72.78 (LiveStar 71.92, +1.19%); OVO-Bench overall 50.97 (LiveStar 50.34, +1.25%). Small by design — these score static short-answer accuracy, not synchrony.
- **OVBench (Table III):** best among open-source *online* MLLMs at **46.8** avg (LiveStar 45.7; MovieChat 30.9; Flash-Vstream 31.2), narrowing the gap to offline InternVL2-7B (48.7) and Gemini-1.5-Flash (50.7).
- **Offline general understanding (Table IV):** LyraV vs LiveStar — MVBench 67.10 vs 66.95; LongVideoBench 56.80 vs 57.00; VideoMME 60.90/64.10 vs 60.80/64.40. Essentially preserved (control modules inactive offline).
- **Ablations (Table VI, OmniStar-RNG):** removing **FDTC** collapses to two states and craters quality — SS −34.4% (3.37→2.21), NF −55.4% (4.19→1.87), RL +15.4% (1.82→2.10); removing **SToP** (fixed 5 tokens/frame) barely hurts caption quality (SS −8.6%, NF −7.9%) because the latency cutoff bounds emission anyway — SToP's load-bearing effect is on streaming fluidity/throughput, not SS/NF.
- **Pacer architecture (Table VII):** Transformer pacer 91.55% acc, 206.13 rFPS, 8.55M params — beats LSTM (82.97%, 114.06, 12.63M) and RNN (78.14%, 187.53, 6.32M).
- Emergent qualitative behavior: **"dynamic reasoning over streaming tokens"** — LyraV incrementally refines its narration as the video unfolds (Sec. VI-G; non-decoded "thought" tokens in the Fig. 4 case study).

No Zotero highlights present.

**Takeaways:** (1) The novelty that actually pays off is the **Continuing State** (prefix-level utterance continuation), not the inherited verification — ablation makes this explicit. (2) Sync Rate alone is gameable by truncation; report it jointly with SS/NF and use the equal-budget truncated setting as the fair test. (3) A frozen backbone + tiny training-free FSM + an 8.55M pacer converts "when to respond" gating into fine-grained "speak-while-watching" without touching the backbone — a cheap retrofit path for any online Video-LLM.

## How it connects (evolution)
- [[livestar]] — the frozen backbone and the single-pass verification-based response-timing method LyraV extends into a stateful three-state FSM (its most direct predecessor).
- [[videollm-online]] — the EOS-token proactive-response paradigm LyraV positions against (pause-and-decode baseline).
- [[mmduet]] — the response-gating (binary classifier) family LyraV contrasts with; a Table I/V baseline.
- [[dispider]] — online proactive-response baseline compared across Tables I–III.
- [[livecc]] — streaming captioning baseline in the synchrony benchmark (Table V), close on Sync Rate but weaker on content.
- [[proactive-response]] — the sub-topic hub: LyraV reframes proactive response as continuous synchrony rather than discrete trigger points.

## Open questions / limitations
- **Pacing supervision is a proxy.** Ground-truth tempo = human speaking rate from ASR timestamps; inherits speaker-specific habits + ASR noise and may diverge from an "ideal" visual-event-density tempo. Human-annotated or multi-narrator-consensus targets are left as future work.
- **QA/perception gains are marginal** (~1%) and hallucination detection lags — LyraV trails VideoLLM-online on HLD (OVO-Bench) and Flash-Vstream on Action Persistence / Object Existence (OVBench); the control layer stabilizes reasoning but doesn't raise static perception.
- **Narrative Fluency trade-off:** FDTC's aggressive re-triggering can cut utterances short (NF slightly below LiveStar on OmniStar-RNG), trading smoothness for richer multi-frame reasoning.
- **Bolt-on, not co-trained.** FDTC+SToP wrap a frozen backbone; whether joint fine-tuning of backbone + pacer would compound the synchrony gains is untested.

*Verification: Equations (SVLS Eq. 1-3, FDTC PPL, SToP Eqs. 4-6) transcribed from the arXiv PDF (2606.06991v1) method Sections III-V; all numbers cross-checked against the paper's own Tables I (online), III (OVBench), IV (offline), V (synchrony), VI (ablation), VII (pacer). Crux figure is Fig. 3 (page 5). Zotero not running; no external repo/project page consulted.*

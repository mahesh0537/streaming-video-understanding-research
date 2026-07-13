---
zotero_key: null
authors: Ao Li et al. (AIM3 Lab, Renmin University of China; MiLM Plus, Xiaomi; Peking University)
year: 2026
arxiv: 2605.16381
pdf: https://arxiv.org/pdf/2605.16381
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# StreamPro: From Reactive Perception to Proactive Decision-Making in Streaming Video

**Lineage role:** Reframes streaming "proactive" evaluation from *reactive* see-then-answer (which the authors argue collapses to *delayed perception*) into genuine *Proactive Agency* — timely, reliable decisions under partial observation — and ships both a benchmark (StreamPro-Bench) and a two-stage FAR-oriented training recipe (CB-Stream SFT + multi-grained GRPO) that lifts proactive F1 far above prior open-source proactive models.

## Problem — what was limited before this paper
Existing streaming/"proactive" benchmarks (e.g. [[ovo-bench]], [[streamingbench]], the tasks used by [[mmduet]]/[[videollm-online]]) predominantly follow a *see-then-answer* protocol: a response is scored only *after* the triggering evidence has already appeared on screen. The authors argue this quietly reduces "proactivity" to **delayed perception** — the model is rewarded for passively waiting until enough evidence is observable, so it is really a reactive task. What such benchmarks never test is decision-making *before* evidence fully materializes: anticipating a future step, inferring a latent user need, or issuing an early warning before a risk completes. On the training side, streaming trajectories are dominated by *silence* steps (only a small fraction demand a response), so plain cross-entropy SFT is swamped by silence tokens and biases the model toward staying quiet; and proactive behavior has a dual objective (say the *right* thing at the *right* time) that a single CE loss does not shape.

## Key idea
Define a third evaluation axis, **Proactive Agency** — "the ability to perform timely and reliable decision-making under partial observations" — alongside Perceptual Understanding and Temporal Reasoning, and build a benchmark (StreamPro-Bench: 7 tasks, 577 videos, 1,285 QA) whose proactive tasks (Goal Planning, Risk Forecasting) genuinely require acting under incomplete evidence, scored by a timeliness×accuracy F1 metric. Then train a model that succeeds on this axis via two stages: (1) SFT with a **CB-Stream loss** that class-balances the rare `</Response>` decision token against the abundant `</Silence>` token so the model stops defaulting to silence; (2) GRPO RL with a **multi-grained reward** (format + turn-level F1 + trajectory-level LLM rubric) that jointly optimizes response timing, factual correctness, and global coherence.

![[streampro.png]]
> **Crux (Figure 1).** The paper's two contributions in one view: (top) it distinguishes Real-Time Streaming and the reactive "see-then-answer" proactive paradigm from the proposed *Proactive Agency*, where the model must plan ahead / anticipate needs before evidence appears; (bottom-left) the StreamPro training framework — SFT (CB-Stream loss) then GRPO on StreamPro-RL-3K; (bottom-right) StreamPro-Bench's three evaluation dimensions. *Ao Li et al. (2026), arXiv:2605.16381. Embedded for personal research reference.*

![[streampro-framework.png]]
> **Crux (Figure 4).** The training mechanism: SFT reweights the frequent `</Silence>` down and the sparse `</Response>` up (CB-Stream loss); GRPO then scores each triggered turn by an *additive* step score $S' = S_{\text{time}} + S_{\text{acc}}$ (LLM-judged accuracy + time-decayed timeliness) aggregated into a turn-level F1, plus a trajectory-level rubric (granularity/coverage/sequencing). *Ao Li et al. (2026), arXiv:2605.16381. Embedded for personal research reference.*

## Method + math

### Proactive streaming decision format
The model runs online over frames sampled at 1 FPS. At every time step it emits a **decision token** from $\mathcal{S} = \{\texttt{</Silence>}, \texttt{</Response>}\}$: `</Silence>` when nothing should be said, or `</Response>` followed by the answer text when a response is triggered. $\mathcal{T}$ is the ordinary language-token set. So one streaming trajectory is a long sequence dominated by `</Silence>` with sparse `</Response>` + text bursts.

### Stage 1 — SFT with CB-Stream loss
To fight the silence/response imbalance the authors apply **class-balanced reweighting** based on the *effective number of samples*. For each decision-token class $k\in\mathcal{S}$ with in-batch frequency $n_k$:

$$E_k = \frac{1-\beta^{\,n_k}}{1-\beta}, \qquad \hat{w}_k^{\text{CB}} = \frac{1/E_k}{\sum_{j\in\mathcal{S}} 1/E_j}\cdot|\mathcal{S}|$$

$E_k$ is the effective sample size (saturates as $n_k$ grows, so the very frequent `</Silence>` is down-weighted and the rare `</Response>` up-weighted); $\beta\in[0,1)$ tunes how aggressive the reweighting is. The final SFT objective is a reweighted token NLL where decision tokens use the class-balanced weight and language tokens use a constant scale $\lambda_{\text{text}}$:

$$\mathcal{L}_{\text{CB}} = \frac{1}{N}\sum_{i=1}^{N} w_i^{\text{CB}}\cdot\big[-\log p_i\big], \qquad w_i^{\text{CB}} = \begin{cases}\hat{w}_{y_i}^{\text{CB}}, & y_i\in\mathcal{S}\\ \lambda_{\text{text}}, & y_i\in\mathcal{T}\end{cases}$$

Hyperparameters: $\beta = 0.9999$, $\lambda_{\text{text}} = 2$.

### Stage 2 — GRPO with multi-grained rewards
SFT still suffers exposure bias and does not directly optimize the timing/accuracy trade-off, so stage 2 runs **GRPO** (Group Relative Policy Optimization, via veRL + vLLM, $G=8$ samples per context). For a generated trajectory $\mathcal{Y}$ the reward is a weighted sum of three signals:

$$R(\mathcal{Y}) = w_{\text{fmt}}R_{\text{fmt}} + w_{\text{turn}}R_{\text{turn}} + w_{\text{traj}}R_{\text{traj}}$$

with weights $w_{\text{fmt}}=0.1,\; w_{\text{turn}}=0.45,\; w_{\text{traj}}=0.45$.

- **Format reward** $R_{\text{fmt}}$: step-level 0/1 — 1 if a step emits a standalone `</Silence>` or a `</Response>` followed by non-empty text, else 0; averaged over trajectory length $K$.
- **Turn-level F1 reward** $R_{\text{turn}}$: reuses the benchmark's StreamPro-F1 idea but with three fixes for reward sparsity. (1) An **additive** step score $S'_i = S_{\text{time},i} + S_{\text{acc},i}$ instead of the multiplicative $S_{\text{time}}\!\cdot\!S_{\text{acc}}$, so one bad component doesn't null the whole reward. (2) A larger, universal temporal tolerance $\tau$. (3) Non-greedy matching: for each GT timestamp $t_{gt}$, consider *all* predictions in $[t_{gt}-\tau,\,t_{gt}+\tau]$ and match the one with highest $S'$. Aggregated:

$$R_{\text{turn}} = \frac{2\sum_i S'_i}{N_{\text{pred}} + N_{\text{gt}}}, \qquad S'_i = S_{\text{time},i} + S_{\text{acc},i}$$

- **Trajectory-level rubric reward** $R_{\text{traj}}$: an LLM designer writes $N_c$ binary checkpoints per sample from the GT; an LLM evaluator scores the whole predicted trajectory, giving $c_i\in\{0,1\}$ per checkpoint, verifying **granularity** (not too fragmented / too coarse), **sequencing** (chronological consistency), **coverage** (essential points present, hallucinations penalized):

$$R_{\text{traj}} = \frac{1}{N_c}\sum_{i=1}^{N_c} c_i$$

### Eval protocol — StreamPro-F1
Each triggered response gets a **time score** decayed by distance from the ideal timestamp and an **accuracy score** (LLM-judge, or IoU for temporal grounding):

$$S_{\text{time}} = \max\!\Big(0,\; 1 - \frac{|t_{\text{pred}} - t_{\text{gt}}|}{\tau}\Big), \qquad S = S_{\text{time}}\cdot S_{\text{acc}}$$

Trajectory scoring is a score-weighted F1 over matched predictions vs GT events:

$$P = \frac{\sum_i S_i}{N_{\text{pred}}}, \qquad R = \frac{\sum_i S_i}{N_{\text{gt}}}, \qquad F_1 = \frac{2PR}{P+R}$$

Note the benchmark metric is **multiplicative** ($S=S_{\text{time}}S_{\text{acc}}$) with per-task tolerance $\tau$; the RL reward relaxes both (additive, single large $\tau=8$) precisely to avoid zero-gradient sparsity. Judge/rubric LLM: Gemini 2.5 Pro. Human validation via a Bradley-Terry preference model on 100 sampled videos.

### Task taxonomy (StreamPro-Bench: 7 tasks / 3 dimensions)
- **Perceptual Understanding (PU):** Event Understanding, Object Understanding, Anomaly Alert.
- **Temporal Reasoning (TR):** Temporal Perception, Temporal Grounding.
- **Proactive Agency (PA):** Goal Planning, Risk Forecasting — the genuinely anticipatory tasks.

## Explicit design choices
- **Decision-token interface:** proactivity is expressed as a per-step binary `</Silence>` vs `</Response>`+text emission — a token-based end-to-end proactive model, not a module/gate on top of an offline VLM.
- **Class-balanced (not focal) loss on the decision token** via effective-number reweighting ($\beta=0.9999$), plus a constant language-token weight $\lambda_{\text{text}}=2$ to keep answer fluency while rebalancing timing.
- **Additive step score in RL, multiplicative in eval:** deliberately different so RL gets dense signal while the benchmark stays strict.
- **Non-greedy windowed matching** in the turn reward (best-$S'$ within $[t_{gt}\!-\!\tau, t_{gt}\!+\!\tau]$) to avoid punishing exploratory near-misses.
- **Three-tier reward** (format 0.1 / turn-F1 0.45 / trajectory-rubric 0.45) so timing, factuality, and global coherence are all shaped; trajectory rubric added specifically for multi-step tasks (Goal Planning, Event Understanding).
- **Data:** StreamPro-SFT-63K + StreamPro-RL-3K constructed from the bench pipeline; SFT jointly trains on ~429K including TimeChat-Online-139K, VideoChat-Flash-3K, and 287K filtered from Streamo-Instruct-465K; RL trains *only* on proactive StreamPro-RL-3K.
- **Backbones:** Qwen2.5-VL-3B and Qwen3-VL-4B. 1 FPS sampling; inference uses a sliding window of 200 dialogue turns.
- **Compute:** SFT 64×H100 / 24h / 1 epoch, lr 1e-5, batch 512; RL 8×H100 / 24h / 1 epoch, lr 1e-6, batch 16, $G=8$, $\tau=8$.

## Key results / what to remember
No Zotero highlights present.

- **Proactive tasks (Table 2, StreamPro-F1).** StreamPro-GRPO-4B reaches **SPB Overall W-Avg F1 = 41.5** (Avg 32.3), vs the previous best open-source proactive model Streamo-3B at **10.4** — roughly 4×. On OVO-Bench **FAR** (Forward Active Responding) it hits **F1 = 20.6**, vs Streamo 5.4 / MMDuet2-3B 6.5 / MMDuet-7B 5.7. StreamPro-SFT-4B (no RL) is SPB 24.7 / FAR 17.3, so GRPO adds a large proactive lift.
- **3B variant:** StreamPro-GRPO-3B SPB Overall W-Avg = 16.3 (SFT-3B = 12.5), FAR = 9.5 — the 4B backbone is clearly stronger.
- **Real-time streaming (Table 3, W-Avg).** StreamPro-SFT-4B: OVO-Bench **58.5**, StreamingBench **71.8**, Overall Avg 65.2; StreamPro-GRPO-4B: OVO **57.6**, StreamingBench **71.2**, Overall W-Avg 67.0. Competitive with dedicated streaming models (best non-proactive StreamingVLM OVO 62.0; module-based StreamBridge OVO W-Avg 69.9) — i.e. proactivity training does not wreck ordinary real-time perception (small drop vs SFT after RL).
- **Offline (Table 4).** VideoMME: baseline Qwen2.5-VL-3B 58.6 → StreamPro-SFT 60.7 / GRPO 60.4. LongVideoBench: baseline 55.2 → SFT 54.6 / GRPO 52.9 (slight offline regression — the proactive tuning costs a little on pure offline long-video).
- **Ablations.** Loss (Table 5): Cross-Entropy 6.6 < Focal 14.2 < **CB-Stream 16.3** on SPB. Temporal tolerance (Table 6): $\tau=3$ → 25.8, $\tau=8$ → **28.1**. Reward mix (Table 7): turn-only ($w_{\text{turn}}=0.9$) 25.5 < balanced (0.45/0.45) **28.1**. Data (Table 9): open-source-only SFT gives 8.9 SPB, adding StreamPro-SFT-63K is what unlocks proactivity.
- **Takeaway:** the win is mostly on the *genuinely proactive* axis (SPB PA / OVO FAR), which prior models score near-zero on; the CB-Stream loss is the SFT unlock and the multi-grained GRPO reward is the proactive unlock.

## How it connects (evolution)
- [[proactive-response]] — this note's sub-topic hub; StreamPro is a flagship reactive→proactive reframing + trained model.
- [[ovo-bench]] — supplies the FAR (Forward Active Responding) proactive task StreamPro is tuned/evaluated on; StreamPro argues FAR-style see-then-answer under-tests true anticipation.
- [[streamingbench]] — real-time streaming benchmark StreamPro reports on (RTVU/CU) to show non-proactive perception is preserved.
- [[mmduet2]] / [[mmduet]] — prior token-based proactive-response baselines StreamPro compares against and beats on SPB/FAR.
- [[streamo]] — the previous best open-source proactive model (SPB 10.4) that StreamPro surpasses; its Streamo-Instruct data is partly reused for SFT.
- [[videollm-online]] — the original decision-token (silence/response) streaming formulation StreamPro's per-step interface descends from.

## Open questions / limitations
- **Judge dependence:** both the RL trajectory reward and the benchmark accuracy score lean on an LLM judge (Gemini 2.5 Pro); rubric generation/scoring quality bounds both training and evaluation, and self-consistency of the judge is unverified beyond the Bradley-Terry human check on 100 videos.
- **Small proactive benchmark:** 577 videos / 1,285 QA is modest; the headline PA improvements come off very low baselines (prior models near-zero), so absolute proactive competence is still low (SPB PA F1 only 7.6 for the best model).
- **Offline trade-off:** proactive tuning slightly regresses LongVideoBench (55.2 → 52.9), and GRPO trades a little real-time W-Avg vs SFT — the timing/accuracy/coherence balance isn't free.
- **Reward/eval mismatch by design:** the RL reward (additive, large $\tau$) is deliberately looser than the benchmark metric (multiplicative, per-task $\tau$), which helps optimization but leaves open how tightly RL gains transfer to the strict metric.

*Verification: equations (CB-Stream $E_k/\hat{w}^{CB}/\mathcal{L}_{CB}$, GRPO $R,R_{turn},R_{traj}$, StreamPro-F1 $S_{time}/P/R/F_1$) transcribed from the arXiv HTML method section and confirmed against the rendered PDF pages 6-7; all headline numbers (SPB 41.5 vs 10.4, FAR 20.6, Table 3 OVO 57.6/58.5 & StreamingBench 71.2/71.8, Table 4 VideoMME/LongVideoBench, ablation Tables 5-7) read directly from the cropped PDF Tables 2-3 and page renders — arXiv:2605.16381v1.*

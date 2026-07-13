---
zotero_key: null
authors: Jinming Liu, Jianguo Huang et al. (SJTU / Eastern Institute of Technology / Microsoft Research Asia)
year: 2026
arxiv: 2605.17921
pdf: https://arxiv.org/pdf/2605.17921
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# An Efficient Streaming Video Understanding Framework with Agentic Control

**Lineage role:** the Remember → Respond → Reason (R3) agentic control loop — casts a streaming turn as three *sequentially cascaded* decisions (compress memory → judge readiness → route compute), with a readiness head for proactive response and a target-balanced RL objective (TB-GRPO) for stable compute routing.

## Problem — what was limited before this paper (short)
Streaming video-LLMs must handle continuously arriving frames under strict latency budgets, where information density varies wildly turn to turn. Prior systems optimize only *one* part of the pipeline in isolation and with *static* policies: fixed memory/token compression, a single "always-on" reaction model, or a hand-crafted decision-reaction trigger. This forces a bad trade-off — fast models fail on hard queries, while always invoking a heavy model violates real-time constraints and wastes compute on easy queries. Memory retention, response timing, and reasoning depth are never *jointly* coordinated, so quality collapses under latency pressure.

## Key idea — the core insight, 2-4 sentences
Reframe a streaming turn not as a passive feed-forward pass but as an **agentic control problem**: at each step the agent makes three coupled decisions in sequence — **Remember** (how much history to keep in memory), **Respond** (whether current evidence supports a reliable answer, else defer), and **Reason** (whether to answer on the fast model or escalate to a heavier slow/thinking model). The design is grounded in two empirical findings: (1) important signal is strongly concentrated in *recent* frames — historical tokens hog attention yet deleting them barely shifts the output distribution, so history can be compressed aggressively (even *improving* accuracy); (2) model-scale gains are non-monotonic — a 3B model beats a 7B/thinking model on some streaming tasks, so always-on heavy inference is wasteful. Each decision consumes the refined state of the previous one, so errors compound and coordinated (not isolated) control is needed.

![[r3-streaming.png]]
> **Crux (Figure 3).** Prior streaming methods (a) route through a single decision → reaction model, giving a sub-optimal efficiency-performance trade-off; R3-Streaming (b) decomposes each streaming step into the three cascaded decisions — Remember (compress historical vs. nearby frames) → Respond (readiness: Routine-defer vs. Fast-Response) → Reason (Answer on fast model vs. Escalate to slow reasoning). *Liu, Huang et al. (2026), arXiv:2605.17921. Embedded for personal research reference.*

## Method + math — the mechanism, then the objectives in full

**Problem formulation.** At streaming step $t$ the system observes history $x_{1:t}$, receives query $q_t$, maintains memory state $M_t$, and acts over the space
$$\mathcal{A} = \{\texttt{<Answer>},\ \texttt{<Escalate>},\ \texttt{<Routine>}\}$$
where `<Answer>` returns a grounded response $y_t$ on the fast path, `<Escalate>` delegates to a slow/thinking model, and `<Routine>` defers output when evidence is insufficient. The objective is to maximize answer quality while minimizing latency and unnecessary slow-model calls under streaming causality. The three stages execute in the order Remember → Respond → Reason.

**Remember (Active Forgetting).** A *training-free*, age-aware memory compressor. History is partitioned by a temporal window $W$ into a nearby zone and a historical zone, compressed at different thresholds:
$$
\begin{aligned}
M_t = {}& \text{Compress}(x_{t-W+1:t},\ \tau_{near})\\
&\cup\ \text{Compress}(x_{1:t-W},\ \tau_{hist}),
\end{aligned}
$$
with $\tau_{near} \gg \tau_{hist}$ — near-range memory keeps fine-grained tokens, far-range memory is consolidated into compact episodic slots. No trainable parameters; controls are the window size, ratio schedule, and compression operator (they use the DTD operator from [[timechat-online]]). Motivated by Finding 1: nearby context is critical, stale history injects misleading attention.

**Respond (Proactive Response).** A lightweight readiness *head* $h$ placed before expensive reasoning estimates whether the current memory grounds a reliable answer:
$$p_{\text{ready}} = h(q_t, M_t), \qquad p_{\text{ready}} \in [0,1]$$
Threshold action rule:
$$
a_t = \begin{cases} \text{emit } \texttt{<Routine>}, & p_{\text{ready}} < 0.5,\\ \text{continue to Reason}, & \text{otherwise.} \end{cases}
$$
Respond is trained *last* — after Reason's SFT cold-start + TB-GRPO — by freezing the fast VLM and training only the readiness head with SFT labels, so it adds minimal parameters and does not perturb the optimized router.

**Reason (Adaptive Thinking).** Two-stage learning of a binary routing policy on the fast model.

*Stage 1 — SFT cold-start* teaches the output *format* (`<Answer>` / `<Escalate>`). For each training query the fast model samples $K=4$ responses $\{y^{(k)}\}$; open-ended answers are scored by an external LLM (Qwen3-14B), objective answers get binary correctness $s^{(k)}\in\{0,1\}$. Averaging $\bar s = \frac{1}{K}\sum_k s^{(k)}$, the routing target is
$$
\begin{cases} \texttt{<Answer>}, & \bar s \ge T,\\ \texttt{<Escalate>}, & \bar s < T. \end{cases}
$$
This fixes format but gives no reward for *good* routing decisions.

*Stage 2 — Target-Balanced GRPO (TB-GRPO).* Vanilla GRPO on this binary task collapses to always-`<Escalate>` (sparse reward makes "always escalate" easier than discriminating). TB-GRPO adds **target-band control** around a user-set operating point $(\eta, \gamma)$. For a group $\{y_i\}_{i=1}^G$, let $e_i=\mathbb{I}[y_i=\texttt{<Escalate>}]$, $c_i$ the action-dependent correctness, and $\rho = \frac{1}{G}\sum_i e_i$ the group escalation ratio. A base reward encodes deployment priorities (a correct *direct* answer beats a correct *escalation* — quality at lower latency):
$$
r_i^{\text{naive}} = \begin{cases} 2, & e_i=0,\ c_i=1,\\ -1, & e_i=0,\ c_i=0,\\ 1, & e_i=1,\ c_i=1,\\ 0, & e_i=1,\ c_i=0. \end{cases}
$$
Deviations from the target band $[\eta-\gamma,\ \eta+\gamma]$:
$$
\begin{aligned}
\delta_{\text{esc}} &= \mathrm{clip}\!\left(\rho-(\eta+\gamma),\,0,\,1\right),\\
\delta_{\text{ans}} &= \mathrm{clip}\!\left((\eta-\gamma)-\rho,\,0,\,1\right).
\end{aligned}
$$
$\delta_{\text{esc}}$ fires only when escalation is excessive ($\rho > \eta+\gamma$); $\delta_{\text{ans}}$ only when it is insufficient ($\rho < \eta-\gamma$); both zero inside the band. These modulate the reward:
$$
r_i = \begin{cases} (1-\delta_{\text{esc}})r_i^{\text{naive}}, & e_i=1,\ c_i=1,\\ (1-\delta_{\text{esc}})r_i^{\text{naive}}-\delta_{\text{esc}}, & e_i=1,\ c_i=0,\\ (1-\delta_{\text{ans}})r_i^{\text{naive}}, & e_i=0,\ c_i=1,\\ (1-\delta_{\text{ans}})r_i^{\text{naive}}-2\delta_{\text{ans}}, & e_i=0,\ c_i=0. \end{cases}
$$
Group-normalized advantages $A_i = (r_i-\bar r)/(\mathrm{std}(\{r_j\})+\epsilon)$ feed a clipped GRPO objective with KL regularization:
$$
\begin{aligned}
\mathcal{L}_{\text{TB-GRPO}} = {}& \mathbb{E}\!\left[ \min\!\left( w_i A_i,\ \mathrm{clip}(w_i, 1-\epsilon_c, 1+\epsilon_c) A_i \right) \right]\\
&- \beta_{\mathrm{KL}} D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}}),
\end{aligned}
$$
with $w_i = \pi_\theta(y_i|x)/\pi_{\theta_{\text{policy}}}(y_i|x)$. Over-escalation shrinks escalation rewards; under-escalation penalizes direct answers — proportional feedback control that steers $\rho$ back into the band. The $(\eta,\gamma)$ pair makes escalation frequency directly tunable to a compute budget. Empirically vanilla GRPO hits $\rho=1.0$ by ~step 40; TB-GRPO converges to a lower, stable ratio and higher reward.

## Explicit design choices
- **Cascaded, order-fixed control:** Remember → Respond → Reason, each consuming the previous refined state; justified because errors compound (noisy memory misleads readiness, which triggers needless escalation).
- **Two-segment age-aware memory:** temporal window $W$ splits nearby vs. historical; $\tau_{near}\gg\tau_{hist}$. Default: Nearby threshold $=1.0$ (no compression), Historical threshold $=0.01$, Nearby window $=3$ frames. Compression operator = DTD from TimeChat-Online. Training-free, zero added parameters.
- **Offline vs. online reconfiguration:** for offline long-video eval they raise the Historical threshold to $0.5$ (~45% token drop) instead of $0.01$.
- **Readiness head is a small MLP-style gate** trained *after* the router, with the fast VLM frozen; threshold $0.5$ on $p_{\text{ready}}$ decides `<Routine>` (defer) vs. proceed.
- **Router SFT cold-start with $K=4$ samples per query**, external LLM (Qwen3-14B) scoring for open-ended, binary correctness for objective; threshold $T$ assigns Answer/Escalate.
- **TB-GRPO defaults:** target escalation ratio $\eta=0.3$, tolerance $\gamma=0.2$; asymmetric base reward (2 for correct fast answer, 1 for correct escalation).
- **Model pairing notation** R3-Streaming-[Fast]|[Slow]: fast = Qwen2.5-VL-3B/7B; slow = Qwen3-VL-4B/8B-Thinking or Qwen2.5-VL-32B.
- **Policy training data:** Respond and Reason sets built from TimeChat-Online-139K.

## Key results / what to remember
Headline (from abstract + Tables 1-2): SOTA among streaming MLLMs at **57.92 on OVO-Bench** and **76.36 on StreamingBench**, with **95%–96% visual-token reduction**.
- **OVO-Bench overall avg (Table 1):** R3-Streaming-7B|4B-Thinking = **57.92** (Real-Time 71.89 / Backward 51.27 / Forward 50.60); 3B|4B-Thinking = 56.55; 3B|7B-Instruct = 54.07. Beats Streamo-7B (55.61), StreamAgent-7B (49.40), Dispider-7B (41.78), TimeChat-Online-7B (45.60), FluxMem-7B (53.40).
- **StreamingBench "All" (Table 2):** 7B|4B-Thinking = **76.36**; 3B|4B-Thinking = 74.36; 3B|7B-Instruct = 73.84. Beats StreamAgent (74.28), TimeChat-Online (73.64), Dispider (67.63), and offline Qwen2.5-VL-7B (73.28) and even proprietary GPT-4o (73.28) / Claude 3.5 Sonnet (72.44).
- **Long-video generalization (Table 3):** R3-Streaming (3B|4B) = **70.6 MLVU / 65.5 Video-MME**, beating StreamAgent-7B (67.2 / 62.9), TimeChat-Online (62.9 / 63.3), AdaReTaKe (65.5? — reports 63.1 Video-MME).
- **Base-module ablation (Table 4, on 3B):** baseline 68.6 / 51.4 (SB / OVO) → w/ Remember 71.0 / 52.8 → w/ Reason 73.4 / 55.9 → w/ Both 74.4 / 56.6 (complementary).
- **Remember compression (Table 5, on 7B):** their DTD-based age-aware variant = **75.90 StreamingBench at 95.0% drop**, vs. standalone DivPrune 68.48, VisionZip 68.32, plain DTD 65.16 — the gain is from the age-aware *policy*, not the operator.
- **Respond (Table 6, StreamingBench Proactive split):** readiness head **0.328** proactive output vs. 0.216 (3B) / 0.204 (7B) same-backbone baselines.
- **Reason routing (Table 7):** TB-GRPO = **74.36 SB at 24.0% escalate ratio**, vs. SFT-only 73.16 @ 100%, Vanilla GRPO 73.16 @ 100% (collapsed), AutoThink 70.00 @ 53.2%.
- **TB-GRPO $(\eta,\gamma)$ sweep (Table 8):** e.g. $\eta{=}0.3,\gamma{=}0.2$ → 74.4 acc / 24% esc; higher $\eta$ raises both accuracy and escalation up to a point (operating point is tunable).

No Zotero highlights present.

Takeaways: (1) the three-decision decomposition is the reusable idea — memory, timing, and depth become *coordinated* dynamic policies rather than static heuristics; (2) aggressive historical forgetting can *raise* accuracy, not just save compute (attention-redundancy finding); (3) target-band RL is a clean fix for the collapse pathology of binary routing under a budget — the escalation ratio is directly dial-able.

## How it connects (evolution)
- [[proactive-response]] — the sub-topic hub; R3's Respond head is a proactive readiness gate (Routine/defer vs. answer).
- [[timechat-online]] — R3 reuses its DTD compression operator for Remember and builds Respond/Reason training sets from TimeChat-Online-139K.
- [[dispider]] — decision-reaction streaming baseline that inspired the Respond gate; R3 outperforms it.
- [[streamagent]] — a directly compared streaming agent baseline (Yang et al. 2025a), also cited as inspiration for the readiness idea.
- [[streamingbench]] — primary streaming benchmark (76.36 headline; Proactive split used for Respond).
- [[ovo-bench]] — the second primary streaming benchmark (57.92 headline; task taxonomy drives the non-monotonic-scale finding).
- [[thinkstream]] — sibling work on adaptive/deliberate reasoning within streaming, related to R3's Reason routing.

## Open questions / limitations
- Remember is training-free but its thresholds are hand-set and *setting-dependent* ($\tau_{hist}=0.01$ online vs. $0.5$ offline) — no learned/adaptive schedule; robustness to arbitrary stream statistics is untested here.
- Respond is a single scalar readiness gate with a fixed $0.5$ threshold and is trained/evaluated mainly on the Proactive split — how well the defer decision calibrates across diverse query types is thinly evaluated.
- The slow "reasoning" path still requires a separate heavy model in memory; the efficiency claim is about token/latency per frame, not total system footprint (two models must be hosted).
- TB-GRPO's operating point $(\eta,\gamma)$ must be chosen to match a compute budget; the paper tunes it but does not learn it online, so a shifting budget or workload would need re-tuning.

*Verification: equations (1)–(8) and all reported numbers transcribed directly from the arXiv PDF text (v2, 1 Jun 2026) — abstract, Sec. 3–4 method, and Tables 1–8 on pp. 6–7; figure cropped from p. 3 (Figure 3). Zotero not running (skipped); no external claim-check performed.*

---
zotero_key: null
authors: Weicai Yan et al. (Microsoft Research Asia)
year: 2026
arxiv: 2603.03447
pdf: https://arxiv.org/pdf/2603.03447
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# Proact-VL: A Proactive VideoLLM for Real-Time AI Companions

**Lineage role:** A learned trigger-policy proactive streaming VideoLLM — a lightweight per-second "should I speak now?" classification head on top of a streaming decoder — jointly optimizing *when to speak* and *what to say*, and introducing the Live Gaming Benchmark (ICML 2026 submission).

## Problem — what was limited before this paper
Prior streaming video assistants split into two camps that each solve half the companion problem. **Proactive models** (e.g. [[videollm-online]], [[mmduet]]) decide *when* to speak but tend to emit long, offline-style answers ill-suited to live audio. **Real-time models** ([[livecc]], StreamingVLM) optimize low-latency generation but lack an autonomous mechanism to control *whether/when* to speak — they need an explicit query or fire continuously. Neither jointly controls **timing** and **utterance style** under the strict constraint of a real-time AI companion (e.g. a live game commentator) that must interject short, well-timed remarks, stay silent otherwise, and run unbounded. There was also no benchmark that stresses proactive timing across solo commentary, multi-agent co-commentary, and goal-directed user guidance.

## Key idea
Process the stream in **fixed 1-second chunks**, and at every second let the model make an explicit binary "speak vs. stay silent" decision *before* generating. A special `<|FLAG|>` decision token is appended after each user turn; its hidden state is fed to a small gated-MLP **response head** that outputs a speaking probability, thresholded at $\tau$. Only if triggered does the decoder generate a short clip-level utterance; otherwise it emits a silence placeholder. Crucially, the timing decision is framed as a **sequence** problem (learning when to *keep* vs. *switch* state), not independent per-step classification, and is trained with a transition-smoothed loss plus a stability regularizer alongside the standard LM loss.

![[proact-vl.png]]
> **Crux (Figure 4).** At each second Proact-VL ingests multi-source tokens (background/visual/query + a `FLAG` token), the decoder produces a hidden state at `FLAG`, and a gated-MLP response head scores it; `>τ` → append assistant prefix and speak a short clip-level text, `<τ` → append a Silence token and stay quiet — all over a persistent KV cache. *Yan et al. (2026), arXiv:2603.03447. Embedded for personal research reference.*

## Method + math

**Chunk-wise streaming with a persistent cache.** At timestep $t$ the model consumes a triplet of token streams — visual content $V_t$, an optional user query $Q_t$, and background/context $B_t$ (prior commentary) — serialized in a ChatML-style role format (system / user / assistant). It produces an utterance $U_t$ and an updated KV cache:
$$ (U_t, \mathcal{K}_t) = f_\theta(V_t, Q_t, B_t;\; \mathcal{K}_{t-1}), $$
where $\mathcal{K}_t$ is the running key-value cache. Generated utterances are appended into future context, so $U_t$ serves both as the current output and as context for coherence. This incremental caching gives true online processing with full temporal context.

**Proactive response mechanism.** A `<|FLAG|>` decision token is inserted at the end of each user message. After processing the full context at step $t$, the hidden state $\mathbf{h}_t$ at that token is passed through a gated MLP + sigmoid to get a speaking probability:
$$ p_t = \sigma\big(\mathrm{MLP}(\mathbf{h}_t)\big). $$
The binary decision compares $p_t$ to a fixed threshold $\tau$:
$$ a_t = \mathbb{I}[\,p_t \ge \tau\,], $$
with $a_t=1$ triggering utterance generation and $a_t=0$ maintaining silence. This lightweight "decide-then-generate" gatekeeping is more controllable than treating silence as an ordinary output token.

**Training objectives — jointly learning *what to say* and *when to speak*.** The overall loss combines the causal LM loss and a response (timing) loss:
$$ \mathcal{L} = \mathcal{L}_{\text{main}} + \alpha\,\mathcal{L}_{\text{resp}}, \qquad \alpha = 0.2. $$
$\mathcal{L}_{\text{main}}$ is standard next-token cross-entropy on assistant tokens, masked so supervision only lands on positions corresponding to actual assistant responses (silence positions excluded).

$\mathcal{L}_{\text{resp}}$ has two parts. The insight is that the dominant imbalance is not silence-vs-response *count* but **state transitions vs. state persistence** (transitions are rare, ~1:5). A *transition-smoothed classification loss* up-weights the rare switching steps: with ground-truth speaking labels $y_t \in \{0,1\}$, set the per-step weight $w_t = \gamma$ (with $\gamma = 5$) when $y_t \neq y_{t-1}$ (a transition) and $w_t = 1$ otherwise, then
$$ \mathcal{L}_{\text{cls}} = \frac{1}{\sum_t w_t} \sum_t w_t\big(-y_t \log p_t - (1-y_t)\log(1-p_t)\big). $$
A *stability regularizer* enforces smooth probabilities inside a continuous speak/silence segment (local term) and calibrates the global speaking rate to the human baseline (global term):
$$ \mathcal{L}_{\text{reg}} = \mathbb{E}\big[(p_t - p_{t-1})^2 \mid y_t = y_{t-1}\big] + \big(\mathbb{E}[p_t] - \mathbb{E}[y_t]\big)^2, $$
where $\mathbb{E}[y_t]$ is treated as a constant human-baseline speaking rate. The two combine as $\mathcal{L}_{\text{resp}} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{reg}}$.

**Infinite inference (dual-cache + reverse-RoPE).** To run unbounded under a fixed context length, Proact-VL keeps a persistent **system cache** (initial prompt) plus a dynamic **streaming cache** (ongoing user/assistant tokens). When the budget is hit, it evicts the oldest ~20% of the streaming cache and applies a lightweight **reverse-RoPE** correction to re-base the remaining cached keys' positions, avoiding positional discontinuity without re-encoding.

**Live Gaming Benchmark — the eval protocol (this is the "math" for the benchmark side).**
- *Scale/data:* ~561 hours across 12 game titles; training set is 128,000 samples from 12 sources (10 games + [[livecc]]-style data + Ego4D). Data pipeline: WhisperX-large-v3 for ASR + speaker ID → Qwen3-Omni-Flash for paralinguistic labels (pauses/laughter) → DeepSeek-V3.2 for game-terminology polish → persona/commentator-profile synthesis.
- *Three scenarios:* **Solo Commentary** (autonomous narration), **Co-Commentary** (multi-agent social coordination), **User Guidance** (goal-directed player assistance).
- *Splits:* clip-level in-domain benchmark of 2,640 clips (10 games, stratified by response rate into 0–30% / 30–70% / 70–100% bins) + 240 Black Myth: Wukong (out-of-domain game) + 134 Ego4D (general); plus a **Streaming** long-form set of 10 full videos (30 min–2 h).
- *Text-quality metrics:* **CC** = win-rate of outputs vs. Gemini 2.5 Pro (LLM-as-judge); **LiveU** = streaming usability, mean of Time/Rate/TextU (coverage of salient moments with minimal overlap, pacing, immediate spoken clarity); **FinalQ** = concatenated-script quality (mean of Fidelity/Continuity/Substance).
- *Proactivity metrics:* **TimeDiff** (↓) = temporal distance between predicted and ground-truth onset, penalizing off-window predictions; **PAUC** (↑) = proactive area-under-curve integrating response quality over time; **F1** (↑) = temporal binary classification over the full timeline (response windows positive). The scoring judge is GPT-5.1 (for PAUC, CC win-rate, LiveU/FinalQ).

## Explicit design choices
- **Base model:** initialized from **LiveCC-7B-Base**; also trained from Qwen-series backbones (appendix) which likewise beat the base model.
- **1-second fixed chunk** as the atomic streaming step; utterances are short "clip-level" texts (~1 s of speech), not long answers.
- **`<|FLAG|>` decision token + gated-MLP response head + sigmoid**, fed from the **penultimate** transformer layer's hidden state — a separate scoring head rather than modeling silence as an LM output token.
- **Threshold:** $\tau = 0.3$ for the main results (Secs 5.2–5.4); $\tau = 0.5$ for ablations/analyses.
- **Loss weighting:** $\alpha = 0.2$; transition weight $\gamma = 5$ (matched to the ~1:5 transition:persistence ratio); gradient clipping `max_grad_norm = 1.0`.
- **Optimization:** lr 1e-5 cosine schedule, batch size 64, 2,000 steps, ~200 H100 GPU-hours total.
- **ChatML serialization** of (system/user/assistant) with background, visual, and query token types; assistant utterances appended to context.
- **Dual-cache sliding window + reverse-RoPE**, evicting oldest ~20% of the streaming cache at capacity for unbounded runtime.

## Key results / what to remember
No Zotero highlights present.

Verified against the paper's Tables 1–2 (Live Gaming Benchmark, printed p.7). All Proact-VL (Real-Time Proactive Model row):
- **Overall:** CC **49.23**, LiveU **6.52**, FinalQ **5.03** (best across all categories) | TimeDiff **1.71**, PAUC 18.10, F1 **64.87** (best F1).
- **Solo Commentary:** CC **53.62**, LiveU **6.89**, FinalQ **5.48**; TimeDiff 1.20, PAUC 20.36, F1 **63.25**.
- **Co-Commentary:** CC **51.46**, LiveU 5.15, FinalQ **3.59**; TimeDiff **0.71**, PAUC 7.01, F1 **77.44** (much higher than any baseline).
- **Guidance:** CC 42.60 (best), LiveU **7.52**, FinalQ 6.02; TimeDiff 3.21, PAUC 26.92, F1 53.91.
- **Comparisons:** beats offline GPT-4o (overall CC 39.42, F1 6.54) and Gemini 2.5 Pro on proactivity F1; beats proactive baselines [[videollm-online]] (overall CC 13.78, F1 6.54), [[mmduet]] (CC 20.08, F1 0.18), Livestar (CC 8.59, F1 0.20) on both text and timing; and real-time LiveCC-7B-Base (CC 38.88, F1 36.10) / StreamingVLM (CC 14.89, F1 50.67) on text quality and triggering. Honest weakness: in Guidance its PAUC/FinalQ sit slightly below the strongest offline model, and its Solo PAUC (20.36) is below GPT-4o's (25.14).
- **Ablation (Table 6):** removing $\mathcal{L}_{\text{cls}}$ collapses timing (F1 drops sharply, TimeDiff blows up); removing $\mathcal{L}_{\text{reg}}$ leaves F1 roughly intact but degrades CC (47.53 → 45.54) — both terms needed.
- **Efficiency (Table 7):** per-chunk latency ~0.35–0.43 s; peak memory ~16–17 GB on a single H100; sustains real-time (well under the 1 s chunk budget).
- **Generalization (Table 3):** strong on Ego4D (general) and Black Myth: Wukong (out-of-domain game) — top-tier LLM-judged text quality, indicating the trigger policy transfers.

Takeaways: (1) Framing timing as a **sequence** problem with transition-aware weighting is what makes the trigger accurate — plain per-step classification fails. (2) A **separate scored decision token** decouples "when" from "what," giving controllable short utterances. (3) The **CC = win-rate-vs-Gemini** and PAUC/TimeDiff/F1 metric suite is a reusable proactive-eval template. (4) reverse-RoPE + dual cache is a cheap route to unbounded streaming.

## How it connects (evolution)
- [[videollm-online]] — the streaming-dialogue + EOS-token proactivity ancestor Proact-VL improves on (short, timely utterances vs. long answers).
- [[mmduet]] — proactive "informative head" per-frame speak decision; a direct proactive baseline it beats.
- [[livecc]] — real-time low-latency commentary and the **base model** Proact-VL fine-tunes from.
- [[streammind]] — proactive streaming with a learned gating/"perceive-then-respond" trigger; closest methodological sibling on the timing head.
- [[proassist]] — proactive assistant framing (when to interject) in the companion setting.
- [[proactive-response]] — the sub-topic hub aggregating learned-trigger proactive VideoLLMs.
- [[streaming-benchmarks]] — where the Live Gaming Benchmark's proactive metrics (TimeDiff/PAUC/F1) sit among streaming eval protocols.

## Open questions / limitations
- **Judge-dependent metrics:** CC is a win-rate vs. Gemini 2.5 Pro and quality scores use GPT-5.1 as judge — headline text numbers inherit that judge's bias/stochasticity (paper adds robustness tables, but the ceiling is judge-defined).
- **Guidance is the weak scenario:** PAUC/FinalQ trail the strongest offline model, suggesting the model under-incorporates external/goal guidance — the harder "assistant" half of companionship is not solved.
- **Threshold sensitivity:** a single global $\tau$ (0.3 main / 0.5 ablation) governs all speaking; per-context or adaptive thresholds are unexplored, and the speaking-rate calibration assumes a fixed human baseline $\mathbb{E}[y_t]$.
- **Domain scope:** trained/evaluated heavily on gaming streams; transfer beyond Ego4D/games (e.g. sports, tutoring) is untested.

*Verification: equations ($p_t$, $a_t$, $\mathcal{L}$, $\mathcal{L}_{\text{cls}}$, $\mathcal{L}_{\text{reg}}$, caching) read off the rendered method pages (printed pp.5–6); all headline numbers cross-checked against the rendered Tables 1–2 (printed p.7); hyperparameters ($\tau$, $\alpha$, $\gamma$, base model, GPU-hours) confirmed from the Implementation-Details paragraph on printed p.6. Full-text via arXiv HTML; figures via the downloaded PDF.*

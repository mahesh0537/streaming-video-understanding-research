---
zotero_key: null
authors: Yueqian Wang et al. (Peking University WICT; Meituan)
year: 2025
arxiv: 2512.06810
pdf: https://arxiv.org/pdf/2512.06810
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# MMDuet2: Enhancing Proactive Interaction of Video MLLMs with Multi-Turn Reinforcement Learning

**Lineage role:** direct successor to [[mmduet]] — it removes the hand-tuned response-head threshold and trains proactive *when-to-speak* behaviour end-to-end with multi-turn GRPO under a timing-aware reward (PAUC), landing at ICLR 2026.

## Problem — what was limited before this paper (short)
Proactive video assistants must decide, frame by frame, *whether and when* to reply while a video streams in. Prior streaming approaches ([[videollm-online]], [[mmduet]], [[dispider]]) attach a lightweight "response head" that emits a scalar probability at each frame, and a human must hand-tune a decision **threshold**: set it too high and the model stays silent, too low and it floods the stream with redundant, repetitive replies. Worse, supervising that head needs **frame-level reply-timestamp annotations**, which are expensive and imprecise — real caption/scene data is only coarse (scene-level), so there is no reliable ground-truth "reply now" frame to imitate. The paper's premise is that timing should be *learned* from a reward, not distilled from brittle timestamp labels or gated by a magic number.

## Key idea — the core insight, 2-4 sentences
Cast proactive interaction as a pure **text-to-text** problem: at every incoming frame the model either produces a natural-language reply or emits a literal `NO REPLY` token, so the decision *when to speak* is just ordinary autoregressive generation with no external head and no threshold. Then optimize this behaviour with **multi-turn reinforcement learning** (GRPO), where the reward — anchored on a **Proactive-AUC (PAUC)** metric — rewards saying the right thing *as early as possible within the valid span* while penalizing duplicate, off-span, and verbose-prefix replies. Because the reward only needs the coarse scene span (a start/end window), not an exact reply frame, it dodges the annotation-precision problem entirely; the model discovers its own timing.

![[mmduet2.png]]
> **Crux (Figure 2).** The MMDuet2 chat template: a system prompt frames the proactive task, each user turn feeds 1–2 new video frames (`<image>` tokens), and the assistant autonomously answers *or* emits `NO REPLY` — the whole "should I speak now?" decision is text-to-text, replacing MMDuet's thresholded response head. *Wang et al. (2025), arXiv:2512.06810. Embedded for personal research reference.*

## Method + math — mechanism, then objectives in full

### Interaction mechanism
Frames are sampled at a fixed interval (2 s) and each is encoded to ~128 tokens; **1–2 frames constitute one user turn** (2 for the longer [EGO]/[TV]/[VAD] streams, 1 for short [WEB]/StreamingBench-PO). After each user turn the assistant decodes greedily: a text span becomes a timestamped proactive reply (timestamp = frame position ÷ sampling rate), while `NO REPLY` advances the stream with no output. A custom system message (Figure 2) separates the proactive regime from ordinary offline video-QA, which the paper uses to curb catastrophic forgetting of offline understanding.

**Training data / dialogue construction.** Two dialogue schemas are built over segmented, captioned scenes: **1QnA** (one question $q^j$ asked up front, whose multiple scene-answers $[a^j_1,\dots,a^j_n]$ must each be replied within their scene span $v_i$) and **nQnA** (questions can arrive anytime; an LLM summarizes all answers of scenes ending before the question into one *immediate answer* $a^j_{\{1,\dots,t-1\}}$ to be spoken at once, with later spans replied as their scenes play). Dataset: **50,228 web videos** (avg 92.7 s, Live-WhisperX) + **2,543 ego-centric videos** (avg 164.4 s, Ego-Exo4D + EgoExoLearn).

**Two-stage pipeline.** SFT on ~52k videos (minus 1,900 held out) + 25k offline video-QA (LLaVA-Video) + 25k captions (Tarsier2) — answers are placed at the *end* of each reply span to avoid hallucinating ahead of evidence (16 H800, ~8 h). Then RL on the 1,900 held-out videos with SGLang + veRL (8 H800, ~20 h). Base model: **Qwen2.5-VL 3B**.

### PAUC — the timing-aware correctness signal
For a ground-truth reply turn with valid span $(t^{start}, t^{end})$, let the model emit $P$ replies at timestamps $\tau_1<\dots<\tau_P$ with per-reply correctness scores $s_1,\dots,s_P$ (an LLM/judge score in $[0,S]$). Plot the step function $s(\tau)$ and integrate the area under it, normalized by the maximum attainable area $(t^{end}-t^{start})\cdot S$:

$$
\mathrm{PAUC} = \frac{(\tau_1 - t^{start})\cdot 0.5 \;+\; \sum_{p=1}^{P-1}(\tau_{p+1}-\tau_p)\, s_p \;+\; (t^{end}-\tau_P)\, s_P}{(t^{end}-t^{start})\cdot S}
$$

The leading $0.5$ credits a small default before the first reply; the sum accumulates *correctness held over time*, so both **higher correctness** and **earlier** replies enlarge the area. Implementation uses $S=4$ and reports at $\omega=0.5$.

### Multi-turn RL objective
Optimization is **GRPO** with **4 rollouts** per prompt. Crucially, to avoid the sparse-reward / credit-assignment blow-up of rolling out an entire long video, training samples a **short span of 20–60 s** from a video and provides the *prior dialogue history as context* — so each episode is a tractable multi-turn segment. The scalar reward combining the four components is

$$
r = \omega_{\text{PAUC}}\, r_{\text{PAUC}} + \omega_{\text{rep}}\, r_{\text{rep}} + \omega_{\text{in\_span}}\, r_{\text{in\_span}} + \omega_{\text{pfx}}\, r_{\text{pfx}}
$$

with weights $\omega_{\text{PAUC}}=3,\ \omega_{\text{rep}}=2,\ \omega_{\text{in\_span}}=0.5,\ \omega_{\text{pfx}}=2$, where:
- $r_{\text{PAUC}}$ — the timing-aware correctness term above (be right, be early).
- $r_{\text{rep}}$ (replication) — inverse of (already-covered info entries ÷ total entries); suppresses re-saying information already delivered.
- $r_{\text{in\_span}}$ (in-span) — inverse of (out-of-span replies ÷ total replies); suppresses off-topic / mistimed replies outside any valid scene window.
- $r_{\text{pfx}}$ (prefix) — penalizes long common prefixes shared with previous replies (stops verbose, boilerplate-repeating openings).

GRPO forms group-relative advantages over the 4 rollouts, so no learned value/critic is needed. Figure 5 documents three empirical RL phases: a **transition** dip (steps ~0–180), rapid **growth** (~180–450), and a **plateau** (~450–489) where over-training risks degrading long-form generalization.

## Explicit design choices
- **No response head, no threshold**: `NO REPLY` is a generated token; the speak/stay-silent decision is folded into ordinary decoding.
- **Coarse-span supervision, not frame-precise labels**: reward needs only scene start/end windows → sidesteps costly reply-timestamp annotation.
- **Span-sampled RL episodes (20–60 s)** with prior dialogue as context → dense, learnable reward instead of one terminal reward over a whole video.
- **GRPO, 4 rollouts**, critic-free group-relative advantages.
- **Four-term reward** with PAUC dominant ($\omega=3$); duplicate + prefix penalties weighted 2 each; in-span 0.5.
- **PAUC metric with $S=4$**, reported at $\omega=0.5$ — same measure used as the training reward *and* the eval metric.
- **Frame protocol**: 2 s sampling, ~128 tokens/frame, 1–2 frames per user turn (2 for long streams).
- **Answers placed at end of reply spans in SFT** to prevent premature hallucination.
- **Custom proactive system prompt + offline-mixed SFT** to preserve offline video-QA ability (anti-forgetting).
- **Base = Qwen2.5-VL 3B**; SFT→RL two-stage on 16→8 H800 GPUs.

## Key results / what to remember
Verified against the paper's tables.

**ProactiveVideoQA (Table 2) — PAUC ($\omega{=}0.5$) ↑ / reply-duplicate proportion ↓:**
- [WEB]: MMDuet2-RL **53.3 / 4.2** vs MMDuet 38.9 / 81.3 vs VideoLLM-Online 25.9.
- [EGO]: MMDuet2-RL **33.6 / 8.1** vs MMDuet 46.0 / 99.4 (MMDuet scores higher PAUC but by spamming ~99% duplicate replies) vs VideoLLM-Online 25.0.
- [TV]: MMDuet2-RL **43.4 / 1.0** vs MMDuet 21.1 / 92.8.
- [VAD]: MMDuet2-RL **28.9 / 15.2** vs MMDuet 27.4 / 99.2 (all models weak on surveillance video).
- MMDuet2-SFT alone: [WEB] 37.6 / 1.7, [EGO] 26.4 / 4.4 — RL supplies the large PAUC jump while keeping duplicates near-zero.

**StreamingBench Proactive Output (Table 5) — accuracy:** MMDuet2-RL **34.69** vs MMDuet 29.44 vs Dispider 25.34 vs MMDuet2-SFT 19.59 vs VideoLLM-Online 1.96.

**Offline understanding retained (Table 4, † = their impl):** MMDuet2-RL† Video-MME (w/wo sub) 67.5 / 58.1, MVBench 66.4, LongVideoBench 52.7 — essentially matching the Qwen2.5-VL 3B† base (66.5 / 57.3, 65.6, 53.1), i.e. proactive training barely dents offline skills.

**Efficiency (Table 3, [WEB], H100):** MMDuet2 issues **3.3 (±1.9) reply turns** vs MMDuet's 5.7 (±3.4) — far fewer redundant replies — at comparable wall time (2m52s vs 2m27s, MMDuet2 fully generates "NO REPLY" every decision).

**Ablations (Table 6):** removing $r_{\text{rep}}$ raises raw PAUC (55.5 [WEB]) but explodes duplicates (17.3) and reply count (4.9); removing $r_{\text{in\_span}}$ makes the model reply at almost every turn ([WEB] PAUC 62.7 but 8.4 reply turns, [EGO] FAIL — inference >20 min/sample). Frame density (Table 7): SFT 2 s / RL 2 s / infer 1 s gives best [WEB] 53.3; mismatched RL-vs-inference intervals hurt (RL 1 s / infer 2 s → 39.4).

No Zotero highlights present.

Takeaways: (1) the headline story is *quality of timing, not just PAUC* — MMDuet's high [EGO] PAUC is an artifact of ~99% duplicate spam; MMDuet2 wins on the joint PAUC-and-low-duplicate frontier. (2) RL, not SFT, delivers proactive competence — SFT alone underperforms even MMDuet on StreamingBench-PO. (3) The reward's non-PAUC terms are load-bearing guardrails: without them the model degenerates into over-replying.

## How it connects (evolution)
- [[mmduet]] — the direct predecessor; MMDuet2 keeps its dense per-frame proactive framing but swaps the thresholded response head for text-to-text `NO REPLY` + RL.
- [[videollm-online]] — the earlier streaming-dialogue baseline it consistently beats; shares the "reply during playback" formulation.
- [[dispider]] — contemporaneous proactive streaming model used as a StreamingBench-PO baseline.
- [[proactivevideoqa]] — the benchmark (and PAUC metric) this paper trains and evaluates on ([WEB]/[EGO]/[TV]/[VAD] splits).
- [[streamingbench]] — supplies the Proactive-Output task where MMDuet2-RL leads at 34.69.
- [[proactive-response]] — the sub-topic hub: MMDuet2 is a keystone example of RL-driven "when to speak."

## Open questions / limitations
- **Weak on surveillance ([VAD]: 28.9 PAUC, 15.2 duplicates)** — proactive timing on long, low-event streams is still largely unsolved for all models.
- **Reward-metric circularity**: PAUC is both the training reward and the primary eval metric, so gains partly measure reward-optimization; StreamingBench-PO is the more independent check.
- **Over-training risk**: Figure 5's plateau phase shows long-form generalization can degrade past ~450 steps — the RL stopping point is delicate.
- **Coarse-span reward ceiling**: because timing is only supervised at scene granularity, sub-scene precision (exact best frame to speak) is left to emergent behaviour, not directly optimized.

*Verification: equations (PAUC, combined reward, weights), the RL setup (GRPO/4 rollouts, 20–60 s spans, Qwen2.5-VL 3B, dataset sizes), and all numbers cross-checked against the arXiv:2512.06810 HTML full text and the rendered PDF Tables 2–7; Figure 2 read from the rendered page 3.*

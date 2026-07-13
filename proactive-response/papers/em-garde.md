---
zotero_key: null
authors: Yikai Zheng, Xin Ding et al. (AIR/IIIS Tsinghua, USTC, Microsoft Research Asia, Nanjing University)
year: 2026
arxiv: 2603.19054
pdf: https://arxiv.org/pdf/2603.19054
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# Em-Garde: A Propose-Match Framework for Proactive Streaming Video Understanding

**Lineage role:** Reframes proactive triggering from a per-frame "reason-and-decide" loop into a two-phase **propose-then-match** design — parse the query into candidate visual proposals *once* (offline the streaming loop), then cheaply match those proposals against the live stream to decide *when* to respond.

## Problem — what was limited before this paper (short)
Proactive streaming VideoLLMs must decide, at *every* incoming frame, whether now is the moment to answer a standing query (e.g. "tell me when the water boils"). Prior systems (VideoLLM-Online, MMDuet, Dispider, StreamForest) fold this into a per-frame decision made by a heavy LLM that re-runs visual perception + event-transition detection + query-relevance reasoning on each timestep. Because the loop must keep up with the frame rate (5-10+ fps), the per-frame compute budget is tiny — forcing an **efficiency-accuracy dilemma**: rich enough reasoning to trigger correctly is too slow, and fast enough models trigger poorly. Context also grows unboundedly, so KV-cache-based approaches slow down as the video lengthens.

## Key idea — the core insight, 2-4 sentences
The expensive part (understanding *what* the query means and *what visual evidence* would justify a response) does not need to be recomputed every frame — it can be done **once at query time**. So Em-Garde decouples semantic understanding from streaming perception: an **Instruction-Guided Proposal Parser (IGPP)** runs a full 7B VLM once to turn the instruction into a set of concrete, perceptually-groundable *visual proposals* (short textual descriptions of trigger-worthy events). Then a training-free **Lightweight Proposal Matching Module (LPMM)** runs in the streaming loop, embedding a short sliding window of frames and firing a response when any proposal's text-video similarity **sharply rises**. This converts a semantic decision problem into a cheap embedding-matching problem on constant-length input.

![[em-garde.png]]
> **Crux (Figure 2).** IGPP (orange) parses the query + low-fps context into grounded visual proposals once at query time; LPMM (blue) runs per-frame in the streaming loop, tracking text-video cosine similarity of each proposal and triggering the off-the-shelf responder MLLM the instant similarity spikes — separating semantic parsing from the real-time loop. *Zheng, Ding et al. (2026), arXiv:2603.19054. Embedded for personal research reference.*

## Method + math — the mechanism, then the central objective/equations in full

**Problem formulation.** An instruction $I$ arrives once at query time $t_0$; the stream is an unbounded frame sequence $\{x_t\}_{t\ge 0}$. At each step $t\ge t_0$ the model emits a binary triggering decision
$$D_t = \pi_{\text{trigger}}\big(I, \{x_\tau\}_{0\le\tau\le t}\big) \in \{0,1\},$$
where $D_t=1$ means "respond now." On a trigger at $t_1$, a responder generates $R_{t_1}=G(I,\{x_\tau\}_{0\le\tau\le t_1})$. Let $T_R=\{t\mid D_t=1\}$ be the predicted response times; performance matches these against ground-truth event times within a tolerance window.

**Two-stage decomposition.** Instead of computing $\pi_{\text{trigger}}$ from scratch every frame, Em-Garde splits it:

1. **Semantic parsing (once, at $t_0$).** Given $I$ and a short low-fps context window $\{x_\tau\}_{t_0-h_c\le\tau\le t_0}$, the proposer produces a proposal set
$$P = \pi_{\text{propose}}\big(I, \{x_\tau\}_{t_0-h_c\le\tau\le t_0}\big).$$
Each proposal describes observable visual evidence for a trigger-worthy event.

2. **Visual perception (every frame, in the loop).** At step $t$, over a short sliding window $\{x_\tau\}_{t-h_s\le\tau\le t}$ of length $h_s$, compute matching scores against $P$ and derive the decision by simple temporal processing of the score history:
$$S^t = \pi_{\text{match}}\big(P,\{x_\tau\}_{t-h_s\le\tau\le t}\big), \qquad D_t = \pi_{\text{tp}}\big(\{S^\tau\}_{\tau\le t}\big).$$

**Proposal design properties.** IGPP is trained to emit proposals that are (i) *temporally localizable* — matched only when a response is actually due; (ii) *perceptually groundable* — recognizable from a short segment by simple perception, no long-context reasoning; (iii) *redundant* — multiple complementary cues describe the same underlying event, hedging the inherent uncertainty of a streaming video rather than betting on one hypothesis.

**IGPP training (two stages).**
- **SFT** on the curated **Parse2Prop-1K** dataset teaches the proposal format and pre-defined proposal methods.
- **RL** (GRPO) directly optimizes *downstream triggering behavior*: the proposer runs the actual perception module, tests its proposals against the stream, and is rewarded for triggering near each ground-truth event onset. A trigger is *correct* if within a **4-second** tolerance window after event start (accounting for information-gathering latency); triggers outside these windows are false positives. The reward balances recall vs. false-trigger suppression:
$$r = \Big(1-\lambda\, r_{fp}\Big)\,\frac{n_c}{n}, \qquad r_{fp} = 1 - 2^{-n_{fp}/n},$$
where $n_c$ = correct triggers, $n_{fp}$ = false triggers, $n$ = number of ground-truth events, and $\lambda$ is the false-positive penalty coefficient (default $\lambda=1$). $n_c/n$ rewards recall; $(1-\lambda r_{fp})$ discounts it as false triggers accumulate.

**LPMM matching (training-free).** LPMM is an off-the-shelf VLM multimodal embedding model (**Ops-MM-V1-2B**), used with no fine-tuning. At each $t$ it embeds the recent sliding window into a video representation $e_v^t$; each proposal $p_i$ is pre-embedded as $e_{p_i}$. The per-proposal matching score is cosine similarity
$$s_i^t = \cos\!\big(e_v^t,\, e_{p_i}\big).$$
Triggering keys off the *temporal change* in similarity, not its absolute value — a trigger fires when at least one proposal's score jumps sharply:
$$D_t = \mathbb{1}\Big[\max_i\big(s_i^t - s_i^{t-1}\big) > \theta\Big],$$
with default threshold $\theta=0.04$. Using the *derivative* (score increase) rather than absolute similarity makes triggering stable across videos, and $\theta$ is exposed as an explicit sensitivity knob (higher $\theta$ → fewer spurious triggers but more misses).

**Computational efficiency.** The streaming loop runs a lightweight model on **constant-length** input, so inference time does not grow linearly with video length. A **visual encoding cache** stores encodings of overlapping frames in the sliding window so only one new frame is encoded per step — a further $2\times$-$3\times$ speedup. Result: up to **10-15 fps** on A100 GPUs over arbitrarily long sessions. The heavy proposal parsing and response generation run infrequently and asynchronously, so they never block the loop.

## Explicit design choices — concrete decisions (raw material for new systems)
- **Decouple semantic parsing from the streaming loop:** run the heavy 7B VLM *once* at query time, keep the per-frame loop lightweight and training-free.
- **IGPP proposer:** initialized from **Qwen2.5VL-7B**; input = instruction + ~5-second low-fps (1 fps) context; output = set of textual visual-cue proposals.
- **Proposals as a redundant, groundable set** (avg **3.99 cues/query**, avg **13.5 words/cue**) rather than a single target description — hedges streaming uncertainty.
- **LPMM matcher:** off-the-shelf **Ops-MM-V1-2B** multimodal embedding model, *no training*; consumes ~2-second, ~2 fps sliding windows.
- **Trigger on similarity *increase* (derivative) not absolute similarity**, threshold $\theta=0.04$ default — robust across videos and to threshold choice.
- **RL (GRPO) on downstream triggering** with a recall × false-positive-penalty reward, correctness within a 4-second window, $\lambda=1$.
- **Parse2Prop-1K training data:** 668 queries over 92 videos from **COIN, Ego4D, BEHAVIOR**; half the proposal examples human-or-GPT-5-authored, diverse styles/lengths.
- **Visual encoding cache** to encode only the newest frame each step ($2$-$3\times$ speedup).
- **Off-the-shelf MLLM as the responder** (Respond & Check), invoked only after a trigger — the framework is modular around the responder.
- **New online eval metric for OVO-Bench** (recall/precision within a tolerance window) replacing its indirect offline-accuracy proxy.

## Key results / what to remember — exact numbers with setting (verified vs. the paper's tables)

**OVO-Bench Future/Forward Active Responding (online recall/precision → F1), Table 1.** Em-Garde (2B, 2 fps) **Avg. F1 = 30.99**, vs. MMDuet-2 (3B) 20.51, StreamForest (7B) 13.95, VideoLLM-Online (8B) 11.07, FVStream (7B) 4.77. Per-task F1: CRR 26.40, SSR 23.40, REC 43.16.

**StreamingBench PO task accuracy, Table 2.** Em-Garde 2B = **37.6 @1 fps / 38.0 @2 fps**, vs. MMDuet-2 34.6, StreamAgent 28.9, Dispider 25.3, VideoLLM-Online 4.0, FVStream 2.0 — "more than 3% accuracy" over the best prior.

**ProactiveVideoQA, PAUC↑ / reply-duplicate-proportion↓, Table 3.** Em-Garde WEB 44.3/4.5, EGO **52.3**/17.4, VAD 27.4/1.4 — beats MMDuet-2 on EGO (33.6) and is competitive on WEB (MMDuet-2 53.3) despite MMDuet-2 directly optimizing PAUC and training on related data.

**Online video understanding (responder quality), Table 4.** Em-Garde (7B responder): StreamingBench Real-time VU **76.7**, OVO Real-time VP **63.0**, Backward Tracing **52.2** — on par with SOTA StreamForest (77.3 / 61.2 / 52.0), confirming the framework preserves the responder's QA ability.

**Ablations (OVO-Bench FAR F1 / StreamingBench PO), Table 5.** Full 31.0/38.0; w/o IGPP 24.0/28.8 (−7.0 F1); w/o LPMM 14.6/23.2 (−16.4 F1); w/o both 18.6/17.2; plain sliding-window baseline 21.0/26.8. Both modules are load-bearing; LPMM's matching signal matters most.

**Threshold robustness, Table 6.** Across train/test $\theta\in\{0.03,0.035,0.04\}$, OVO-Bench F1 stays ~26.6-31.0, best 30.99 at train/test $\theta=0.04/0.04$ — not overly sensitive to $\theta$.

**RL coefficient $\lambda$, Table 7.** OVO-Bench F1 / StreamingBench acc: $\lambda=0$ → 27.7/30.8, $\lambda=0.75$ → 28.6/36.4, **$\lambda=1.0$ → 31.0/38.0 (best)**, $\lambda=1.5$ → 19.3/33.6. Moderate false-positive control helps; too much hurts.

**Efficiency (Figure 3).** ~**13 fps** max throughput while topping all baselines on OVO-Bench F1; unlike VideoLLM-Online/MMDuet-2, no throughput degradation as context grows.

No Zotero highlights present.

Takeaways: (1) You can beat much larger proactive models with a 2B streaming loop by moving semantics *out* of the loop. (2) Triggering on the *derivative* of text-video similarity (with a redundant proposal set) is a simple, robust surrogate for per-frame reasoning. (3) RL on *downstream triggering* (not just proposal format) is what lifts F1 from ~24 to ~31 on OVO-Bench. (4) The paper also argues OVO-Bench's original offline-accuracy metric can be gamed by degenerate policies, motivating its online recall/precision protocol.

## How it connects (evolution)
- [[mmduet2]] — the closest per-frame-triggering proactive baseline Em-Garde beats on OVO/StreamingBench; contrast: single dense-head decision vs. propose-match.
- [[videollm-online]] — the original per-frame streaming-dialogue formulation Em-Garde's problem setup inherits and outperforms.
- [[dispider]] — decoupled perception/decision/reaction streaming design; a different route to the same efficiency-accuracy tension.
- [[ovo-bench]] — the FAR benchmark Em-Garde re-scores with a new online recall/precision metric (and critiques its offline proxy).
- [[streamingbench]] — the PO-task benchmark used for the headline accuracy comparison.
- [[proactivevideoqa]] — the PAUC/duplicate-response benchmark where Em-Garde is competitive with metric-tuned MMDuet-2.

## Open questions / limitations
- **Responder quality bounds accuracy:** LPMM only decides *when*; the *what* still relies on an off-the-shelf MLLM whose answer correctness (Respond & Check) is outside the trigger loop's control.
- **Proposal coverage risk:** if IGPP fails to enumerate the right visual cues at query time, no amount of streaming matching recovers the missed event — the once-at-$t_0$ parse is a single point of failure.
- **Small, task-shaped training data:** Parse2Prop-1K is 668 queries / 92 videos on COIN/Ego4D/BEHAVIOR; generalization to open-world queries and non-instructional streams is untested here.
- **Threshold/embedding transfer:** triggering on similarity *change* with a fixed $\theta$ and a frozen general embedder (Ops-MM-V1) may need re-tuning for domains (e.g. safety-critical low-latency settings) far from the eval distribution.

*Verification: equations (Eq. 1-9), the reward $r=(1-\lambda r_{fp})n_c/n$ with $r_{fp}=1-2^{-n_{fp}/n}$, the trigger rule $\max_i(s_i^t-s_i^{t-1})>\theta$, and all numbers in Tables 1-7 checked against the arXiv PDF (2603.19054) full text extracted via PyMuPDF; author/affiliation and config (Qwen2.5VL-7B IGPP, Ops-MM-V1-2B LPMM, θ=0.04, λ=1, GRPO, Parse2Prop-1K) read from pages 1/4/5/6.*

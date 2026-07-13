---
tags: [streaming-video-understanding, hub, synthesis]
---

# Streaming Video Understanding — Cross-Sub-Topic Overview

The connecting map across the three sub-topics of this workspace:
[[proactive-response]] (**when to speak** — trigger mechanisms), [[streaming-benchmarks]]
(**how timing and streaming ability are measured**), and [[streaming-memory]] (**the
substrate** that keeps an unbounded stream answerable in bounded compute). Each sub-topic
has its own four synthesis artifacts; this note is the layer above them — it argues that
the field is organized by **two capability axes joined by a measurement referee**, traces
how the three pushed each other from 2024→2026, maps which trigger family actually pairs
with which memory family in shipped systems, and names the open problems.

Per-sub-topic deep dives:
[[evolution-of-proactive-response]] · [[proactive-response-design-axes]] · [[proactive-response-benchmark-table]] · [[proactive-response-concept-graph]] ·
[[evolution-of-streaming-benchmarks]] · [[streaming-benchmarks-design-axes]] · [[streaming-benchmarks-benchmark-table]] · [[streaming-benchmarks-concept-graph]] ·
[[evolution-of-streaming-memory]] · [[streaming-memory-design-axes]] · [[streaming-memory-benchmark-table]] · [[streaming-memory-concept-graph]]

---

## 1. The two-axis framing (and where measurement sits)

A streaming video model, unlike an offline one, is defined by two questions it must answer
*continuously* while frames keep arriving:

- **Axis A — When to act (the [[proactive-response]] axis).** Each frame/second, emit a
  response or stay silent. This is a *timing* decision under partial observation: the model
  sees only `Video[0:t]`, never the future, and a wrong trigger time is a wrong answer even
  if the words are right. The seed is [[videollm-online]]'s streaming-EOS loss (CVPR 2024),
  which first turned "when to respond" into a learned per-frame target.
- **Axis B — How to sustain (the [[streaming-memory]] axis).** Keep hours of stream inside a
  fixed token / KV / weight budget so the model can *still* answer about the past without
  cost growing with stream length. The seed is fixed-size memory that decouples VRAM from
  length — [[streamingdvc]]'s K-means cluster memory (CVPR 2024) and [[flash-vstream]]'s
  bounded STAR memory (~1 s flat latency, ~16 GB regardless of length).

**Measurement is not a third axis — it is the referee that sits between them and defines the
objective for both.** The [[streaming-benchmarks]] sub-topic does three things at once:
it scores *when-to-act* with **timing-aware metrics** (PAUC in [[proactivevideoqa]]; the
`2^{-(m'-m)p}` time-decay of [[ovo-bench]] forward-active; SW-F1 in [[streaming-video-wild]];
ESTP-F1 in [[videollm-eyewo]]; ARS in [[streamready]]; IA-QTF1 in [[omniinteract]]); it scores
*how-to-sustain* with **memory-probing metrics** (the forgetting curve of [[river-bench]],
backward-tracing in [[ovo-bench]], the byte-budget deployment score of [[streamingeval]]); and
its metrics **become the training signal** — PAUC is literally the GRPO reward in
[[mmduet2]], RealStreamEval drives the self-evolution in [[evostreaming]], SW-F1's asymmetric
weights shape the trigger loss in [[streaming-video-wild]]. So the referee closes the loop:
a benchmark exposes a gap → the gap's metric is turned into a loss → the next model closes it.

**The rosetta stone between the axes and the benchmarks** is the temporal-regime taxonomy the
benchmarks converged on independently: *where the evidence lives in time relative to the query*
— **backward / real-time / forward**. [[ovo-bench]] named it, [[river-bench]] restates it as
retrospective/live/proactive, [[ovbench-videochat-online]] as Past/Current/Future. The mapping
onto model modules is exact: **backward ⇔ memory, real-time ⇔ perception/compression,
forward ⇔ trigger.** A model's OVO-Bench column profile is therefore a direct readout of which
axis its architecture invests in — [[streaming-model-remember]] lifts Backward Tracing
54.00→62.20 with a latent evidence graph while real-time perception stays flat, and [[stride]]
lifts Forward Active 46.30→59.70, the proactive dimension. Reading benchmark tables
column-by-column, not row-by-row, is how to see which axis a paper actually moved.

![[videollm-online.png]]
*The origin of Axis A: [[videollm-online]]'s LIVE framework interleaves a standard LM loss on
reply tokens with a streaming-EOS loss on silent frames — "when to speak" as a learned
per-frame decision. Every trigger mechanism below descends from or reacts against this.*

---

## 2. Co-evolution timeline, 2024→2026

How trigger mechanisms, eval protocols, and memory designs pushed each other. The recurring
pattern: **a benchmark exposes a failure → a trigger or memory design answers it → a stricter
benchmark exposes the answer's own failure.**

### 2024 — seeds on both axes, first benchmarks

- **Trigger:** [[videollm-online]] streaming-EOS loss (the paradigm origin);
  [[stream-vlm-qevd]] makes timing an in-band decodable `<next>`/`<feedback>` action and pairs
  it with a two-axis (content + Temporal-F-Score) eval; [[sdqes]] (NeurIPS 2024 D&B)
  formalizes proactive *onset* detection with latency-aware SR@k / SMD@k.
- **Memory:** [[streamingdvc]] fixed K-means cluster memory + streaming decoding points;
  [[videostreaming]] and [[videollamb]] (NeurIPS 2024) trained recurrent fixed-memory
  propagation; [[flash-vstream]] bounded four-granularity STAR memory at flat ~1 s latency.
- **Measurement:** [[streamingbench]] — first comprehensive timestamped streaming benchmark
  (900 videos / 4,500 MCQ), reporting a ~25-pt human gap (Human 91.66% vs Gemini 1.5 Pro
  67.07%). Its most pointed finding: the *dedicated streaming models scored far below offline
  generalists* — [[videollm-online]] 32.48 and [[flash-vstream]] 24.04 overall vs LLaVA-OneVision
  56.36 — i.e. as of late 2024, streaming-specific architecture wasn't yet buying anything.
  [[svbench]] (ICLR 2025 spotlight) frames streaming as multi-turn dialogue chains.

### Early–mid 2025 — trigger heads proliferate; benchmarks sharpen timing; memory branches into KV

- **Trigger heads bloom:** [[mmduet]] (video-text *duet*, two per-frame informative+relevance
  heads), [[dispider]] (disentangle perception / decision / reaction), [[streammind]]
  (Cognition Gate over a ~56M-param SSM perception token, ~100 FPS), [[lion-fs]] (CVPR 2025
  fast/slow decoupling), [[livecc]] (dense ASR-frame interleaving).
- **The pivot benchmark:** [[ovo-bench]] (CVPR 2025) conditions on `Video[0:t]` across three
  regimes — backward tracing, real-time perception, and **forward active responding** — and
  its forward-active regime *isolates response timing*, exposing a ~30-pt gap (best Gemini 1.5
  Pro 63.00% vs Human 92.81%). This is the benchmark that made the timing deficit undeniable
  and set up the 2026 trigger wave.
- **Over-emission is caught:** [[proactivevideoqa]]'s PAUC ("be right, *and* early") reveals
  that proactive models win high scores by **spamming** — [[mmduet]]'s duplicate-turn
  proportion is 81.3 / 99.4 / 92.8 / 99.2% across its four domains. Timing metrics now have to
  *penalize verbosity*, not just reward early answers.
- **The cross-axis bridge:** [[timechat-online]] shows the memory-compression signal and the
  trigger can be the *same* signal — its Differential-Token-Drop ratio drops ~80% of redundant
  tokens, and the *valleys* of that same drop-ratio timeline (scene changes) become the
  proactive Trigger Times. Axis A and Axis B fuse for the first time.
- **Memory branches into KV:** [[rekv]] (store-all KV, offload the tail, retrieve top-k per
  query — training-free, grafts onto any decoder Video-LLM); [[ovbench-videochat-online]]
  (CVPR 2025) ships a KV-synced Pyramid Memory Bank + OVBench.
- **Omni + adaptation:** [[omnimmi]] adds audio + proactive turn-taking (nearly all reactive
  models score ✗); [[vispeak]] defines visual-instruction-feedback with a binary informative
  head; [[streambridge]] retrofits offline models with a decoupled activation model.

### Late 2025 — verification gates, event memory, the KV-compression wave

- **Verification gates answer over-emission.** [[livestar]] replaces the EOS/silence token with
  a *training-free perplexity-verification* gate (SVeD: speak iff PPL spikes), directly
  targeting the duplicate-spam failure PAUC exposed; [[livestarpro]] adds a tree-structured
  hierarchical memory (TSHM) and, tellingly, **reuses the PPL signal as the memory's
  keyframe-salience score** — the trigger signal drives memory eviction. [[lyrav-dont-pause]]
  wraps this in a three-state synchrony FSM.
- **Event-structured memory matures.** [[streamforest]] (event-tree PEMF forest merged under
  similarity / merge-count / recency penalties) and [[eventmemagent]] (histogram-boundary event
  segmentation + reservoir sampling) reorganize memory around *events*, not frames — which is
  precisely what later lets triggers fire on *event boundaries*.
- **The KV-compression wave.** [[infinipot-v]] (TaR+VaN, fixed budget), [[streammem]],
  [[streamkv]], [[hermes-kv]] (layer-stratified cache), [[livevlm]] (sink bucketing),
  [[streamingtom]] (quantized), [[streamingvlm]] (attention-sink + sliding window ported from
  StreamingLLM, *training-aligned*), [[decouple-and-cache]], [[cacheflow]] — all training-free,
  all holding peak memory constant over unbounded streams.
- **Stricter timing eval.** [[omni-duplexeval]] grades content *and* timing jointly and splits
  the failure cleanly: streaming-commentary models **over-fire** ([[livecc]]-Base 91.1% wrong,
  [[streamingvlm]] 96.7% wrong on proactive) while the strongest models **over-stay-silent**
  ([[mmduet2]] 75.8% no-answer, MiniCPM-o 4.5 49.2% no-answer). Best proactive score 20.0 vs
  Human-Duplex 92.8. *Neither* regime is solved.

![[ovo-bench.png]]
*The measurement pivot: [[ovo-bench]]'s three timestamp-conditioned regimes — backward tracing
(past), real-time perception (present), forward active responding (future). The forward-active
regime scores response **timing** with a `2^{-(m'-m)p}` decay and exposed the ~30-pt gap that
launched the 2026 trigger wave.*

### 2026 — the trigger wave, reasoning-as-memory, deployment-grade eval

- **RL-trained timing (the trigger wave).** [[mmduet2]] (ICLR 2026) drops the hand-tuned
  response head entirely — `NO REPLY` is a generated token — and trains *when-to-speak* with
  multi-turn GRPO under the PAUC reward, cutting [[mmduet]]'s duplicate rate from ~81–99% to
  1–8% while raising PAUC (WEB 38.9→53.3). The fine print matters: on the EGO split
  [[mmduet]]'s 99.4%-duplicate spam still *out-scores* MMDuet2 on raw PAUC (46.0 vs 33.6) —
  raw PAUC is gameable, and the real win is the joint PAUC-and-low-duplicate frontier; the
  reward's anti-spam terms ($r_{\text{rep}}$) are load-bearing guardrails, not decoration.
  [[streampro]] reframes it as "proactive agency," lifting proactive F1 ~4× over the prior
  open-source best (SPB 10.4→41.5) via class-balanced SFT + multi-grained GRPO.
  [[thinkstream]] and [[r3-streaming]] fold timing into RLVR / budget-controlled routing.
- **Training-free retrieval-gated triggers, enabled by event/graph memory.**
  [[response-g1]] gates the trigger by top-k retrieval over an online *scene-graph* memory
  (frozen Qwen3-VL, +15.1 on StreamingBench-PO); [[em-garde]] moves semantic parsing out of the
  loop and fires on a *similarity-spike* over cached proposals; [[fluxmem]] reuses its memory's
  TAS backward scores as a zero-overhead trigger — the same memory-signal-as-trigger idea
  [[timechat-online]] started, now over hierarchical memory.
- **Reasoning-as-memory.** [[thinkstream]], [[vst]], [[tays]], [[tww]] make the model's own
  chain-of-thought / streaming thoughts *be* the compressed long-term memory (a textual or
  KV anchor), amortizing reasoning into pre-query watching time.
- **New memory paradigms.** Parametric / fast-weight memory ([[video-salmonn-s]] test-time-
  trained MLP; [[ovg-hq-unify]] parametric memory block); agentic frame-recoverable memory
  ([[visual-agentic-memory]], [[streammeco]]); budgeted latent-evidence graphs
  ([[streaming-model-remember]]).
- **Deployment-grade and specialized eval.** [[streamingeval]] folds throughput / latency /
  byte-budget into one StreamingScore and shows accuracy-only tables hide throughput collapse;
  [[streaming-video-wild]]'s SW-F1 + prefix-cache harness targets <1 s TTFT over 2-hour video;
  [[omnipro]] / [[omniinteract]] stress omni (audio) triggers; [[egopro-bench]] adds
  personalization; [[streameqa]] / [[streamgaze]] add embodied / gaze grounding;
  [[evostreaming]]'s RealStreamEval enforces strict causality + a verbosity penalty and
  critiques the polling protocols of earlier suites.
- **The dissolution challenge — architecture, or just data + a harness?** Two 2026 results
  attack the premise of the whole streaming-architecture program. [[evostreaming]]: a *frozen
  offline backbone* + LoRA on ~1,000 self-generated timing trajectories (139× less data than
  [[timechat-online]]'s ~139K) beats the architecture specialists on RealStreamEval — 54.6 vs
  [[dispider]]-7B 33.3, [[flash-vstream]] 24.0. [[streamingeval]] lands the same punch from the
  eval side: under an equal byte budget, *offline* Qwen3-VL-8B tops both OVO (58.00) and
  StreamingBench (77.31) with a better StreamingScore than any native-online model. The variable
  that resolves the apparent contradiction with [[rtv-bench]] (where online-designed VITA-1.5
  *does* beat offline VideoLLaMA2, 44.51 vs 39.55) is **who controls timing**: when the eval
  polls or prefixes the clip, offline models win; when the model must self-trigger on a raw
  stream ([[omniinteract]], [[omni-duplexeval]]), everything collapses. The architecture
  question is really the trigger question in disguise.

---

## 3. The design-space map: which trigger family pairs with which memory family

Cross-tabulating the two axes over *actual shipped systems*. The analytical payoff is in the
**Coupling** column: the field's most influential move is not a better trigger *or* a better
memory in isolation — it is **fusing the two signals**, so the memory-redundancy statistic
*is* the trigger, or the timing/salience score *drives* memory eviction.

| System | Trigger family (Axis A) | Memory family (Axis B) | Coupling of the two axes |
|---|---|---|---|
| [[videollm-online]] | streaming-EOS token | bounded LLM context (EOS never stored) | *decoupled* — the seed; trigger and memory independent |
| [[mmduet]] | per-frame informative+relevance heads | plain streaming context | decoupled; over-emits (dup 81–99%) |
| [[streamo]] | 3-state token (Silence/Standby/Response) | plain streaming window | decoupled, one-pass |
| [[mmduet2]] / [[streampro]] | RL-trained decision token (GRPO) | plain streaming window | decoupled; timing learned from the eval metric |
| [[streammind]] | Cognition Gate over SSM perception token | constant per-frame recurrent SSM state | trigger reads the *same* compact perception token the memory is |
| **[[timechat-online]]** | drop-ratio *valleys* = triggers | DTD-compressed FIFO KV bank | **fused** — the redundancy signal that compresses memory *is* the trigger |
| **[[fluxmem]]** | TAS backward-score threshold | 3-tier short/mid/long token cascade | **fused** — memory's per-frame Otsu statistics reused as the trigger |
| **[[livestarpro]]** | SVeD perplexity-verification gate | tree-structured hierarchical memory (TSHM) | **fused** — PPL doubles as the memory's keyframe-salience score |
| [[streamready]] | `<RDY>` readiness gate | 3-tier visual memory tree | gate lives *inside* the long-term Q-Former, shares the answer rep |
| [[streamov]] | cross-attn trigger head on prefill states | evidence-guided long-short memory | trigger reads memory-selected evidence |
| [[response-g1]] | scene-graph top-k retrieval gate | online scene-graph memory bank | trigger *is* a retrieval over the memory |
| [[em-garde]] | similarity-spike over proposals | query-side proposal set + newest-frame cache | timing decoupled from a light per-frame memory |
| [[r3-streaming]] | Respond readiness gate | age-aware two-zone memory (DTD, from [[timechat-online]]) | cascade: Remember→Respond→Reason, memory feeds gate |
| [[streamagent]] | anticipation-horizon LLM-judge | hierarchical layer-adaptive KV + rolling captions | decision on cheap text memory, answer on visual KV |
| [[thinkstream]] / [[vst]] / [[tww]] | response/silent in generation | reasoning-/thought-/note-as-memory | the memory (CoT) and the decision share one decode stream |
| [[streamforest]] / [[eventmemagent]] | (reactive; event boundaries available) | event-tree / event-segmented memory | event structure *enables* later event-gated triggering |

**Reading the map.** Systems cluster into (i) *decoupled* early designs — a trigger head bolted
onto plain context ([[videollm-online]]→[[mmduet]]→[[mmduet2]]); (ii) *fused* designs where one
signal serves both axes — the strongest recent line ([[timechat-online]], [[fluxmem]],
[[livestarpro]], [[response-g1]]); and (iii) *reasoning-native* designs where a single decode
stream carries both the memory and the speak/silent decision ([[thinkstream]], [[vst]],
[[tays]], [[tww]]). Event-structured memory ([[streamforest]], [[eventmemagent]])
is the substrate that makes fusion (iii→event-gated triggers) natural.

**The systematic hole in the map: KV-retrieval memory is reactive-only.** The entire
store-all-retrieve branch ([[rekv]], [[streamkv]], [[cacheflow]], [[livevlm]], [[hermes-kv]],
[[streamingtom]], [[v-rex]]) has *no proactive trigger anywhere* — structurally so: retrieval
needs a query to retrieve against, and until the user speaks there is none. The only systems
that bridge it do so by making the query *standing*: [[response-g1]] parses the query once into
a scene-graph retrieval target and gates the trigger on retrieval hits (OVO forward-active 58.2
vs prior open-source best [[streamagent]] 45.4), and [[em-garde]] parses once into cached
proposals then does cheap per-frame similarity matching. Both win precisely because parsing the
query once moves the heavy semantics *out of the per-frame loop*. Retrieval-memory systems with
standing queries are the most productive pairing nobody in the memory line has built natively.

**The dominant architecture motif: two-speed compute.** A cheap always-on process plus an
expensive on-demand one recurs independently across camps — [[dispider]] (light
perception/decision, heavy reaction), [[lion-fs]] (fast/slow paths), [[streammind]] (~56M SSM
gate before the LLM), [[em-garde]] (small matcher / large parser), [[streamagent]] (small
planner / large responder), [[r3-streaming]] (fast/slow VLM routing), [[streambridge]]
(separate small activation model), [[flash-vstream]] (writer/reader processes), [[wat]]
(watch/think). The counter-camp is single-model in-band decoding ([[streamo]], [[streampro]],
[[thinkstream]], [[streaming-video-wild]]): with the right loss reweighting and RL, one
autoregressive pass does it all. As of the frontier the split is clean — the RL'd single-model
camp holds the proactive-accuracy records ([[streampro]] SPB F1 41.5), the two-speed camp the
efficiency records ([[streammind]] ~100 FPS perception loop; [[em-garde]] ~13 FPS at 2B).

**The in-band formulation manufactures its own class-imbalance problem — rediscovered five
times.** Silence tokens swamp response tokens (310:1 on Ego4D, 71:1 on SoccerNet, per
[[streammind]]), so every in-band-token paper independently reinvents a reweighting fix:
weighted CE with tuned silence weight ([[streammind]]), focal loss ([[streamo]]: CE→focal moves
OVO forward-active REC 6.45→27.94), class-balanced effective-number weights ([[streampro]]:
CE 6.6 < focal 14.2 < CB 16.3 on SPB), negative-frame subsampling ([[proassist]] ρ=0.1:
narration F1 30.1→58.7), role-weighted loss ([[streaming-video-wild]]). Five independent
solutions to one problem — an imbalance the decoupled-head and training-free-statistic camps
never face, which is itself an argument in the family choice.

![[streamforest.png]]
*The memory substrate that enables event-gated triggering: [[streamforest]]'s bounded Persistent
Event Memory Forest — meta-events become tree nodes merged under similarity / merge-count /
temporal-recency penalties, kept beside a sharp real-time window. Event-level memory is what lets
later systems fire triggers on *event boundaries* rather than raw frames.*

---

## 4. Open problems as of July 2026

Each grounded in a specific paper's verified result.

1. **When-to-speak is still nowhere near human, in either direction.** [[omni-duplexeval]]:
   best model 20.0 proactive vs Human-Duplex 92.8, with the failure cleanly split into
   *over-fire* ([[streamingvlm]] 96.7% wrong) vs *over-silent* ([[mmduet2]] 75.8% no-answer).
   [[ovo-bench]] forward-active and [[streampro]]-Bench (prior open-source best F1 10.4) show
   the same. Verification gates ([[livestar]]) and RL ([[mmduet2]], [[streampro]]) narrow it but
   do not close it. One calibration note on the ceiling itself: *even humans* lose ~10 points
   under the live constraint — [[omni-duplexeval]]'s Human-Duplex is 81.8 overall vs
   Human-Offline 91.5 (the 92.8 above is the proactive-reminder subtask). The duplex-human
   number, not the offline one, is the honest target for proactive systems — and almost no
   model paper cites it.

2. **Omni-modal (audio-driven) triggering barely functions.** [[omnipro]]: 84% of samples need
   audio, yet best *online* mean F1 is 20.9% ([[mmduet2]] 3B only 11.3%), with non-speech
   sound triggers the bottleneck (15.3–22.3). [[omniinteract]]: nested-query resumption fails
   almost totally (Gemini/Qwen miss the outer query in 119/120 and 116/120 pairs). The
   vision-blindness runs deeper than benchmark scores: for conversational turn-taking onset,
   [[egospeak]] finds audio-only *matches or beats* audio+vision on Ego4D (mAP 69.2 vs 69.0 —
   vision adds nothing there, though it helps on EasyCom), yet the trigger literature is
   overwhelmingly vision-only. The field may be optimizing the less informative modality for
   conversational timing.

3. **Long-horizon memory forgets, and current benchmarks confirm it.** [[livestarpro]] reports
   73.4% of its long queries span beyond the active window, and the >30 min bucket collapses
   (LiveStar 21.1). [[river-bench]] makes the forgetting curve first-class (GPT-4o MC
   63.26 short → 52.21 very-long). [[visual-agentic-memory]]'s month-scale MM-Lifelong: best
   VAM 17.11% vs human 82.5% *(human ceiling row)*.

4. **Accuracy tables hide deployability collapse.** [[streamingeval]]: [[flash-vstream]]'s
   accuracy *rises* (OVO 33.15→50.31) while its StreamingScore *collapses* (2.34→0.74) as
   MaxFPS drops 8→1; [[ovbench-videochat-online]]'s VideoChat-Online sustains only 0.14 FPS —
   a deployment-blocking failure invisible in accuracy-only leaderboards. [[streaming-video-wild]]
   answers with prefix-caching (<1 s TTFT over 2-hour video) but only for its own harness.

5. **Reasoning-while-streaming trades latency for correctness, unevenly.** [[tays]] gets a
   near-zero TTFT but ~12 s end-to-end delay; [[thinkstream]] >5× decode speedup but a ~20
   tokens/video-second budget; and [[egopro-bench]] finds thinking *hurts* action-timing
   (RL-with-think F1 79.66 vs RL-without-think 85.03), forcing a hard "short-thinking" length
   budget. The reconciliation with [[thinkstream]]'s +8.66 OVO overall (Backward ↑10.4) and
   [[vst]]'s gains is the temporal-regime split again: reasoning helps *backward* (memory-side)
   tasks and hurts *forward* (latency-critical) firing. When to reason at all remains unsolved.

6. **Personalization and embodied grounding are near-floor.** [[egopro-bench]] intent branch:
   Response-Quality 3.23/5. [[streameqa]] embodiment penalty (Qwen3-VL Real-time 74%→54.26%
   moving from general to embodied streaming). [[streamgaze]]: gaze-conditioned proactive OAA
   for GPT-4o just 0.149 vs human 0.780 *(human ceiling row)*.

7. **The metrics themselves are not standardized — a meta-problem.** PAUC ([[proactivevideoqa]]),
   SW-F1 ([[streaming-video-wild]]), ARS ([[streamready]]), ESTP-F1 ([[videollm-eyewo]]),
   IA-QTF1 ([[omniinteract]]), RealStreamEval ([[evostreaming]]), and the byte-budget
   StreamingScore ([[streamingeval]]) all measure "when + what" differently, and
   [[evostreaming]] argues the older *polling* protocols ([[streamingbench]]-style ±window
   sampling) are not causally valid. Cross-paper timing numbers are therefore only loosely
   comparable. Even *accuracy* numbers on one benchmark aren't: "StreamingBench" scores span
   ~24 to ~83 across papers because some report the 18-task Overall ([[dispider]] 53.12), others
   the Real-Time subset only ([[timechat-online]] 75.28, [[vst]] 79.5,
   [[streaming-model-remember]] 82.67). A cross-paper table that doesn't fix the subset is
   meaningless — a metrology paper is missing.

8. **No consensus memory-eviction principle, and no referee for one.** The field runs at least
   five incompatible keep-rules — keep-*dissimilar* (DTD in [[timechat-online]], [[cacheflow]]),
   keep-*attended* ([[streammem]] proxy-query attention; [[hermes-kv]] layer-stratified),
   keep-*surprising* ([[videoscaffold]], [[streaming-model-remember]]), keep-*semantically-salient*
   ([[savemem]] pseudo-question priors), and *order-matters-more-than-selection*
   ([[weavetime]]) — and each paper's ablation beats the others' method as its own baseline.
   The closest things to third-party bake-offs are [[streamingeval]] and [[r3-streaming]]'s
   compression comparison (age-aware DTD 75.90 vs DivPrune 68.48 at 95% drop, showing the gain
   is in the *policy*, not the operator).

9. **The compression numbers quietly indict the benchmarks.** 80–99% of visual tokens can be
   dropped at negligible cost: [[timechat-online]] 82.6% drop → 73.64 vs 75.36 full-token;
   [[r3-streaming]] 95% drop → 75.90; [[videoscan]] 1 token/frame (>99% cut) keeps ~85% of base
   accuracy. Either streaming video is spectacularly redundant, or current benchmarks probe only
   sparse high-salience content — the deployment framings of [[streamingeval]] and
   [[omniinteract]] suggest the latter. No benchmark yet *punishes* aggressive dropping (dense
   fine-grained counting over long spans would).

### How to read the numbers (caveats that apply across sections)

- **Reward = metric circularity.** [[mmduet2]] trains GRPO *on* PAUC and reports PAUC;
  [[streampro]] trains on a turn-F1 reward mirroring its own bench's scoring. Gains on the
  shared metric should be discounted relative to third-party-metric gains.
- **Judge circularity.** RealStreamEval's semantic judge is Qwen3-VL-235B scoring mostly
  Qwen backbones ([[evostreaming]] runs a five-judge robustness check, but the constants are
  hand-tuned).
- **Own-benchmark headline wins.** [[livestar]]/[[livestarpro]] headline on self-built
  OmniStar; [[streamo]] on Streamo-Bench; [[streaming-video-wild]] on its own harness. All
  plausibly real, none independently replicated yet.
- **Extraction fragility.** [[weavetime]]'s numbers come from the arXiv HTML only — its PDF
  endpoint mis-served a different paper, so they were never PDF-verified. Treat as unconfirmed
  until re-checked.

---

## 5. Suggested reading path (10 papers, in order)

Ordered so understanding builds: seed → measurement → substrate → trigger → the fusion → the
metric → memory maturity → RL endpoint. One line each on *why it's here*.

1. **[[videollm-online]]** — the origin of Axis A: streaming-EOS loss makes "when to speak" a
   learned per-frame decision; every later trigger descends from or rejects it.
2. **[[streamingbench]]** — the first comprehensive timestamped benchmark; defines the streaming
   eval axis and the ~25-pt human gap (Human 91.66 vs Gemini 1.5 Pro 67.07).
3. **[[ovo-bench]]** — the pivot: its forward-active-responding regime isolates *timing* and
   exposes the ~30-pt gap (63.00 vs 92.81) that drove the 2026 trigger wave.
4. **[[flash-vstream]]** — the canonical Axis B substrate: bounded STAR memory, flat ~1 s
   latency, VRAM decoupled from stream length.
5. **[[rekv]]** — the training-free KV-retrieval memory branch: store-all + offload + retrieve,
   graftable onto any decoder Video-LLM; the ancestor of the KV-compression wave.
6. **[[mmduet]]** — the per-frame trigger-head paradigm — and, via its ~81–99% duplicate spam,
   the over-emission failure that motivated everything after it.
7. **[[timechat-online]]** — the cross-axis bridge: the memory-compression redundancy signal
   (drop-ratio valleys) *is* the proactive trigger. Read it right after a pure-memory and a
   pure-trigger paper to see them fuse.
8. **[[proactivevideoqa]]** — PAUC: the timing-aware metric ("early AND accurate") that made
   over-emission measurable and later became the RL reward.
9. **[[streamforest]]** — event-structured (tree-forest) memory: the substrate that makes
   event-gated triggering and long-horizon retrieval tractable.
10. **[[mmduet2]]** — the 2026 endpoint: GRPO on the PAUC reward removes the response-head
    threshold *and* the duplicate spam — the metric-becomes-loss loop closed. (Read [[streampro]]
    next for the ~4× proactive-F1 scaling of the same idea.)

---

*This hub is re-runnable: when a sub-topic's `papers/` changes, re-synthesize that sub-topic's
four artifacts first, then revisit sections 2–4 here. Numbers are drawn from the verified
per-paper notes; human-ceiling and baseline rows are marked inline.*

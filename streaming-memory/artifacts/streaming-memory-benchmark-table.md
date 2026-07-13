---
tags: [streaming-video-understanding, streaming-memory, synthesis]
---

# Streaming-memory: verified benchmark table

A collation of the **headline numbers each streaming-memory paper reports for itself**,
organized per benchmark. Every value is lifted verbatim from that paper's own note
(`papers/<slug>.md`), which in turn cites the paper's own table. Sibling artifacts:
[[evolution-of-streaming-memory]] · [[streaming-memory-design-axes]] · [[streaming-memory-concept-graph]].

> [!warning] These tables are NOT a head-to-head leaderboard
> Each row comes from a **different paper's own experiments**. Backbones differ
> (LLaVA-OV-0.5B/7B, LLaVA-Video-7B, Qwen2-VL / Qwen2.5-VL-3B/7B/32B, Qwen3-VL-8B,
> CLIP+Vicuna-7B, video-SALMONN-2+-8B, Gemini 3 Flash, Llama-3-8B …), and so do frame
> rate, token/KV budget, sampling, prompt, and the LLM **judge** used to score open-ended
> answers. Even the "same" benchmark is reported under different aggregate **splits**
> (e.g. StreamingBench *Real-Time* ≈ 70–83 vs *Overall/all-task* ≈ 41–63). Read gaps
> **within a single source paper** (method vs its own baselines); treat cross-row gaps as
> indicative only. `n/r` = not reported in that paper. `(base)` = a backbone / closed-model
> reference point, not a streaming-memory method. Idea-family letters (A–F) match the
> sibling artifacts (see key at the bottom).

![[streamingvlm-designspace.png]]
*The streaming-memory design space, from [[streamingvlm]] Fig.1: the whole sub-topic is a
race to approximate full-attention quality at bounded (sink+window / retrieval / compressed)
cost — which is exactly why the numbers below are not directly comparable across the
different budgets each paper picks.*

---

## 1. StreamingBench

Scores are accuracy (%). The **Split** column names the aggregate exactly as the source
note labels it — *RT* = Real-Time Visual Understanding, *Omni* = Omni-Source, *Ctx* =
Contextual, *Overall* = full all-task aggregate. Do not compare an RT number against an
Overall number.

| Method | Fam | Backbone / setting | Score | Split / metric (as reported) |
|---|---|---|---|---|
| [[streaming-model-remember]] SelectStream | F | Qwen3-VL-8B / Qwen2.5-VL-7B | **82.67 / 81.42** | StreamingBench (Table 1) |
| [[decouple-and-cache]] DSCache | A | Qwen2.5-VL-7B / LLaVA-OV-7B | **82.32 / 79.12** | RT visual understanding acc |
| [[hermes-kv]] HERMES | D | Qwen3-VL-8B / Qwen2.5-VL-7B-4K / LLaVA-OV-7B-4K | **81.32 / 79.44 / 73.23** | ⚠ paper's own "Avg." = mean of RT + Backward-Tracing, not the standard RT split |
| [[vst]] VST | B | Qwen2.5-VL-7B | **79.5** | RT, Overall |
| [[omnimem]] OmniMem | A | video-SALMONN-2+-8B | 78.5 / 60.9 / 40.7 | RT / Omni / Ctx |
| [[video-salmonn-s]] video-SALMONN S | D· | Qwen3-VL-8B | 78.9 / 57.5 / 41.5 (avg 67.1) | RT / Omni / Ctx (avg) |
| [[streamforest]] StreamForest | D | SigLIP+Qwen2-7B | 77.3 | RT, All |
| [[eventmemagent]] EventMemAgent | D | Qwen3-VL-8B, ≤32 frames | 77.00 | RT, All |
| [[fluxmem]] FluxMem | D | Qwen2.5-VL-7B | 76.4 (+SFT 76.7) | RT subtasks |
| [[infinipot-v]] InfiniPot-V | A | Qwen2.5-VL-7B, 4K | 76.4 | StreamingBench (Table 4) |
| [[savemem]] SAVEMem | E | Qwen2.5-VL-7B / 3B | 76.0 / 70.7 | Real-Time |
| [[streamingassistant]] StreamingAssistant | A | TimeChat-Online-7B, 91.5% dropped | 70.24 | Real-Time |
| [[livevlm]] LiveVLM | C | LLaVA-OV-7B | 63.10 | Overall (all-task) |
| [[tww]] TWW | B | Qwen3-VL-8B / 4B | 62.04 / 60.04 | Overall, single-turn S3 |
| [[streamkv]] StreamKV | C | LLaVA-OV-7B, ↓60% KV | 58.9 | Overall |
| [[weavetime]] WeaveTime | D | LLaVA-OV-7B (over ReKV 53.56) | 57.57 | Real-Time |
| [[videoscaffold]] VideoScaffold | D | Vicuna-7B, 0.5 fps | 41.0 | RT, Overall |
| *GPT-4o* (base) | — | closed | 73.3 (RT) / 62.50 (Overall) | ref. cited by [[vst]] / [[livevlm]] |
| *Gemini 1.5 Pro* (base) | — | closed | 75.7 (RT) | ref. cited by [[vst]] |
| *Qwen2.5-VL-7B* (base) | — | backbone-only | 73.9 / 73.31 | ref. cited by [[fluxmem]] / [[hermes-kv]] |
| *LLaVA-OV-7B* (base) | — | backbone-only | 58.85 (Overall) | ref. cited by [[livevlm]] |

Reading the column honestly: the >80 leaders ([[streaming-model-remember]],
[[decouple-and-cache]], [[hermes-kv]]) all sit on **Qwen2.5/3-VL** backbones and report the
RT-style aggregate; the ~57–63 rows ([[weavetime]], [[streamkv]], [[livevlm]]) are on the
weaker **LLaVA-OV-7B** and/or report the harder **Overall** split. So the ~25-point spread
is mostly backbone + split, not method quality — the honest signal is each method's **delta
over its own base** (e.g. [[decouple-and-cache]] +2.2/+3.8 vs uniform cache, [[hermes-kv]]
+6.13 over Qwen2.5-VL-7B, [[fluxmem]] +2.5 over Qwen2.5-VL-7B, [[livevlm]] +4.25 over
LLaVA-OV-7B).

**The ReKV Rashomon — one baseline, four numbers.** ReKV on LLaVA-OV-7B "StreamingBench"
is cited at **53.5** by [[streamkv]] (Overall, 0.5 FPS), **53.56** by [[weavetime]] (labeled
Real-Time), **69.22** by [[hermes-kv]], and **71.06** by [[decouple-and-cache]] — a 17.6-pt
spread for the *same method on the same backbone on the same benchmark*, driven by silent
split/FPS/budget differences. Every cross-paper "we beat ReKV by +X" claim inherits this
ambiguity. Relatedly, [[decouple-and-cache]]'s "StreamingVLM 76.92" is the exact value of
its own uniform-cache baseline — cited baselines are often re-implementations (sink +
sliding window on the citing paper's backbone), not the published model.

**StreamingBench-RT is recency-dominated — it barely tests memory.** A plain **4-frame
recent window** scores 78.47 (Qwen2.5-VL-7B) / 80.59 (Qwen3-VL-8B) in
[[streaming-model-remember]]'s baselines — *above* the ~73.3–73.9 full-history base rows.
And [[decouple-and-cache]]'s ablation: removing the *cumulative* (history) cache costs
0.06 pts (79.12 → 79.06) while removing the *instant* (recent) cache costs 4.85 (→ 74.27);
its own limitations note concedes it is close to "just re-encode the recent window well."
A memory contribution validated primarily on this column has validated recency handling;
the history-dependent evidence lives in OVO Backward-Tracing (§2).

**Composability — the one stacking experiment.** [[decouple-and-cache]] + ReKV → 79.41,
+ InfiniPot-V → 79.60, above each alone: cache *construction*, KV *compression*, and
*retrieval* exploit different redundancies and are closer to orthogonal modules than
competitors — yet this is the only composition test in all 37 papers.

---

## 2. OVO-Bench

OVO-Bench splits into Real-Time perception (RT), Backward-Tracing (BT, history-dependent),
and Forward-active (FW). The **Overall** column is the paper's headline aggregate; the
sub-scores column reports RT / BT / FW where the note gives them. Metric = accuracy (%).

| Method | Fam | Backbone / setting | Overall | RT / BT / FW sub-scores (as reported) |
|---|---|---|---|---|
| [[visual-agentic-memory]] VAM | E | Gemini 3 Flash agent | 68.41 (RT+BT avg) | RT 78.94 / BT 57.89 / STU 82.19 |
| [[streaming-model-remember]] SelectStream | F | Qwen3-VL-8B / Qwen2.5-VL-7B | **67.03 / 65.71** | BT avg 62.20 / 61.05; RT 82.76 / 80.85 |
| [[hermes-kv]] HERMES | D | Qwen2.5-VL-32B-6K / -7B-4K / LLaVA-OV-7B-4K | 64.82 / 59.21 / 58.27 | ⚠ paper's own "Avg." = RT+BT mean |
| [[savemem]] SAVEMem | E | Qwen2.5-VL-7B / 3B | 62.69 / 57.34 | RT avg 74.93 / BT avg 50.44 |
| [[eventmemagent]] EventMemAgent | D | Qwen3-VL-8B, ≤32 frames | 60.75 | RT-VP 68.29 / BW 58.03 / FW 55.92 |
| [[fluxmem]] FluxMem | D | Qwen2.5-VL-7B | 53.3 (+SFT 61.4) | RT 67.2 — ⚠ [[savemem]] cites FluxMem at 57.22; the two configs don't reconcile |
| [[vst]] VST | B | Qwen2.5-VL-7B | 59.3 | RT 67.2 / BT 56.7 / FW 54.0 |
| [[decouple-and-cache]] DSCache | A | LLaVA-OV-7B / Qwen2.5-VL-7B | 57.5 / 55.4 | RT 71.5 / BT 47.8 / FW 53.1 (LLaVA-OV) |
| [[streamforest]] StreamForest | D | SigLIP+Qwen2-7B | 55.6 | (OVBench 60.5, see §5) |
| [[infinipot-v]] InfiniPot-V | A | Qwen2.5-VL-7B, 4K | 53.6 (avg) | RT 65.9 / BW 47.6 / FW 47.9 |
| [[streamingassistant]] StreamingAssistant | A | TimeChat-Online-7B, 93% dropped | 43.49 | — |
| [[streamingvlm]] StreamingVLM | B | Qwen2.5-VL-7B | n/r | RT only: 56.00 → **61.96** |
| [[weavetime]] WeaveTime | D | LLaVA-OV-7B | n/r | RT only: 66.15 → **72.13** |
| [[tww]] TWW | B | Qwen3-VL-8B / 4B | 55.02 (4B, S3) | multi-turn 51.80 |
| *GPT-4o* (base) | — | closed | 59.5 | ref. cited by [[vst]] / [[eventmemagent]] |
| *Gemini 1.5 Pro* (base) | — | closed | 63.00 | ref. cited by [[eventmemagent]] |
| *Gemini 3 Flash e2e* (base) | — | closed | 67.46 (RT+BT avg) | ref. cited by [[visual-agentic-memory]] |

The consistent story across sources: **Backward-Tracing is where memory earns its keep —
and it is the only widely-reported subset that actually discriminates memory designs**. RT
is near-saturated (75–83 across methods), while BT spans base 44.65 ([[savemem]]'s
Qwen2.5-VL-7B) → 47.6 ([[infinipot-v]]) → 50.44 ([[savemem]]) → 56.7 ([[vst]]) → 62.20
([[streaming-model-remember]]). The ordering aligns with mechanism: methods that *train* a
selection/consolidation policy (SelectStream's gated writes, VST's textual thoughts) beat
training-free eviction heuristics by ~6–12 BT points while staying within noise on RT. Even
the best BT (62.20) sits ~30 pts below OVO's human BT (92.33) — memory remains the open
problem. [[eventmemagent]]'s tool-use (OCR/detection) adds ~+3–4 to Overall.
[[visual-agentic-memory]] is the outlier setting — a Gemini-3-Flash *agent* over
recoverable raw frames, so its 68.41 is not comparable to the open 7-8B rows.

**Baseline drift bounds the readable delta.** Plain Qwen2.5-VL-7B OVO-Overall appears as
~49.8 (implied by [[fluxmem]]'s +3.5 → 53.3), 52.27 ([[savemem]]), and 52.28
([[hermes-kv]]) — a ~2.5-pt spread for the *same untouched backbone* (FPS / frame cap /
prompt undisclosed). Any cross-paper delta under ~3 pts on this benchmark is protocol
noise, not method signal.

---

## 3. RVS-Ego / RVS-Movie (RealTime VStream-QA streaming split)

The most nearly-comparable cluster: nearly all rows are the **same LLaVA-OV-7B backbone**,
same RVS streaming protocol, GPT-judge acc + 1–5 score — but token/KV **budgets** and CPU
offloading still differ, so it is still not a clean leaderboard. [[rekv]] (store-all with
offloading) is the accuracy **upper-bound reference** the compressive methods chase.

| Method | Fam | Backbone | RVS-Ego (acc / score) | RVS-Movie (acc / score) |
|---|---|---|---|---|
| [[rlivs]] rLiVS | C | Qwen2.5-VL-7B | **68.1** / n/r | 56.1 / n/r |
| [[rlivs]] rLiVS | C | LLaVA-OV-7B | 65.3 / 4.0 | 57.7 / 3.6 |
| [[rekv]] ReKV *(store-all, base upper-bound)* | C | LLaVA-OV-7B (internal) | 63.7 / 4.0 | 54.4 / 3.6 |
| [[cacheflow]] CacheFlow | C | LLaVA-OV-7B | 61.6 / 3.94 | 50.5 / 3.44 |
| [[videoscan]] VideoScan (M=128) | D | LLaVA-Video-7B | 60.9 / 4.0 | 54.1 / 3.5 |
| [[hermes-kv]] HERMES (6K) | D | LLaVA-OV-7B | 60.3 / 4.0 | 54.4 / 3.6 |
| [[decouple-and-cache]] DSCache | A | LLaVA-OV-7B | 59.5 / 3.97 | 49.4 / 3.50 |
| [[streamingtom]] StreamingTOM | A | LLaVA-OV-7B | 58.3 / 3.9 | 53.2 / 3.5 |
| [[infinipot-v]] InfiniPot-V (offload-free) | A | LLaVA-OV-7B | 57.9 / 3.5 | 51.4 / 3.5 |
| [[livevlm]] LiveVLM (no offload) | C | LLaVA-OV-7B | 57.8 / 3.9 | 53.4 / 3.6 |
| [[streammem]] StreamMem (6K) | A | LLaVA-OV-7B | 57.6 / 3.8 | 52.7 / 3.4 |
| [[flash-vstream]] Flash-VStream *(memory base)* | D | CLIP+Vicuna-7B | 57.3 / 4.0 | 53.1 / 3.3 |
| [[cacheflow]] CacheFlow (0.5B) | C | LLaVA-OV-0.5B | 54.3 / 3.87 | 42.6 / 3.34 |

Caveats that dominate this table: (i) [[rekv]]'s 63.7/54.4 needs unbounded stored KV
(~18.8 GB/h CPU offload); the compressive rows match or beat it at a **fixed** budget with
far less memory/latency — that trade, not the raw accuracy, is each paper's real claim.
(ii) reported values for the **same** baseline drift across papers (e.g. "Flash-VStream-7B"
RVS-Ego appears as 57.3, 56.3, 57.0, and 55.0 in different notes because each author re-ran
it under their own settings) — the single sharpest reason not to read this as a leaderboard.
(iii) [[flash-vstream]] uses a different (CLIP+Vicuna) backbone entirely. (iv) Read the
column vertically and the accuracy story is flat: everything from [[hermes-kv]] down
clusters within ~3 pts of the 2024 [[flash-vstream]] baseline — on this benchmark family,
two years of training-free KV innovation moved accuracy by roughly the GPT-judge's noise
floor, and the *real* progress is the resource axis: [[rekv]]'s 38 GB + 18.8 GB/h offload
down to [[cacheflow]]-0.5B's 2.9 GB at comparable accuracy. (v) The strongest small-model
result: [[rlivs]] at **0.5B** (57.6 Ego / 51.3 Movie) beats *7B* ReKV on Movie by +6.7 —
and it does so by answering from stored **captions** rather than visual tokens (its own
ablation: captions-only beats visual-token retrieval, which performs near random).

---

## 4. Offline long-video generalization (MLVU / VideoMME)

Streaming-memory methods report offline scores to show memory doesn't cost general
long-video ability. MLVU is M-Avg unless noted; VideoMME is w/o-subtitles Overall unless
noted. Backbone/budget varies heavily — again read deltas over each paper's own base.

| Method | Fam | Backbone / budget | MLVU | VideoMME (w/o subs) |
|---|---|---|---|---|
| [[video-salmonn-s]] video-SALMONN S | D· | Qwen3-VL-8B | n/r | **76.9** (Long 71.3) |
| [[tww]] TWW | B | Qwen3-VL-4B | n/r | 73.41 |
| [[streaming-model-remember]] SelectStream | F | Qwen2.5-VL-7B | 73.0 | n/r (abs); +2.7 over base |
| [[fluxmem]] FluxMem | D | Qwen2.5-VL-7B | 73.1 | 65.3 |
| [[streamforest]] StreamForest | D | SigLIP+Qwen2-7B | 70.0 | 61.4 |
| [[rekv]] ReKV | C | LLaVA-OV-7B | 68.5 (dev) | n/r |
| [[livevlm]] LiveVLM | C | LLaVA-OV-7B | 68.1 | 59.6 (Long 51.3) |
| [[streamingtom]] StreamingTOM | A | LLaVA-OV-7B | 67.9 | 59.9 |
| [[cacheflow]] CacheFlow | C | LLaVA-OV-7B | 66.9 | n/r |
| [[streammem]] StreamMem (6K) | A | LLaVA-OV-7B | 66.9 (Med) / 63.0 (Long) | 59.4 |
| [[infinipot-v]] InfiniPot-V (6K) | A | Qwen2-VL-7B | 65.8 | 62.8 |
| [[omnimem]] OmniMem | A | video-SALMONN-2+-8B | n/r | Long 69.6 |
| [[streamingassistant]] StreamingAssistant | A | TimeChat-Online-7B | n/r | 61.6 (Long 53.1) |
| [[hermes-kv]] HERMES (4K) | D | LLaVA-OV-7B | n/r | 58.85 |
| [[videoscan]] VideoScan (M=128) | D | LLaVA-Video-7B | 61.3 | 55.1 |
| [[videoscaffold]] VideoScaffold | D | Vicuna-7B, 60 fr | 49.5 | 43.3 |
| [[videollamb]] VideoLLaMB | D | Vicuna-7B | n/r | 41.41 (compr. 0.06) |
| *LLaVA-Video-7B* (base) | — | backbone-only | 70.8 | 63.3 | — cited by [[videoscan]] |
| *full-KV (50K)* (base) | — | Qwen2-VL-7B | 65.8 | 63.9 | — cited by [[infinipot-v]] |

The near-lossless story for the compressive family is clearest here: [[infinipot-v]] holds
65.8 MLVU / 62.8 VideoMME at 6K vs full-KV 65.8 / 63.9 (~1/8 the memory); [[streammem]]
at 24K actually *edges past* full-KV (66.3 vs 65.9 on Qwen2-VL MLVU — mild pruning acts as
regularization, not loss); [[videoscan]] keeps ~85% of its base at >99% token cut. And the
convergence is tight: the LLaVA-OV-7B training-free rows span just 66.9–68.5 MLVU (~1.6
pts), i.e. on offline long-video QA every KV/token-compression method is
accuracy-equivalent and the whole comparison is an efficiency-at-iso-accuracy race.

One depth note on the "query-agnostic is nearly free" claim these rows support: the oracle
gap is a **benchmark property**, not a method property. [[streammem]]'s chat-template proxy
costs only −1.2 MLVU vs a true-query oracle (66.9 vs 68.1), but [[rekv]]'s internal
retrieval loses −8.4 to its oracle on QAEgo4D (56.0 vs 64.4; relevant-frame recall 70.5%)
and [[cogreasoner]] loses −5.1 to ground-truth context on CogStream. MLVU-style evals —
where evidence is diffuse — are the easy case; papers that argue query-agnostic compression
is free by citing them are choosing it.

---

## 5. Native / niche benchmarks (single-source, no cross-comparison)

Several papers headline on their own or specialized benchmarks; these are **each reported by
exactly one paper**, so there is nothing to compare against — recorded here for completeness.

| Paper | Fam | Benchmark (backbone) | Headline result |
|---|---|---|---|
| [[streamchat-mem]] StreamChat | D | StreamBench, online (LongVA) | Acc 64.7 / Score 3.48 (Slow) vs Video-online 56.4 |
| [[cogreasoner]] CogReasoner | E | CogStream test (VideoLLaMA3) | 72.26 avg (0–100); GPT-4o 73.90, ReKV 43.18 |
| [[streamingvlm]] StreamingVLM | B | Inf-Streams-Eval (Qwen2.5-VL-7B) | 66.18% win-rate vs GPT-4o-mini (GPT-5 judge) |
| [[ovg-hq-unify]] OVG-HQ | D | QVHighlights-Unify, text query | oR@1,IoU=.5 = 23.26 / omAP@.5 = 23.09 |
| [[streamvln]] StreamVLN | E | R2R / RxR Val-Unseen (LLaVA-Video-7B) | R2R SR 57.4 / SPL 51.1; RxR SR 54.4 / SPL 45.4 |
| [[streamingdvc]] StreamingDVC | D | ActivityNet-Captions (GIT) / YouCook2 (Vid2Seq) | CIDEr 41.2 / SODA_c 6.6 ; YouCook2 CIDEr 32.9 |
| [[videostreaming]] VideoStreaming | D | EgoSchema / NExT-QA / MovieChat-1K (Vicuna-7B) | 44.1 / 66.2 / Global 90.4 (score 4.42) |
| [[providellm]] ProVideLLM | D | EgoExo4D keystep / COIN next-step (Llama-3.1-8B) | 50.74% ego acc ; 53.6% Top-1 forecast |
| [[videollamb]] VideoLLaMB | D | EgoSchema / NExT-QA (Vicuna-7B) | 53.8 / 71.1 |
| [[video-salmonn-s]] video-SALMONN S | D· | ELViM episodic / LVBench (Qwen3-VL-8B) | 46.7% (w/ reading) ; LVBench 55.6 |
| [[omnimem]] OmniMem | A | LV-Omni-Bench / LVBench (video-SALMONN-2+-8B) | 42.5% ; 53.3% |
| [[streammeco]] StreamMeCo | E | M3-Bench + VideoMME-Long, 70% compr. (M3-Agent) | 45.7% avg (base 44.7) at 1.87× retrieval speedup |
| [[visual-agentic-memory]] VAM | E | MM-Lifelong, month run (Gemini 3 Flash) | 17.11% acc, stores 0.06% of raw frames |
| [[v-rex]] V-Rex | C | COIN accuracy-drop (Llama-3-8B) | −0.8% vs ReKV −2.0%; 2.2–7.3× per-frame speedup |
| [[tays]] TaYS | B | extended VideoEspresso CoT (Qwen2.5-VL-3B/7B) | avg 33.45 / 36.86; GPT-5 win-rate 43.7% |

**OVBench** (the one benchmark named in the brief that barely recurs): only [[streamforest]]
reports it — **60.5** (PEMF, in its own ablation) — so it gets no table, just this note.

---

## 6. What these tables structurally miss

Cross-cutting blind spots that no per-benchmark row can show — each verified against the
source notes.

- **The wall-clock reality check.** Every accuracy row above silently assumes the model
  keeps up with the stream for free. Under [[streamingeval]]'s matched byte-budget +
  wall-clock protocol (cross-topic, 0.5 GB / 1 FPS), plain *offline* Qwen3-VL-8B tops both
  substrates (OVO 58.00 / SB 77.31, MaxFPS 8, TTFT 0.20 s) while the best native-online
  model, [[streamforest]], matches accuracy (55.57 / 77.26) at MaxFPS 4 / TTFT 0.98 s —
  and VideoChatOnline-4B sustains 0.14 FPS, i.e. it cannot actually watch at 1 FPS. Part
  of the streaming-memory accuracy record is an offline-in-disguise regime.
- **Memory and timing are optimized by disjoint communities.** No memory paper here reports
  proactive/response-timing metrics (those tables live in [[proactive-response]]);
  [[fluxmem]] even builds a proactive trigger from its pruning statistics and never
  evaluates it on a proactive benchmark.
- **No multi-turn evaluation of the KV methods** — the setting query-agnostic compression
  exists for. The one datapoint is [[cogreasoner]]: feeding all dialogue history loses ~4.7
  pts under 30% distractor turns while selective retrieval holds steady.
- **Parametric memory never meets the KV line.** [[video-salmonn-s]] (TTT fast-weights) and
  [[ovg-hq-unify]] (weights-as-memory) headline only on their own benchmarks — there is
  zero head-to-head evidence on the tokens-vs-weights question.
- **Text vs visual memory is untested at matched budget.** Text-side memories lead the
  history-dependent subsets ([[vst]]'s thoughts, [[tww]]'s notes, [[rlivs]]'s captions)
  while visual-KV memories lead the efficiency tables; nobody has run both on one backbone
  at one token budget.
- **Only [[rekv]] prices memory per query** (5.6 vs 13.8 TFLOPs/QA at 360 QA/h — encoding
  amortizes as query rate rises); "memory pays off when queried often" is otherwise untested.
- **Elaborate structure earns little online, a lot offline.** [[eventmemagent]]'s
  hierarchical-vs-fixed ablation is worth +0.59 OVO / +0.20 SB while its OCR/detection
  *tools* are worth +2.9–3.9; [[streamforest]]'s PEMF beats FIFO by only +1.8–2.7 on the
  online splits — but rescues offline MLVU by +13.3 (56.7 → 70.0). Current online
  benchmarks reward recent-window fidelity plus coarse retrieval, not structure.

---

## How to use this table honestly

1. **Never rank across rows.** The dominant variable is the backbone (a Qwen3-VL-8B row will
   beat a Vicuna-7B row regardless of memory design), then the reported split, then the
   token/KV budget and judge.
2. **Read within-paper deltas.** The publishable claim is always method-vs-its-own-baseline
   under matched settings; those deltas (cited in each row's source note) are the real signal.
3. **Efficiency is the co-metric.** Most of these methods trade a small accuracy delta for a
   large memory/latency win at fixed budget — see [[streaming-memory-design-axes]] for the
   memory-footprint / latency side of the ledger that this accuracy table deliberately omits.

**Idea-family key** (shared across the four streaming-memory artifacts):
**A** budgeted KV compression/eviction · **B** streaming attention layouts (sink+window;
think-while-stream) · **C** KV / segment retrieval · **D** hierarchical & event-structured
memory (D· = parametric fast-weight memory, grouped with D) · **E** agentic / semantic
memory · **F** principled selection (what to remember at all).

---
tags: [streaming-video-understanding, proactive-response, synthesis]
---

# Proactive Response — Verified Benchmark Table

Every number here is copied from **each paper's own tables** (cross-checked against the
per-paper deep notes in `../papers/`). This is a *map of what each paper claims on which
benchmark*, grouped by the same five idea-families as the sibling
[[proactive-response-design-axes]] and [[evolution-of-proactive-response]] artifacts.

> [!warning] These numbers are **NOT** head-to-head comparable.
> Rows come from different papers, run under **different backbones** (2B → 30B; Qwen2-VL /
> Qwen2.5-VL / Qwen3-VL / InternVideo2.5 / LLaMA / Omni), **different FPS** (1 vs 2), **different
> token budgets / drop-ratios**, and — critically for the proactive benchmarks — **different
> scoring protocols on the same benchmark name**. Two traps in particular:
> - **StreamingBench** is reported both as its *reactive* "Real-Time Visual Understanding /
>   All" accuracy **and** as the *proactive* "Proactive Output (PO)" subtask accuracy — different
>   axes; never read one against the other.
> - **OVO-Bench Forward Active Responding (FAR)** is reported under **three incompatible
>   scorings**: (a) the original **accuracy-style** time-decayed average (used by [[dispider]],
>   [[streamagent]], [[stride]], [[vispeak]], [[wat]], [[response-g1]]); (b) [[em-garde]]'s
>   **online recall/precision F1**; (c) [[streampro]]'s **turn-level windowed F1**. A FAR
>   *accuracy* ~55 and a FAR *F1* ~30 are **not** on the same scale — and the two F1s aren't
>   comparable to *each other* either: the same MMDuet2 scores **20.51** under (b) but **6.5**
>   under (c).
>
> Treat this as "what the authors report," not a leaderboard. `n/r` = not reported in that
> paper. Proprietary/offline reference ceilings (GPT-4o, Gemini 1.5 Pro, Human) are in footnotes,
> not as rows.

**Idea-families** (consistent across the sub-topic's artifacts):
**A** In-band token gating (EOS/action tokens in the LM) ·
**B** Dedicated decision heads / trigger policies ·
**C** Training-free / decoding-time triggers ·
**D** Offline-to-streaming adaptation ·
**E** Agentic / interaction-infrastructure systems.

---

## 1. StreamingBench

Two distinct columns. **PO** = Proactive Output subtask accuracy (reply must land within ~2 s
of the answering scene) — the *proactive* signal. **RT/Overall** = Real-Time Visual
Understanding "All" or Overall accuracy — a *reactive* QA score. Backbone noted because it
swings the RT column by >20 pts.

| Fam | Paper | Backbone | PO (proactive, acc ↑) | RT / Overall (reactive, acc ↑) |
|---|---|---|---|---|
| A | [[videollm-online]] *(baseline)* | LLaMA-8B | 3.92 | 32.48 / 35.99 |
| A | [[thinkstream]] | Qwen2.5-VL-3B | n/r | 75.00 (RT) |
| A | [[mmduet2]] | Qwen2.5-VL-3B | 34.69 | n/r |
| B | [[mmduet]] | LLaVA-OV-7B | 29.44 | n/r |
| B | [[dispider]] | 7B | 25.34 | 53.12 overall / 67.63 (RT All) |
| B | [[streamagent]] | Qwen2.5-VL-3B+7B | 28.9 *(from [[response-g1]] Tab. 2)* | 57.02 overall / 74.28 (RT All) |
| B | [[stride]] | Qwen3-VL-8B | 32.40 → **42.80** | 46.84 → **59.29** |
| B | [[vispeak]] | Qwen2.5-7B (omni) | n/r | 62.58 (s3) / 62.00 (s2) |
| B | [[roma]] | Qwen2.5-Omni | n/r | ~76 avg |
| B | [[streamready]] | Qwen2-VL-7B | 48.2 *(proactive subset)* | n/r |
| C | [[response-g1]] | Qwen3-VL-8B (frozen) | **44.0** | 77.5 (RT All) |
| C | [[timechat-online]] | Qwen2.5-VL-7B | n/r | 75.28 (↓44% tok) / 73.64 (↓83% tok) |
| C | [[em-garde]] | 2B trigger / 7B responder | 38.0 @2fps / 37.6 @1fps | 76.7 (RT, 7B responder) |
| C | [[lyrav-dont-pause]] | InternLM2.5-7B (frozen) | n/r | 72.78 (RT) |
| C | [[querystream]] | n/r | claimed SOTA (n/r) | claimed SOTA (n/r) |
| D | [[wat]] | Qwen2.5-VL-7B | n/r | **77.70** |
| D | [[streambridge]] | Qwen2-VL-7B | n/r | 77.04 (RT AVG) |
| E | [[r3-streaming]] | 7B ‖ 4B-Thinking | *(Proactive split used for Respond)* | **76.36** (All, 95% tok drop) |
| E | [[streampro]] | Qwen3-VL-4B | *(see StreamPro-Bench)* | 71.8 (SFT) / 71.2 (GRPO) |
| E | [[streamov]] | Qwen3-Omni-30B (frozen) | n/r | 68.6 (Omni avg) / 86.2 (Visual-Only) |

Papers reporting *no* StreamingBench number (own or otherwise): [[stream-vlm-qevd]], [[livecc]],
[[proassist]], [[assistpda]], [[lion-fs]], [[videollm-eyewo]], [[streamo]], [[egospeak]],
[[streammind]], [[proact-vl]], [[sdqes]], [[livestar]], [[livestarpro]], [[evostreaming]],
[[streamchat-nvidia]] — they headline their own bespoke benchmarks (§5–6).

*Reference ceilings — read the axis carefully: the oft-quoted **GPT-4o 73.28 / Gemini 1.5 Pro
75.69** are the **RT-subgroup** scores; the **full 18-task overall** is GPT-4o **60.15** /
Gemini **67.07** / Human **91.66** ([[streamingbench]] Tab. 2, [[dispider]] Tab. 1). So "beats
GPT-4o" claims by [[wat]]/[[thinkstream]]/[[r3-streaming]] are earned on the easiest (RT) axis —
where the offline Qwen2.5-VL-7B base already scores **73.68** ([[timechat-online]] Tab. 1), i.e.
streaming fine-tuning is worth only ~+2–4 pts there and the axis no longer separates methods.*

![[dispider.png]]
*The proactive-output result that made StreamingBench-PO a standard column: [[dispider]]'s
disentangled decision head lifts PO to 25.34 (~6× over [[videollm-online]]'s 3.92) — later
overtaken by [[mmduet]] 29.44 → [[mmduet2]] 34.69 → [[em-garde]] 38.0 → [[stride]] 42.80 →
[[response-g1]] 44.0. Source: [[dispider]] Fig. 1.*

---

## 2. OVO-Bench

**Overall** = accuracy-style overall average (Real-Time + Backward + Forward). **FAR** = Forward
Active Responding, the proactive "answer when the event arrives" axis — but read the metric tag:
*(acc)* rows and *(F1)* rows are **on different scales and cannot be compared**.

| Fam | Paper | Backbone | Overall (acc ↑) | FAR — proactive |
|---|---|---|---|---|
| A | [[videollm-eyewo]] | VideoLLM-Online (LLaMA3) | 24.16 | n/r |
| A | [[streamo]] | Qwen2.5-VL-7B | 55.61 (1fps) / 57.86 (2fps) | CRR 83.33 *(forward-active lead)* |
| A | [[thinkstream]] | Qwen2.5-VL-3B | **59.66** | *(BW 52.30 sub)* |
| A | [[mmduet2]] *(baseline in others)* | Qwen2.5-VL-3B | n/r | 20.51 *(F1, em-garde proto)* / 6.5 *(F1, streampro proto)* |
| A | [[livecc]] | Qwen2-VL-7B | 59.8 (OVOBench avg) | n/r |
| B | [[dispider]] *(baseline)* | 7B | 41.78 | 34.7 *(acc)* |
| B | [[streamagent]] | Qwen2.5-VL-3B+7B | 49.4 | 45.4 *(acc)* |
| B | [[stride]] | Qwen3-VL-8B | 51.77 → **59.07** | 46.30 → **59.70** *(acc)* |
| B | [[vispeak]] | Qwen2.5-7B (omni) | 61.08 (s2) | 54.25 *(acc)* |
| B | [[roma]] | Qwen2.5-Omni | ~59 (reactive) | narration F1 14.54 / GPT 0.42 |
| B | [[streamready]] | Qwen2-VL-7B | *(proactive subset)* | n/r |
| C | [[timechat-online]] | Qwen2.5-VL-7B | 45.6 (↓85% tok) / 47.6 (↓45%) | *(strong FAR sub)* |
| C | [[response-g1]] | Qwen3-VL-8B (frozen) | 61.3 | **58.2** *(acc)* |
| C | [[em-garde]] | 2B trigger / 7B responder | 63.0 (RT-VP, 7B) | **30.99** *(F1, own proto)* |
| C | [[lyrav-dont-pause]] | InternLM2.5-7B (frozen) | 50.97 | n/r |
| D | [[wat]] | Qwen2.5-VL-7B | 55.2 | 55.8 *(acc)* |
| D | [[streambridge]] | Qwen2-VL-7B | 71.30 (RT AVG) | ET-Bench (see §4) |
| D | [[evostreaming]] | Qwen2-VL (frozen) | 43.8 → 54.6 *(RealStreamEval re-score)* | 12.6 → 30.6 *(RealStreamEval)* |
| E | [[r3-streaming]] | 7B ‖ 4B-Thinking | **57.92** (96% tok drop) | 50.60 *(acc)* |
| E | [[streampro]] | Qwen3-VL-4B | 58.5 (SFT) / 57.6 (GRPO) | **20.6** *(F1, own proto)* |
| E | [[streamov]] | Qwen3-Omni-30B (frozen) | 64.0 (visual-only) | n/r |

Notes on protocol:
- **FAR *(acc)*** and **FAR *(F1)*** are mutually incomparable — and so are the two F1s
  (em-garde's online recall/precision vs streampro's turn-level windowed matching; MMDuet2 =
  20.51 vs 6.5 under the respective protocols). Both F1s were introduced to penalize degenerate
  always-fire policies, which [[em-garde]] and [[streampro]] argue the original accuracy metric
  rewards. The smoking gun: [[streamo]] posts CRR **83.33** under the official scoring but FAR F1
  **5.4** under [[streampro]]'s — the same model, top-of-column and near-floor at once.
- **The polling protocol itself inflates streaming models.** [[evostreaming]]'s causal,
  verbosity-penalized re-score drops [[dispider]] 41.78 → **33.3**, [[timechat-online]]
  45.6–47.6 → **27.5**, [[flash-vstream]] ~33 → **24.0** — roughly **8–18 pts** of every official
  OVO number is protocol, not capability. Three papers ([[em-garde]], [[streampro]],
  [[evostreaming]]) independently rebuilt the FAR metric because they didn't trust it; none
  adopted another's rebuild.
- **Backbone drift does much of the "overall" work**: zero-shot Qwen3-VL-8B already scores
  **51.77** ([[stride]] Tab. 1) — above [[dispider]]'s entire trained system (41.78). A 2026
  paper claiming "+15 over Dispider" inherited ~10 of those points from the newer base model.
- [[streambridge]]'s 71.30 is an **OVO-Bench-RT AVG** under an offline→streaming *multi-turn*
  setting (history retained), not the single-turn overall — noted so it isn't misread against
  the others.
- [[evostreaming]] re-scores OVO tasks under its own causal, verbosity-penalized
  **RealStreamEval** protocol; its 54.6 is not the vanilla OVO overall.

*Reference ceilings: the origin paper's official numbers are GPT-4o **59.54**, Gemini 1.5 Pro
**63.00**, Human **92.81** ([[ovo-bench]] Tab. 1); the ≈58.6 / ≈65.3 figures circulating in
[[streamagent]]/[[wat]] tables are re-quotes — even the *ceilings* drift between papers.
[[streamo]]/[[r3-streaming]] report [[streamforest]] 55.61-class baselines; [[flash-vstream]]-7B
≈ 24–33 is the weak floor. The ~30-pt human–model gap has barely moved since [[ovo-bench]]
measured it.*

---

## 3. OVBench (VideoChat-Online)

Only two rows carry a verified OVBench number — both from [[lyrav-dont-pause]]'s Table III,
where [[livestar]] is the in-paper baseline.

| Fam | Paper | Backbone | OVBench avg (↑) |
|---|---|---|---|
| C | [[lyrav-dont-pause]] | InternLM2.5-7B (frozen) | **46.8** (best open-source online) |
| C | [[livestar]] *(baseline in lyrav)* | InternVideo2.5-7B | 45.7 |

*Reference in the same table: MovieChat 30.9, Flash-VStream 31.2, offline InternVL2-7B 48.7,
Gemini-1.5-Flash 50.7 — the open-source-online row still trails offline/proprietary.*

---

## 4. ET-Bench (proactive: DVC / SLC / TVG / TAL)

Event-localization F1 / similarity for streaming captioning & grounding. Watch the metric:
[[stride]]'s number is an **activation-window F1**, not the same as [[streambridge]]/[[dispider]]'s
generation F1.

| Fam | Paper | Backbone | DVC-F1 | DVC-Sim | SLC-F1 | TVG-F1 | TAL-F1 |
|---|---|---|---|---|---|---|---|
| A | [[videollm-online]] *(baseline)* | LLaMA-8B | 24.0 | n/r | 9.9 | 13.2 | n/r |
| B | [[dispider]] | 7B | 33.8 | 18.9 | 18.8 | 36.1 | 27.3 |
| B | [[stride]] | Qwen3-VL-2B | n/r | n/r | n/r | 62.8 *(activation F1)* | avg 32.6 |
| D | [[streambridge]] | Qwen2-VL-7B | **38.3** | **25.1** | 22.6 | 34.3 | 24.3 |

Read-off: [[streambridge]] wins the *description-similarity* metrics (decoupled timing → richer
captions); [[dispider]]'s dedicated grounding still edges the pure temporal-grounding F1s.

---

## 5. Ego4D streaming narration (online "when + what to say")

Heterogeneous metrics — each paper reports its own subset, so cells name the metric. All are on
Ego4D (or Ego-Exo4D) narration streams; **lower is better** for PPL/TimeDiff, **higher** for the
rest.

| Fam | Paper | Backbone | LM-PPL ↓ | TimeDiff ↓ | Fluency % ↑ | Token/Trigger metric |
|---|---|---|---|---|---|---|
| A | [[videollm-online]] | LLaMA-8B | 2.43 | 2.32 | 42.6 | — |
| A | [[lion-fs]] | LLaMA-3-8B | 2.09 (Ego4D) / 2.04 (Ego-Exo4D) | 2.15 / 0.74 | 46.1 / 36.5 | LM-Correctness 52.4 / 48.2 |
| A | [[assistpda]] | Qwen2-VL-2B | 1.68 (LM-PPL, VAPDA) | 1.07 | 53.81 | *(surveillance domain)* |
| B | [[streammind]] | (CLIP+SSM+LLM) | n/r | 1.89 | 60.2 | TriggerAcc 43.34 / TimVal 39.73 |
| C | [[livestar]] | InternVideo2.5-7B | 1.97 | 1.76 | n/r | Token-Acc 61.1 (+8.7 over LION-FS) |
| C | [[livestarpro]] | InternVideo2.5-7B | n/r | n/r | n/r | Token-Acc +18.1% over LION-FS |

[[streammind]] also reports SoccerNet online (TriggerAcc 52.18 / TimVal 47.36 / BLEU-4 66.70 /
ROUGE-L 82.04). [[lyrav-dont-pause]] reports the related **OmniStar synchrony** benchmark:
Sync-Rate **98.29%** un-truncated (vs [[livestar]] 78.93, [[livecc]] 92.41).

**Every row above is self-run, and even the shared baseline drifts**: [[videollm-online]]'s own
paper reports 2.43 / 2.32 / 42.6, but [[lion-fs]]'s Ego4D re-run of the same baseline gets
2.40 / **2.04** / 45.3 — different frame rates and splits. And [[lion-fs]] honestly *loses*
TimeDiff to its own baseline (2.15 vs 2.04) even while winning PPL/Fluency/Correctness — a
timing-vs-content trade its headline omits. This lineage supports **within-paper deltas only**.

---

## 6. Bespoke benchmarks (each paper's own headline)

Most proactive papers ship their **own** benchmark to score timing — so these single-row entries
are the cleanest statement of "what the paper proves," and are the *least* comparable across
rows. Baseline/teacher comparisons live in the per-paper notes.

| Fam | Paper | Benchmark (own) | Headline verified result |
|---|---|---|---|
| A | [[stream-vlm-qevd]] | QEVD-FIT-COACH | Temporal F-Score **0.56**, LLM-Acc 2.45/5 (best on all axes) |
| A | [[assistpda]] | VAPDA-127K | VAP weighted-F1 **64.69** / AAT **29.19 s** (vs Holmes-VAD-7B 47.91 / 15.68 s) |
| A | [[proassist]] | PROASSIST | Action-Narration F1 **61.96**; Dialogue-Gen F1 36.25 (timing-aware) |
| A | [[livecc]] | LiveSports-3K-CC | CC win-rate 43.2 (Base) vs GPT-4o anchor 72.2; beats LLaVA-Video-72B 35.0 |
| A | [[videollm-eyewo]] | ESTP-Bench | ESTP-F1 overall **34.7** (vs best offline-polling 22.9, [[mmduet]] 17.8) |
| A | [[streamo]] | Streamo-Bench | avg **55.3** (vs StreamingVLM 24.6, [[dispider]] 14.6) |
| A | [[thinkstream]] | *(uses OVO/StreamingBench)* | — (see §1–2) |
| A | [[mmduet2]] | ProactiveVideoQA (PAUC) | see §7 |
| B | [[egospeak]] | EasyCom / Ego4D | Target-speaker AP **52.7 / 66.8** (~2× naive; A+V Transformer) |
| B | [[mmduet]] | MAGQA / QVHighlights | MAGQA in-span 3.13 (LLaMA judge); QVHighlights 31.3 mAP / 49.6 HIT@1 |
| B | [[vispeak]] | ViSpeak-Bench | overall **2.76** (TAcc 80.42%, TS 3.25); GPT-4o 2.99, Human 3.69 |
| B | [[roma]] | OmniMMI | PA 37.50 / PO 53.60 / CRR 35.42 / REC 33.81 (speak-head ablation → collapse) |
| B | [[proact-vl]] | Live Gaming Benchmark | CC **49.23** / F1 **64.87** (best) / PAUC 18.10 / TimeDiff 1.71 |
| B | [[streamready]] | ProReady-QA | Acc 56.4 / ARS 0.69 / **Acc_e 0.53** (best; [[streambridge]] 0.42) |
| B | [[sdqes]] | EgoSDQES | 1-min SR@1 **29.1** (LaViLa+QR-Adapter), SMD@1 18.1 s |
| B | [[stride]] | *(uses OVO/StreamingBench/ET-Bench)* | — (see §1–2, §4) |
| B | [[streamagent]] | *(uses OVO/StreamingBench)* | — (see §1–2) |
| C | [[livestar]] | OmniStar (5 task) | RNG SemCor 3.19 / TimDiff 1.91 (+19.5% SemCor over prior online); FPS 3.82 |
| C | [[livestarpro]] | OmniStarPro-Long | >30-min-bucket recall LMR **37.2** (vs [[livestar]] 21.1, [[mmduet]] 9.1) |
| C | [[lyrav-dont-pause]] | OmniStar synchrony | Sync-Rate **98.29%**; rFPS 3.89 |
| C | [[em-garde]] | *(OVO/ProactiveVideoQA)* | OVO-FAR F1 30.99 (§2); ProactiveVideoQA (§7); ~13 fps |
| C | [[response-g1]] | *(uses OVO/StreamingBench)* | — (see §1–2) |
| C | [[timechat-online]] | *(uses OVO/StreamingBench)* | — (see §1–2) |
| C | [[querystream]] | *(claims SOTA, unverifiable)* | n/r |
| D | [[evostreaming]] | RealStreamEval | Overall 43.8 → **54.6** on Qwen2-VL (+10.8) with ~1,000 self-gen trajectories (139× less data) |
| D | [[wat]] | *(uses OVO/StreamingBench)* | — (see §1–2) |
| D | [[streambridge]] | *(uses OVO/Streaming/ET-Bench)* | — (see §1–2, §4) |
| E | [[streamov]] | SOVBench | SOVBench-O avg **83.8%**; Proactive-QA 78.6% / F1 90.5; trigger P/R 86.1/98.3 |
| E | [[streampro]] | StreamPro-Bench | SPB W-Avg F1 **41.5** (GRPO-4B) vs previous-best [[streamo]]-3B 10.4 (~4×) |
| E | [[r3-streaming]] | *(uses OVO/StreamingBench)* | — (see §1–2) |
| E | [[streamchat-nvidia]] | StreamEval (win/tie/loss) | 7B vs LLaVA-Video-72B **37/32/31**; vs VILA-1.5-40B 53/24/23 |

![[streampro.png]]
*[[streampro]] reframes proactive evaluation from reactive see-then-answer into "Proactive
Agency" (decisions under partial observation) — its StreamPro-Bench F1 41.5 vs the prior best
open-source proactive model's 10.4 shows how far apart the *proactive* axis still is from the
saturated reactive columns in §1. Source: [[streampro]] Fig. 1.*

---

## 7. ProactiveVideoQA (PAUC)

The one *shared* proactive-timing benchmark beyond StreamingBench/OVO. **PAUC** (↑) rewards being
right *and early within the valid span*; **dup** (↓) = reply-duplicate proportion — the guardrail
that exposes always-fire spam.

| Fam | Paper | Backbone | WEB (PAUC / dup) | EGO (PAUC / dup) | TV | VAD |
|---|---|---|---|---|---|---|
| B | [[mmduet]] *(baseline)* | LLaVA-OV-7B | 38.9 / 81.3 | 46.0 / 99.4 *(spam)* | 21.1 / 92.8 | n/r |
| A | [[mmduet2]] | Qwen2.5-VL-3B | **53.3 / 4.2** | 33.6 / 8.1 | 43.4 / 1.0 | 28.9 / 15.2 |
| C | [[em-garde]] | 2B / 7B | 44.3 / 4.5 | **52.3** / 17.4 | n/r | 27.4 / 1.4 |
| — | *Human (ref)* | — | 38.6 | 38.2 | 47.0 | 53.6 |
| — | *GPT-4.1 (offline ref)* | — | 51.7 | 58.8 | 56.8 | 46.2 |

Read-off: [[mmduet]]'s high EGO PAUC (46.0) is an artifact of ~99% duplicate spam; [[mmduet2]]
and [[em-garde]] win on the **joint PAUC-and-low-duplicate** frontier. All models stay weak on
surveillance (VAD ≈ 27–29), confirming proactive timing on long low-event streams is unsolved.
Two construct-validity caveats from the benchmark's own tables ([[proactivevideoqa]] Tab. 3):
**humans score below GPT-4.1 on WEB/EGO/TV** (confounded by the pause-and-type annotation
interface, so PAUC has no sane human ceiling), and all correctness comes from a GPT-4.1 judge
with modest human-agreement kappas (~0.3–0.5). Still, PAUC is the **only** proactive-timing
metric adopted by a second group ([[mmduet2]] trains on it as an RL reward, [[proact-vl]]
reports it) — every other bespoke timing metric in §6 has exactly one user: its creator.

---

## 8. Offline capability retention (VideoMME, no-collapse check)

Recurs because the proactive papers must show streaming training didn't wreck base QA. VideoMME
overall accuracy (metrics/subtitles vary — see notes); marked where it's a preserved-vs-base check.

| Fam | Paper | VideoMME (↑) | Note |
|---|---|---|---|
| A | [[lion-fs]] | *(narration-only paper)* | n/r overall |
| A | [[thinkstream]] | 61.9 | +5.0 avg over base Qwen2.5-VL-3B |
| A | [[streamo]] | offline avg 63.9 | +3.3 over Qwen2.5-VL-7B |
| A | [[mmduet2]] | 67.5 / 58.1 | ~matches Qwen2.5-VL-3B base |
| B | [[dispider]] | 57.2 | offline preserved |
| B | [[streamagent]] | 62.9 / long 50.6 | matches backbone |
| B | [[streamready]] | 65.8 | offline avg 69.8 |
| C | [[livecc]] | 64.1 (w/o sub) / 70.3 (w sub) | — |
| C | [[livestarpro]] | 60.8 (w/o sub) | +8.0 over VideoChat-Online |
| C | [[lyrav-dont-pause]] | 60.90 | MVBench 67.10, preserved |
| C | [[timechat-online]] | 62.5 (↓85% tok) | training-free DTD |
| D | [[streambridge]] | 64.4 | +1.1, general video preserved |
| D | [[wat]] | 62.4 | no offline collapse |
| D | [[evostreaming]] | drops <2 pts | offline avg preserved across 5 backbones |
| E | [[streampro]] | 60.7 (SFT) / 60.4 (GRPO) | slight LongVideoBench regression |
| E | [[streamov]] | 73.5 | Long 63.4 (+5.0) |
| E | [[streamchat-nvidia]] | 63.1 / 66.3 (14B); 58.6 / 62.8 (7B) | beats larger VILA-40B |

The consistent story across families: streaming/proactive adaptation **preserves offline QA to
within ~1–2 pts** — the offline-to-streaming thread ([[streambridge]], [[evostreaming]], [[wat]])
makes this an explicit design goal, and the in-band/head families back into it via LoRA + frozen
encoders.

---

## 9. Cross-cutting confounds the tables hide

Verified in the papers' own ablations — these swing scores as much as the architectures do:

- **Class-imbalance handling is the most load-bearing "detail" in the literature.** Four
  independent confirmations: [[streamo]] focal-vs-CE on OVO Forward Active — REC **27.94 vs
  6.45** (4.3×, Tab. 4); [[streampro]] loss ablation CE 6.6 → focal 14.2 → CB-Stream **16.3**
  (2.5×, Tab. 5); [[proassist]] negative-frame sub-sampling F1 **30.1 → 58.7** (2×, Tab. 6);
  [[streammind]] weighted CE against a **310:1** silence:response ratio. No paper's abstract
  leads with this, yet it's worth more than most proposed modules.
- **Trigger thresholds are hidden per-paper operating points.** [[vispeak]] and [[streamready]]
  fire at a fixed 0.35; [[em-garde]] at θ=0.04 — and em-garde is the *only* paper publishing a
  sensitivity sweep (F1 26.6–31.0 across θ). Every cross-model timing comparison implicitly
  compares tuned operating points, not policies.
- **fps is a free, mostly unstated gain**: [[streamo]] goes 55.61 → 57.86 OVO overall just by
  evaluating at 2 fps instead of 1, no retraining; [[em-garde]] reports 38.0 @2fps vs 37.6
  @1fps. Comparison tables rarely state fps.
- **Judge fragmentation**: GPT-4o judges OVO open-ended tasks, GPT-4.1 judges PAUC,
  Gemini 2.5 Pro judges [[streampro]]'s rubric and [[proact-vl]]'s win-rate, GPT-5.1 scores
  [[proact-vl]]'s quality metrics. Judge-scored numbers are incomparable across benchmarks and
  unreproducible once the judge API deprecates.
- **Baseline drift ≈ method gains.** Five different "Dispider" numbers circulate: 53.12 (full
  StreamingBench), 67.63 (RT All), ~72 ([[roma]]'s re-run), 34.7 (FAR acc), 33.3
  ([[evostreaming]] causal re-score) — the spread across re-runs is comparable in size to many
  claimed improvements over it.
- **System-size confounds are ignored**: top rows are increasingly two-model cascades
  ([[streamagent]] 3B+7B, [[r3-streaming]] 7B‖4B-Thinking, [[em-garde]] 2B trigger + 7B
  responder) ranked flat against single 7Bs; nobody normalizes for total params/FLOPs, and only
  latency side-notes ([[em-garde]] ~13 fps, [[lyrav-dont-pause]] rFPS 3.89) hint at cost.

---

## How to read this artifact

- **The proactive columns are the young, unsaturated ones.** StreamingBench-PO climbs 3.92 →
  44.0; OVO-FAR-F1 sits at 20–31; ProactiveVideoQA VAD ≈ 27–29; [[streampro]]'s SPB is 41.5 vs a
  prior 10.4. The *reactive* RT/Overall columns are near-saturated (73–78, brushing proprietary
  ceilings) — which is exactly the [[streampro]] critique that "reactive streaming" has collapsed
  into delayed perception while genuine proactivity is still open.
- **Family ≠ score tier.** Every family has a top-of-column entry ([[wat]]/[[streambridge]] in D,
  [[response-g1]]/[[em-garde]] in C, [[stride]]/[[streamagent]] in B, [[r3-streaming]]/[[streampro]]
  in E). The mechanism (in-band token vs head vs training-free vs adaptation vs agentic) is a
  *design* choice, not a performance ranking — see [[proactive-response-design-axes]].
- **But the PO leaderboard has a mechanism trend**: the 2026 leaders — [[response-g1]]
  (training-free retrieval over frozen Qwen3-VL-8B, 44.0), [[stride]] (2B masked-diffusion
  activation net + frozen 8B, 42.80), [[em-garde]] (2B embedder trigger + separate responder,
  38.0) — all **keep the LLM frozen and solve timing outside the weights**, while the trained
  end-to-end trigger lineage ([[mmduet]], [[dispider]], [[vispeak]]) plateaued in the 25–35
  band. The when-to-speak decision is empirically separating from generation — [[dispider]]'s
  disentanglement argument taken to its limit: no gradient into the backbone at all.
- **Memory and proactivity are orthogonal capabilities** (cross-link to [[streaming-memory]]):
  memory specialists sit near the proactive floor — [[flash-vstream]] PO 1.96 / FAR-F1 4.77,
  [[streamforest]] RT 77.3 but FAR-F1 13.95 (the largest reactive/proactive spread in these
  tables). No amount of retrieval fixes a broken trigger; conversely [[wat]]/[[r3-streaming]]
  show good memory plus an explicit trigger compose additively.
- **Bespoke benchmarks dominate.** 20+ of the 34 papers headline a benchmark they authored (§5–6);
  the shared benchmarks (§1–3, §7) are the only cross-paper anchors, and even those carry
  protocol caveats. Trust the *within-paper* deltas (each note's baseline rows) over any
  cross-row read here.

*Sources: every value verified against the corresponding `../papers/<slug>.md` deep note (each
note carries its own arXiv/table-level verification line). See [[evolution-of-proactive-response]]
for the narrative and [[proactive-response-concept-graph]] for the mechanism map.*

# Baseline model — training-free methods to max benchmark performance

> **Goal (2026-07-17):** strongest possible *training-free* baseline on the three streaming
> benchmarks — the bar any future trained method must clear, and a map of where training-free
> saturates. **Backbone fixed: frozen Qwen3-VL-8B** (decided 2026-07-17). Everything bolted on at
> inference. Compute: 1–2× A100/H100.

## Benchmarks
- [[streamingbench]] — StreamingBench
- [[ovo-bench]] — OVO-Bench
- [[river-bench]] — RIVER

---

## 1. What each benchmark demands

| | [[streamingbench]] | [[ovo-bench]] | [[river-bench]] |
|---|---|---|---|
| Scale | 900 vids / 4,500 MCQ, 18 tasks | 644 vids / ~2,800 Q, 12 subtasks | 1,067 vids / 4,278 Q |
| Task groups | Real-Time (10) · Omni-Source **(audio)** (4) · Contextual (4, incl. Proactive Output) | Backward Tracing · Real-Time Perception · Forward Active Responding | Retro-Memory (forgetting curve) · Live-Perception · Pro-Response |
| Feed protocol | 1 fps, timestamp burned into frame, prefix V[0, t_q] | prefix Video[0:t]; forward tasks = dense multiple-triggering queries | windowed dialogue; multiple EOS = silence allowed |
| Metric | MCQ acc; PO = fire within ±2 s (polled every 1 s) | MCQ acc; Forward = acc + 2^(−Δt·p) lateness decay, GPT-4o judge | MC/OE (Qwen2.5-72B judge); Pro-Response: **early → hard 0**, late → linear decay |
| Best proprietary | Gemini 1.5 Pro 67.07 overall (75.69 RT) | Gemini 1.5 Pro 63.00 (human 92.81) | GPT-4o: Retro 59.56 MC, Live 61.05 MC, Loc 1.63 |
| Universal weak spot | Contextual ≤48.7 even for Gemini | Forward Active; HLD hallucination ≤52.6 | Pro-Response near floor (Loc ≤ ~6) for everyone |

**Structural insight:** all three factor into the same three axes — (past) memory/retrieval ·
(present) perception under budget · (future) when-to-speak. Baseline = frozen Qwen3-VL-8B + one
training-free bolt-on per axis, composed (§4). RIVER has **0 external adopters** — first credible
baseline there is cheap novelty.

---

## 2. The backbone: frozen Qwen3-VL-8B — what we actually know

**Why it's right:** highest plain-model ceiling in the vault (SB-RT 77.31 / OVO 58.00 per
[[streamingeval]], MaxFPS 8, TTFT 0.20 s); the only backbone with a *fully training-free* proactive
result ([[response-g1]]); fits one 80 GB GPU with headroom for caches.

**⚠ Adversarial finding #1 — the published "frozen Qwen3-VL-8B baseline" is not one number.**
Across vault notes the *same frozen model* scores:

| Source note | OVO overall | OVO Forward (FAR) | StreamingBench | Protocol |
|---|---|---|---|---|
| [[streamingeval]] | **58.00** | – | **77.31** (RT) | official-style, MaxFPS 8 |
| [[eventmemagent]] (its baseline row) | 55.81 | – | 70.20 | its harness |
| [[stride]] (its baseline row) | 51.77 | **46.30** | 46.84 (PO 32.40) ⚠ anomalous — subset/protocol unclear | its harness, cadence/prompt undocumented |
| [[evostreaming]] | – | **15.8** | – | RealStreamEval (causal + verbosity/premature penalties) |

Spread: ±6 pts OVO overall, **3× on FAR** purely from protocol. Consequence: **M0 must establish
*our own* baseline under the official harnesses before any method gets credit**, and every number
we ever report carries a protocol label.

**⚠ Adversarial finding #2 — architecture novelty.** Qwen3-VL uses interleaved-MRoPE, DeepStack
multi-layer visual injection, native-resolution tokenization, and text-timestamp alignment. **No
memory-method note in the vault even mentions DeepStack or interleaved-MRoPE** — every KV method
below was derived on 1D-RoPE (LLaVA-OV) or plain M-RoPE (Qwen2.5-VL) with fixed tokens-per-frame
grids. Port risk is real and method-specific (§3A table).

**Context for targets:** the *trained* ceiling on this backbone is SelectStream
([[streaming-model-remember]]): SB 82.67 / OVO overall 67.03. Our training-free stack lives between
plain (77.31/58.00) and that.

---

## 3. Adversarial review — within each axis

### Axis A — Memory (OVO Backward, RIVER Retro-Memory, long-video items)

Verdicts after fact-checking every claim against the notes:

| Method | Measured on Qwen3-VL? | Number reality-check | Architecture risk on Qwen3-VL | Code | Verdict |
|---|---|---|---|---|---|
| [[hermes-kv]] | **Yes — sole datapoint: SB-RT 81.32** (no OVO, no ablations on it) | SB numbers = RT subset ✓; but its "OVO Avg" is a **nonstandard 2-regime mean (RT+BT, excludes Forward)** — not comparable to true overalls | HIGH: position re-indexing proven only for 1D-RoPE/M-RoPE; layer bands 10/60/30 + anchor spacing N=196 are backbone-profiled and **unreported for Qwen3-VL** | **Yes** (github haowei-freesky/HERMES) | Port candidate #1 — code + a Qwen3-VL datapoint exist, but expect re-profiling work, and re-derive re-indexing for interleaved-MRoPE |
| [[savemem]] | No (Qwen2.5-VL only; note: "portability unverified") | OVO 62.69 is a **true overall** ✓; but note warns gains shrink on 3B and on SB (+0.7–2.1) — "the large OVO jump may be partly benchmark/backbone-specific" | **LOW** — no position surgery at all (cosine-on-features, tiered tokens); only feature-layer choice matters under DeepStack | **Yes** (github wuhang03/savemem) | Port candidate #2 — safest port, biggest OVO-overall evidence, must retune ρ/tiers |
| [[decouple-and-cache]] | No | SB 82.32 = RT subset on Qwen2.5-VL; OVO 57.5 true overall ✓; **only method with measured stacking** (+ReKV 79.41, +InfiniPot 79.60); own ablation: history contributes little on SB/OVO | **HIGHEST**: pre-RoPE caching proof assumes RoPE + single-layer visual injection — **DeepStack breaks the proof directly** | No | Don't port the mechanism. **Steal the idea training-free**: keep a raw-frame buffer and *re-prefill* the recent window at query time — same effect, zero position surgery, costs prefill latency |
| [[infinipot-v]] | No | SB = RT subset; OVO Avg = 3-regime ✓ (+1.9); note admits "a token discarded before a query arrives is gone" (BW only 44.5→47.6) | HIGH: 3D time×H×W cache reshape assumes fixed patches/frame — native-resolution tokenization breaks the grid; note itself: "newer native-resolution MLLMs untested" | No | Fallback only; its TaR/VaN *signals* are reimplementable but the grid machinery isn't worth porting |
| [[rekv]] | No | **No SB/OVO in its own note at all** (RVS/MLVU etc.); "ReKV ≈57.2 SB" is other papers' citation, unverified | MED-HIGH: discards original positions on retrieval; per-layer retrieval semantics shift under DeepStack; unbounded memory (~18.8 GB/h) | Not noted | Use as the *concept* reference for query-time KV retrieval; don't build on it directly |
| [[fluxmem]] | No (Qwen2.5-VL @1fps only) | SB-RT +2.5, OVO overall +3.5 ✓ labeled correctly; Otsu = parameter-free (portability plus) | LOW-MED: no position surgery; 3×3 grid matching complicated by native-res | No | Idea donor (Otsu thresholds); below SAVEMem on evidence |
| [[weavetime]] | No | **⚠ numbers unverified** — note says the arXiv PDF served a *different paper*; SB baseline 53.56 inconsistent with every other note | MED (rides ReKV) | Project page only | **Drop from plan.** Not training-free (needs LoRA) and evidence is broken |
| [[streamingtom]] | No | No SB/OVO; hardcodes LLaVA-OV geometry (L=28, N=196, H_kv=4) | HIGH (fixed-N grid) | No | Drop; 4-bit KV quantization idea noted for memory pressure only |

**Within-axis conclusion:** the earlier plan's "HERMES/InfiniPot eviction + DSCache construction +
SAVEMem retrieval" composition was over-optimistic — DSCache's mechanism and InfiniPot's grid don't
survive the port, and the stacking evidence (79.41/79.60) is all Qwen2.5-VL/LLaVA-OV. Revised Axis A:
**SAVEMem-style salience compression + retrieval (safe port, code) as primary; HERMES as the
ambitious alternative (code + only Qwen3-VL datapoint); recent-window re-prefill (DSCache's idea,
zero-risk reimplementation) as the cheap always-on component.** At most **one** position-touching
method in any configuration.

### Axis B — Perception front-end (SB Real-Time, OVO Real-Time, RIVER Live)

- **Adversarial check on [[timechat-online]] DTD:** the celebrated +5.7 zero-fine-tuning gain is on
  **VideoMME-long — an offline long-video benchmark, not SB/OVO**. Its note also warns the
  patch-correspondence assumption breaks under camera motion, and pruning is change-based, not
  relevance-based. Demote: DTD is a *latency/budget* tool, not an accuracy tool; on 80 GB with
  ≤30-min videos we may not need pruning at all for SB/OVO accuracy runs.
- **What actually moves present-tense accuracy** (per DSCache's own ablation): encoding the recent
  window cleanly at adequate resolution. So Axis B = fps/resolution/frame-budget sweeps + recent-window
  re-prefill, with pruning only if RIVER's long streams force it.
- Hard sub-tasks (OVO STU 58.4 even for Gemini; SB Counting/Text-Rich): prompt-level mitigation only.
  Cheap, worth one config each.

### Axis C — Proactive trigger (OVO Forward, SB PO, RIVER Pro-Response)

**⚠ Adversarial finding #3 — metric families were being mixed.** OVO Forward numbers in the vault
come in three incomparable protocols:

| Family | Numbers in it | Comparable to leaderboard? |
|---|---|---|
| OVO Table-1 **accuracy** | Response-G1 **58.2** · STRIDE 59.70 (trained) · STRIDE's frozen baseline **46.30** · StreamAgent 45.4 | **Yes** — this is our reporting family |
| Em-Garde's own **online F1** | Em-Garde 30.99 | No — bespoke protocol built to *criticize* the accuracy family |
| [[evostreaming]] **RealStreamEval** | Qwen3-VL 15.8 plain → 37.6 tuned | No — causal + verbosity multiplier (talk-rate ≥0.4 → down to ×0.2) + premature penalty |

Method verdicts:

| Trigger | Truly training-free? | On Qwen3-VL-8B? | Adversarial concerns | Verdict |
|---|---|---|---|---|
| Per-window polling (official protocols) | Yes | Effectively what STRIDE's 46.30 FAR / 32.40 PO baseline is | It *is* the official protocol on both OVO (dense triggering) and SB PO — but [[evostreaming]] shows it's the *protocol* rewarding this, and RealStreamEval re-scoring craters it. Report as floor, never as the story | **Day-one floor** |
| [[response-g1]] scene-graph gate | **Yes — fully frozen** | **Yes** — FAR 58.2 acc, SB PO 44.0, OVO overall 61.3 | ~825 ms/frame (1.2 fps); no code recorded; 2026 retro-dated arXiv (numbers vault-transcribed, unreproduced); open-set SGG hallucination risk; weak on causal queries | **The training-free ceiling to reproduce** — biggest single prize, biggest build |
| Em-Garde LPMM similarity-derivative | Matcher yes; **the IGPP cue-proposer is RL-trained** (−7 F1 without it) | No (2B embedder loop) | Our version = zero-shot *prompted* proposer + off-the-shelf embedder → performance unmeasured anywhere; θ=0.04 domain-sensitive; F1 numbers can't calibrate expectations | Worth building (cheap, ~13 fps, immune to cache mutation) but treat as unproven |
| PPL-drift gate ([[livestar]]/[[lyrav-dont-pause]]) | Gate yes (α only) | **No** — designed on a *streaming-instruction-tuned* InternVideo backbone | On a non-streaming-tuned Qwen3-VL, held-answer PPL may track language priors, not vision; α=1.03 backbone-tuned; also **couples to Axis A**: cache eviction events shift PPL → false fires | Deprioritize; test only after memory config is frozen |
| EOS/yes-no logit threshold | Yes (prompted token) | No published numbers | Same cache-coupling caveat; cheapest to try after polling | Quick second |
| [[streamagent]] sufficiency self-check prompt | Yes (as prompting) | No (Qwen2.5-VL 3B+7B) | Its 45.4 FAR is *below* STRIDE's frozen Qwen3-VL baseline 46.30 — the scaffold may add nothing on a stronger backbone | Cheap A/B, low expectations |

**RIVER-specific:** early answer = hard zero, and the decay constant τ is unpublished (n/r in note)
→ tune triggers conservative on a held-out slice, expect trial-and-error; OVO's lateness-only decay
wants the eager operating point. Same trigger, two thresholds, both reported.

### Axis D — Prompt & protocol layer
Survives adversarial review intact (all mechanisms benchmark-protocol-native): timestamp text echo
(SB burns them in-frame anyway; Qwen3-VL has native timestamp alignment — echo, don't re-invent),
SQA history formatting, HLD "unable to answer" guard + self-check, MCQ hygiene. One caution:
after any KV re-indexing (HERMES port), **text timestamps become the only reliable temporal signal**
— never strip them.

---

## 4. Adversarial review — across the axes (composition conflicts)

The earlier "stack best-per-axis" assumption fails in four specific places:

1. **B→A information destruction.** Query-agnostic pruning (DTD/spatial) before the cache means
   tokens a *later* query needs are unrecoverable — directly attacks OVO Backward and RIVER
   Retro-Memory, the tasks Axis A exists for. SAVEMem's own ablation is the warning shot: wrong
   gating collapses Backward-Tracing to 45.18. → Pruning enters only if memory pressure forces it,
   and always upstream-conservative.
2. **A→C signal contamination.** PPL-drift and logit-threshold triggers read model internals
   computed over the mutated cache; every eviction/re-index event perturbs PPL → phantom triggers.
   External-signal triggers (LPMM embedder, Response-G1 graphs) are immune. → If the memory config
   does aggressive eviction, the trigger must be external-signal.
3. **A→C latency coupling.** Recent-window re-prefill (our DSCache substitute) at every 1 Hz poll
   multiplies prefill cost. → Trigger polls the *cheap* signal per second; full re-prefill only on
   fire or on query arrival.
4. **Position-surgery exclusivity.** HERMES re-indexing, ReKV position-discard, pre-RoPE caching
   are mutually incompatible remappings, all unproven on interleaved-MRoPE. → **At most one**
   position-touching component per configuration, ported deliberately.

Also a protocol split that no stack removes: SB/OVO backward+realtime allow prefix (re-)encoding;
RIVER + all proactive tasks need genuinely incremental inference. The harness runs two modes;
the *components* stay the same.

**Revised composed system:**
```
frames @1fps ──► raw-frame ring buffer (recent) + SAVEMem-style salience-compressed KV/token memory (past)
                 [alt config: HERMES-ported eviction instead of SAVEMem compression — never both]
query ────────► re-prefill recent window + query-aware retrieval over compressed memory ──► frozen Qwen3-VL-8B
no query ─────► external-signal trigger (LPMM-style derivative; stretch: Response-G1 graphs)
                polling floor always measured alongside
all wrapped in Axis D prompt layer; TTFT/fps logged per config (StreamingEval-proofing)
```

---

## 5. Execution plan (revised)

**M0 — Baseline truth (≈1 wk).** Official harnesses; frozen Qwen3-VL-8B plain, both runtime modes.
Resolve the published scatter (OVO 51.77/55.81/58.00) with *our* numbers; pin fps, frames, prompts,
judges (D4), seeds. Subset-sampling mode (~15–20% stratified) for iteration. **Nothing counts before this.**

**M1 — Free points (≈3 dy).** Axis D only. Establishes the "prompting-only" row.

**M2 — Memory (≈1.5–2 wk).** Order by port risk: recent-window re-prefill (days, zero risk) →
SAVEMem port + retune (code exists) → HERMES port (code exists; re-profile bands, re-derive
re-index for interleaved-MRoPE) — SAVEMem vs HERMES is an either/or bake-off, not a stack.
Readouts: OVO Backward, **RIVER forgetting curve** (the money plot), SB overall.

**M3 — Trigger (≈1.5 wk).** Polling floor (two operating points: OVO-eager / RIVER-conservative) →
prompted yes/no logit threshold → LPMM-style with zero-shot prompted cue proposer → Response-G1
reproduction only if headroom and time allow. Readouts: OVO FAR (accuracy family only; floor ≈46.3,
training-free ceiling on record 58.2), SB PO (floor ≈32.4, ceiling on record 44.0), RIVER Loc
(anything >6.2 beats everything published).

**M4 — Composition + full runs (≈1 wk).** Best memory + external trigger + prompt layer;
one-axis-out ablations; both runtime modes; full runs. Deliverables into the SVU workspace
`artifacts/`: baseline table (protocol-labeled), forgetting curve, polled-vs-own-timing proactive gap.

**Targets (frozen Qwen3-VL-8B, honest labels):** SB-RT ≥ 81 (HERMES's 81.32 is the existing TF
mark) · OVO overall ≥ 62 (SAVEMem's 62.69 was on a weaker backbone) · OVO FAR ≥ 58 acc
(match/beat Response-G1) · RIVER Retro ≥ 60 MC + Loc > 6.2 (first external baseline) ·
Context: trained ceiling SelectStream 82.67 SB / 67.03 OVO.

## 6. Risks (updated by the review)
- **Every 2026 method is code-light and unreproduced** (several retro-dated arXiv ids; WeaveTime's
  PDF literally serves a different paper). Only HERMES + SAVEMem have recorded repos. Budget
  reimplementation time; verify repos exist before M2 ordering (D1).
- **Interleaved-MRoPE / DeepStack port failures** — the biggest technical unknown; mitigated by
  leading with position-safe methods (re-prefill, SAVEMem).
- **Protocol dominates method** on proactive tasks (46.30 vs 15.8 same model) — always co-report
  the RealStreamEval-style honest number or the baseline claim won't survive review.
- **Judge pinning** (GPT-4o on OVO Forward, Qwen2.5-72B on RIVER OE): pin versions, cache outputs.
- **Composed ≠ sum of parts** — conflicts §4; one-axis-out ablations are mandatory, not optional.

## 7. First-run decisions (user, 2026-07-17)
- **First run = M0 calibration only.** Plain frozen Qwen3-VL-8B, official harnesses, both runtime
  modes, no methods — pin our own baseline against the published scatter before anything counts.
- **OVO-Bench comes up first** (best-documented protocol, most Qwen3-VL reference points:
  51.77 / 55.81 / 58.00 overall). StreamingBench second, RIVER third.
- **M2 bake-off order: SAVEMem port first** (position-safe, code recorded), HERMES second if
  SAVEMem underdelivers or the port surfaces feature-layer issues.

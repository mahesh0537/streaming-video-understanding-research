# M1 — prompting-only layer (axis D) on frozen Qwen3-VL-8B

> **Headline: d2-hld-selfcheck takes OVO subset total 51.54 → 59.87** (+8.33) by fixing
> HLD alone (10.71 → 85.71). Every other task is bit-identical to M0. **Read the
> confound section before quoting this number.** d1 (instruction-only) failed and is
> the informative negative that motivated d2. 2026-07-20.

All rows are the same seed-17 15% stratified subset as M0, greedy decoding, so
columns are directly comparable and unchanged tasks come out bit-identical.
Layer ids and full injected text live in `src/svu/prompts/layers.py`; each run's
protocol block records the exact text.

## OVO-Bench (subset, 528 q)

| Row | EPM | ASI | HLD | **Backward** | **Realtime** | **Forward** | **Total** |
|---|---|---|---|---|---|---|---|
| M0 baseline | 48.89 | 60.87 | 10.71 | 40.16 | 61.26 | 53.20 | 51.54 |
| d1-hld-guard (embedded instruction) | 48.89 | 60.87 | 14.29 | 41.35 | 59.92 | 53.20 | 51.49 |
| **d2-hld-selfcheck (two-pass evidence gate)** | 48.89 | 60.87 | **85.71** | **65.16** | 61.26 | 53.20 | **59.87** |

## d1 verdict: FAILED (informative negative)

The uniform "choose 'Unable to answer' when evidence is missing" instruction moved HLD
by one item (+3.6 at n=28) and cost small realtime noise (STU −3.7, OCR −4.3); total
unchanged. **The HLD failure is not instruction-following — the model genuinely believes
it saw the queried event** (perception-level false positive under 64-frame sampling: the
queried object/event is plausible given what it did see). A passive instruction cannot
fire against a confident false belief. This is what forced the escalation to a separate
pass.

## d2 verdict: LARGE WIN, with a confound that must be co-reported

d2 is not a text transform but a two-pass method (the OVO runner detects it by id and
drives the gate itself). Pass 1 asks the evidence question directly — *"Does the video
contain enough visual information to determine the answer?"* → Yes/No — and a "No"
overrides the MCQ answer with the unable-option. Decoupling the evidence judgment from
the MCQ answer prior works where the passive instruction did not: **gate recall 23/28
(82.1%) on HLD, converting a below-random 10.71 into 85.71.**

**Confound (blocking for any published claim):** the gate is only applied to items
carrying an "unable to answer" option, and in this subset those are *exactly* the 28 HLD
items — gate firings on the other 500 queries: zero. Per [[findings]], HLD ground truth
is **always** "Unable to answer". So no item in this eval exists where firing the gate is
the wrong call, and the measured quantity is **gate recall on a single-label task, not
evidence discrimination**. The false-positive cost is untested, and it is the entire
risk: a gate that fires on answerable items would trade Realtime/Forward accuracy for
HLD. The +8.33 is a true official-harness number and a not-yet-earned method claim.

**Required before d2 enters the M4 composition:** a held-out probe of *answerable* items
that also carry an unable-option, measuring the gate's false-positive rate. Until that
exists, report d2 as "HLD-only, single-label task" and keep the Backward macro-average
improvement labeled as such.

## In progress (interrupted 2026-07-20 by workstation crash)

| Layer | Target | State |
|---|---|---|
| d3-spatial | OVO STU (55.56 — weakest realtime subtask) | ⏳ died at 5/178 items; resumable |
| d4-counting | SB Counting (59.07 full ctx — weakest RT subtask, and the only task that prefers full context) | ⏳ died at 1/173 items; resumable |

Both runners append to `outputs.jsonl` and resume from it, so relaunching the same
config continues rather than restarting. d4 additionally needs the SB pre-cutter
(`scripts/sb_precut.py`) since StreamingBench re-encodes every prefix.

d4's layer text is deliberately written to avoid explanation-first output: SB answer
extraction is literally `response[0]`, so any preamble scores 0 (see [[findings]]).

## Open questions for the rest of M1

1. **Does the d2 pattern generalize beyond HLD?** The two-pass evidence gate is a general
   abstention mechanism; OVO is just where the benchmark rewards it degenerately. RIVER
   OE and the long buckets are where an honest abstention test lives.
2. **Is d3/d4 worth finishing at all, or does axis D end here?** The M1 budget was ~3
   days; d1 says passive instructions don't move perception-level failures, which lowers
   the prior on d3 (spatial) and d4 (counting) — both are passive instructions. If they
   land flat, that is a clean "axis D is exhausted at 1 real win" result and M2 starts.

Raw runs: `/data/mahesh/svu-workspace/results/runs/m1-*`.
Workspace table: `/data/mahesh/svu-workspace/artifacts/m1-prompt-layer.md`.

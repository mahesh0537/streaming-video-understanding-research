# M0 adversarial findings — what execution taught us that the notes didn't know

Findings from actually running the official harnesses, 2026-07-17/18. Each one changes
how a number in this program must be read. Complements the plan's §3 adversarial review
([[Jul_17th]]).

## Published scatter resolved

None of the three published "frozen Qwen3-VL-8B" OVO numbers came from the official
harness: 58.00 is [[streamingeval]]'s memory-bounded causal protocol, 55.81 and 51.77 are
method papers' bespoke baseline rows. **Our official-harness M0 = 51.59** — the anchor
all method credit is measured against. The same holds on [[streamingbench]]: 77.31 for
this backbone is StreamingEval's harness; official shipped-code protocol gives RT-All
69.21 (subset). Protocol dominates method even at M0, exactly as the plan feared.

## OVO current edition ≠ the paper (and has no judge)

The official repo's README states the codebase is modified vs the arXiv paper and should
be preferred. Concretely: 1,640 items / 3,035 queries (vs paper's ~2,800), and **scoring
is fully rule-based — the GPT-4o judge of the paper's Forward protocol no longer exists
in the pipeline** (letter-substring MCQ, digit-extraction REC, Yes/No substring SSR/CRR).
Consequences: no API keys needed anywhere in M0; but Forward numbers are NOT the paper's
Table-1 FAR family — comparisons to Response-G1 58.2 / STRIDE 46.30 must be re-derived or
labeled cross-family.

## StreamingBench fps schedule is dead code

`model/Qwen2VL.py` parses clip duration from the cut-clip filename with
`re.findall(r'\d+', basename)[-1] - [-2]` — the trailing ".mp4" contributes a digit 4, so
the computed duration is always negative and **every clip runs the fps=1.0 branch. The
paper's stated 1/0.5/0.2 fps schedule never executes in the released code.** All shipped
open-source leaderboard rows produced with this code ran 1 fps on full prefixes. Our M0
replicates shipped behavior (labeled `fps=1.0-always (shipped)`); the paper-stated
schedule would be a separate labeled variant, not a bugfix.

## StreamingBench has no official "overall", and scoring is response[0]

- Per-category "All" = micro question-weighted (verified against the paper's Gemini row:
  75.69 = size-weighted, not the 75.01 macro). The repo ships NO cross-category
  aggregator; the paper's 67.07 "Overall" can't be bit-reproduced from public numbers.
  We define and report `overall_mcq_micro` explicitly, PO separately (timing metric).
- Answer extraction is literally `response[0]` exact-match — no regex, no substring. The
  strictest rule in any of the three benchmarks; models that prefix anything score 0.

## HLD hallucination floor is real behavior, not artifact

OVO HLD ≈ 10.7 (below 25% random): ground truth is always "Unable to answer" and the
frozen model picks a concrete option every single time (verified by inspection). The M1
axis-D guard targets ~10+ pts of Backward macro-average.

## Reproduction economics (matters for anyone running these harnesses)

- OVO: pre-chunked clips (143 GB) make inference cheap (~2 h full run on 1× H100);
  video decode dominates inference 4:1.
- StreamingBench: the official protocol re-encodes the full prefix with ffmpeg libx264
  for EVERY query and never cleans its cache — sequentially this is ~75% of wall time
  (~70 h full-run estimate). Our parallel pre-cutter (same official encoder settings,
  atomic renames, bounded look-ahead: `svu-workspace/scripts/sb_precut.py`) makes the
  full run feasible (~16-20 h). The re-encode also means models see re-compressed
  pixels, not source pixels — inherent to the official protocol.

## Subset methodology validated

Stratified 15% (seed 17, min 10/task) tracked the OVO full run within 0.1 total
(per-task drift up to ~12 pts on small tasks). Iterate on subsets, report full runs.

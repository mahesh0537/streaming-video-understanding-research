---
tags: [streaming-video-understanding, streaming-memory, synthesis]
---

# Streaming-Memory: Design-Axis Comparison

The decision matrix for [[streaming-memory]] — how 37 methods answer the same
question: *given an unbounded video stream and a frozen or lightly-trained
Video-LLM, what do you store, how do you bound it, and how do you get it back at
query time?* Rows are grouped by idea-family (consistent with the sibling
[[evolution-of-streaming-memory]], [[streaming-memory-benchmark-table]], and
[[streaming-memory-concept-graph]]). Every value is read off the per-paper deep
notes; `n/r` = not reported. **Ancestor** rows predate the streaming-Video-LLM
line but define the primitive.

## The nine axes

1. **Substrate** — what is physically held in memory: full KV (offloaded) ·
   compressed/pruned KV · distilled visual tokens · text notes · parametric
   weights · event tree/forest · memory graph · raw recoverable frames.
2. **Core op** — the memory operation: *evict/compress* (lossy shrink),
   *retrieve* (fetch a query-relevant subset), *both* (compress-then-retrieve),
   or *parametric* (write into weights).
3. **Bounded footprint** — how peak GPU state is held constant: *fixed budget*
   (lossy, constant GPU) vs *offload-all* (lossless store on CPU/disk, constant
   GPU) vs *fixed tokens/weights* (constant by construction).
4. **Query-aware write?** — is the compression/write conditioned on the question
   (*conditioned*) or done blind so the memory is reusable across turns
   (*agnostic*)? (Retrieval is almost always query-conditioned; this axis is
   about the *write*.)
5. **Training-free?** — drop-in on a frozen backbone vs SFT/RL/adapter training.
6. **Structure** — flat · short/long tiers · multi-tier cascade · layer-stratified
   · event-tree/forest · graph.
7. **Keep/evict signal** — what decides survival: cosine-redundancy · attention
   importance · recency · motion/flow · prediction-error/surprise · learned.
8. **Backbone**.
9. **Omni (audio)?**

![[streamingvlm-designspace.png]]
*The design space in one figure: [[streamingvlm]]'s four attention/cache regimes,
their asymptotic cost, and their quality — the axis-2/axis-3 tradeoff that most of
this table is navigating (source [[streamingvlm]]).*

## Decision matrix

| Paper | Substrate | Core op | Bounded footprint | Query-aware write? | Training-free? | Structure | Keep/evict signal | Backbone | Omni? |
|---|---|---|---|---|---|---|---|---|---|
| **A. Budgeted KV compression / eviction** |||||||||
| [[infinipot-v]] | compressed KV | evict/compress (continual) | fixed budget \|M\|→\|C\|, all-on-GPU (no offload) | agnostic | yes | flat + keep recent *r* frames | temporal-redundancy (Key cosine) × value-norm (L2) | Qwen2-VL-7B, LLaVA-OV-7B | no |
| [[streammem]] | compressed KV (prune+merge) | evict/compress | fixed budget *M* split per-layer | agnostic (chat-template proxy query) | yes | flat, 1 prototype/frame | proxy-token cross-attention + cosine redundancy | LLaVA-OV-7B, Qwen2-VL-7B | no |
| [[streamingtom]] | 4-bit quantized KV groups + pre-LLM token cut | both (CTR compress → OQM retrieve) | fixed per-frame quota + OQM budget | agnostic write, conditioned retrieval | yes | flat groups (group = frame) | causal cosine static/dynamic split + saliency | LLaVA-OV-7B, Qwen2.5-VL-7B | no |
| [[streamingassistant]] | pruned tokens in streaming buffer | evict/compress (pre-buffer) | compress-then-query buffer | agnostic | yes | flat buffer | temporal DTD + 4-connected spatial max-cosine (MSSAVT) | TimeChat-Online-7B | no |
| [[decouple-and-cache]] | two KV caches: cumulative past + re-encoded instant | evict/compress (decouple, pre-RoPE) | fixed window sizes | agnostic (rebuilt on demand) | yes | two-part (cumulative + instant) | recency (re-encode recent buffer) — fixes recency bias | LLaVA-OV, Qwen2.5-VL | no |
| [[omnimem]] | two KV caches: audio + visual (AVBA-split) | evict/compress | per-modality/per-layer budgets | agnostic | yes (opt. SFT) | modality-split flat | attention-importance × non-redundancy ψ=aᵏ(1−s) | video-SALMONN 2+ 8B | **yes** |
| **B. Streaming attention layouts (sink + sliding window; think-while-stream)** |||||||||
| [[streamingvlm]] | compact reusable KV: sink-text + recent-text + recent-vision | evict (sliding window, vision-first) | fixed window, O(TW) | n/a (streaming captioning) | **no** (overlapped-chunk full-attn SFT) | sink + window (StreamingLLM layout) | recency + attention-sinks; Contiguous RoPE | Qwen2.5-VL-7B | no |
| [[tays]] | dual KV cache: video cache + reasoning cache | structural split/merge (zero-copy) | sliding-window mask | n/a (streaming CoT) | **no** (SFT) | modality-split dual cache | recency (sliding window) + decoupled RoPE | Qwen2.5-VL-3B/7B | no |
| [[vst]] | short visual buffer (last *L*) + long **textual** thought memory (FIFO) | evict (FIFO) + generate thoughts | fixed token budget + FIFO text | agnostic write (pre-query thoughts) | **no** (SFT + GRPO) | short-visual / long-textual | token-capacity clip boundary + FIFO recency | Qwen2.5-VL 3B/7B/32B | no |
| **C. KV / segment retrieval** |||||||||
| [[rekv]] | **all** frame KV (tail offloaded to RAM/disk) | retrieve (top-k blocks, lossless) | offload-all, ~15K local window on GPU | agnostic store, conditioned retrieval | yes | flat blocks | internal cosine (mean-pooled attention keys) | LLaVA-OV 0.5/7/72B, Video-LLaVA, LongVA | no |
| [[livevlm]] | compressed KV (VSB) + retrievable pages | both (compress + PaR retrieve) | fixed budget, **no** CPU offload | agnostic compress, conditioned retrieve | yes | short FIFO + long-term split | vision-to-vision attention importance | LLaVA-OV-7B | no |
| [[streamkv]] | per-segment compressed KV + retrievable blocks | both (guidance-prompt compress once → question retrieve) | per-layer budget (layer-adaptive) | agnostic compress, conditioned retrieve | yes | semantic segments + per-segment summary | guidance-prompt attention + segment cosine | LLaVA-OV-7B | no |
| [[cacheflow]] | full KV offloaded to CPU + frozen-GRU block summaries | both (token-drop compress + top-k retrieve) | offload-all + fixed 196-token blocks | agnostic drop, conditioned retrieve | yes (frozen GRU) | fixed blocks + local window | per-patch cosine to prev frame + GRU-summary cosine | LLaVA-OV 0.5/7B | no |
| [[rlivs]] | recurrent FIFO visual tokens + stored clip captions | both (attention-select compress + MMR caption retrieve) | fixed FIFO (16 clips) | agnostic select, conditioned retrieve | yes | recurrent FIFO + caption store | caption→visual cross-attention (4 of 28 layers) | LLaVA-OV 7/0.5B, Qwen2.5-VL-7B | no |
| [[v-rex]] | full KV offloaded to CPU/storage + LSH cluster index | retrieve (dynamic, per-frame) | offload-all, adaptive fraction fetched | conditioned retrieval | yes | LSH hash-bit clusters | hash-bit LSH clustering + WiCSum attention-mass threshold | Llama-3-8B + SigLIP | no |
| **D. Hierarchical & event-structured memory** |||||||||
| [[flash-vstream]] | 4-granularity STAR visual tokens + 300-frame raw buffer | both (cluster/consolidate + nearest-raw retrieve) | fixed 681-token budget | agnostic write, on-query read | **no** (2-stage train) | 4 granularities (spatial/temporal/abstract/retrieved) | K-means clustering + semantic-attention momentum | CLIP ViT-L + Vicuna-7B | no |
| [[videostreaming]] | fixed-length per-clip learned memory tokens | both (streaming-encode + Gumbel-Topk select) | constant 256 tokens (V=4 clips) | conditioned selection | **no** (train Phi-2 encoder) | per-clip memories + timestamp prompt | learned pooling + question relevance | CLIP ViT-L + Phi-2 + Vicuna-7B | no |
| [[videollamb]] | fixed recurrent memory-token pool + retrieval cache | both (recurrent bridge compress + cross-attn retrieve) | fixed pool (no growth) | agnostic | **no** (train memory-bridge only) | SceneTiling segments + recurrent pool | CLS-token cosine scene segmentation | frozen enc + Vicuna-7B | no |
| [[streamchat-mem]] | tree of k-means visual clusters + text clues + dialogue mem | both (tree merge + cosine/FAISS retrieve) | short-term S=5 slots + tree | conditioned retrieval | yes | tree (hierarchical) + 3 parallel threads | optical-flow motion gate + Ebbinghaus forgetting | LongVA + CLIP-L/14 | no |
| [[videoscan]] | one carrier token/frame (mean-pool embed + KV) bank | evict (1 token/frame + cosine-pair drop) | fixed bank M=64/128 | agnostic (reusable) | **no** (2-stage LoRA) | flat carrier bank | cosine-similarity redundancy eviction | LLaVA-Video-7B | no |
| [[providellm]] | interleaved FIFO: verbalized **text** past + visual present | both (verbalize compress + FIFO evict) | single interleaved FIFO N=N_L+N_S | agnostic | **no** (2-stage, LoRA) | modality-asymmetric (text past / visual present) | online verbalization (semantic step grouping) | DINOv2 + Llama-3.2-1B/3.1-8B | no |
| [[ovg-hq-unify]] | **parametric** memory (network weights, PMB) | parametric (1 gradient step/timestep) | fixed footprint (weights) | hybrid-modal query | **no** (train + distillation) | 2 PMB instances (fusion / refinement) | test-time reconstruction loss + input-adaptive LR | CLIP feats + custom grounder | no |
| [[video-salmonn-s]] | **parametric** memory (fast weights of 2-layer MLP, TTT) | parametric (online GD) + prompt-dependent reader | fixed 16k memory tokens | conditioned read (prompt-dependent) | **no** (trained; TTT live at inference) | fast-weight memory | reconstruction + long-span prediction loss | Qwen3-VL 8B + Whisper-v3 + Q-Former | **yes** |
| [[videoscaffold]] | self-scaling multi-level event tree | structural (grow) + bottom-up consolidate/retrieve | elastic tree (grows with events) | conditioned (query-time consolidation) | **no** (trained plug-in) | 3-layer event tree | prediction-error (cosine>ε) boundary | EVA-CLIP + Vicuna-7B | no |
| [[eventmemagent]] | event-segmented STM (reservoir) + structured LTM tuples | retrieve (agentic search) + OCR/detect tools | STM cap 32, ≤32 frames at inference | conditioned (agentic search) | **no** (GRPO) | event-segmented hierarchical | histogram-Pearson boundaries + reservoir sampling | Qwen3-VL-8B + G-DINO + DeepSeek-OCR | no |
| [[fluxmem]] | 3-tier cascaded visual tokens (short/mid/long) | evict/compress on overflow (temporal+spatial) | per-tier capacity | agnostic | yes (opt. SFT) | 3 cascaded tiers | Otsu-adaptive per-frame thresholds (TAS + SDC) | Qwen2.5-VL-7B | no |
| [[tww]] | one **text** memory note per segment (persistent bank) | both (write note + implicit-attention retrieve) | compact notes (fixed footprint) | agnostic write | **no** (SFT curriculum) | segment-notes bank | segment-level streaming mask + decoupled MRoPE | Qwen3-VL 2B/4B/8B | no |
| [[weavetime]] | KV cache + coarse-to-fine Past-Current focus cache | retrieve (uncertainty-triggered C2F) | over existing cache/ReKV | conditioned (entropy-gated recall) | **no** (LoRA SOPE; plug-and-play) | past-current dynamic focus | entropy gate + late-interaction maxSim + temporal-recon objective | LLaVA-OV-7B, Qwen2-VL-7B (+ReKV) | no |
| [[streamingdvc]] *(ancestor)* | fixed-size K-means cluster-center memory | evict/compress (cluster update) | fixed K=514 | agnostic | **no** (trained) | flat cluster memory + decoding points | K-means clustering (momentum) | CLIP ViT-L + GIT / Vid2Seq | no |
| [[streamforest]] | real-time window (FSTW) + Persistent Event Memory Forest | evict/merge (event-node merge on overflow) | bounded PEMF cap (L_q=8192) | agnostic | **no** (5-stage train) | two-branch (window + event forest) | 3 merge penalties (similarity/merge-count/temporal) | SigLIP + Qwen2-7B | no |
| [[hermes-kv]] | KV cache as layer-stratified hierarchical memory | evict (layer-matched importance) | per-layer budget 4K/6K | agnostic (generic guidance pseudo-query) | yes | layer-stratified (shallow/middle/deep) | recency-decay (shallow) + attention (deep) + interp (middle) | LLaVA-OV, Qwen2.5-VL, Qwen3-VL | no |
| **E. Agentic / semantic memory** |||||||||
| [[visual-agentic-memory]] | event summaries + **raw RGB frames** + temporal/spatial index | retrieve (agent: plan → retrieve → inspect) | bounded via dedup (~0.06% frames kept) | conditioned (agent) | yes | hierarchical (moments→events), age-tiered | Otsu-adaptive cosine dedup + event boundaries | Gemini Embedding 2 + Gemini 3 Flash | no |
| [[streammeco]] | agent memory **graph** text nodes | both (connectivity-aware compress + time-decay retrieve) | compression ratio (e.g. 70%) | agnostic compress, recency retrieve | yes (plug-in over M3-Agent) | memory graph (isolated vs connected nodes) | minmax diversity + entity-importance/redundancy + time-decay | M3-Agent (plug-in) | no |
| [[savemem]] | 3-tier tokens (short FIFO / mid pruned / long spatial) | both (3-tier compress + recency-gated retrieve) | global budget B~2048, O(1) | pseudo-question probe bank + recency gate | yes | 3 tiers | semantic salience max cos(v,q-probe) + EMA recency gate | Qwen2.5-VL-7B/3B | no |
| [[cogreasoner]] | compressed visual events + retrieved history QAs (dialogue) | both (LLM-Select retrieve + dual visual compress) | dual compression (full vs 1 pooled token) | conditioned (LLM Select step) | **no** (2 LoRA adapters) | timestamp-interleaved events + dialogue | LLM relevance selection + time-based k-means events | shared LLM + 2 LoRA | no |
| [[streamvln]] *(embodied)* | SlowFast: fast sliding-window KV + slow voxel-pruned memory | both (3D voxel prune + KV reuse) | sliding window N=8 + slow mem 8×196 | agnostic | **no** (2-stage DAgger) | fast window + slow 3D memory | voxel-based 3D pruning (keep-recent-per-voxel) | LLaVA-Video-7B (Qwen2-7B) | no |
| **F. Principled selection (what to remember at all)** |||||||||
| [[streaming-model-remember]] | fixed-capacity latent memory **graph** (M evidence tokens) | budgeted: surprise-write + priority-consolidate + graph-read | 3 decoupled budgets N/B/M, O(1) | conditioned (evidence retrieval) | **no** (frozen VLM; train encoder/gates/GAR) | latent memory graph | surprise windowing (attention JS-divergence + cosine) | Qwen2.5-VL-7B / Qwen3-VL-8B | no |

![[infinipot-v.png]]
*Family A archetype: [[infinipot-v]] compresses the KV cache back to a fixed
target whenever a budget is hit — temporal-redundancy on Keys, value-norm on
Values — holding peak GPU memory constant (cut up to ~94%) with ~0.5%
compression overhead and no CPU offload (source [[infinipot-v]]).*

## Reading the axes

**1. Two substrates split the field, and training-free-ness picks the side.**
The KV-cache-native methods — compress, evict, or retrieve on the decoder's *own*
key/value cache (families A, C, and the KV rows of D) — are almost all
**training-free** ([[infinipot-v]], [[streammem]], [[rekv]], [[streamkv]],
[[hermes-kv]], [[cacheflow]], [[livevlm]], [[v-rex]], [[fluxmem]]). The moment a
method wants a *different* substrate — distilled visual tokens, text notes, an
event tree, or parametric weights — it pays for training ([[flash-vstream]],
[[videoscan]], [[providellm]], [[streamforest]], [[tww]], [[ovg-hq-unify]]). KV is
the free lunch precisely because it is already there in a frozen model; anything
richer must be learned.

**2. Training-free ⟺ query-agnostic write is the recurring pairing.** To be a
drop-in *and* survive multi-turn QA, you must compress before you know the
question — so the training-free camp invents a stand-in query: chat-template
proxy tokens ([[streammem]]), a question-agnostic guidance prompt ([[streamkv]]),
a generic pseudo-query ([[hermes-kv]]), a fixed pseudo-question probe bank
([[savemem]]). [[streammem]] quantifies what faking the query costs: chat-template
proxy 66.9 vs true-query oracle 68.1 on MLVU — ~1.2 points. That small gap is quietly
devastating for query-conditioned writes: if a free generic proxy recovers nearly all
of oracle compression quality, an elaborate query-aware write path is buying
multi-turn *inconvenience* for tiny accuracy. Query-*conditioned* compression is the
deprecated design anyway (it forces re-prefill per question, the exact failure
[[infinipot-v]] calls out in SnapKV). Trained methods sidestep this by learning the
pooling ([[videostreaming]], [[flash-vstream]]) or by only conditioning at *read* time.

**3. Compress-then-retrieve has become the default; pure eviction and pure
retrieval are the endpoints.** The two extremes are lossless offload-all retrieval
([[rekv]], [[v-rex]] — keep everything, fetch top-k) and pure budgeted eviction
([[infinipot-v]], [[hermes-kv]], [[streammem]] — shrink, never look back). Most
2025 work occupies the middle: compress to a fixed budget *and* retrieve a
query-relevant subset from what survived ([[streamingtom]], [[livevlm]],
[[streamkv]], [[rlivs]], [[savemem]], [[streamforest]]; [[cacheflow]] sits just
inside — it drops ~70–87% of tokens as redundant *before* offloading the
survivors losslessly). The bet is that redundancy-pruning and relevance-retrieval
attack orthogonal waste. But the middle ground disagrees on the lossy operator
itself: [[streammem]] finds weighted *merging* beats plain discarding (66.9 vs
65.6 on MLVU), [[video-salmonn-s]] explicitly rejects merging as over-smoothing
on very long streams and discards instead, and [[decouple-and-cache]] does
neither — it discards the recent slice and *re-encodes from raw features on
demand*. These can't all be right in general; the resolution is probably
budget-dependent — [[streamkv]]'s segmentation advantage *widens* with
compression (+1.75 at ↓50% → +5.31 at ↓90%), i.e. structure only pays under
pressure — but no paper sweeps the ratio to settle merge-vs-discard.

**4. Bounded footprint is now table stakes — the real choice is lossy-fixed-budget
vs lossless-offload.** Almost every row holds GPU state constant over an unbounded
stream; the schism is *how*. The offload camp ([[rekv]], [[v-rex]], and — after
heavy pruning — [[cacheflow]]) keeps a recoverable store on CPU/disk and pays
transfer latency; the fixed-budget camp ([[infinipot-v]], [[livevlm]],
[[hermes-kv]], [[streammem]]) keeps everything on GPU and pays lossy compression.
[[infinipot-v]] and [[livevlm]] explicitly argue *against* offload for
edge/single-GPU deployment; [[rekv]] argues *for* it to stay lossless. Parametric
memory ([[ovg-hq-unify]], [[video-salmonn-s]]) sits outside both: footprint is
constant by construction because history lives in fixed weights. The schism may
be empirically moot on today's benchmarks: [[infinipot-v]] matches the 50K full
cache at a 6K budget (MLVU 65.8 = 65.8), and [[streammem]] at 24K *surpasses*
full-KV (66.3 vs 65.9) — either the offload machinery is unnecessary, or the
evals cannot detect what lossy budgets destroy. The one month-scale datapoint
([[visual-agentic-memory]]: best systems ~17% vs human 82.5% on MM-Lifelong)
says the ceiling is far enough away that the question is unresolved.

**5. The keep/evict signal has migrated from hand-set thresholds to data-driven
ones.** Early redundancy pruning used a fixed cosine cutoff ([[streammem]] δ=0.95,
[[streamkv]] 0.99, [[cacheflow]] τ=0.5). The frontier makes the threshold itself
adaptive: [[fluxmem]] and [[visual-agentic-memory]] set it per-frame by Otsu;
[[videoscaffold]] and [[streaming-model-remember]] fire on *prediction-error /
surprise*; [[weavetime]] gates recall on answer *entropy*; [[v-rex]] replaces
fixed top-k with an attention-mass fraction. "How much to drop" is becoming a
learned or self-calibrating quantity rather than a hyperparameter. The same
statistics are also starting to do double duty across sub-topics: [[fluxmem]]
reuses its TAS compression scores as a zero-overhead when-to-speak trigger —
echoing [[timechat-online]]'s drop-ratio valleys — a sign that streaming-memory's
compression signal and [[proactive-response]]'s salience/trigger signal are
converging on the same quantity.

**6. Text-as-memory is a fast-growing camp that outsources compression to
language.** A cluster of recent methods stores history as words, not tokens:
[[providellm]] verbalizes the procedural past, [[vst]] and [[tww]] write streaming
thoughts / segment notes, [[rlivs]] and [[streamchat-mem]] retrieve over clip
captions, [[cogreasoner]] retrieves prior dialogue QAs, [[streammeco]] compresses
a graph of text nodes. Language is a lossy-but-semantic codec the LLM already
speaks — this is where much of the D/E frontier lives, and it is almost entirely
post-2025. And it has scoreboard evidence where memory is actually exercised:
[[vst]]'s textual thoughts hit OVO Backward-Tracing 56.7, +4.7 over
[[streamforest]]'s visual event trees, and [[cogreasoner]] shows retrieved text
history is robust where raw accumulation is fragile (its all-context baseline
drops −4.72 under 30% distractor turns while its retrieval holds at ~71.94).
No one has run the obvious bake-off — same backbone, same budget, KV vs token
vs text vs fast-weights substrate.

**7. Hierarchy has converged on event structure.** The progression is flat FIFO →
fixed tiers ([[flash-vstream]], [[fluxmem]], [[savemem]]) → **event trees /
forests** ([[streamforest]], [[videoscaffold]], [[eventmemagent]],
[[streamchat-mem]]) and **graphs** ([[streammeco]], [[streaming-model-remember]],
[[visual-agentic-memory]]). Online *event segmentation* — by motion, histogram
correlation, prediction-error, or surprise — is the shared new primitive that
gives these structures their nodes. [[hermes-kv]] is the odd hierarchy: it
stratifies by transformer *layer* (sensory/working/long-term) rather than by time.

**8. Parametric ("weights-as-memory") is a real but tiny camp — and no one has
made it training-free.** Only [[ovg-hq-unify]] and [[video-salmonn-s]] store
history in fast-updated weights via test-time training. It is the cleanest
fixed-footprint story on the table, yet both require training and neither is
audio-first-class beyond [[video-salmonn-s]]. Training-free parametric memory is
open whitespace.

**9. Omni-audio memory is nearly empty.** Of 37 methods only [[omnimem]]
(modality-split KV budgets) and [[video-salmonn-s]] (audio bypasses the TTT
memory) treat audio-visual streams as first-class; every other row is
vision-only, even though the streaming benchmarks increasingly score omni-source
tasks. Audio-aware eviction, audio event-trees, and audio in the retrieval
signal are all untried.

**10. The ablations vindicate recency and the read path — not the write path the
titles advertise.** The sharpest single result in the sub-topic is
[[decouple-and-cache]]'s: deleting its *instant* (recent, cleanly re-encoded)
cache collapses StreamingBench 79.12 → 74.27, while deleting the *cumulative*
long-term cache costs 0.06 points (79.12 → 79.06) — on real-time suites,
long-term memory is nearly decorative. The pattern repeats wherever gains are
split by task: [[streaming-model-remember]] improves almost only Backward-Tracing
(OVO BT 54.0 → 62.2, Real-Time flat at 82.76%); [[streamforest]]'s PEMF-vs-FIFO
gap lives on history-heavy MLVU (70.0 vs 56.7), not OVO (55.6 vs 52.9);
[[eventmemagent]]'s hierarchical-vs-fixed ablation is worth −0.59 / −0.20.
Meanwhile *read-path* ablations are huge: [[rekv]]'s internal retrieval lifts
relevant-frame recall 6.1% → 70.5% (accuracy 53.0 → 56.0, oracle 64.4 — an
8.4-point unclaimed retrieval gap); [[streamingdvc]] without its caption prefix
collapses 40.6 → 23.1 CIDEr; [[savemem]]'s always-retrieve gate collapses
Backward-Tracing to 45.18 (EMA gate: 62.69); [[streammeco]]'s compression alone
drops *below* baseline (41.2 vs 44.7) until time-decay retrieval restores the
win (45.7). Two consequences: memory papers evaluated mainly on real-time splits
are measuring their recency handling, not their memory; and the cheapest
headroom in the field is better retrieval, not cleverer eviction.

**11. RoPE surgery is an unacknowledged tenth axis.** Every KV-substrate method
must repair positions after eviction/retrieval, and no two do it the same way:
[[rekv]] re-indexes retrieved KV as consecutive tokens; [[cacheflow]] applies a
rotary phase offset to rehydrated blocks; [[decouple-and-cache]] stores KV
*pre-RoPE* and re-injects positions at use time; [[livevlm]] strips positions to
score pages, then restores them; [[streamingvlm]] left-shifts indices
(Contiguous RoPE) so they saturate in-distribution; [[tays]] and [[tww]]
decouple the input/output position axes entirely. This is not a detail —
[[streamingvlm]]'s ablation swings 25.09% → 66.18% win rate on position handling
alone. Same problem, seven mutually incompatible fixes, zero cross-comparisons:
a systematic position-repair study is the cheapest high-value paper nobody has
written.

**What nobody has combined.** Training-free + parametric memory (all parametric
rows are trained); omni-audio + event-tree/graph structure (the two omni methods
are both flat/modality-split); and query-agnostic write that *also* preserves
raw, visually-verifiable frames — only [[visual-agentic-memory]] keeps raw frames,
and it does so agentically at query time, not as a cheap streaming write. The
richest under-explored corner is a training-free, adaptive-threshold, event-
structured memory that carries audio and keeps recoverable evidence — every
ingredient exists in isolation, none are assembled together.

Four more empty cells the matrix exposes:
- **RL over the write path.** RL appears only for reading/tool-use
  ([[eventmemagent]]) and thought generation ([[vst]]); every
  forgetting/eviction policy in all 37 rows is a hand-designed heuristic.
- **Composability.** The families are orthogonal by construction (pruner →
  budget compressor → retriever → RoPE fix), yet only [[decouple-and-cache]]
  ever stacks (+[[rekv]] → 79.41, +[[infinipot-v]] → 79.60 on StreamingBench).
- **Multi-turn interference.** Only [[cogreasoner]] measures memory under
  distractor dialogue and only [[streamchat-mem]] keeps a separate dialogue
  store; multi-turn memory corruption is otherwise unmeasured.
- **The lifelong regime.** [[visual-agentic-memory]] alone runs at month scale
  (105.6 h over 51 days); everything else stops at hours — yet month-plus
  streams are exactly where fixed-budget vs offload vs parametric should
  finally diverge. Relatedly, uncertainty-*gated* reading — recall history only
  when the cheap answer is unsure — appears exactly twice ([[weavetime]],
  [[savemem]]) while the retrieval family pays retrieval latency on every query.

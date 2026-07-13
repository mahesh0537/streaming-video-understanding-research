---
zotero_key: null
authors: Ruyi Xu, Guangxuan Xiao, ..., Song Han (MIT Han Lab — StreamingLLM lineage)
year: 2025
arxiv: 2510.09608
pdf: https://arxiv.org/pdf/2510.09608
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# StreamingVLM: Real-Time Understanding for Infinite Video Streams

**Lineage role:** Ports the StreamingLLM attention-sink + sliding-window KV cache to video-language models, and — crucially — closes the train/test gap with an *overlapped-chunk full-attention* SFT that teaches the recency bias the streaming cache enforces at inference; runs stably at 8 FPS on one H100 over 2+ hour videos (ICLR 2026).

## Problem — what was limited before this paper (short)
A VLM that must caption/answer over a near-infinite live stream faces a bad menu of options. **Full attention** over the whole video is $O(T^2)$ in tokens, blows memory/latency, and degrades once the context exceeds the training length (OOM on long video). **Sliding window without overlap** bounds memory but chops the stream into chunks whose boundaries break narrative coherence and lose long-term memory. **Sliding window with overlap** restores coherence but recomputes the overlap every step ($O(TW^2)$), so latency spikes and real-time inference is impossible. Worse, even if you build a good streaming KV cache, a model trained on short clips has never seen the recency-biased attention pattern that the cache imposes, so it drifts, repeats, and cannot decide *when to speak vs. stay silent*.

## Key idea — the core insight, 2-4 sentences
Keep a compact, **reusable** KV cache with three asymmetric regions — a fixed set of **attention-sink** tokens (system + earliest text), a **long window of recent text** (long-term memory), and a **short window of recent vision** (current action) — and never recompute evicted tokens. Fix the positional drift that eviction causes with **Contiguous RoPE**: after eviction, position indices are left-shifted so retained + incoming tokens stay numerically contiguous and, once the video outgrows the window, the effective RoPE indices stop growing and remain in a bounded, in-distribution range. Then align *training* to this inference behavior with **overlapped-chunk full attention** SFT — supervise on consecutive video chunks with temporal overlap, interleaving vision/text at 1 s intervals — so the model learns the exact recency bias without ever training on prohibitively long, quadratic-cost contexts.

![[streamingvlm.png]]
> **Crux (Figure 3).** The streaming inference scheme across four rounds: a green attention-sink token is always kept, older vision tokens (yellow) are evicted first (greyed), a long recent-text window (blue) is preserved for memory, and Contiguous RoPE re-indexes so positions never run past the training length — reusing KV states instead of recomputing gives $O(TW)$ cost at coherent quality. *Xu et al. (2025), arXiv:2510.09608. Embedded for personal research reference.*

## Method + math — the mechanism, then the central objective/equations
**Streaming-aware KV cache (inference).** As frames arrive, the model reuses cached key/value states of three token groups instead of recomputing. Let $T_\text{sink}$, $T_\text{window}$, $V_\text{window}$ be the retained lengths of sink text, recent text, and recent vision. The paper's deployed sizes are $T_\text{sink}=512$ tokens (system + earliest text), $T_\text{window}=512$ recent text tokens, and $V_\text{window}=16\text{ s}$ of recent vision (the schematic in Fig. 3 uses tiny $T_\text{sink}=1, T_\text{window}=3, V_\text{window}=4$ for illustration). Eviction is asymmetric: **vision tokens are evicted first** (they are the bulk and go stale fastest), and early text is evicted only when the budget overflows. This retention keeps per-step compute lowest while preserving enough context for coherent generation, matching the quality of overlapping sliding window at a fraction of the cost.

**Cost.** For video length $T$ and window $W$: full attention is $O(T^2)$; sliding window (no overlap) $O(TW)$ but incoherent; sliding window (overlap) $O(TW^2)$; StreamingVLM (reuse KV) $O(TW)$ *and* coherent. Because states are reused, per-token latency is bounded and roughly constant as $T \to \infty$.

**Contiguous RoPE.** Standard RoPE would assign ever-growing absolute positions; after eviction the surviving tokens carry large, out-of-distribution indices. Contiguous RoPE left-shifts indices so that after each eviction the retained tokens and the newly arriving tokens are numerically contiguous with the last retained position. Once $T$ exceeds the total window size, the effective indices saturate inside a bounded range $\le$ the training length — keeping every position in-distribution. For the Qwen-VL family (3D positional embeddings for vision), a **contiguous 3D RoPE** variant left-shifts the time axis while re-assembling $(t,h,w)$ indices under the 3D rule, respecting the interleaved vision/text layout.

**Overlapped-chunk full-attention SFT (training).** To teach the recency bias cheaply, split a long stream into consecutive chunks $\{\mathcal{C}_1,\mathcal{C}_2,\dots\}$ of length $W$ frames with temporal overlap $O$ frames, $0<O<W$. Within a chunk, apply **full attention** (every token attends to all tokens in the same chunk), each chunk keeping the attention sinks and overlapping later in time. The right panel of Fig. 4 shows this overlapped full-attention supervision *approximates the effective attention pattern at inference* — sink + long recent-text window + short recent-vision window — so the model learns the intended recency bias without training on quadratic-length contexts. Deployed values: $W=24\text{ s}$, $O=12\text{ s}$. Vision and text tokens are **interleaved at 1 s intervals** (not all-vision-then-text), and **loss is computed only on text positions** aligned to per-second narration; when a second has no narration a placeholder token `"..."` is inserted so the model learns *when to speak and when to remain silent*.

**Benchmark protocol (Inf-Streams-Eval).** A new benchmark of 20 full sports games, average length 2.12 hours. Each game is split into 100 s segments, keeping segments with $\ge 200$ words of commentary as ground truth. Scoring is **pairwise win rate**: a larger judge model (GPT-5) sees the ground-truth reference and votes between two model outputs; higher win share = better commentary. Two settings: **chunk** ($\dagger$, model gets previous text + current chunk) for models that cannot do infinite inference, and **infinite** ($\infty$, model runs on the full stream, keeping its own past outputs as previous text).

## Explicit design choices — concrete decisions
- **Base model:** fine-tune Qwen2.5-VL-7B-Instruct; two-stage SFT (Step 1 = streaming pattern on 525K SFT samples + LiveCC Live-WhisperX-526K; Step 2 = 14K high-quality real-time annealing samples). Total ~**128 H100-days**.
- **Cache geometry:** $T_\text{sink}=512$, $T_\text{window}=512$, $V_\text{window}=16\text{ s}$; evict vision-first, early-text last.
- **Contiguous (3D) RoPE** so positions saturate inside the training length — the single biggest ablation lever (see results).
- **Overlapped-chunk full-attention SFT** with $W=24\text{ s}$, $O=12\text{ s}$; ≥ $2W$ words min-words filtering per sample.
- **Vision/text interleaving at 1 s** with loss only on text; `"..."` placeholder for silent seconds — teaches speak/silent timing.
- **Data pipeline (Inf-Streams-Train):** 6,000+ hrs of English sports commentary (basketball 712, soccer 544, ice hockey 402, baseball 399, American football 392) → WhisperX ASR → GPT-5 rule-based clean per 120 s segment (keep/edit/delete: 46.32% kept, 37.89% edited, 15.79% deleted) → 2,449 cleaned games (4,000+ hrs) → 525K overlapped-chunk SFT samples.
- **Annealing subset:** slice non-overlapping 16–64 s clips, internal silence ≤ 3 s, ≥ $2D$ words; keep only clips whose GPT-5-judged real-time-action share ≥ 80% → 52,530 candidates → **14,786** retained.
- **Eval as pairwise win rate** judged by GPT-5 against ground-truth commentary; chunk vs. infinite settings.

## Key results / what to remember — exact numbers with setting
**No Zotero highlights present.**

- **Captioning, Inf-Streams-Eval (infinite ∞), win rate vs. GPT-4o mini (judge GPT-5): StreamingVLM = 66.18%.** For contrast, sliding-window-with-overlap reaches 66.54% but at $O(TW^2)$ cost, while full attention is 3.89% and sliding-window-no-overlap 23.54% (Figure 1) — StreamingVLM matches the expensive overlap quality at $O(TW)$ cost.
- **Captioning, LiveCC-Sports-3K CC: 56.19%** (Table 1).
- **Efficiency:** stable real-time up to **8 FPS on a single NVIDIA H100**; per-token latency bounded/flat as video length grows (Figure 7). Full attention OOMs; overlapping sliding window shows recomputation latency spikes.
- **VQA (no VQA-specific fine-tuning; gains from streaming SFT alone, Table 3):** MVBench 67.34 → **69.16** (+1.82); LongVideoBench 54.70 → **59.00** (+4.30); OVO-Bench Realtime 56.00 → **61.96** (+5.96).
- **Ablation — Contiguous RoPE (Table 4):** native RoPE on infinite streams **25.09%** vs. contiguous RoPE **66.18%** win rate — the mechanism that makes infinite streaming work.
- **Ablation — vision window (Table 5):** 0 s → 52.90%; **16 s → 66.18%** (chosen); 32 s → 65.49% (diminishing returns).
- **Ablation — SFT strategy (Table 7):** non-overlapping chunks 62.51% vs. **overlapped 66.18%** — train/test alignment matters.
- **Ablation — data staging (Table 6):** base + Live-WhisperX-526K only 32.17% → + Inf-Streams-Train 63.46% → + annealing 66.18%.

Takeaways: (1) the streaming KV cache + Contiguous RoPE is what removes OOM/drift and gives flat latency; (2) but the *quality* comes from aligning SFT to inference (overlapped chunks) — the same cache with mismatched training or native RoPE collapses to ~25–32%; (3) streaming SFT transfers to standard VQA with no task-specific tuning.

## How it connects (evolution)
- [[streamingvlm]] is the video port of the **StreamingLLM** attention-sink idea; within this vault the closest KV-cache-for-streaming-video neighbors are [[rekv]] (training-free streaming KV — a baseline it cites as ReKV), [[infinipot-v]], [[streamkv]], and [[hermes-kv]].
- Shares the **compact/evicting memory** goal with [[flash-vstream]], [[streammem]], and [[streamforest]] (memory management for infinite streams).
- On the **online captioning / speak-timing** axis it builds on [[livecc]] (its Live-WhisperX data source) and relates to [[videollm-online]], [[mmduet]], and [[dispider]] (learning when to emit vs. stay silent).
- Its benchmark **Inf-Streams-Eval** sits alongside streaming eval efforts [[streamingbench]], [[ovo-bench]], and [[svbench]].
- Hub: part of [[streaming-memory]] under [[streaming-video-understanding]].

## Open questions / limitations
- **Domain is narrow:** all training + the Inf-Streams benchmark are English **sports** commentary — the learned recency bias and "when to speak" timing may not transfer to conversational, egocentric, or instructional streams without new data.
- **Judge-model evaluation:** headline quality is a GPT-5 pairwise win rate vs. GPT-4o mini, not a grounded factual metric — win rate can reward fluent commentary style over correctness, and both the judge and a baseline are closed models.
- **Fixed heuristic cache:** vision-first eviction with a fixed 16 s vision window is hand-tuned; there is no content-adaptive retention, so a slow-developing event outside the window is unrecoverable (no long-range vision memory).
- **Cost/base coupling:** ~128 H100-days on a specific 7B base; unclear how the train/test-alignment recipe scales to other backbones or smaller compute.

*Verification: numbers checked against the arXiv HTML full text and the paper's own Tables 1/3/4/5/6/7 and Figures 1/3/4/7 (win rates 66.18/66.54/23.54/3.89, RoPE 25.09 vs 66.18, VQA deltas, 8 FPS/H100, W=24s/O=12s, cache 512/512/16s, 14,786 annealing samples, 128 H100-days); crux figures cropped from the downloaded PDF (pages rendered directly).*

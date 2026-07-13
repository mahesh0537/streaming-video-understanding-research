---
zotero_key: null
authors: Linli Yao et al. (TimeChat lineage; ACM MM 2025)
year: 2025
arxiv: 2504.17343
pdf: https://arxiv.org/pdf/2504.17343
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# TimeChat-Online: 80% Visual Tokens are Naturally Redundant in Streaming Videos

**Lineage role:** the visual-change/redundancy-trigger approach to proactive response — a purely vision-based Differential Token Drop prunes ~80% of redundant streaming tokens *before* the LLM, and the same drop-ratio signal doubles as a training-free trigger: valleys (low drop ratio = scene change) mark when to proactively speak.

## Problem — what was limited before this paper (short)
Streaming video-LLMs face two coupled costs. (1) **Redundancy:** at the dense frame rates streaming needs (~1 FPS), consecutive frames are almost identical, so feeding every frame's tokens to the LLM wastes most of the compute on visually-repeated content — and prior token-pruning methods run *inside* the LLM (post-encoding) or re-process history. (2) **Proactive timing:** a streaming assistant must answer at *any* timestamp, including future-oriented questions ("what does he do next?"), which requires deciding *when* enough new visual evidence has arrived to respond — most prior systems either answer only reactively or need a trained decision head.

## Key idea — the core insight, 2-4 sentences
Successive streaming frames are naturally redundant, so instead of pruning inside the LLM, TimeChat-Online drops temporally-redundant visual tokens **right after the ViT encoder, before the LLM**, using a training-free Differential Token Drop (DTD) that compares each patch/token to its spatially-aligned counterpart in the previous frame. Dropped tokens keep their original 3D (temporal, height, width) M-RoPE positions so spatial-temporal reasoning is undisturbed. Crucially, the per-frame **drop ratio becomes a free scene-change signal**: valleys (many *new* tokens = low drop ratio) mark scene transitions, which serve as "Trigger Times" for proactive responding — no separate decision module needed.

![[timechat-online.png]]
> **Crux (Figure 2).** DTD in three steps — patchify+encode dense frames, compute static redundancy between temporally-consecutive spatially-identical tokens, drop the redundant ones while reserving M-RoPE (t,h,w) positions (~82.5% dropped) — and (bottom) the drop-ratio curve over the timeline whose valleys are the proactive Trigger Times. *Yao et al. (2025), arXiv:2504.17343. Embedded for personal research reference.*

## Method + math — the mechanism, then the central objective/equations IN FULL

**Pipeline.** Each frame $f_t$ is patchified into $\mathcal{P}_t=[p_1,\dots,p_{H\times W}]$ and ViT-encoded into visual tokens $\mathcal{V}_t=[v_1,\dots,v_{H\times W}]$. DTD then compares frame $t$ against the previous frame $t{-}1$ at each spatial location $(h,w)$ and drops redundant tokens before the LLM. Two redundancy tests (either can drive dropping; feature-level is the default):

**Pixel-level redundancy** (L1 distance of temporally-consecutive, spatially-identical patches, inspired by RLT):
$$\mathrm{Sim}(p^{t-1}_{hw},\,p^{t}_{hw}) \;=\; \big\lVert p^{t-1}_{hw}-p^{t}_{hw}\big\rVert_1 \;<\; \tau_{\text{pixel}}$$
A small L1 distance ⇒ visually near-identical ⇒ mark redundant. Because the ViT keeps a one-to-one patch↔token correspondence, a dropped *patch* maps directly to the token to drop.

**Feature-level redundancy** (cosine similarity of the spatially-aligned visual tokens):
$$\mathrm{Sim}(v^{t-1}_{hw},\,v^{t}_{hw}) \;=\; \frac{v^{t-1}_{hw}\cdot v^{t}_{hw}}{\big\lVert v^{t-1}_{hw}\big\rVert_2\,\big\lVert v^{t}_{hw}\big\rVert_2} \;>\; \tau_{\text{feat}}$$
High cosine ⇒ redundant. Both operate *before* the LLM (vision-only), unlike feature-level pruning methods that act inside the LLM — hence greater efficiency.

**Position-aware token dropping.** Redundant locations in the current frame form a binary keep-mask $M^t_{\text{drop}}$. Retained tokens and their M-RoPE positions are selected by element-wise masking, so the reserved tokens keep the vanilla 3D positions computed **before** dropping:
$$\widetilde{\mathcal{V}}_t=\mathcal{V}_t\circ M^t_{\text{drop}},\qquad \widetilde{\mathcal{P}}_t=\mathcal{P}_t\circ M^t_{\text{drop}}$$
M-RoPE indexes each token by $p=\{t,h,w\}$; reserving these positions avoids disrupting fine-grained spatial/temporal tasks (spatial localization, action recognition). The operation is parallelized over all consecutive frame pairs in $\mathcal{F}_{1:t}$ to yield the dropped stream $\widetilde{\mathcal{V}}_{1:t}$; because it only inspects the newest incoming frame, it never re-processes history (streaming-friendly).

**Threshold ⇒ drop ratio.** A single $\tau$ controls the overall drop ratio and is empirically *consistent across datasets*: $\tau_{\text{feat}}=0.25$ ⇒ ~85% tokens dropped (across three datasets), $\tau_{\text{feat}}=0.5$ ⇒ ~45% dropped. During training the report uses $\tau_{\text{feat}}=0.7$.

**Proactive response via drop-ratio valleys (the eval/inference protocol for future-responding).** For future-oriented questions the model must pick *when* to speak. TimeChat-Online defines **Trigger Times** as timestamps where the video transitions to a new scene — which show up as **valleys** (locally low drop ratio, i.e. many new/non-redundant tokens) in the drop-ratio-vs-timeline curve. The hard drop-ratio threshold for declaring a Trigger Time is 60%. At each trigger the model answers using only the most recent slimmed context; trained on 20K **negative/"unanswerable"** samples, it learns to reply "unanswerable" (i.e. *wait for the next trigger*) when current evidence is insufficient. This makes proactive timing a training-free by-product of DTD plus a learned unanswerable behavior, rather than a dedicated decision head.

**Streaming memory.** After dropping, the most-recent ≤6K slimmed tokens are held in a first-in-first-out memory bank (≤2 s response latency). The paper stresses the memory design is *not* its contribution and is replaceable by memory- or KV-cache-retrieval streaming methods; DTD is orthogonal and complementary to them.

## Explicit design choices
- **Prune before the LLM, not inside it:** DTD operates on ViT outputs / raw patches (vision-only), so the LLM never sees redundant tokens — cheaper than in-LLM feature pruning (VisionZip etc.).
- **Two interchangeable redundancy signals:** pixel-level (L1 of patches) and feature-level (cosine of tokens); **feature-level is the default** for all experiments (consistently better) with $\tau_{\text{feat}}=0.25/0.5$ for ~85%/45% drop.
- **Video-aware (not frame-aware) dropping:** compare against the running previous kept frame across the whole video rather than resetting per frame — beats frame-aware in the ablation.
- **M-RoPE position reservation:** compute (t,h,w) positions *before* dropping and carry them on the survivors, so dropping doesn't corrupt spatial/temporal geometry.
- **Backbone:** long-context Qwen2.5-VL-7B; vision encoder frozen; full-parameter projector + LLM fine-tuned 1 epoch on 8×A800 80GB, batch 128, lr 1e-5, 448×448 input, max 64 frames for training.
- **Training mix:** TimeChat-Online-139K (streaming) + LLaVA-Video-178K (100K subset) + Tarsier2 (100K) + VideoChat-Flash (3K); DTD applied with 50% probability per training batch (so the model sees both dropped and full inputs).
- **Inference:** 1 FPS dense frames, max frame length extended to 1016, ~85% tokens dropped, FIFO memory bank ≤6K tokens.
- **TimeChat-Online-139K dataset:** 11,043 visually-informative videos (avg 11.1 min, 87.8 key frames each, ~7.14 s between key frames), scene-oriented dense captions (~176 words) via GPT-4o; 139K QA across three streaming task families — **Backward Tracing, Real-Time Visual Perception, Forward Active Responding** — plus 20K negative "unanswerable" samples. Pipeline: source long videos → PySceneDetect scene cuts → key frames → drop DINO-v2-redundant frames → keep videos with >5 distinct scenes → GPT-4o dense captions → GPT-4o streaming QA.

## Key results / what to remember
Backbone throughout is Qwen2.5-VL-7B; "↓X%" = percent of video tokens dropped. Numbers verified against the paper's Tables 1–4.

- **StreamingBench — Real-Time Visual Understanding (Table 1):** TimeChat-Online-7B at 1 FPS scores **All = 75.28 at ↓44.2% tokens** and **73.64 at ↓82.6% tokens**, vs its own full-token setting **75.36 (100%)** and the Qwen2.5-VL-7B baseline **73.68**. So it *matches* full-token Qwen2.5-VL while dropping ~83% of tokens, and beats online baselines Dispider-7B (67.63), VideoLLM-online-8B (35.99), Flash-VStream-7B (23.23). Its 75.28 is on par with proprietary models in the same table (GPT-4o 75.36, Gemini-1.5-Pro 75.69) and above Claude-3.5-Sonnet (72.44) — while using only ~56% of the tokens.
- **OVO-Bench (Table 2):** Overall **45.6 at ↓84.8%** (+12.4 over Flash-VStream-7B's 33.2) and **47.6 at ↓44.6%** (+14.4); full-token = 46.7. Strong on Forward Active Responding subtasks.
- **Offline long-video (Table 3), TimeChat-Online-7B at ↓85.0%:** MLVU **65.4**, LongVideoBench **57.7**, VideoMME overall **62.5** (long split 49.2); at ↓46.3%: MLVU 62.9, LongVideoBench 57.1, VideoMME overall **63.3** (long 52.4); full-token: MLVU 62.6, LongVideoBench 55.4, VideoMME overall 62.4.
- **Training-free DTD on vanilla Qwen2.5-VL-7B (Table 3):** applying DTD *without any fine-tuning* at ↓84.6% actually *raises* VideoMME long from 50.4 → **56.1** and overall 63.2 → 64.9 — evidence that dropping redundant tokens can *help*, not just save compute.
- **Ablation (Table 4, StreamingBench, no SFT):** vs 22K-token vanilla (73.7 = 100%): at ↓44.1% (12K tokens) feature-level video-aware drop retains **99.6%** (73.4) vs VisionZip 98.1% (72.3); at ↓82.5% (4K tokens) feature-level video-aware retains **97.7%** (72.0) vs VisionZip 93.4% (68.8). Feature-level > pixel-level; video-aware > frame-aware.
- **Efficiency:** the headline framing — "**82.8% reduction in video tokens while maintaining ~98% performance**" ⇒ over 80% of streaming visual content is naturally redundant; ~**1.76× inference speedup** at ~81% drop ratio.

No Zotero highlights present.

Takeaways: (1) redundancy removal and proactive timing are the *same* signal — the drop ratio; (2) pruning *before* the LLM with position reservation is a clean, backbone-agnostic, training-free front-end that composes with memory/KV-cache streaming methods; (3) dropping tokens can improve, not just cheapen, long-video accuracy.

## How it connects (evolution)
- [[dispider]] — a strong online video-LLM baseline TimeChat-Online beats on StreamingBench (67.63 → 75.28); contrast its explicit decision module with DTD's drop-ratio trigger.
- [[videollm-online]] — the online streaming-dialogue paradigm and a baseline here (VideoLLM-online-8B 35.99); TimeChat-Online is the redundancy-pruning + proactive-trigger successor.
- [[mmduet]] — another proactive/dense per-frame response line; both address *when to speak*, MMDuet via a response head vs DTD via scene-transition valleys.
- [[streammind]] — proactive speaking driven by a change/perception signal; closely comparable "visual-change trigger" philosophy.
- [[vispeak]] — visual-trigger proactive response, a sibling in the [[proactive-response]] cluster.
- [[streamingbench]] — the primary evaluation benchmark for real-time visual understanding used throughout.

## Open questions / limitations
- **Static-redundancy assumption:** DTD keys on spatially-aligned pixel/feature similarity, so it presumes a roughly stationary camera/grid; heavy camera motion or global scene shift could break the one-to-one patch correspondence and inflate the drop budget with non-redundant tokens.
- **Trigger-time recall vs precision:** using drop-ratio valleys as the *only* scene-transition signal ties proactive timing to visual change — slow/gradual transitions or semantically-important-but-visually-static moments may not register as valleys (missed triggers).
- **Single global $\tau$:** one threshold controls drop ratio; the "consistent across datasets" claim is empirical and may not transfer to very different domains (egocentric, synthetic, screen content).
- **Memory bank is deliberately simple:** the ≤6K FIFO bank caps long-horizon backward tracing; the paper explicitly defers real memory design to complementary methods, so long-range QA quality depends on the plug-in memory, not measured here in depth.

*Verification: equations (1)–(3) and the DTD mechanism transcribed from the method pages of arXiv:2504.17343; all headline numbers cross-checked against rendered Tables 1 (StreamingBench), 2 (OVO-Bench), 3 (offline long-video), and 4 (ablation) of the PDF; dataset statistics from the Section 3.3 / implementation pages. Crux figure is Figure 2 (page 3).*

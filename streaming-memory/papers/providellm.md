---
zotero_key: null
authors: Dibyadip Chatterjee et al. (Meta Reality Labs; FAIR, Meta; National University of Singapore)
year: 2025
arxiv: 2504.13915
pdf: https://arxiv.org/pdf/2504.13915
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# ProVideLLM: Memory-efficient Streaming VideoLLMs for Real-time Procedural Video Understanding

**Lineage role:** The interleaved-multimodal-memory design point for streaming procedural video — one FIFO cache mixing compressed *text* tokens (verbalized long-term past) with fine-grained *visual* tokens (short-term present), giving ~22x token reduction and 2 GB, 10-25 FPS streaming (ICCV 2025).

## Problem — what was limited before this paper (short)
Procedural videos (cooking, assembly, furniture building) run from minutes to hours and are *causal*: each step depends on earlier ones, so a streaming assistant must remember a long past while reacting online with no access to future frames. Prior streaming VideoLLMs cache raw per-frame visual tokens, so token count and KV-cache memory blow up with video length, triggering long-context degradation and making per-frame streaming infeasible. LSTR-style long/short-term visual memories still use only visual tokens (redundant and expensive), and language-aligned encoders (CLIP/SigLIP) suffer "temporal collapse" — nearly identical features for distinct fine-grained steps — plus a bias toward background over the hand-object interactions that actually define a procedural step.

## Key idea — the core insight, 2-4 sentences
Represent the *long-term* past not as visual tokens but as **verbalized text**: the LLM decoder continuously predicts step labels online, groups semantically similar consecutive actions, and stores the compressed language summary (e.g. "get -> crack -> whisk eggs") in the cache — language is a dense, information-preserving encoding of step structure. Keep only the *short-term* present (~8-16 s) as uncompressed fine-grained visual tokens from a DINOv2 + **DETR-QFormer** connector that focuses on hands and objects-in-contact. Both token types live in a single **multimodal interleaved cache** — one FIFO queue, one entry, two exits — so no separate caches and no redundant attention recomputation, yielding sub-linear memory/compute scaling with video length.

![[providellm.png]]
> **Crux (Figure 1).** The interleaved cache feeding one LLM: green `<L>...` entries are verbalized text tokens summarizing the long-term past; yellow `<v>` are fine-grained visual tokens of the recent present from DINOv2 + DETR-QFormer; the same model answers "what's my goal / what am I doing / what next" for multiple procedural tasks. *Chatterjee et al. (2025), arXiv:2504.13915. Embedded for personal research reference.*

## Method + math — mechanism, then the equations
**Architecture (LLaVA-style).** Visual encoder $E_V$ = DINOv2 (ViT-S for the 1B config, ViT-L for the 8B); vision-language connector $C_{VL}$ = the novel DETR-QFormer; language decoder $D_L$ = Llama-3.2-1B or Llama-3.1-8B. Configs are written `ProVideLLM-{1B,8B}/{m}` where $m$ is the number of visual tokens kept per short-term frame (e.g. `-8B/11`). Streaming output at timestep $t$ over frames $V=\{v_0,\dots,v_t\}$:

$$o_t = \text{ProVideLLM}(v_{0:t}) = D_L\!\left(\;\big\|_{i=0}^{t}\; C_{VL}\big(E_V(v_i)\big)\right)$$

where $\|$ is concatenation over frames. K-V caching makes this $O(N)$ per frame.

**Long-term = online verbalization (Sec. 3.2).** Instead of building expensive step graphs, the decoder emits step-label predictions on the fly and merges semantically similar consecutive actions into a text summary that is written back into the cache as language tokens. On Ego4D Goal-Step, one hour of long-term past is stored in **~630 verbalized text tokens** on average — a **22x** token reduction vs. long/short-term visual caching at the same sampling rate.

**Short-term = fine-grained visual tokens (Sec. 3.3).** DINOv2 (not CLIP/SigLIP) is chosen because language-aligned encoders exhibit *temporal collapse* — low per-class temporal variance conflates distinct steps (Fig. 3a); DINOv2 preserves temporal variation and localizes hands better (PCA, Fig. 3b). The **DETR-QFormer** connector is a cross-attention transformer decoder over the $16^2$ DINOv2 patch tokens with three query groups: $m$ learnable visual queries $Q_v$ (compress patches to $m\times d$ tokens), hand queries $Q_h$ (left/right), and object queries $Q_o$ (objects-in-contact). The hand/object queries additionally regress bounding boxes, forcing the compressed tokens onto hand-object interactions and away from background distractors.

**Multimodal interleaved cache (Sec. 3.4).** A single FIFO queue holds interleaved text (`<L>`) and visual (`<v>`) tokens with capacity $N = N_L + N_S$ (long-term budget + short-term budget). `Entry` appends new tokens; `Exit_short` evicts short-term tokens beyond $N_S$; on eviction the short-term visual span is *verbalized* and re-inserted as a long-term text entry (prepended `<L>`), and `Exit_long` evicts text beyond $N_L$. Because both modalities share one cache, causal attention is computed once (no re-running attention over a separate long-term cache). Fig. 2 contrasts three strategies: (a.1) visual-only progressive cache — $O(1)$ conversion but unbounded tokens for long video; (a.2) online verbalization alone — $O(N_S^2 + N_L)$ conversion cost, unstable for streaming; (b) **interleaving** — reduces conversion to $O(N)$ and enables streaming on long videos. Since $N_L + N_S$ grows sub-linearly with frames, per-frame cost stays low.

**Training (Sec. 3.5).** *Stage-1 (alignment):* freeze $E_V$ and $D_L$, train only DETR-QFormer with a captioning LM loss plus a hand-object box loss.

$$\mathcal{L}_{LM} = -\frac{1}{|W|}\sum_{i=1}^{|W|} \log P\big(w_i \mid w_{<i}, V;\, C_{VL}^{\theta}\big)$$

$$\mathcal{L}_{HO}\big(b_i, \hat{b}_{\sigma(i)}\big) = \mathcal{L}_{iou}\big(b_i, \hat{b}_{\sigma(i)}\big) + \big\|\,b_i - \hat{b}_{\sigma(i)}\,\big\|_1$$

$$\mathcal{L} = \mathcal{L}_{LM} + \lambda_1\, \mathcal{L}_{HO}$$

where $b_i$ are pseudo ground-truth boxes and $\hat{b}_{\sigma(i)}$ the matched predictions. *Stage-2 (instruction tuning):* fine-tune DETR-QFormer + decoder via LoRA ($r=128$, $\alpha=256$) with $E_V$ frozen, using a special `<L>` marker so the decoder distinguishes long-term text entries from short-term visual tokens.

## Explicit design choices
- **Two token types in one cache**: verbalized text (`<L>`, long-term) + raw visual (`<v>`, short-term) — asymmetric compression matched to how much fidelity each horizon needs.
- **Online verbalization by the model itself** — the decoder's own step predictions become the long-term memory; semantically similar consecutive actions are grouped before storing.
- **Single interleaved FIFO cache** (one entry, two exits: short and long) instead of two separate caches — kills redundant attention recomputation, gives $O(N)$ per-frame conversion.
- **DINOv2 over CLIP/SigLIP** to avoid temporal collapse on fine-grained steps.
- **DETR-QFormer connector**: learnable visual queries + hand + object queries with a box-regression aux loss to focus tokens on hand-object interactions; compresses $16^2$ patches to a small $m$ (e.g. 5 or 11) tokens/frame.
- **Two-stage training**: Stage-1 connector alignment (frozen encoder+decoder, LM + $\mathcal{L}_{HO}$); Stage-2 LoRA instruction tuning of connector+decoder.
- **Small decoders on purpose** (Llama-3.2-1B / 3.1-8B) to hit real-time FPS at low memory.
- **Single model, six procedural tasks** (online step detection, step forecasting, step recognition, task recognition, long-term forecasting, multi-task) across Ego4D Goal-Step, EgoExo4D, Assembly101, COIN.

## Key results / what to remember
Numbers verified against the HTML tables (paper's own tables); config `-{size}/{m}`.
- **Efficiency (headline):** ~**22x** fewer tokens than long/short-term visual caching to represent one hour of long-term past (~630 text tokens for 1 h on Ego4D Goal-Step); **~2 GB** GPU for the 1B config.
- **Runtime (Table 9, A6000):** ProVideLLM-1B/5 = **10.4 FPS** per-frame streaming @ 2.0 GB, **24.6 FPS** streaming dialogue @ 2.2 GB; ProVideLLM-8B/11 = 4.2 FPS per-frame @ 16.2 GB, 17.2 FPS dialogue @ 16.9 GB. Connector is the per-frame bottleneck.
- **Online step detection — Ego4D Goal-Step (Table 1, per-frame mAP, test):** 12.9% with verbalization+interleaving (12.2% without) vs. LSTR 8.1%, EgoOnly 10.9%.
- **Step forecasting — Assembly101 (Table 2):** 13.8% class-mean Top-5 recall (8B/11) vs. 12.0% baseline.
- **Next-step forecasting — COIN (Table 3, Top-1):** 53.6% (8B/11+, with long-term verbalization) vs. 52.1% without, VideoLLM-MoD 49.7%, VideoLLM-online 49.1%.
- **Fine-grained keystep recognition — EgoExo4D (Table 4, ego acc, test):** 50.74% (8B/11), reported as a **9.2%** absolute gain over the second-best; VideoLLM-MoD 42.62% (val), View-Invariant Encoder 41.53% (test).
- **Step recognition — COIN (Table 5, Top-1):** 67.3% (8B/11+) vs. VideoLLM-MoD 63.4%, VideoLLM-online 63.1%.
- **Multi-task on COIN (Table 6, 8B/11+):** step 66.9%, task 95.0%, next-step 50.5%, long-term forecast 51.0%, long-term+task 55.9% — beats VideoLLM-MoD / -online on all five.
- **Cross-dataset generalization (Table 7, EgoExo4D keystep, trained on Ego4D Goal-Step):** 12.4% vs. LLaVA-OneVision+Llama-3.1-8B 8.3%, VideoLLaVA 3.6%.
- **Connector ablation (Table 8, EgoExo4D ego acc):** DINOv2+DETR-QFormer 40.7% > DINOv2+QFormer 38.1% > DINOv2+MLP 32.4% > SigLIP+QFormer 33.2% > SigLIP+MLP 28.7% (17 tok/frame for QFormer variants, 5 for MLP).

No Zotero highlights present.

Takeaways: (1) the memory win comes from *modality-asymmetric* compression — text for the distant past, pixels for the present — not from dropping frames; (2) verbalization only becomes streaming-viable once interleaved into one cache ($O(N)$); (3) a self-supervised, hand-object-focused encoder (DINOv2 + DETR-QFormer) matters more than a language-aligned one for fine-grained procedural steps.

## How it connects (evolution)
- [[videollm-online]] — the online-decoding streaming-dialogue paradigm ProVideLLM builds on; VideoLLM-online/-MoD are its main baselines.
- [[mmduet]] — another streaming VideoLLM baseline for dense online response; contrasts in memory strategy.
- [[dispider]] — streaming perception/decision decoupling; related design axis for real-time online video LLMs.
- [[flash-vstream]] and [[rekv]] — memory-compression approaches for long streaming video; ProVideLLM's text-verbalization is an alternative compression to their visual-memory schemes.
- [[timechat-online]] — streaming temporal grounding; adjacent online procedural-understanding line.
- [[streamingbench]] — the kind of streaming benchmark on which such memory-efficient online models are measured.

## Open questions / limitations
- Verbalized long-term memory is only as good as the decoder's own online step predictions — errors are baked into the cache and can compound over an hour-long video; the paper does not quantify long-horizon error accumulation from bad verbalizations.
- DETR-QFormer relies on pseudo ground-truth hand/object boxes for the Stage-1 loss; quality/availability of those pseudo-labels bounds the fine-grained gains and may not transfer beyond egocentric hand-object domains.
- Gains are shown on structured procedural datasets (cooking/assembly/instructional); generalization to open-domain streaming video or non-procedural tasks is untested.
- The 8B config still needs ~16 GB and runs at ~4 FPS per-frame — the truly cheap 2 GB / 10 FPS regime is the small 1B model, whose accuracy trails the 8B.

*Verification: equations (Eq. 1 output, LM loss, $\mathcal{L}_{HO}$, combined $\mathcal{L}$) and all headline numbers checked against the arXiv HTML (arxiv.org/html/2504.13915) tables/abstract; title, authors and affiliations confirmed from the rendered PDF first page; crux figure cropped from the PDF (Figure 1, p.1).*

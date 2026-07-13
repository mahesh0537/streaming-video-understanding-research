---
zotero_key: null
authors: Guangzhi Sun, Yixuan Li, Yudong Yang, Chao Zhang (Tsinghua University · ByteDance · University of Cambridge)
year: 2026
arxiv: 2606.07577
pdf: https://arxiv.org/pdf/2606.07577
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# OmniMem: Perturbation-aware Memory Compression for Streaming Audio-Visual LLMs

**Lineage role:** Jun-2026 frontier point in streaming KV-cache memory compression that makes the compressor *modality-aware* — it splits the audio and visual KV budgets and scores tokens by a perturbation-driven attention×redundancy criterion, targeting audio-visual (omni) LLMs rather than vision-only streams.

## Problem — what was limited before this paper (short)
Audio-visual LLMs choke on long video: the visual token stream and its KV cache grow linearly with duration, so hour-long clips blow past memory limits. Prior KV-cache compressors ([[infinipot-v]], [[streammem]], HERMES) treat every token uniformly with a single shared budget. That ignores modality imbalance — visual tokens outnumber audio tokens by 30–40×, so a uniform budget starves the rare-but-important audio tokens (e.g. human speech, which is *less* compressible than visual redundancy), degrading exactly the omni-source questions that need both streams.

## Key idea — the core insight, 2-4 sentences
Compress the streaming KV cache with two modality-aware moves. First, keep tokens by a *perturbation-driven* score that multiplies accumulated attention importance by non-redundancy (low cosine similarity to neighbors), so a retained token must be both frequently attended and not a near-duplicate. Second, allocate *separate* memory budgets to the audio and visual caches via Audio-Visual Budget Allocation (AVBA), sizing each modality's (and each layer's) budget by its measured compressibility — entropy of attention × redundancy. An optional budget-aware fine-tune consolidates information under the tight budget.

![[omnimem.png]]
> **Crux (Figure 1).** Chunk-by-chunk streaming: at each layer, MHSA runs over the past KV cache plus the new chunk, then the cache is pruned to a fixed size by attention score and similarity, with audio and visual positions stored in two separate caches whose sizes come from AVBA. *Sun et al. (2026), arXiv:2606.07577. Embedded for personal research reference.*

## Method + math — the mechanism, then the objective in full

**Streaming setup.** Video is processed chunk by chunk. For chunk $t$, multi-head self-attention at each layer $l$ attends over the retained past KV cache concatenated with the current chunk's new K/V pairs. After the chunk, the cache is updated with the new pairs and then *pruned back down to a fixed size* — so memory stays bounded regardless of video length. Audio-position and visual-position KV pairs are kept in **two separate caches**.

**Perturbation-aware selection (Sec. 3.1).** Each candidate key $k$ gets an importance score = accumulated attention mass it receives from all queries:
$$a_k = \sum_{q \in |Q|} A_{q,k}$$
and a redundancy score = mean cosine similarity to its neighbors (computed on value/hidden vectors):
$$s_k = \tfrac{1}{2}\big[\cos(H_k, H_{k+1}) + \cos(H_{k-1}, H_k)\big].$$
The selection score combines them **multiplicatively**:
$$\psi_k = a_k^{\lambda}\,(1 - s_k),$$
and the top-$K$ tokens by $\psi_k$ are retained. The exponent $\lambda$ (default $0.02$) is a normalization factor because the dynamic ranges of $a_k$ and $s_k$ differ sharply. The multiplicative form means a token survives only if it is *both* heavily used ($a_k$ high) *and* non-redundant ($s_k$ low) — an additive score would let one term dominate. "Perturbation-driven" = redundant tokens (high similarity to neighbors) perturb the representation little when dropped, so they are the safe ones to evict.

**Audio-Visual Budget Allocation (Sec. 3.2).** Define per-modality compressibility scores from the entropy $H$ of the normalized attention distribution and the average similarity:
$$\mathcal{C}_v = \frac{H[\bar a_v]\,(1-\bar s_v)}{\log K_v}, \qquad \mathcal{C}_a = \frac{H[\bar a_a]\,(1-\bar s_a)}{\log K_a},$$
where $\bar a_v,\bar a_a$ are the normalized attention probabilities over visual/audio positions, $\bar s_v,\bar s_a$ their average cosine similarities, and $K_v,K_a$ the pre-eviction token counts (used to normalize entropy). High entropy + low redundancy ⇒ hard to compress ⇒ deserves more budget. A fixed prior visual-to-audio ratio $r=5$ anchors the split, and the actual proportions float around it:
$$w_v = \frac{\mathcal{C}_v\, r}{\mathcal{C}_v\, r + \mathcal{C}_a}, \qquad w_a = \frac{\mathcal{C}_a}{\mathcal{C}_v\, r + \mathcal{C}_a}.$$
Budget is also distributed **across layers** by a temperature-softmax over each layer's compressibility:
$$\mathcal{C}^{(l)} = H[\bar a^{(l)}]\,(1-\bar s^{(l)}), \qquad \mathcal{B}^{(l)} = \mathrm{Softmax}\!\big(\mathcal{C}^{(l')}/T\big)[l],$$
with temperature $T=0.2$ controlling how much the per-layer budget is allowed to vary (small $T$ ⇒ sharper concentration on high-compressibility-need layers).

**Budget-aware fine-tuning (Sec. 3.3, optional).** Same chunked processing, but gradients are truncated at a middle layer (layer 18 for video-SALMONN 2+) for the intermediate chunks and only flow through all layers on the final chunk — cutting training cost while still shaping deep-layer behavior. Selection during fine-tuning uses cosine-similarity-only (cheaper) rather than the full $\psi_k$. This teaches the model to consolidate information into the surviving tokens under the tight budget; it is a strict add-on to the training-free default.

## Explicit design choices
- **Two separate KV caches**, one for audio positions and one for visual positions, each with its own AVBA-assigned budget — the central departure from uniform single-cache compressors.
- **Multiplicative** attention×non-redundancy score $\psi_k = a_k^{\lambda}(1-s_k)$, not additive weighting — both conditions must hold.
- **Similarity from value/hidden vectors**, importance from accumulated attention — reusing quantities already computed, so the method is cheap and **training-free by default**.
- **Fixed visual:audio prior $r=5$**; budgets float around it via compressibility, rather than a hard 5:1 split.
- **Per-layer budgets** via temperature-softmax ($T=0.2$) over layer compressibility — later layers (lower entropy, high redundancy for visual) get less.
- **Chunk-by-chunk streaming** with fixed post-chunk cache size — bounded memory for arbitrarily long video.
- **Backbones:** video-SALMONN 2+ (4B, 8B) and Qwen-2.5-Omni (7B). Inference at 1 FPS, 360p, ~8K KV entries per layer (variable under per-layer budgets), scalable to 48K.
- **Optional SFT** with middle-layer gradient truncation (layer 18) and similarity-only selection for efficiency.

## Key results / what to remember
Verified against the paper's Tables 1–4. Accuracies are percentages; baselines are Uniform, InfiniPot-V, StreamMem, HERMES.

**Long-video understanding, video-SALMONN 2+ 8B (Table 1):**
- OmniMem: Video-MME Long **69.6%**, LVBench **53.3%**, LV-Omni-Bench **42.5%**.
- OmniMem + SFT: **70.2% / 55.7% / 43.1%**.
- vs HERMES (65.0 / 49.8 / 40.2): **+4.5 / +3.0 / +2.2** absolute (training-free), with a further **+0.6 / +2.4 / +0.6** from SFT.

**Smaller / other backbones (Table 1):**
- video-SALMONN 2+ 4B: OmniMem 64.4 / 50.3 / 39.8; +SFT 65.2 / 54.4 / 40.7 (≈ +2.5 / +3.2 / +2.3 over HERMES).
- Qwen-2.5-Omni 7B: OmniMem 51.9 / 38.8 / 34.3 (only ≈ +1.4–1.7 over baselines — attributed to weaker audio modeling in this backbone).

**StreamingBench, video-SALMONN 2+ 8B (Table 2):**
- OmniMem **78.5% Realtime / 60.9% Omni-source / 40.7% Contextual** vs HERMES 77.9 / 57.8 / 40.1.
- Largest gain (**+3.1**) on the **Omni-source** partition — direct evidence the separate audio budget helps when both modalities matter.

**Ablations (Table 4, 8B):** Full 69.6 / 53.3 / 42.5. Removing separate A/V budgets → 66.8 / 52.2 / 40.7; removing AVBA entirely (uniform single cache) → 66.3 / 51.7 / 40.5 (AVBA worth ≈2.8–3.3 abs); similarity-only (drop $\psi_k$) → 67.9 / 52.0 / 41.4 (perturbation scoring worth ≈1.6–1.7).

**Ratio ablation (Table 3):** $r=5$ is the sweet spot on Video-MME Long (69.6) and LV-Omni-Bench balance; No-Split baseline 66.8 / 52.2 / 40.7. $r=2$ pushes LV-Omni-Bench highest (43.2) but hurts LVBench (51.5).

**Cost (Figure 4):** <1 GB extra memory for hidden-state retention; first-token latency ~0.2 s, comparable to baselines. **Budget scaling (Figure 6):** beats StreamMem across 4K–48K per-layer budgets, advantage largest at 8K–16K; VideoMME-Long/LV-Omni plateau ~32K while LVBench (longest videos) keeps improving.

No Zotero highlights present.

Takeaways: (1) modality-blind KV compression is the real bottleneck for omni-LLMs — separating the audio/visual budgets (AVBA) is the single biggest win. (2) A multiplicative attention×non-redundancy criterion beats similarity-only pruning. (3) It is training-free, with SFT as a modest additional gain (biggest on LVBench). (4) Benefit tracks how much audio actually matters (largest on Omni-source; muted on a weak-audio backbone).

## How it connects (evolution)
- [[infinipot-v]] — a prior streaming KV-cache compressor used as a direct baseline; OmniMem generalizes the bounded-cache idea to two modality-specific caches.
- [[streammem]] — query-agnostic streaming KV memory; the head-to-head baseline OmniMem beats across the 4K–48K budget sweep.
- [[hermes-kv]] — HERMES KV-cache method, the strongest baseline in Tables 1–2 that OmniMem improves on.
- [[streamkv]] — sibling KV-cache-for-streaming approach in this sub-topic; contrast the selection criterion.
- [[video-salmonn-s]] — audio-visual streaming LLM lineage; OmniMem's backbone family (video-SALMONN 2+) sits here.
- [[streaming-memory]] — sub-topic hub tying these bounded-memory streaming methods together.

## Open questions / limitations
- The visual:audio prior $r=5$ is a hand-set hyper-parameter tuned per model; a poorly chosen $r$ (see $r=2$ vs $r=5$ trade-offs) shifts which benchmark wins — no automatic setting.
- Gains collapse to ≈1.4–1.7 on Qwen-2.5-Omni, suggesting the method's payoff is bounded by the backbone's audio quality rather than being universally large.
- Compressibility scores rely on attention entropy and cosine similarity computed on a small calibration set (Fig. 2); robustness of those layer/modality patterns to very different domains (music, non-speech audio) is untested.
- SFT's benefit is uneven (+2.4 LVBench but only +0.6 elsewhere at 8B) and adds a training stage that the "training-free" pitch otherwise avoids.

*Verification: equations (8)–(14) and all reported accuracies checked against the paper's Tables 1–4 and Figures 1–2/4/6 via the arXiv HTML full text plus the downloaded PDF (page renders for Fig. 1/2); no external project page used. arXiv id 2606.07577 is as given in the task.*

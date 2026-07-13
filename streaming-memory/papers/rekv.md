---
zotero_key: null
authors: Shangzhe Di, Weidi Xie et al. (Shanghai Jiao Tong University; Alibaba Group)
year: 2025
arxiv: 2503.00540
pdf: https://arxiv.org/pdf/2503.00540
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# Streaming Video Question-Answering with In-context Video KV-Cache Retrieval (ReKV)

**Lineage role:** Origin of the KV-cache-retrieval line for streaming video LLMs (ICLR 2025) — reframes streaming memory as *retrieval over an already-computed KV cache* rather than a learned/compressed memory bank, and does it training-free by grafting onto any decoder-based Video-LLM.

## Problem — what was limited before this paper
Offline VideoQA assumes the whole video is available and can be sampled/encoded before a question arrives. Streaming VideoQA (StreamingVQA) instead needs to (1) encode a continuous stream chunk-by-chunk with no access to future frames, (2) preserve information from arbitrarily-far-back frames for long-horizon reasoning, and (3) answer a question about any moment with low, stable latency. Prior streaming approaches (e.g. [[flash-vstream]]) compress history into a fixed-size memory, which loses detail and re-processes/re-samples context per question, so accuracy and cost scale poorly with video length. ReKV's insight: the Video-LLM already produces per-frame key/value vectors during encoding — keep them and *retrieve* only the question-relevant ones, so nothing is destructively compressed and per-question compute is bounded.

## Key idea — the core insight
Encode the stream once with sliding-window attention, storing every frame's KV cache (offloading out-of-window blocks to RAM/disk). When a question arrives, score each cached frame/block against the question by cosine similarity, reload only the top-`r` relevant KV blocks onto GPU, and answer by attending over just those retrieved KV vectors plus the question. No retraining, no extra parameters (in the "internal" variant), and latency stays flat as the video grows because the answer attends to a fixed budget of retrieved frames rather than the entire history.

![[rekv.png]]
> **Crux (Figure 2).** ReKV's three-stage attention modification of a decoder Video-LLM: (a) the stream is encoded with sliding-window attention and out-of-window KV-caches are offloaded to RAM/disk; (b) on a question, compressed frame vectors are scored by cosine similarity to retrieve relevant KV blocks; (c) the retrieved KV is reloaded to GPU and used as context for autoregressive answer generation. *Di, Xie et al. (2025), arXiv:2503.00540. Embedded for personal research reference.*

## Method + math
ReKV is a training-free wrapper that modifies the attention of an existing decoder-based Video-LLM. Three components: streaming encoding, KV-cache retrieval, and QA over retrieved KV.

**1. Video stream encoding with sliding-window attention.** The stream $\mathcal{V}^T$ is processed chunk by chunk. At a step the past key–value cache is $\mathbf{P} = \{(\mathbf{k}_j, \mathbf{v}_j)\}_{j=1}^{l_P}$ and the current chunk tokens are $\mathbf{X} = \{\mathbf{t}_{i+l_P}\}_{i=1}^{l_X}$. Only the local window of the last $l_L$ cached pairs, $\mathbf{L} = \mathbf{P}_{[l_P - l_L + 1 : l_P]}$, is attended to:
$$\mathbf{O} = \mathrm{Attn}\!\left(\mathbf{W_Q X},\; [\mathbf{L}_k, \mathbf{W_K X}],\; [\mathbf{L}_v, \mathbf{W_V X}]\right) \tag{1}$$
Every frame's KV is *kept* (not just the window) — out-of-window blocks are offloaded to RAM or disk (default local window $l_L = 15\text{K}$ tokens), so encoding cost per chunk is bounded while nothing is discarded.

**2a. External retrieval (baseline).** A CLIP-like model (SigLIP-SO400M) embeds each frame $\mathbf{v} = f_v(v) \in \mathbb{R}^D$ and the question $\mathbf{q} = f_t(q) \in \mathbb{R}^D$; retrieve by cosine similarity
$$\mathrm{Sim}(\mathbf{v}, \mathbf{q}) = \frac{\mathbf{v} \cdot \mathbf{q}}{\tau\, \|\mathbf{v}\|\, \|\mathbf{q}\|} \tag{2}$$
with $\tau$ a learnable temperature. Optionally group $b$ consecutive frames into a block (average their vectors) and score at block level; keep the top-$r$ frames (or $\lceil r/b\rceil$ blocks). This needs an extra model and re-encodes frames, so it is mainly a comparison point.

**2b. Internal retrieval (the proposed variant).** Reuse the Video-LLM's own self-attention states — no external encoder, no new parameters. A frame's representative vector is the mean of its key vectors, concatenated across heads into dimension $D'$:
$$\mathbf{v} = \frac{1}{N_f}\sum_{j=1}^{N_f} \mathbf{k}_j \in \mathbb{R}^{D'}, \qquad \mathbf{q} = \frac{1}{N_q}\sum_{k=1}^{N_q} \mathbf{q}_k \in \mathbb{R}^{D'}$$
where $N_f$ is tokens per frame and $N_q$ tokens in the question. Score with Eq. (2) but $\tau = 1$. Crucially retrieval runs *independently in each self-attention layer*, so different layers can pull different video blocks — broadening the effective context captured. Because it reuses already-computed hidden states, it adds essentially no compute or memory over vanilla decoding.

**3. QA over retrieved KV.** Let $\mathbf{R}_k, \mathbf{R}_v$ be the key/value vectors of the retrieved blocks. Answer generation attends over the retrieved context (retrieved video KV + question + previously generated tokens):
$$\mathbf{O} = \mathrm{Attn}\!\left(\mathbf{W_Q X},\; [\mathbf{R}_k, \mathbf{W_K X}],\; [\mathbf{R}_v, \mathbf{W_V X}]\right) \tag{3}$$
where $\mathbf{X}$ is either the question tokens or the current decoding token.

**Positional encoding.** The base models use RoPE. During streaming, RoPE operates normally inside the local window but is bounded by a "distance ceiling" for far tokens (LM-Infinite style). For QA, retrieved KV are re-indexed as *regular consecutive tokens* (their original positions are discarded). ReKV found standard RoPE on retrieved tokens beats the Inf-LLM variant (all retrieved tokens sharing one position), because relative temporal ordering matters for video.

## Explicit design choices
- **Training-free graft** onto any decoder-based Video-LLM (LLaVA-OV 0.5B/7B/72B, Video-LLaVA, LongVA all demonstrated) — no fine-tuning, ReKV explicitly does not claim the base model's capability.
- **Store all KV, offload the tail** (RAM/disk) rather than compress — memory managed by offloading, not by lossy summarization; default local window 15K tokens.
- **Two retrieval modes:** external (SigLIP-SO400M, learnable $\tau$) vs internal (self-attention keys, $\tau=1$, no params). Internal is the default: faster, cheaper, and per-layer independent.
- **Frame-level or block-level retrieval:** default block size $b=1$ frame; default retrieved frames $r=64$; question padded to 64 tokens; answers fixed at 128 tokens.
- **Compress before scoring:** mean-pool a frame's key vectors to one vector so similarity is computed on compact representatives, accelerating retrieval over long histories.
- **RoPE re-indexing** of retrieved KV as consecutive tokens (keep relative order, drop absolute positions).
- **Eval FPS 0.5** (1,800 frames for a 1-hour video), FP16 on A100-80GB, aligned with GPT-4o's MLVU protocol.

## Key results / what to remember
Verified against Tables 4 and 5 of the paper.

**Offline VideoQA (Table 4), LLaVA-OV-7B + ReKV (internal, 0.5 FPS → 64 frames):**
- MLVU-dev 68.5 (+3.8 over 64.7 base) · QaEgo4D-test 56.0 (+3.2) · EgoSchema 60.7 (+0.9) · ActivityNet-QA acc 60.4 (+3.8), open-ended score 3.52 (+0.23, gpt-3.5-turbo-0613 rating 1–5).
- LLaVA-OV-0.5B + ReKV: MLVU 56.1 (+2.9) · QaEgo4D 50.0 (+7.4) · EgoSchema 31.0 (+1.4) · ActivityNet-QA 52.1 (+1.6), score 3.15 (+0.13).

**Streaming VQA (Table 5), 1-hour videos, RVS-Ego / RVS-Movie (Acc):**
- LLaVA-OV-7B **Internal**: RVS-Ego 63.7 (score 4.0), RVS-Movie 54.4 (3.6); video enc 11 FPS; latency 3.3 s; GPU 38 GB; KV-cache 18.8 GB/h.
- LLaVA-OV-7B External: RVS-Ego 62.4, RVS-Movie 53.6; 11 FPS; latency 5.8 s; GPU 55 GB. (Internal is faster and lighter than external at ≈ same accuracy.)
- Baseline Flash-VStream-7B: RVS-Ego 57.3 (4.0), RVS-Movie 53.1 (3.3); 14 FPS; latency 2.4 s; GPU 20 GB. ReKV-Internal beats it by +6.4 (Ego) / +1.3 (Movie).
- LLaVA-OV-0.5B Internal: RVS-Ego 54.7, RVS-Movie 44.6; 17 FPS; latency 1.6 s; GPU 19 GB; KV-cache 4.0 GB/h.

**Retrieval-quality ablation (Table 2, QaEgo4D, LLaVA-OV-7B):** uniform sampling 53.0 acc / 6.1 recall → external 54.2 / 58.1 → **internal 56.0 / 70.5** → oracle 64.4 / 100. Internal retrieval nearly doubles relevant-frame recall over external and closes much of the gap to oracle.

**Cost scaling:** per-QA FLOPs/MACs *drop* as questions get more frequent because encoding is amortized — at 360 QAs/hr internal ReKV uses ~5.6 TFLOPs/QA vs ~13.8 for Flash-VStream (Table 8), i.e. retrieval cost is bounded while the fixed-memory baseline stays flat/high.

No Zotero highlights present.

Takeaways: (1) retrieval over a preserved KV cache beats lossy fixed-memory compression on both accuracy and per-question cost for long streams; (2) the model's *own* attention keys are a strong, free retriever — no external CLIP needed; (3) per-layer independent retrieval is a distinctive lever that broadens captured context; (4) it is a training-free wrapper, so it rides base-model improvements.

## How it connects (evolution)
- [[flash-vstream]] — the fixed-size compressed-memory streaming baseline ReKV is measured against and outperforms; contrast: compress-memory vs retrieve-from-KV.
- [[streaming-memory]] — sub-topic hub; ReKV is the origin of the KV-retrieval branch of streaming memory.
- [[hermes-kv]] — closely related KV-cache-based streaming memory line (sibling in the KV-retrieval family).
- [[streamkv]] — KV-centric streaming memory descendant sharing the retrieve-relevant-KV idea.
- [[infinipot-v]] — long-video KV-cache management (offload/compress) addressing the same memory-growth problem.
- [[video-salmonn-s]] — streaming Video-LLM with a related memory/retrieval mechanism for long streams.

## Open questions / limitations
- Retrieval granularity is frame/block-level cosine similarity on mean-pooled keys — coarse; fine-grained or temporally-structured queries (counting, ordering across distant events) may under-retrieve, and oracle at 64.4 vs internal 56.0 on QaEgo4D shows a real retrieval-quality ceiling remains.
- Storing all KV and offloading to RAM/disk grows unbounded (18.8 GB/h for 7B) — feasible for hours, but truly endless streams still need eviction that the paper does not solve.
- Per-layer independent retrieval adds retrieval calls per layer; its accuracy benefit vs a single shared retrieval is asserted but the compute/accuracy trade across layer counts is not deeply ablated.
- Evaluated at 0.5 FPS; behavior for high-motion content needing dense temporal sampling (and its memory cost) is untested here.

*Verification: equations (1)–(3), internal-retrieval formulation, and RoPE/offload details checked against rendered PDF pages 3–4; all headline numbers checked against Table 4 and Table 5 as rendered from arXiv:2503.00540 page 7; ICLR 2025 venue confirmed from the page header.*

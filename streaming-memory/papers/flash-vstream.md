---
zotero_key: null
authors: Haoji Zhang, Yiqin Wang, Yansong Tang et al. (Tsinghua Shenzhen / ByteDance)
year: 2024
arxiv: 2406.08085
pdf: https://arxiv.org/pdf/2406.08085
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# Flash-VStream: Memory-Based Real-Time Understanding for Long Video Streams

**Lineage role:** Canonical streaming-memory video LLM — introduces the STAR (Spatial-Temporal-Abstract-Retrieved) compressed memory and a lock-free two-process pipeline that keeps VRAM and answer latency flat while a stream grows unbounded; also ships VStream-QA, the first timestamp-grounded online-QA benchmark on 30-60 min videos.

## Problem — what was limited before this paper (short)
Offline video LLMs (Video-ChatGPT, Chat-UniVi, LLaMA-VID, MovieChat) sample or store per-frame tokens for the *whole* clip and only answer once the video ends. On a live stream this fails twice: (1) token/VRAM cost grows with frame count (Chat-UniVi needs 77 GB at 1000 frames), and (2) query latency scales linearly with how much video has arrived, so a question on an hour-long stream can take >10 s. There was also no benchmark that scores answers using *only* the frames seen before the question's timestamp, so "online" behavior could not be measured properly.

## Key idea — the core insight, 2-4 sentences
Separate the two things offline models fuse: continuously *encode + consolidate* frames into a small fixed-size memory, and *decode* an answer on demand from that memory. Flash-VStream runs a **frame handler** process that never stops writing incoming CLIP features into a bounded 681-token STAR memory (four sub-memories at different granularities), and a **question handler** process that, whenever a user asks, reads the current memory and generates an answer with an LLM. Because memory size is capped and reading is the only cross-process link, both peak VRAM and per-query latency stay roughly constant (~1 s) no matter how long the stream runs.

![[flash-vstream.png]]
> **Crux (Figure 3).** The two asynchronous processes: a *frame handler* (visual encoder → STAR memory {S, T, A, R} + feature buffer) writes continuously, and a *question handler* (projector → LLM) reads the shared memory to answer anytime — encoding is decoupled from decoding so latency is independent of video length. *Zhang, Wang, Tang et al. (2024), arXiv:2406.08085. Embedded for personal research reference.*

## Method + math — the mechanism, then the central objective/equations IN FULL

**Encoder.** A frozen CLIP ViT-L/14 (224 px) maps each incoming frame $V^t \in \mathbb{R}^{H\times W\times 3}$ to a patch feature map $e^t \in \mathbb{R}^{P\times P\times D}$ with $P\times P = 196$ patches (14×14) and $D=1024$. Only patch tokens are kept.

**STAR memory (four components, capped sizes).** The memory holds tokens at four levels of granularity, each with a per-frame spatial resolution $P_\ast$ (tokens obtained by average-pooling the $14\times14$ grid down to $P_\ast\times P_\ast$) and a temporal length $N_\ast$ (number of frame-slots retained):

- **Spatial** $M_{spa}\in\mathbb{R}^{N_{spa}\times P_{spa}^2\times D}$, with $N_{spa}=1,\ P_{spa}=8$ → 64 tokens. Newest, highest-resolution frame; a FIFO queue holding the most recent frame(s):
$$ M_{spa}^t = M_{buff}^t[0:N_{spa},:,:] $$
- **Temporal** $M_{tem}\in\mathbb{R}^{N_{tem}\times P_{tem}^2\times D}$, $N_{tem}=25,\ P_{tem}=4$ → 16 tokens/slot. Compresses history by **weighted K-means clustering** (Algorithm 1): pool the new frame, concatenate with the previous $N_{tem}$ centroids to get $N_{tem}+1$ items, then re-cluster back to $N_{tem}$ so recurring events survive as centroids:
$$ M_{tem}^t = g_{\text{wkmeans}}\!\big(\operatorname{concat}(g_{\text{pool}}(e^t, P_{tem}),\ M_{tem}^{t-1}),\ N_{tem}\big) $$
The weighted centroid of cluster $S_j$ uses the cluster occupancy counts $w_i$ as weights:
$$ c_j \leftarrow \frac{\sum_{x_i\in S_j} w_i\, x_i}{\sum_{x_i\in S_j} w_i} $$
- **Abstract** $M_{abs}\in\mathbb{R}^{N_{abs}\times P_{abs}^2\times D}$, $N_{abs}=25,\ P_{abs}=1$ → 1 token/slot. High-level semantics maintained by a learnable **Semantic Attention** update (Algorithm 2). New pooled features $e$ act as keys/values, the current abstract memory as query; attention weights $W=\operatorname{softmax}(QK^\top)$ (with $K=f_{k}(e)$, $Q=f_{q}(M_{abs})$) route information, and memory is updated by momentum:
$$ M_{abs} \leftarrow (1-\alpha)\,M_{abs} + W e, \qquad M_{abs}^t = f_{SA}\!\big(M_{abs}^{t-1},\ g_{\text{pool}}(e^t, P_{abs}),\ N_{abs}\big) $$
- **Retrieved** $M_{ret}$, $N_{ret}=3,\ P_{ret}=8$ (spatial resolution). Restores spatial detail to the most important events (Algorithm 3): take the top-$N_{ret}$ *largest* temporal clusters, then pull the nearest *raw* frame features from the feature buffer to those centroids:
$$ M_{ret}^t = g_{\text{retrieve}}(M_{buff}^t,\ M_{tem}^t,\ N_{ret}) $$

A raw **feature buffer** $M_{buff}$ keeps the last $N_{buff}=300$ un-pooled frames so retrieval can recover fine detail. Total memory is capped at **MAXSIZE = 681 tokens**.

**Memory reading.** All four sub-memories are simply concatenated/summed into a single token set fed to the LLM:
$$ M^t = M_{spa}^t \,\Vert\, M_{tem}^t \,\Vert\, M_{abs}^t \,\Vert\, M_{ret}^t $$

**Decoding (question handler).** On a query $Q^t$ at time $t$: project memory $I_{vision}^t = f_{proj}(M^t)$ (2-layer MLP into Vicuna-7B embedding space), embed the question $I_{text}^t = f_{embed}(Q^t)$, and generate
$$ A^t = f_{LLM}(I_{text}^t,\ I_{vision}^t). $$

**Asynchronous execution.** The frame handler owns writes to the shared STAR memory; the question handler only reads. No lock is needed because writer is single and readers are non-blocking, so answering never stalls encoding and vice-versa — the source of the flat ~1 s latency.

## Explicit design choices — concrete decisions
- **Four granularities, not one buffer:** spatial (detail/recency), temporal (event summary via clustering), abstract (semantics via learned attention), retrieved (detail re-injected onto key events). Each targets a different failure of naive token stacking.
- **Fixed token budget (681):** decouples memory/VRAM from stream length; sizes chosen by ablation ($N_{tem}=N_{abs}=25$, $P_{spa}{=}8,P_{tem}{=}4,P_{abs}{=}1$).
- **Weighted K-means for temporal compression** — weights = cluster sizes so frequent/persistent content dominates centroids; O(1) memory over time.
- **Semantic Attention over Q-Former** for the abstract update: momentum + learned attention beats (Sequential) Q-Former in ablation.
- **Two-process, single-writer, lock-free** design — encoding runs continuously; answers computed on demand.
- **Frozen CLIP ViT-L/14 encoder + Vicuna-7B decoder + 2-layer MLP projector.**
- **Two-stage training:** Stage 1 modality alignment (LLaVA-558K images + LLaMA-VID-232K videos, lr 1e-3, bs 256; train only semantic attention + projector). Stage 2 instruction tuning (LLaVA-665K + Video-ChatGPT-98K, lr 2e-5, bs 128; everything except the encoder open). ~15 h on 8× A100-80G, BF16.
- **VStream-QA benchmark:** every QA carries a **timestamp** and is answerable from frames *before* that timestamp only — enforcing the online setting. Built from Ego4D (10×1 h) + MovieNet (22×30 min), 21 h total, ~3.5K QA. Pipeline: GPT-4V dense captions per 30 s segment → GPT-4 dedup/summarize → GPT-4 generates 5 QA types → human filtering. Five question types: Scene Summary, Action Description, Event Occurrence (yes/no), Ordered Event Narrative, Sequence Validation (yes/no). GPT-based judging yields Accuracy + a 0-5 Score.

## Key results / what to remember — exact numbers WITH setting (verified vs the paper's tables)
No Zotero highlights present.

Zero-shot VideoQA, Accuracy / Score (Table 3), Flash-VStream (Vicuna-7B):
- **MSVD-QA: 80.3 / 3.9** (best vs MovieChat 75.2, LLaMA-VID 69.7).
- **MSRVTT-QA: 72.4 / 3.4** (best vs Vista-LLaMA 60.5, LLaMA-VID 57.7).
- **ActivityNet-QA: 51.9 / 3.4** (best vs Vista-LLaMA 48.3, LLaMA-VID 47.4).
- **NExT-QA: 61.6 / 3.4** (best vs Chat-UniVi 60.8).
- **VStream-QA-Ego: 59.0 / 3.9**; **VStream-QA-Movie: 56.1 / 3.4** (both best).

Real-time / long-stream (Table 1, RealTime-VStream, 1000 frames unless noted):
- **RVS-Ego 57.3 / 4.0, RVS-Movie 53.1 / 3.3**, at **16.03 GB VRAM** — best accuracy *and* lowest memory. Chat-UniVi needs 77.56 GB; LLaMA-VID 33.64 GB; MovieChat 16.90 GB.
- **Latency ~1 s regardless of frame count** (Figure 2); competitors scale linearly (>10 s at ~10k frames).

Ablations:
- Drop **temporal** memory → worst (VS-Movie 51.4 vs 56.1 full) — temporal is the most load-bearing; every component adds (Table 4).
- **Semantic Attention 59.0 / 56.1** vs Q-Former 57.1 / 50.4, Sequential Q-Former 56.0 / 51.4 (Table 5) — "by large margin," especially on Movie.
- Balanced sizes win: $P_{spa}8/P_{tem}4/P_{abs}1$ and $N_{tem}=N_{abs}=25$ optimal; bigger doesn't help and costs more (Table 6).

Takeaways: compressing a stream into a *bounded* four-granularity memory is enough to (a) beat much heavier offline models on standard VideoQA, (b) run at flat ~1 s latency and ~16 GB on arbitrarily long streams, and (c) the temporal (clustering) memory carries most of the accuracy while retrieved memory restores detail on key events cheaply.

## How it connects (evolution)
- [[videollm-online]] — the other 2024 pillar of online video LLMs; VideoLLM-online streams per-frame with a decision to speak, whereas Flash-VStream streams into a compressed memory and answers on query.
- [[rekv]], [[infinipot-v]], [[streamingvlm]] — successor **KV/memory-compression** approaches for unbounded streams; STAR is the token-side ancestor of these cache-side methods.
- [[streammem]], [[streamkv]], [[hermes-kv]] — later streaming-memory designs that similarly bound state to keep VRAM/latency flat.
- [[streamingbench]], [[ovo-bench]], [[svbench]] — streaming-QA benchmarks that follow VStream-QA's timestamp-grounded "answer from frames-so-far" protocol.
- [[streaming-memory]] — sub-topic hub; [[streaming-video-understanding]] — topic hub.

## Open questions / limitations
- Fixed 681-token budget is tuned on Ego4D/MovieNet; whether the same sizes hold for denser or faster-changing streams (sports, driving) is untested.
- Weighted K-means and momentum updates are lossy and irreversible — once an event is merged away it cannot be recovered beyond the 300-frame raw buffer, so very-fine long-past detail is gone.
- Evaluation leans on GPT-judged Accuracy/Score, which is noisy and biased toward short factual answers; temporal-order question types are where all methods (including this one) score lowest.
- The single-writer/reader design assumes one stream and one query at a time; true concurrent multi-query or multi-stream serving isn't addressed.

*Verification: numbers and equations checked against the paper's own Tables 1, 3, 4, 5, 6 and the STAR-memory equations/Algorithms 1-3 via the arXiv HTML (arxiv.org/html/2406.08085) and PDF; Figure 3 cropped from PDF page 3.*

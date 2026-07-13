---
zotero_key: null
authors: Haomiao Xiong, Zongxin Yang, Jiazuo Yu, et al. (Dalian University of Technology; Harvard University)
year: 2025
arxiv: 2501.13468
pdf: https://arxiv.org/pdf/2501.13468
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# Streaming Video Understanding and Multi-round Interaction with Memory-enhanced Knowledge (StreamChat)

**Lineage role:** A *training-free* streaming video-LLM that stores the past as a **hierarchical (tree-structured) long-short-term + dialogue memory**, decoupled from generation across three threads, and ships **StreamBench** — a long-video, multi-round, memory-probing benchmark (ICLR 2025).

## Problem — what was limited before this paper (short)
Video-LLMs choke on three things at once: (1) fitting an **arbitrarily long, growing** frame stream into bounded GPU memory; (2) sustaining **multi-round dialogue** where a later question depends on earlier turns and on events seen long ago; and (3) doing this **online**, at low latency, instead of ingesting the whole clip offline. Existing benchmarks compounded the blind spot — short, single-domain clips with one-shot questions — so they never tested recall over minutes of video, cross-turn coherence, or streaming latency.

## Key idea — the core insight, 2-4 sentences
Treat the video as a **compressible, storable repository** rather than a token block to re-encode. A human-memory-inspired hierarchy holds it: a **short-term memory** of recent frame embeddings (kept via an Ebbinghaus forgetting curve), a **long-term memory tree** whose leaves are k-means-compressed visual clusters paired with text-caption "clues," and a **dialogue memory** of past Q-A pairs. Frames are filtered by optical-flow motion before storage, and the three stages — **selective frame stacking, memory formation, contextual summarization** — run as **parallel threads** so generation never blocks ingestion, giving sub-second latency at up to 32 FPS. All of it is **training-free** on top of a frozen LongVA backbone.

![[streamchat-mem.png]]
> **Crux (Figure 4).** The three-thread StreamChat pipeline: selective frame stacking encodes/filters frames into a vision buffer, memory formation organizes them into the hierarchical memory (update), and contextual summarization retrieves relevant tokens + dialogue history to answer a query and records the new turn. *Xiong et al. (2025), arXiv:2501.13468. Embedded for personal research reference.*

## Method + math — the mechanism, then the objective/equations in full

**(i) Selective frame stacking.** To avoid storing redundant frames, each incoming frame $F^i \in \mathbb{R}^{H\times W\times 3}$ is compared to $F^{i-1}$ with **Lucas-Kanade optical flow**, solving for the motion vector $(u,v)$:
$$\begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} \sum_i I_x(i)^2 & \sum_i I_x(i)I_y(i) \\ \sum_i I_x(i)I_y(i) & \sum_i I_y(i)^2 \end{bmatrix}^{-1} \begin{bmatrix} -\sum_i I_x(i)I_t(i) \\ -\sum_i I_y(i)I_t(i) \end{bmatrix}$$
with $I_x,I_y,I_t$ the frame's spatial/temporal partials. The motion magnitude $\lVert\theta\rVert=\sqrt{u^2+v^2}\in[0,1]$ gates storage: only if $\lVert\theta\rVert>t$ (threshold) is the frame encoded to a vision embedding $e^i\in\mathbb{R}^{n\times d}$ and pushed to buffer $\mathcal{B}_{\text{vision}}$. A high $t$ keeps fewer frames → faster but lossier.

**(ii) Hierarchical memory storage.** The memory is $M_l \cup M_s = \{l_i\}_{i=0}^{T/L}\cup\{s_i\}_{i=0}^{S}$ plus dialogue memory $M_d=\{d_i\}_{i=0}^{D}$ ($T$ = video duration, $S$ = short-memory length, $D$ = #dialogues, $L$ = chunk size).

- **Short-term memory $M_s$** (Atkinson-Shiffrin flavored). From $N$ recent candidates $\mathcal{C}$, each weighted by a normalized **Ebbinghaus forgetting probability** $\sigma_i$, randomly sample $S$ embeddings:
$$\mathcal{C}=\{\sigma_{N-1}e^{i-(N-1)},\,\sigma_{N-2}e^{i-(N-2)},\,\dots,\,\sigma_0 e^{i}\}\;\xrightarrow[\text{select}]{\text{random}}\;M_s=\{s_i\in\mathbb{R}^{n\times d}\}_{i=0}^{S}$$

- **Long-term memory tree $M_l$.** The vision buffer is chunked (length $L$); each chunk is k-means-compressed into a visual cluster $v_i\in\mathbb{R}^{C\times d}$ and captioned into a **text clue** $t_i$ by a prompt model $p_\theta$:
$$\mathcal{B}_{\text{vision}}=\{\mathcal{K}_i\}_{i=0}^{T/L},\quad \mathcal{K}_i=\{e^i\}_{i=0}^{L},\quad v_i=f_{\text{k-means}}(\mathcal{K}_i),\quad t_i=p_\theta(x_i\mid \mathcal{K}_i)$$
Each cluster+clue is a **leaf (base) node** $l_i=\{v_i,t_i\}$, pushed in chronological order:
$$[l_0,l_1,\dots,l_{i-1}]_{\text{nodes}}\;\xleftarrow{\text{push}}\;l_i=\{v_i,t_i\}$$
Groups of $g$ base nodes are recursively merged into higher-level nodes — again k-means over child visuals and $p_\theta$ over child clues — until the tree is complete:
$$M_l=\{l_i\}_{i=0}^{T/L},\quad l^1_k=\{\,f_{\text{k-means}}(\{v_i\}_{i=0}^{g}),\; p_\theta(x_i\mid\{t_i\}_{i=0}^{g})\,\}$$
Text clues act as a searchable **index** over the compressed visual memory.

- **Dialogue memory $M_d$.** Each turn $(Q_i,A_i)$ is a fragment pre-encoded by MiniLM-L6 $E(\cdot)$: $\;d_i=E(\langle Q_i,A_i\rangle)$, appended to $M_d=\{d_0,\dots,d_{i-1}\}$.

**(iii) Retrieval + contextual summarization.** On query $Q_i$: encode $Q_i$ with the LLM's tokenizer/embedding layer and compute **cosine similarity** against every text clue $t_i$; traverse the highest-similarity path down $M_l$ and return the matched visual features and short-term memory as **retrieved tokens** $\tilde{M}_s\cup\{v_r\in\mathbb{R}^{C\times d}\}_{r=0}^{\mathcal{L}}$ ($\mathcal{L}$ = layer number). In parallel, encode $Q_i$ with $E(\cdot)$ and search $M_d$ over a **FAISS** index for the most relevant prior turn $\langle Q_{\text{retrieved}},A_{\text{retrieved}}\rangle$ as history context. The LLM answers on *question + retrieved visual tokens + history context*, then records the new turn back into $M_d$.

**System scheduling.** The three stages run as independent threads — (i) stacking populates $\mathcal{B}_{\text{vision}}$; once full it clears to (ii) formation, which builds $M_l$ / refreshes $M_s$; (iii) summarization retrieves and generates — so no stage blocks another, keeping latency <0.9 s.

**StreamBench (the eval "math").**
- **Task taxonomy — 6 question types** stressing time-to-recall: **OS** Object Search (object visible <5 s, asked >30 s later), **LM** Long-term Memory (event >5 s, asked >1 min after it ends), **SM** Short-term Memory (recent event, <20 s delay), **CI** Conversational Interaction (needs dialogue context from >2 turns prior), **KG** Knowledge-based QA (encyclopedic knowledge tied to the video), **SF** Simple Factual (asked within 30 s of start).
- **Data pipeline.** Sourced from EgoSchema + YouTube-8M, filtered by (1) MLLM classification → (2) human assessment removing static/noisy clips → (3) annotation verification. Final: **306 videos, 24.8 h total, ~4.5 min average, 4 domains, 16 sub-classes, ~1.8K QA pairs.**
- **Metrics.** **Score (Sco.)** ∈ [0,5] from a LLaMA-3 judge rating semantic correctness; **Accuracy (Acc., %)** binary correctness from a score threshold; **Coherence (Coh.)** the score *fluctuation* across dialogue turns (lower = smoother); **RPD** Request Processing Delay in seconds (query submission → start of generation, lower = better).

## Explicit design choices
- **Training-free** wrapper on a frozen **LongVA** LLM + **CLIP-L/14** vision encoder — memory is inference-time machinery, no fine-tuning.
- **Optical-flow gating** (Lucas-Kanade) as a cheap, content-based frame dropper before any encoding — motion threshold $t$ is the main speed/accuracy knob.
- **Tree-structured** long memory (leaves = k-means visual clusters + text-caption clues; internal nodes = recursive merges) — text clues are the retrieval index, visual clusters are the payload.
- **Ebbinghaus forgetting curve** to weight short-term candidates, then **random sampling** to $S$ slots — biologically-motivated recency bias, not a fixed window.
- **Separate dialogue memory** encoded by MiniLM-L6, retrieved via **FAISS cosine** — decouples "what was said" from "what was seen."
- **Three parallel threads** (stack / form / summarize) for non-blocking, sub-second streaming.
- **Three configs** trading speed vs accuracy via memory params (Table 3): Slow $t{=}0.13,L{=}35,g{=}15,C{=}5$; Base $t{=}0.35,L{=}25,g{=}10,C{=}5$; Fast $t{=}0.58,L{=}30,g{=}15,C{=}5$. Selected units $S{=}5$, candidate length 20. Two A800-80GB GPUs.
- For **offline** short-clip benchmarks (MSVD/MSRVTT) they *disable* long-term $M_l$ (videos too short) and *disable* dialogue $M_d$ (single-round) — memory modules are modular/optional.

## Key results / what to remember
Numbers verified against the paper's own Tables 3-7.

- **StreamBench, online (Table 4).** StreamChat-**Slow**: Acc **64.7%**, Score **3.48**, Coherence **1.76**, 15 FPS, RPD **0.90 s**. **Base**: 63.8% / 3.42 / 1.79 / 20 FPS / 0.89 s. **Fast**: 61.7% / 3.28 / 1.81 / **32 FPS** / 0.85 s. Baselines: Video-online 56.4% / 3.11 / 1.94 / 5 FPS / 1.07 s; Flash-VStream 52.1% / 2.89 / 2.21 / 1 FPS / 4.15 s. Human performance 79.4% / 4.03 / 1.16.
- **Headline deltas vs Video-online:** Slow is **+8.3%** Acc and +0.37 Score; best model improves **Coherence by 0.18** and cuts **RPD by 0.17 s**; Fast is **+5.3%** Acc while running 32 vs 5 FPS.
- **Per-task (Table 5), StreamChat-Slow Acc:** OS 51.7% (+10.3 vs Video-online), LM 53.9% (+5.1), SM 57.8% (+4.9), CI 68.5% (+5.8), KG 88.1%, SF 69.3%. Largest gains on the memory-search tasks (OS/LM/SM/CI).
- **Offline benchmarks (Table 6), StreamChat-Base Acc / Score:** ActivityNet 50.1% / 2.78, NExT-QA 50.5% / 2.84, MSVD 58.7% / 3.08, MSRVTT 43.4% / 2.38, **Average 50.6% / 2.77** — **+2.5%** over the LongVA base; surpasses streaming Flash-VStream by 12.8% and offline LLaMA-VID-Hound by 1.4% on ActivityNet, +5.1% on NExT-QA over LongVA.
- **Ablation (Table 7), avg Acc:** no memory 60.3% → +$M_d$ 60.9% (+4.1% on CI) → +$M_s$ 60.7% (+3.2% on SM) → +$M_l$ 62.2% (+6.2% on LM) → $M_l{+}M_s$ 63.1% (+0.9% from combining) → **all three 63.8%**. Each module helps its matched task type.
- **Param sweeps (Fig. 7):** raising $t$ saturates speed at 32 FPS but hurts Acc (64.0→60.7%); chunk $L$ 15→30 helps (61.2→64.0%) then 40 slightly hurts and raises latency (0.84→1.26 s); group $g$ 2→12 helps (62.0→63.9%) but raises RPD; clustering goal $C$ 3→10 helps (59.4→64.0%) but VRAM 20→56 GB.

No Zotero highlights present.

Takeaways: (1) a **training-free hierarchical memory** can turn a frozen short-context video-LLM into a competitive streaming, multi-round agent; (2) **text-clue-indexed, k-means-compressed** visual memory is the recall engine — it drives the big OS/LM gains; (3) **thread decoupling** is what buys the latency; (4) the memory modules are **ablatably modular**, each tied to a task family; (5) StreamBench's **coherence + RPD** metrics operationalize "good streaming dialogue" beyond accuracy.

## How it connects (evolution)
- [[flash-vstream]] — prior streaming-memory video-LLM; StreamChat's main streaming baseline (beats it +12.6% StreamBench Acc, and on offline recall).
- [[videollm-online]] — the "Video-online" online-dialogue baseline StreamChat compares against and outperforms across StreamBench tasks.
- [[streamingbench]] — sibling streaming benchmark; StreamBench is the memory-and-multi-round counterpart in the [[streaming-benchmarks]] cluster.
- [[rekv]] / [[hermes-kv]] — related KV/feature-memory approaches for long streaming video; contrast the tree+forgetting-curve store vs KV eviction.
- [[streammem]] — another explicit streaming-memory design; same [[streaming-memory]] sub-topic axis (compression + retrieval).
- [[svbench]] — multi-round streaming dialogue benchmark; complementary probe of the same interaction capability StreamBench's CI task targets.

## Open questions / limitations
- **Retrieval is single-path greedy** (highest cosine down the tree) — a wrong high-level clue can prune the correct subtree; no backtracking / beam search reported.
- **Caption bottleneck:** long-term recall rides on $p_\theta$-generated text clues, so anything the caption model omits is effectively unindexed (visible in the failure cases, Fig. 13).
- **VRAM vs fidelity tension:** better clustering goal $C$ sharply raises memory (20→56 GB), and richer group size $g$ raises latency — the "compressible unit" story still has a hard hardware ceiling.
- **Judge/metric dependence:** Score/Accuracy/Coherence all rest on a LLaMA-3 grader, so absolute numbers inherit its biases; human ceiling (79.4%) shows substantial headroom remains.

*Verification: title/authors/venue, the method equations (frame-gating, Ebbinghaus short-term, tree eqs 3-5, dialogue+retrieval) and all headline numbers cross-checked against the arXiv:2501.13468 PDF's own Tables 3 (configs), 4 (StreamBench), 5 (per-task), 6 (offline), 7 (ablation) and Figures 4-5, 7; abstract/prose corroborated via the arXiv HTML page.*

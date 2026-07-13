---
zotero_key: null
authors: Lu Wang, Zhuoran Jin, Yupu Hao, Yubo Chen, Kang Liu, Yulong Ao, Jun Zhao (CASIA Institute of Automation + BAAI)
year: 2026
arxiv: 2603.11896
pdf: https://arxiv.org/pdf/2603.11896
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# Think While Watching: Online Streaming Segment-Level Memory for Multi-Turn Video Reasoning in Multimodal Large Language Models

**Lineage role:** A memory-anchored streaming framework (on Qwen3-VL) that writes one text "memory note" per video segment and decouples perception from generation via a segment-level streaming causal mask + separate input/output positional encoding, so an MLLM can keep watching while thinking across many turns — extending KV/state memory lines like [[streamforest]], [[streamagent]] and [[streammind]] toward compact per-segment textual memory with strict streaming causality.

## Problem — what was limited before this paper (short)
Most video MLLMs are offline (whole video available before a single answer) or use an **interleaved** perception-generation loop (as in [[videollm-online]], [[stream-vlm-qevd]]/StreamChat). Interleaving is fundamentally serial and causes two failures: (1) **Memory Erosion** — later multi-turn questions that reference earlier questions/visual cues become unanswerable because early context is forgotten (they measure a 40.39% offline→online multi-round accuracy drop for Qwen3-VL-4B Thinking); (2) **Serialization Bottleneck** — once autoregressive decoding starts, unified positional encoding forces new input frames to wait until the (unknown-length) output finishes, so ingestion stalls and end-to-end latency grows with rounds.

## Key idea — the core insight
Make memory an **explicit online write**: for every arriving segment the model emits exactly one compact text **memory note** appended to a persistent memory bank; when a question arrives it answers by letting attention **implicitly retrieve** the relevant notes, rather than re-reading raw frames. To break serialization, give the **input stream and output stream independent positional encodings** so newly arriving segments always get a valid input position even while an answer of unknown length is still decoding — enabling true input-output parallelism ("watch while thinking"). A **segment-level streaming causal mask** enforces that each generated unit only sees the segment prefix observed so far (strict causality, no future leakage).

![[tww.png]]
> **Crux (Figure 1).** Interleaved baselines (a) serialize perception and generation, causing memory erosion and a queueing bottleneck; Think While Watching (b) instead builds a continuous segment-level memory (one note per SEG) and answers by retrieving notes while still watching, and (c) its decoupled input/output timeline parallelizes segment processing with answering to cut latency. *Wang et al. (2026), arXiv:2603.11896. Embedded for personal research reference.*

## Method + math
**Setting.** A video stream is an ordered sequence of segments $S_{1:T}=\langle S_1,\dots,S_T\rangle$, each a contiguous chunk of frames arriving in order. In a multi-turn interaction with $R$ turns, question $q_r$ is asked after observing prefix $S_{1:\tau_r}$ with nondecreasing arrival $1\le\tau_1\le\dots\le\tau_R\le T$, and dialogue history $H_{r-1}=\langle\langle q_1,a_1\rangle,\dots,\langle q_{r-1},a_{r-1}\rangle\rangle$. **Strict causality:** answer $a_r$ is conditioned only on $S_{1:\tau_r}$, $q_r$, and $H_{r-1}$ — never on future segments.

**Streaming-unit serialization.** The interaction is flattened into a received-unit sequence $R_{1:U}=\langle R_1,\dots,R_U\rangle$ where each $R_u$ is either a visual segment unit $S_t$ or a question unit $Q_r$, and a one-to-one aligned generated sequence $C_{1:U}$: if $R_u=S_t$ then $C_u$ is the memory note $m_t$; if $R_u=Q_r$ then $C_u=\langle\pi_r,a_r\rangle$ (rationale + answer). $\mathrm{idx}[\cdot]$ returns arrival index. Each received unit gets a nonoverlapping input-position span
$$\Delta[R_u]=\begin{cases}\max\{T_u,H_u,W_u\}, & R_u\in\{S\}\\ L[R_u], & R_u\in\{Q\}\end{cases}$$
where $\langle T_t,H_t,W_t\rangle$ are the vision-encoder token-grid sizes (temporal/height/width) of a segment and $L[\cdot]$ is text token length.

**Segment-level memory bank.** After prefix $S_{1:t}$ the bank is $M_t\triangleq\{\langle i,m_i\rangle\}_{i=1}^t$. Each note $m_t$ is a compact text unit grounded in $S_t$ recording reusable evidence (key entities/attributes, salient actions/interactions, scene changes, short-range temporal relations). With the backbone's memory-writing function $\mathrm{Mem}_\theta[\cdot]$:
$$m_t=\mathrm{Mem}_\theta[S_t],\qquad C_{\mathrm{idx}[S_t]}=m_t.$$
Answering a question retrieves implicitly over the memory prefix:
$$\langle\pi_r,a_r\rangle\sim p_\theta\big[\pi_r,a_r \mid q_r, H_{r-1}, M_{\tau_r}\big].$$

**Segment-level streaming attention mask.** The full sequence $\langle R_1,\dots,R_U,C_1,\dots,C_U\rangle$ is fed with a mask $M^{\text{seg}}$. With $A$ the query-contributing unit (arrival index $u$) and $B$ the key/value unit ($v$ if received $R_v$, $k$ if generated $C_k$):
$$M^{\text{seg}}[A,B]=\begin{cases}\mathbb{I}[v\le u], & A=R_u,\ B=R_v\\ \mathbb{I}[v\le u], & A=C_u,\ B=R_v\\ \mathbb{I}[k\le u], & A=C_u,\ B=C_k\\ 0, & \text{otherwise.}\end{cases}$$
So the received stream is causal in arrival order; each generated $C_u$ attends only to the received prefix up to $u$ and to prior generated units $C_{1:u}$; all $R_u\!\to\!C_k$ links and future $C_u\!\to\!R_v\ (v>u)$ links are masked. Token-level masks expand $M^{\text{seg}}$ with standard causal masking inside each $C_u$ (e.g. $C_1$ attends only to $S_1$, $C_3$ to $\langle S_1,Q_1,S_2\rangle$).

**Streaming positional encoding (decoupled MRoPE).** Built on MRoPE but with the **input stream** using the standard cumulative offset and the **output stream** independently restarting from 0. Base offsets for the $k$-th segment input, $k$-th question input, and $k$-th generated unit:
$$B^S_k=\!\!\sum_{u<\mathrm{idx}[S_k]}\!\!\Delta[R_u],\quad B^Q_k=\!\!\sum_{u<\mathrm{idx}[Q_k]}\!\!\Delta[R_u],\quad B^C_k=\begin{cases}0, & k=1\\ \sum_{i=1}^{k-1}L[C_i], & k\ge 2.\end{cases}$$
Because $B^S_k,B^Q_k$ depend only on the received prefix and $B^C_k$ only on previously generated tokens, arriving input segments always get correct positions even while output length is unknown — this is what enables input-output parallelism. A visual token at local grid $\langle t,h,w\rangle$ in $S_k$ maps to global $\langle t+B^S_k, h+B^S_k, w+B^S_k\rangle$; text token $n$ uses $n+B^Q_k$ (question) or $n+B^C_k$ (output).

**Three-stage training** (fine-tuning Qwen3-VL-Instruct): **Stage 1** — write segment-level memory notes + answer single-round queries (5,160 single-round instances from VideoChatOnline-IT temporal-perception subsets, ≤64 frames). **Stage 2** — scale to multi-round dialogues (2,752 dialogues, 8,513 rounds, avg 3.09 rounds, grouping questions over the same video prefix). **Stage 3** — long-range behaviors on long YouTube videos (1,500 instances, 6,000 rounds, avg 4 rounds, 100–300+ frames): long-term evidence recall, **uncertainty handling** (defer commitment when evidence not yet observed), and **distractor-segment learning** (insert irrelevant frames as distractors). CoT with memory notes is synthesized with GPT-5.2 over the original dataset QAs; quality inspection enforces exactly one output item per segment plus one per question ($S+Q$ items).

**Streaming inference.** A **dual KV cache** decouples continuous source ingestion from autoregressive decoding (same mask + MRoPE as training). An **adaptive attention backend**: since the streaming mask can have query length $\ne$ key length, use Flash Attention for source prefilling ($q_{\text{len}}=k_{\text{len}}$) and single-step decoding ($q_{\text{len}}=1$), and switch to memory-efficient attention with the explicit streaming mask when $1<q_{\text{len}}<k_{\text{len}}$.

## Explicit design choices
- **One text memory note per segment**, appended to a persistent bank; answering reads notes (compact state), not raw frames — attention does implicit retrieval (no explicit retriever).
- **Decoupled positional encoding**: input stream cumulative offset, output stream restarts at 0 → new frames get valid positions during decoding of unknown length.
- **Segment-level streaming causal mask** enforcing strict no-future-leakage, expanded to token-level with intra-unit causal masking.
- **Three-stage, stage-matched curriculum**: single-round CoT → multi-round CoT → long-range (recall + uncertainty + distractor robustness).
- **Data**: Stages 1&2 from VideoChatOnline-IT short videos (≤64 frames); Stage 3 from 1,500 long YouTube videos across tutorial/lecture/long-form (500 keywords), balanced 100–200 / 200–300 / 300+ frame buckets, 3–5 rounds; CoT generated by GPT-5.2.
- **Backbones**: Qwen3-VL Instruct at 2B/4B/8B (trained), compared against the Qwen3-VL Thinking variant.
- **Inference engineering**: dual KV cache (perception/generation parallelism) + adaptive Flash/memory-efficient attention by qlen/klen pattern.
- **Two eval protocols**: single-round streaming (many segments, one question) and multi-round streaming (many segments, multiple timed questions); offline "Batch" checkpoints (TWW_Batch,S2/S3) also reported.
- **Segmentation**: split video at question timestamps into consecutive segments; any segment >60s split into 30s chunks (default 60s/30s).

## Key results / what to remember
All numbers verified against the paper's Tables 2–6 (Qwen3-VL family). "Overall" = benchmark overall accuracy.

- **StreamingBench, 4B (Table 2):** TWW_single-turn,S3 = **60.04%** overall vs Thinking-batch 58.52% (+1.52). Naive online Thinking collapses to 18.13% and Instruct_online to 21.47%, showing streaming needs aligned supervision. Multi-turn TWW_S3 = 57.40% at **302.56 avg tokens = 56.10% token reduction** vs Thinking's 689.22.
- **StreamingBench, 8B (Table 2):** TWW_single-turn,S3 = **62.04%** overall (+3.83 vs Thinking 58.21).
- **OVO-Bench, 4B (Table 3):** TWW_single-turn,S3 = **55.02%** overall vs Thinking 50.70 (+4.32). Multi-turn TWW_S3 = 51.80% at 255.91 tokens (**45.80% token reduction**).
- **Headline abstract claims:** +2.6% single-round on StreamingBench and +3.79% on OVO-Bench, and 56% output-token reduction in multi-round. (The +2.6% / +3.79% correspond to comparisons vs the online Thinking baseline / different backbone rows; the 4B single-turn-vs-batch-Thinking deltas above are +1.52 and +4.32, so the abstract's exact figure depends on the reference row — verified the underlying table cells, headline aggregate framing (n/r) for the exact chosen pairing.)
- **Offline transfer (Table 4, 4B):** TWW_single-turn,S3 lifts Video-MME 68.89→**73.41%** and LV-Bench 53.47→**57.68%** — long-range streaming supervision transfers to offline long-video reasoning.
- **Latency (Table 6, 4B):** streaming pipeline cuts TTFT from batch Thinking's 31203.69 to **2304.28 tokens (−92.6%)** at comparable accuracy (57.40% vs 58.52%); interleaved matches TTFT but is less accurate (55.35%).
- **Ablations (Table 5):** removing memory notes drops accuracy 57.40→**52.35%**; segment granularity trades off — 120s/60s cuts tokens to 230.46 but −2.07% acc, 30s/15s keeps acc but 380.50 tokens (+25.8%).
- **Attention analysis (Fig. 3):** Stage-2 checkpoint has strong recency bias; Stage 3 shifts attention mass to more distant segments, more so on MEMORY than FRAME tokens — memory notes act as compact long-range state.

No Zotero highlights present.

Takeaways: (1) compact per-segment text memory + implicit attention retrieval is a strong, cheap alternative to KV-cache retrieval for multi-turn streaming; (2) decoupling input/output positional encoding is the key enabler of watch-while-think parallelism and the big TTFT/token wins; (3) long-video Stage-3 training measurably re-allocates attention away from recency bias.

## How it connects (evolution)
- [[videollm-online]] and [[stream-vlm-qevd]] — the interleaved perception-generation baselines TWW argues against (memory erosion + serialization).
- [[dispider]] — decoupled perception/decision/reaction streaming design; TWW is a baseline-comparison peer (Dispider-7B in Tables 2–3).
- [[streamforest]] and [[streamagent]] — long-horizon streaming memory / agent-memory systems compared head-to-head on OVO-Bench; TWW favors compact textual notes over heavier memory structures.
- [[streammind]] and [[vispeak]] — streaming "know when to respond"/proactive lines; TWW instead focuses on multi-turn memory + latency under fixed question timestamps.
- [[streamingbench]] and [[ovo-bench]] — the two evaluation benchmarks it is trained and measured on.

## Open questions / limitations
- The memory note is a lossy text bottleneck: under severe frame corruption (Fig. 4) accuracy approaches the no-memory regime because note-writing becomes unreliable — visual detail not captured at write time is unrecoverable.
- Evaluation uses benchmark-provided question timestamps to define segments; real deployment must decide segment boundaries and when to answer online (proactivity is out of scope here).
- Gains over strong offline closed-source models remain modest (e.g. Gemini 1.5 Pro 70.26% offline StreamingBench vs TWW-8B 62.04% online) — the win is streaming causality/latency, not raw accuracy ceiling.
- CoT and memory notes are synthesized by GPT-5.2, so quality/biases of the teacher bound the learned memory-writing behavior.

*Verification: equations (Eqs. 1–12, mask + MRoPE offsets) transcribed from the PDF method sections (pp. 4–7); all accuracy/token/TTFT numbers checked against Tables 2, 3, 4, 5, 6 of the arXiv PDF (2603.11896v1); the abstract's exact +2.6%/+3.79% aggregate framing marked (n/r) where it did not map cleanly to a single verified table row.*

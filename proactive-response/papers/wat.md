---
zotero_key: null
authors: Zifan Han, Hongbo Sun, Jinglin Xu et al.
year: 2026
arxiv: 2603.13412
pdf: https://arxiv.org/pdf/2603.13412
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# WAT: Online Video Understanding Needs Watching Before Thinking

**Lineage role:** Structures online video response as two decoupled phases — a *query-agnostic* "watching" phase that continuously builds a hierarchical (short + long term) memory, and a *query-triggered* "thinking" phase that retrieves from it — arguing that comprehensive perception must precede reasoning so the model does not discard transient evidence before a question arrives.

## Problem — what was limited before this paper (short)
Online/streaming video LLMs must process an unbounded frame stream under a fixed compute/memory budget. Two dominant strategies each fail in opposite ways: (1) *aggressive temporal sampling / token pruning* (e.g. differential token dropping) meets the budget but destroys the fine-grained spatiotemporal detail needed for reasoning; (2) *reactive, query-triggered memory updates* only keep or fetch what the current query points at, so any important transient event that occurred *before* a question is issued is already gone — crippling backward tracing and long-horizon reasoning. Both couple "what to remember" to "what is being asked," which is a mistake when the question comes late.

## Key idea — the core insight, 2-4 sentences
Decouple perception from reasoning. In a **Watching** stage the model is query-agnostic: it maintains a *dual* memory — a high-fidelity FIFO **Short-Term Memory (STM)** sliding window for recent frames, plus a *fixed-capacity* **Long-Term Memory (LTM)** whose contents are kept semantically *diverse* by a redundancy-aware eviction policy (evict whatever is most similar to everything else). Only in the **Thinking** stage, when a query arrives, does the model fuse the query with STM context and use that enriched representation to retrieve the top-K most relevant historical frames from LTM for reasoning. A contrastive objective aligns retrieval with the query so the right evidence is fetched.

![[wat.png]]
> **Crux (Fig. 2).** The two-stage WAT pipeline: the left "Watching" block continuously inserts frames into LTM (redundancy-drop) and STM (FIFO-drop); the right "Thinking" block, triggered by a query, cross-attends the query with STM to form $\mathbf{z}_q$, retrieves top-K frames from LTM by similarity, and feeds STM ∥ retrieved ∥ query into the video-LLM — trained with a retrieval-alignment contrastive loss (RACL) plus next-token prediction (NTP). *Han et al. (2026), arXiv:2603.13412. Embedded for personal research reference.*

## Method + math — mechanism then objectives in full

**Frame features.** The stream $V=\{f_1,\dots,f_T\}$ is encoded per frame into feature maps $F_t$. Each memory entry keeps *two* things: the full feature map $R_i=F_t$ (for detailed reasoning) and a compact normalized *descriptor* $\hat{\mathbf r}_i$ obtained by global average pooling over spatial dimensions then L2-normalization:
$$\mathbf r_i = \text{GAP}(F_i),\qquad \hat{\mathbf r}_i = \frac{\mathbf r_i}{\lVert \mathbf r_i\rVert_2}.$$

**Short-Term Memory $\mathcal M_S$.** A FIFO queue of the $N_S = 16$ most recent frames — high-fidelity, updated every step; oldest frame dequeued on overflow. Preserves fine recent temporal dynamics.

**Long-Term Memory $\mathcal M_L$.** Fixed capacity $N_L = 768$. Updated *asynchronously* (less frequently than STM). When full, entries are replaced by the **redundancy-aware eviction policy** (Algorithm 1): compute pairwise cosine similarity over descriptors,
$$S_{ij} = \hat{\mathbf r}_i^{\top}\hat{\mathbf r}_j,$$
then a per-frame **redundancy score** = average similarity to all other entries,
$$\bar s_i = \frac{1}{N_L}\sum_{j=1}^{N_L} S_{ij}.$$
The most recent $\rho N_L$ frames ($\rho = 0.1$) are *protected* (set $E$). The evicted entry is the highest-redundancy non-protected frame,
$$i^{*} = \arg\max_{i\in V\setminus E}\ \bar s_i,$$
and the new frame overwrites slot $i^{*}$; the similarity matrix rows/cols for $i^{*}$ are refreshed. The full $S$ matrix is only recomputed every $U$ steps (an update-frequency knob) to save compute. Net effect: LTM stays a *compact, diverse* summary of the whole video rather than a recency window.

**Thinking — context-aware retrieval.** On query $Q$ (embedding $\mathbf Q$), fuse it with STM context through a cross-attention module $\mathcal A$ (query = $\mathbf Q$, key/value = pooled STM) with a residual:
$$\mathbf z_q = \mathbf Q + \mathcal A\big(\mathbf Q,\ \text{GAP}(\mathcal M_S)\big).$$
Score every LTM entry by cosine similarity to the fused query, take top-K ($K=32$):
$$s_i = \cos(\mathbf z_q,\hat{\mathbf r}_i)=\frac{\mathbf z_q\cdot\hat{\mathbf r}_i}{\lVert\mathbf z_q\rVert\,\lVert\hat{\mathbf r}_i\rVert},\qquad
\mathcal F^{*}=\{e_i \mid i\in\arg\text{TopK}(\{s_i\},K)\}.$$
The reasoning input is the concatenation $\mathcal F = \mathcal M_S \,\Vert\, \mathcal F^{*} \,\Vert\, Q$, fed to a multimodal LLM (Qwen2.5-VL-7B backbone) for the final response.

**Retrieval-Alignment Contrastive Learning (RACL, Algorithm 2).** A dedicated loss teaches the retriever to align the query with the *right* frames. Per batch of size $B$: the positive descriptor $\mathbf r$ is the normalized mean of the ground-truth retrieved frames; negatives $N$ come from (a) batch permutations of $\mathbf r$ and (b) a randomly sampled subset of LTM ($\mathcal F^{*}_R$). With temperature $\tau$:
$$s_i^{+}=\frac{\cos(\mathbf q_i,\mathbf r)}{\tau},\qquad s_{i,j}^{-}=\frac{\cos(\mathbf q_i,\mathbf n_j)}{\tau},$$
$$\mathcal L_{\text{RACL}} = -\frac{1}{B}\sum_{i=1}^{B}\log\frac{\exp(s_i^{+})}{\exp(s_i^{+})+\sum_j \exp(s_{i,j}^{-})}.$$

**Total objective.** Joint with next-token prediction:
$$\mathcal L = \mathcal L_{\text{RACL}} + \mathcal L_{\text{NTP}}.$$

**Training data — WAT-85K.** A synthetic online-VideoQA set of ~85K samples built to emphasize the two capabilities under-covered by prior data: *real-time perception* and *proactive forecasting* (plus backward tracing). Sources: an offline subset curated from LLaVA-Video-178K (balanced short/medium clips) and an online subset from TimeChat-Online-178K (short videos filtered out; complete long-form source videos retained rather than fragmented clips). Sampled at 1 FPS, max training length 4000 s.

## Explicit design choices
- **Two-stage, query-agnostic watching → query-triggered thinking** split — perception never conditioned on the (possibly-late) query, so transient events survive until asked about.
- **Dual memory** with different eviction rules: STM drops by **FIFO** (recency), LTM drops by **redundancy** (diversity). STM $N_S=16$; LTM $N_L=768$.
- **Two representations per LTM entry**: full feature map $R_i$ for reasoning + pooled normalized descriptor $\hat{\mathbf r}_i$ for cheap similarity/eviction.
- **Redundancy score = mean cosine similarity to all entries**; evict the argmax (most redundant), *protecting* the newest $\rho N_L$ ($\rho=0.1$).
- **Amortized eviction**: recompute the full similarity matrix only every $U$ steps; otherwise incremental row/col updates — keeps the watching stage cheap.
- **Query fused with STM before retrieval** via cross-attention + residual → conditioned query $\mathbf z_q$; top-$K=32$ from LTM.
- **Retrieval trained explicitly** with contrastive RACL loss (temperature $\tau=0.07$), negatives = batch permutations + random LTM samples; combined with NTP.
- **Backbone** Qwen2.5-VL-7B; **inference at 1 FPS** for streaming.
- **WAT-85K** deliberately over-weights real-time-perception and forecasting samples missing from prior streaming datasets.

## Key results / what to remember
No Zotero highlights present.

- **StreamingBench (real-time visual understanding), overall:** WAT **77.70** at 1 fps — beats the strongest open-source online prior TimeChat-Online-7B (75.36) and proprietary Gemini 1.5 Pro (75.69), GPT-4o (73.28), Qwen2.5-VL-7B baseline (73.68). Per-category WAT leads on OP 82.93, CS 85.49, ATP 84.24, PR 83.33, SU 73.17, ACP 72.44 (verified, Table I).
- **OVO-Bench, overall:** WAT **55.2** at 1 fps vs TimeChat-Online-7B 46.7 and Gemini 1.5 Pro 65.3 (proprietary still ahead). Sub-scores: Real-Time Visual Perception avg 64.7, Backward Tracing avg 45.2, Forward Active Responding avg 55.8 (verified, Table II).
- **Offline long-video (generalization):** MLVU 63.9, VideoMME overall 62.4, VideoMME 30–60 min **50.8** (best in its table on the long split) — competitive with Qwen2.5-VL-7B (66.9 / 63.2 / 50.4) and TimeChat-Online (65.4 / 62.5 / 49.2), i.e. no collapse on offline QA (Table III).
- **Ablation — hierarchical memory:** LTM-only gives OVO 50.3 / Streaming 75.7; adding STM → **55.2 / 77.7** (STM+LTM), and the Qwen2.5-VL baseline is 73.7 on StreamingBench (Table IV). The recent high-fidelity window contributes ~+4.9 OVO points.
- **Ablation — LTM length:** OVO 52.0→52.6→53.7→**55.2** and Streaming 76.9→77.1→77.2→**77.7** as $N_L$ = 32→128→256→768 (monotone; +3.2 OVO from 32→768) (Table V).
- **Ablation — loss:** NTP-only 50.1 / 76.4; +RACL at $\tau{=}0.01$ → 54.6 / 77.3; at $\tau{=}0.07$ → **55.2 / 77.7** — RACL adds ~+5 OVO points, temperature matters (Table VI).
- Runs in **real time at 1 FPS** and reports competitive throughput vs open-source online baselines (qualitative, paper claim — exact latency numbers n/r in extracted tables).

## How it connects (evolution)
- [[timechat-online]] — direct baseline and a *source* of WAT's online training data (TimeChat-Online-178K); its differential-token-dropping is the "aggressive pruning" WAT argues against.
- [[dispider]] — the disentangled Perception-Decision-Reaction pipeline WAT contrasts with; both decouple perception from response but WAT's split is watching-vs-thinking with explicit dual memory.
- [[videollm-online]] — the paper that first formalized online/streaming video understanding; WAT's reactive-memory critique targets this lineage.
- [[flash-vstream]] — dynamic memory-bank streaming VLM; a listed baseline WAT outperforms by a wide margin on both benchmarks.
- [[ovo-bench]] — one of the two headline evaluation benchmarks (real-time perception / backward tracing / forward active responding), the taxonomy WAT is designed around.
- [[streamingbench]] — the second headline benchmark (10 real-time visual-understanding categories) where WAT reaches 77.7.
- [[proactive-response]] — sub-topic hub: WAT's forecasting/proactive-forecasting emphasis in WAT-85K is squarely a proactive-response contribution.

## Open questions / limitations
- **Still below proprietary on OVO-Bench** (55.2 vs Gemini 1.5 Pro 65.3) — the watching-before-thinking design closes the open-source gap but not the frontier gap, especially on backward tracing (BT avg only 45.2) and hallucination detection (HLD 11.8).
- **Query-agnostic watching assumes the eviction diversity heuristic keeps the "right" transient events.** Redundancy = mean cosine similarity can still evict a rare-but-important frame if its descriptor happens to resemble the corpus; no explicit event/saliency signal guards it beyond recency protection.
- **Retrieval quality bounds reasoning:** top-K=32 is a hard cutoff and RACL is trained on synthetic ground-truth-retrieval labels; robustness to noisy/ambiguous queries and multi-hop temporal questions is not stressed.
- **Latency/throughput advantage is asserted but exact real-time numbers are not in the extracted result tables** (n/r) — the per-step cost of asynchronous LTM similarity updates at large $N_L$ deserves a reported profile.

*Verification: equations (GAP/normalize, $S_{ij}$, $\bar s_i$, $i^{*}$, $\mathbf z_q$, top-K, RACL loss, $\mathcal L=\mathcal L_{\text{RACL}}+\mathcal L_{\text{NTP}}$) and Algorithms 1–2 transcribed from the arXiv HTML method section and cross-checked against the rendered PDF page 3–4; all headline numbers (StreamingBench 77.70, OVO-Bench 55.2, VideoMME 30–60min 50.8, ablation Tables IV–VI) verified against the paper's own Tables I–VI. Fig. 2 cropped from PDF page 3.*

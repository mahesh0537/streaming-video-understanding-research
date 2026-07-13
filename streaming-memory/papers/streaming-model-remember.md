---
zotero_key: null
authors: Haonan Ge, Yiwei Wang, Hang Wu, Yujun Cai
year: 2026
arxiv: 2606.16353
pdf: https://arxiv.org/pdf/2606.16353
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# What Should a Streaming Video Model Remember?

**Lineage role:** Poses selective memory as a *budgeted allocation* problem — not whether to use history but how much of a fixed evidence budget to spend on it — and answers it with SelectStream's write/keep/read triad (surprise windowing, priority consolidation, query-conditioned graph reasoning) over a fixed-capacity latent memory graph.

## Problem — what was limited before this paper (short)
Streaming video models must answer at any moment using only past frames, under fixed memory and compute. The dominant fixes bolt on memory banks, retrieval modules, or token compression to hoard long-range history. But strong *recent-window* baselines (keep only the last few frames) are surprisingly hard to beat: indiscriminate history injection dilutes current-scene perception, so more memory can hurt. The real question is therefore selective — which slice of history is worth a place in a bounded context — not merely whether to retain it.

## Key idea — the core insight, 2-4 sentences
Keep the *current* observation directly visible to a frozen VLM, and expose *history* only through a small, query-conditioned evidence budget of latent tokens. Cast this as **budgeted online latent evidence allocation** over three separately-budgeted knobs — memory capacity $N$, subgraph budget $B$, evidence budget $M$ — and govern it with three coordinated mechanisms answering *when to write* (surprise-driven adaptive windowing), *what to preserve* (priority-preserving consolidation), and *how to read* (query-conditioned graph reasoning). Retrieved evidence is calibrated and injected as latent tokens, never replaying frames or growing context with stream length, so active state is $O(1)$ in processed length.

![[streaming-model-remember.png]]
> **Crux (Figure 2).** The full SelectStream pipeline: a frozen VLM encoder projects frames to embeddings; (a) surprise-driven adaptive windowing decides *when* to close a segment and write, (b) latent visual memory does a gated write into a fixed-capacity dynamic memory graph and consolidates *what* to keep under budget $N$, (c) query-conditioned graph attention reasoning routes a small subgraph to decide *how to read*, and (d) the top-$M$ nodes become calibrated latent evidence tokens injected into a frozen MLLM decoder. *Ge et al. (2026), arXiv:2606.16353. Embedded for personal research reference.*

## Method + math — the mechanism, then the central objective/equations

**Memory as a graph.** SelectStream maintains a dynamic latent evidence graph
$$G_t = (\mathcal{M}_t, E_t),$$
where $\mathcal{M}_t$ is the active node set (each node $i$ holds a latent state $h_i\in\mathbb{R}^d$ plus metadata: temporal span, accumulated surprise, write/read/merge counts) and $E_t$ holds temporal and semantic edges. Three user-facing budgets are decoupled: $N$ (memory capacity / node count), $B$ (retrieved subgraph size), $M$ (injected evidence tokens).

**(1) Surprise-driven adaptive windowing (SAW) — *when to write*.** At each step compute a per-frame surprise from two signals — attention shift and feature change — then smooth it:
$$
s_t = \lambda\, s_t^{\text{attn}} + (1-\lambda)\, s_t^{\text{feat}},\qquad
\bar s_t = \rho\, \bar s_{t-1} + (1-\rho)\, s_t,
$$
with $s_t^{\text{attn}} = \mathrm{JS}(A_t \,\|\, A_{t-1})$ (Jensen–Shannon divergence between successive frozen-backbone attention maps, head-averaged and pooled into visual bins) and $s_t^{\text{feat}} = 1 - \cos(g_t, g_{t-1})$ over pooled projected embeddings $g_t = \mathrm{Pool}(\Phi_v(x_t))$. A segment starting at $t_{\text{start}}$ closes when it is long enough **and** a boundary trigger fires:
$$
t - t_{\text{start}} \ge L_{\min}\ \ \text{and}\ \ \Big(\bar s_t > \theta_{\text{high}}\ \ \text{or}\ \ \sum\nolimits_{k=t_{\text{start}}}^{t}\bar s_k \ge B_s\ \ \text{or}\ \ t - t_{\text{start}} \ge L_{\max}\Big),
$$
i.e. close on a surprise spike, on accumulated surprise energy $B_s$, or on max length $L_{\max}$. This spends memory where the stream actually changes.

**(2) Latent visual memory + priority-preserving consolidation — *what to preserve*.** On segment closure, cached embeddings are encoded into one latent by a lightweight Transformer with temporal positions and learned query pooling:
$$
z_j = \mathrm{SegEnc}(\{g_t : t\in \text{seg}_j\}).
$$
Write decision: compare $z_j$ to active nodes via $r_i = \cos(z_j, h_i)$. If $\max_i r_i > \tau_r$ **and** $\bar s_j < \tau_s$ (similar and not too surprising), update the best node $i^\*$ with a **gated** update; otherwise spawn a new node:
$$
g = \sigma\!\big(\mathrm{MLP}([z_j; h_{i^\*}; \bar s_j; \Delta t])\big),\qquad
h_{i^\*} \leftarrow (1-g)\, h_{i^\*} + g\, f_{\text{write}}(z_j, h_{i^\*}),
$$
where $\Delta t$ is time since that node's last update; the gate prevents unconditional averaging that would erase salient changes. Edges carry event continuity and semantic recurrence:
$$
w_{uv}^{\text{tmp}} = \exp(-c\,\bar s_v)\ \ (\text{lower surprise} \Rightarrow \text{stronger temporal link}),\qquad
w_{uv}^{\text{sim}} = \cos(h_u, h_v).
$$
When $|\mathcal{M}_t| > N$, merge the pair with the **smallest** graph-aware penalty
$$
\pi_{uv} = p_{uv}^{\text{sim}} + p_{uv}^{\text{pri}},
$$
where $p^{\text{sim}}$ favors merging semantically redundant nodes and $p^{\text{pri}}$ *protects* surprising, frequently accessed, or recently updated evidence — so consolidation removes redundancy, not salience, keeping $|\mathcal{M}_t|\le N$.

**(3) Query-conditioned subgraph retrieval + graph attention reasoning (GAR) — *how to read*.** Encode the query $u = \mathrm{Enc}_q(q)$ into the same latent space (learned query tokens over prompt-side hidden states, mean-pooled). Coarse per-node relevance:
$$
\text{score}_i = \cos(u, h_i) + \eta\,\hat s_i - \xi_\ell\,\hat\ell_i - \xi_m\,\hat m_i^{\text{merge}},
$$
which rewards semantic match and salience but *downweights* coarse merged nodes (large span $\hat\ell_i$, high merge count $\hat m_i^{\text{merge}}$). Take top-$k$ seeds, expand via query-conditioned routing $\psi_{ij}(q) = \text{score}_j + \alpha_r\, w_{ij}$ over temporal/semantic neighbors, respecting subgraph budget $B$. Then run relational graph attention over the subgraph:
$$
e_{ij} = q_i^\top k_j + b_{\text{type}} + b_{\Delta t} + b_w,\qquad
\alpha_{ij} = \mathrm{softmax}_j(e_{ij}),\qquad
h_i^{(\ell+1)} = h_i^{(\ell)} + \sum_{j\in\mathcal{N}(i)} \alpha_{ij}\, v_j,
$$
with learned biases for edge type, temporal gap, and edge support. After $K$ layers, re-score refined nodes and keep the top-$M$.

**(4) Evidence calibration + injection.** Each retained node becomes a latent evidence token
$$
E = \{e_m\}_{m=1}^{M},\qquad e_m = \mathrm{LN}(W_e\, \tilde h_{i_m}),
$$
where $W_e$ calibrates distributional drift accumulated through segment encoding, gating, consolidation, and graph reasoning, and projects into the MLLM token-embedding dimension. Inject into the frozen VLM by concatenation:
$$
X_{\text{in}} = [\,X_{\text{prompt}};\, X_{\text{cur}};\, e_1, \dots, e_M\,],\qquad y \sim p_\theta(y \mid X_{\text{in}}).
$$

**Training objective.** Supervised fine-tuning with a frozen backbone; only SelectStream modules update:
$$
\mathcal{L} = \mathcal{L}_{\text{ans}} + \beta\,\mathcal{L}_{\text{ret}} + \gamma\,\mathcal{L}_{\text{spar}},
$$
where $\mathcal{L}_{\text{ans}}$ is the autoregressive answer loss, $\mathcal{L}_{\text{ret}}$ is a supervised evidence-retrieval loss (set to $0$ on answer-only data), and $\mathcal{L}_{\text{spar}}$ regularizes diffuse routing / redundant evidence. A theory sidebar (Prop. 2) lower-bounds expected attention on the relevant evidence node by $\mathbb{E}[\alpha_{h^\*}] \ge r\cdot \frac{\exp(\mu_{\text{sig}})}{\exp(\mu_{\text{sig}}) + (M-1)\exp(\sigma^2/2)}$, tying answer quality to retrieval recall $r$ and evidence separability $\mu_{\text{sig}}$.

## Explicit design choices
- **Current frame stays raw; history is the only thing that gets compressed.** Present observation is fed directly to the frozen VLM; memory is exposed *only* through the $M$ evidence tokens — preserving current-scene perception that recent-window baselines are strong at.
- **Frozen backbone.** Backbone VLM never updates; only trainable parts are the segment encoder, gate/write MLPs, query encoder, GAR module, and the calibration/projection layer $W_e$. Trained separately for Qwen2.5-VL-7B and Qwen3-VL-8B.
- **Three decoupled budgets** $N$ (capacity), $B$ (retrieval scope), $M$ (generation context) — each tuned independently (budget-sensitivity sweep in Fig. 3).
- **Event-adaptive, not fixed, segmentation** via dual surprise (attention JS-divergence + feature cosine), so segment boundaries land on real scene changes.
- **Gated write, not overwrite** — protects salient history from being averaged away; write-vs-create decided by similarity $\tau_r$ and surprise $\tau_s$ thresholds.
- **Priority-protecting consolidation** merges only redundant nodes; surprise / access / recency shields important evidence when $|\mathcal{M}_t| > N$.
- **Latent tokens, no frame replay** — retrieval injects calibrated latents, keeping active state $O(1)$ in stream length (constant query latency + GPU memory, Fig. 4).
- **Trained on Streamo-Instruct-465K**; timestamped examples give retrieval supervision via overlap of annotated evidence intervals with memory-node spans, untimestamped ones use answer loss only.

## Key results / what to remember
Evaluated with a causal streaming protocol against offline VLMs, streaming VLMs, recent-window baselines, and simple memory policies (uniform/FIFO/random/similarity-merge).

- **StreamingBench:** 82.67% (SelectStream-Qwen3-VL-8B), 81.42% (Qwen2.5-VL-7B) — best among all compared; gains of **+2.08% / +2.95%** over the matched recent-window baselines (Qwen3-VL-8B+4f = 80.59, Qwen2.5-VL-7B+4f = 78.47).
- **OVO-Bench:** overall 67.03% (Qwen3-VL-8B) and 65.71% (Qwen2.5-VL-7B). RT/BT Avg. 72.48% / 70.95%. Largest gains on **Backward Tracing** (history-dependent): BT Avg. 51.90→61.05 (Qwen2.5-VL-7B) and 54.00→62.20 (Qwen3-VL-8B), while Real-Time perception stays high (82.76% / 80.85%).
- **Offline generalization (VideoMME / MLVU / MVBench avg):** best average 74.4% (Qwen3-VL-8B, +1.7% over frozen backbone); 70.4% (Qwen2.5-VL-7B, up from 68.3% backbone, with +2.7% VideoMME and +2.8% MLVU). Smaller MVBench gain — memory helps most on long videos needing temporal evidence, not short-range action.
- **Ablations (SelectStream-Qwen2.5-VL-7B):** full allocation = 81.42 StreamingBench / 65.71 OVO-Bench / 73.0 MLVU; replacing SAW with fixed segments drops to 79.86 / 64.21 / 71.2; removing gated writing causes a further consistent drop — both the adaptive segmentation and the gated write are load-bearing.
- **Efficiency:** query latency and peak GPU memory stay ~flat as stream length grows (Fig. 4), confirming the $O(1)$-state claim under fixed $N,B,M$.

No Zotero highlights present.

Takeaways: the paper reframes streaming memory from "store more history" to "allocate a bounded evidence budget selectively," and shows the write/keep/read triad beats both recent-window baselines and prior memory-bank streaming models *without* trading away offline video ability. Backward-Tracing gains are the cleanest evidence that selective latent memory — not just recent frames — is doing the work.

## How it connects (evolution)
- [[streaming-memory]] — this note is a central instance of that sub-topic's core question (capacity allocation / retention strategy).
- [[streamo]] — SelectStream fine-tunes on Streamo-Instruct-465K and lists Streamo among streaming baselines it surpasses.
- [[hermes-kv]] — HERMES is a memory-bank streaming baseline SelectStream compares against and beats on StreamingBench/OVO.
- [[streamforest]] — another streaming memory model used as a recent baseline in Table 1.
- [[thinkstream]] — reasoning-oriented streaming baseline; SelectStream's RT/BT averages exceed it.
- [[streamingbench]] — the primary streaming benchmark (RT perception) SelectStream tops at 82.67%.

## Open questions / limitations
- Results rest on frozen Qwen2.5/Qwen3-VL backbones and Streamo-Instruct training; how much of the gain transfers to other backbones or without retrieval-supervised timestamps is untested here.
- The surprise metric, thresholds ($\theta_{\text{high}}, \tau_r, \tau_s, B_s, L_{\min/\max}$), and budgets ($N,B,M$) are hand-set; robustness to badly-tuned budgets beyond the reported sweep is unclear.
- Prop. 2's attention-concentration bound assumes retrieval recall $r$ and evidence separability as inputs — it characterizes a *good* retriever rather than guaranteeing one; real failure comes from retrieval misses, which the paper only probes via Recall@M.
- Consolidation uses all-pair penalties for small $N$ but top-similarity candidates for real-time — the accuracy cost of that approximation at large $N$ is not quantified in the main text.

*Verification: equations (surprise, boundary trigger, gated write, consolidation penalty, GAR scores, evidence injection, training loss, Prop. 2 bound) transcribed from the arXiv HTML method section and cross-checked against the rendered PDF page 4/6; all headline numbers (StreamingBench 82.67/81.42, OVO-Bench 67.03/65.71, RT/BT 72.48/70.95, BT 51.90→61.05 & 54.00→62.20, offline avg 74.4/70.4, ablation 81.42/65.71/73.0 vs 79.86/64.21/71.2) verified against the paper's own Table 1, Table 2, and Section 4.3 text in the downloaded PDF.*

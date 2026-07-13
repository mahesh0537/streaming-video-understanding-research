---
zotero_key: null
authors: Ke Ma, Jiaqi Tang, Bin Guo, et al. (Northwestern Polytechnical University; Tsinghua University; HKUST; Harbin Engineering University)
year: 2026
arxiv: 2605.07575
pdf: https://arxiv.org/pdf/2605.07575
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# Response-G1: Explicit Scene Graph Modeling for Proactive Streaming Video Understanding

**Lineage role:** A *training-free* proactive streaming pipeline that makes the per-frame silence-vs-respond decision explicit by generating query-guided **scene graphs** online, storing them in memory, and retrieving the top-K query-relevant ones to gate the trigger — turning implicit "have I seen enough evidence?" reasoning into structured evidence-condition alignment.

## Problem — what was limited before this paper (short)
Proactive streaming assistants (e.g. [[mmduet]], [[dispider]], [[streamagent]]) must decide, frame by frame, whether the evidence accumulated so far already satisfies the user's query and it is time to speak, versus staying silent. Prior methods do this with **implicit** modeling: they feed raw frame tokens plus a trigger prompt into a Video-LLM and hope the model's latent state captures both the observed visual evidence and the query's response condition. That coupling is opaque and error-prone — the model has no explicit representation of *which relations it has actually observed* versus *which relations the query is waiting for*, so response timing is inaccurate (early false triggers or missed cues), and improving it usually meant per-frame supervised fine-tuning.

## Key idea — the core insight, 2-4 sentences
Represent **both** the streaming evidence and the query's response condition as structured **scene graphs** (object–predicate–object triplets), so the trigger decision becomes a principled alignment between observed graphs and the expected "query condition graph." Response-G1 generates a query-guided scene graph per clip online, linearizes it to text, embeds and stores it in memory, then at each frame retrieves the top-K graphs most similar to the query and appends them (timestamp-tagged) to the trigger prompt. Everything runs through a frozen Video-LLM's prompting and text-embedding — **no fine-tuning, no frame-wise annotations**.

![[response-g1.png]]
> **Crux (Figure 2).** The three-stage framework: (1) online query-guided scene-graph generation from each video clip into a scene-graph memory, (2) memory-based top-K similarity retrieval against the parsed query condition graph, and (3) a retrieval-augmented streaming pipeline where the retrieved graphs + trigger prompt drive the frozen Video-LLM to output Silence or Response. *Ke Ma et al. (2026), arXiv:2605.07575. Embedded for personal research reference.*

## Method + math — the mechanism, then the equations in full

**Setup and the proactive decision.** Let $\mathcal{F}_{1:t}$ be the frames observed up to time $t$ and $\mathcal{Q}_{t_\text{ask}}$ the user query issued at time $t_\text{ask}$. A joint encoder $\mathcal{J}:(\mathcal{Q},\mathcal{F}_{1:t})\mapsto \mathbf{H}_t$ produces a joint hidden state $\mathbf{H}_t\in\mathbb{R}^{d_h}$, and a scene-graph extraction function $\mathcal{S}:\mathbf{H}_t\mapsto\mathcal{G}_t$ produces the structured graph. The per-frame proactive decision is

$$\mathcal{D}\big(\mathbf{H}_t,\ \mathcal{S}(\mathbf{H}_t)\big)\rightarrow r_t,\qquad r_t\in\mathcal{R}=\{\texttt{silence},\ \texttt{response}\}. \tag{1}$$

There are **no learned parameters** in $\mathcal{D}$ — it is realized by prompting the frozen Video-LLM. The novelty is that $\mathcal{D}$ is conditioned on an *explicit* graph $\mathcal{S}(\mathbf{H}_t)$, not only the opaque hidden state.

**Stage 1 — Online query-guided scene graph generation.** For a clip $\mathcal{C}_t$ centered at $t$, a prompted Video-LLM emits a scene graph $\mathcal{G}_t=(\mathcal{O}_t,\mathcal{P}_t)$ where $\mathcal{O}_t$ are object nodes (with attributes, e.g. *red, large*) and $\mathcal{P}_t$ are predicate edges (spatio-temporal relations, e.g. *next_to, holding*). Equivalently a triplet set:

$$\mathcal{G}_t=\Big\{\ \tau_t^{ij}=(o_t^i,\ p_t^{ij},\ o_t^j)\ \Big|\ o_t^i,o_t^j\in\mathcal{O}_t;\ p_t^{ij}\in\mathcal{P}_t\ \Big\}. \tag{2}$$

Generation is **conditioned on the query** $\mathcal{Q}$ to suppress irrelevant detail and focus on query-salient evidence (contrast with "object-guidance," which injects parsed objects and causes hallucinated triplets → premature triggers):

$$\mathcal{G}_t=\mathcal{S}(\mathcal{C}_t;\ \mathcal{Q}). \tag{3}$$

**Stage 2 — Memory-based scene graph retrieval.** Each graph's triplets are linearized into natural-language phrases and concatenated,

$$\Phi_t=\bigoplus_{i,j}\phi_t^{ij},$$

embedded with the Video-LLM's **text encoder** and mean-pooled into a single vector $\mathbf{g}_t\in\mathbb{R}^{d}$. The query is parsed into a "query condition graph," linearized and embedded the same way into $\mathbf{g}_q$ (Table 5 shows this *query-graph text* format beats using the raw query text — format consistency between video graphs and query graph matters). Relevance is cosine similarity,

$$\mathrm{sim}(\mathcal{G}_t,\mathcal{Q})=\frac{\mathbf{g}_t\cdot\mathbf{g}_q}{\lVert\mathbf{g}_t\rVert\,\lVert\mathbf{g}_q\rVert},$$

and the retrieved context is the top-K most-similar historical graphs,

$$\mathcal{G}_t^{\text{ctx}}=\Big\{\mathcal{G}_\tau\ \Big|\ \tau\in\mathrm{TopK}\big(\{\mathrm{sim}(\mathcal{G}_i,\mathcal{Q})\}_{i=1}^{t},\ K\big)\Big\}.$$

**Stage 3 — Retrieval-augmented streaming pipeline.** At the *trigger* phase, frame embeddings, the timestamp-encoded retrieved graphs, and a trigger instruction $\mathbf{p}_\text{trg}$ are concatenated as the Video-LLM input that yields $r_t$:

$$\big[\mathbf{f}_1,\ldots,\mathbf{f}_t\big]\ \oplus\ \Psi(\mathcal{G}_t^{\text{ctx}})\ \oplus\ \mathbf{p}_\text{trg}. \tag{9}$$

Retrieved graphs are prefixed with **timestamp tokens** to preserve temporal ordering (ablation: this drives the gains on temporally-grounded subtasks like CRR):

$$\Psi(\mathcal{G}_t^{\text{ctx}})=\bigoplus_{i\in\mathcal{I}^{\text{ctx}}}\mathcal{E}_\text{text}\big(\texttt{<}t_i\texttt{s>}\ \oplus\ \Phi_i\big). \tag{10}$$

When the decision fires at $t_\text{res}$, the *response* phase generates the answer with the same augmentation but the encoded original query $\mathbf{q}$ in place of the trigger prompt:

$$\big[\mathbf{f}_1,\ldots,\mathbf{f}_{t_\text{res}}\big]\ \oplus\ \Psi(\mathcal{G}_{t_\text{res}}^{\text{ctx}})\ \oplus\ \mathbf{q}. \tag{11}$$

**Reactive mode is a special case:** set $t_\text{res}=t_\text{ask}$ (respond immediately at query time) and the same architecture answers standard reactive streaming questions — which is why it also improves reactive subtasks.

## Explicit design choices
- **Frozen backbone, no fine-tuning:** Qwen3-VL-8B used as-is; all three stages are prompting + text-embedding retrieval. No frame-wise trigger labels needed.
- **Explicit scene graphs as the evidence/condition interface:** observed evidence and the query's response condition are both structured object–predicate–object graphs, making alignment interpretable.
- **Query-guided (not object-guided) generation** (Eq. 3): conditioning graph generation on the *query text* rather than injecting parsed objects avoids over-focusing on anticipated objects and the resulting false-triplet hallucination / premature triggers (Table 4).
- **Textual linearization + shared text encoder + mean pooling:** graphs become phrases, embedded by the Video-LLM's own text encoder, pooled to one vector per graph — cheap unified similarity space, and consistent with how the query graph is embedded.
- **Query *condition graph* for retrieval, not raw query text** (Table 5): format consistency between video graphs and query graph is what makes the cosine match meaningful.
- **Timestamp-token prefixing** of retrieved graphs (Eq. 10) for temporal reasoning.
- **Top-K memory retrieval:** simple, and robust — performance is stable across K (Figure 3), with K=1 sufficient for latest-frame-focused tasks.
- **Fixed clip size** for scene-graph generation (a stated limitation / future adaptive-trigger direction).
- **Latency budget:** per-frame SGG ≈448 ms, retrieval ≈21 ms, trigger ≈356 ms → ≈825 ms total ≈ 1.2 FPS, ≈2.1 FPS with KV-cache (Table 6), on A100-80GB FP16. Meets the 1 FPS streaming requirement.

## Key results / what to remember
Backbone Qwen3-VL-8B (8B). Best vs. open-source **streaming** Video-LLMs; StreamAgent and TimeChat-Online are the strongest prior open-source baselines.

**OVO-Bench (Table 1).** Forward Active Responding is the *proactive* mode; Real-Time Visual Perception and Backward Tracing are *reactive*.
- **FAR average 58.2** (REC 41.9 / SSR 71.1 / CRR 61.7) vs. second-best StreamAgent **45.4** (REC 35.9 / SSR 48.4 / CRR 52.0) → **+12.8** on the proactive average.
- Real-Time Visual Perception avg **73.6** vs. second-best TimeChat-Online **61.3** → **+12.3** (reactive).
- **Overall avg 61.3** (top among open-source streaming models; StreamAgent 49.4, TimeChat-Online 45.6).

**StreamingBench (Table 2).** PO (Proactive Output) is the *proactive* subtask; the rest reactive.
- **PO 44.0** vs. second-best StreamAgent **28.9** → **+15.1** on proactive output.
- Real-Time Visual Understanding "All" **77.5** vs. second-best TimeChat-Online **75.4** → **+2.1** (reactive).
- **Overall avg 73.7** (StreamAgent 70.2, TimeChat-Online 70.9).

**Ablations.**
- Retrieval augmentation (Table 3, OVO ACR/HLD/CRR + SB CS/PR/PO): W/o Retrieval 66.1 / 28.0 / 55.4 / 83.6 / 79.6 / 36.8 → W/o Timestamp 74.0 / 33.6 / 60.4 / 87.7 / 82.9 / 43.6 → **Full 74.3 / 33.9 / 61.7 / 88.0 / 83.3 / 44.0**. Retrieval is the big lift; timestamp encoding adds the temporally-grounded gains (CRR +1.3, PO +0.4).
- Generation guidance (Table 4, PO/REC/SSR/CRR): W/o Guidance 38.8 / 34.1 / 66.9 / 59.4 → Object-Guidance 43.6 / 40.2 / 67.9 / 61.3 → **Query-Guidance 44.0 / 41.9 / 71.1 / 61.7**. Object-guidance hallucinates and triggers early; query-guidance is best.
- Top-K (Figure 3): roughly flat over K=1..5 (PO ≈44, REC ≈41, SSR ≈70, CRR ≈61).

No Zotero highlights present.

Takeaways: (1) making the silence/respond decision an *explicit graph alignment* — rather than implicit hidden-state reasoning — is worth ~+12–15 points on proactive subtasks with a **frozen** model; (2) the retrieved-graph augmentation *also* helps reactive tasks, so the graph memory is generally useful context, not just a trigger gate; (3) query-guided generation (vs object injection) is the guard against premature-trigger hallucination.

## How it connects (evolution)
- [[streamagent]] — the strongest prior open-source proactive baseline it beats (+12.8 FAR on OVO, +15.1 PO on StreamingBench); Response-G1's explicit-graph gating is the alternative to StreamAgent's implicit approach.
- [[mmduet]] — earlier proactive "respond-or-wait" streaming head; part of the lineage of learned per-frame trigger decisions that Response-G1 replaces with training-free graph alignment.
- [[dispider]] — decoupled perception/decision proactive streaming; a design-space sibling for how to structure the silence-vs-respond loop.
- [[timechat-online]] — reactive streaming baseline and the pixel/config reference it follows; second-best on the reactive subtasks Response-G1 tops.
- [[ovo-bench]] — the proactive (Forward Active Responding) + reactive benchmark it reports on.
- [[streamingbench]] — the second benchmark, source of the PO (Proactive Output) proactive metric.

## Open questions / limitations
- **Causal / "why" queries:** scene graphs capture object-relation evidence well but the authors note limited benefit for questions needing causal or intent reasoning that isn't expressible as triplets.
- **Fixed clip granularity:** the static clip size for SGG is suboptimal; event-level or semantics-level adaptive triggering is left as future work.
- **Generation hallucination trade-off:** open-set LLM-based SGG can still emit false triplets; the authors suggest task-specific fine-tuning of the generator to better balance relevance vs. factuality (which would forgo the training-free property).
- **Cost:** ~825 ms/frame (mostly SGG 448 ms + trigger 356 ms) — meets 1 FPS but leaves little headroom without the KV-cache optimization; scaling to higher frame rates or smaller backbones is unexplored.

*Verification: equations (1)–(3), (9)–(11) and the retrieval/similarity formulas transcribed from the arXiv HTML method section and cross-checked against the rendered PDF page 3; all headline numbers (OVO-Bench Table 1, StreamingBench Table 2, ablation Tables 3–4) read directly off the rendered PDF pages 5–6 and recomputed for internal consistency (FAR avg 58.2 vs 45.4 = +12.8; PO 44.0 vs 28.9 = +15.1). An early WebFetch mis-parse of SSR was corrected against the table image.*

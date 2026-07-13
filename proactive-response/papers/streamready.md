---
zotero_key: null
authors: Shehreen Azad, Vibhav Vineet, Yogesh Singh Rawat (UCF / Microsoft Research)
year: 2026
arxiv: 2603.08620
pdf: https://arxiv.org/pdf/2603.08620
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# StreamReady: Learning What to Answer and When in Long Streaming Videos

**Lineage role:** Frames proactive streaming QA as a *joint* correctness-and-timing problem — contributes the **Answer Readiness Score (ARS)** timing metric, the **ProReady-QA** long-video benchmark with annotated evidence windows, and a learnable `<RDY>`-token readiness gate on top of a hierarchical visual memory (CVPR 2026).

## Problem — what was limited before this paper (short)
Prior streaming-video LLMs answer a question at whatever moment they are asked, and prior benchmarks score only *whether* the answer is correct — never *when* it was produced relative to when the supporting evidence actually appears in the stream. In a proactive setting, a question can be posed **before** its answer is observable, so a model that blurts an answer early is hallucinating from insufficient observation, while one that waits far too long is useless. Accuracy-only metrics reward or ignore both failure modes equally. Existing systems also either keep all frames (unscalable to 30-60 min streams) or aggressively compress and lose the fine-grained evidence a proactive answer depends on, and they lack any explicit "do I have enough evidence yet?" signal.

## Key idea — the core insight, 2-4 sentences
StreamReady treats *what to answer* and *when to answer* as one coupled decision. It maintains a hierarchical **Visual Memory Tree** that keeps recent frames raw and progressively abstracts older ones, does query-aware coarse-to-fine retrieval over that memory through a dual-branch (short-/long-term) Q-Former reasoning module, and attaches a learnable `<RDY>` token whose **Readiness Head** gates output: the model keeps watching until accumulated evidence crosses a readiness threshold, then answers. The paper pairs this with **ARS**, a timing metric that harshly penalizes premature answers and mildly penalizes late ones, and folds it into an *effective accuracy* `Acc_e = Acc × ARS`.

![[streamready.png]]
> **Crux (Figure 2).** The StreamReady pipeline: a frozen visual encoder feeds a Visual Memory Tree; on a question the dual-branch reasoning module produces short-term ($z_s$) and long-term ($z_\ell$) representations, a `<RDY>` token + Readiness Head decide "Ready?" — if yes the long-term representation (enriched by the Contextual Memory Bank of past QA) goes to the frozen LLM, if no it keeps watching. *Azad, Vineet, Rawat (2026), arXiv:2603.08620. Embedded for personal research reference.*

## Method + math — mechanism, then objective/metrics in full

### Memory storage
**Visual Memory Tree $\mathcal{M}_V$ (three levels).** Streaming frames from a *frozen* visual encoder are stored so that recent detail is preserved and old context is compressed:
- **$\mathcal{M}_{V1}$** — a FIFO buffer of the most recent raw frame embeddings (short-term detail).
- **$\mathcal{M}_{V2} = \{c_1,\dots,c_J\}$** — a mid-level centroid set built by EMA K-means over frames evicted from $\mathcal{M}_{V1}$. For an evicted frame $f_o$ with decay factor $\alpha$ and adaptive threshold $\tau_t$:
$$c_j \leftarrow \begin{cases} (1-\alpha)\,c_j + \alpha f_o, & \text{if } \operatorname{sim}(f_o, c_j) \ge \tau_t \\ \text{new centroid}, & \text{otherwise} \end{cases}$$
$\tau_t$ tightens in stable scenes (favoring merges) and relaxes when novelty rises (allowing new clusters).
- **$\mathcal{M}_{V3} = \{s_1,\dots,s_U\}$** — coarse prototypes abstracted from $\mathcal{M}_{V2}$ once it hits capacity $J$ or drifts, updated by EMA over the centroids assigned to prototype $u$ (index set $\mathcal{I}_u$):
$$s_u \leftarrow (1-\alpha)\,s_u + \alpha\Big(\tfrac{1}{|\mathcal{I}_u|}\sum_{j\in\mathcal{I}_u} c_j\Big).$$
A lightweight mini-K-means re-clusters prototypes if they become heterogeneous.

**Contextual Memory Bank $\mathcal{M}_C$.** Stores, per prior QA turn $i$, the question embedding $q_i$ and the learned answer representation $a_i$, giving a semantic history for multi-turn reuse (complementary to the visual $\mathcal{M}_V$).

### Query-aware dual-branch reasoning
When question $q_i$ arrives the model switches from passive encoding to active retrieval over a dual-branch Q-Former.
- **Short-term branch** attends over raw recent frames plus the question:
$$z_s = \mathcal{Q}_s\big(\operatorname{Concat}[\mathcal{M}_{V1}, q_i]\big).$$
- **Long-term branch** does coarse-to-fine retrieval: score prototypes, then centroids, then reason (conditioned on $z_s$):
$$\mathcal{M}_{V3}^{t} = \operatorname{Top\text{-}K}\big((W_s q_i)^\top \mathcal{M}_{V3}\big), \quad \mathcal{M}_{V2}^{t} = \operatorname{Top\text{-}m}_{c \in C_0}\big((W_c q_i)^\top c\big),$$
$$z_\ell = \mathcal{Q}_\ell\big(\operatorname{Concat}[\mathcal{M}_{V3}^{t}, \mathcal{M}_{V2}^{t}, q_i],\ z_s\big),$$
where $C_0$ is the union of centroids under the selected prototypes. The long-term representation is finally enriched with $\mathcal{M}_C$ context and sent to the frozen LLM to generate the answer.

### Readiness mechanism
A learnable **`<RDY>` token** is embedded inside the long-term Q-Former block $\mathcal{Q}_\ell$ so it "sees" exactly the evidence the answer will be built from. A lightweight **Readiness Head** maps it to $R_{\text{pred}}(t) \in [0,1]$ each timestep. Training uses **similarity-based pseudo-labels** (no ground-truth evidence windows needed at train time): centroids highly similar to $z_\ell$ form a pseudo-positive region $P$, dissimilar ones a pseudo-negative region $N$. A pairwise contrastive loss pushes readiness higher inside the evidence region, plus a temporal-smoothness regularizer:
$$\mathcal{L}_{\text{ctr}} = -\log \sigma\big(R_{\text{pred}}(t^+) - R_{\text{pred}}(t^-)\big),\quad t^+\in P,\ t^-\in N,$$
$$\mathcal{L}_{\text{rdy}} = \mathcal{L}_{\text{ctr}} + \lambda_{\text{reg}}\,\big\|\nabla_t R_{\text{pred}}(t)\big\|_1.$$
At inference the LLM is triggered to answer only when $R_{\text{pred}}$ exceeds a readiness threshold (0.35 in the paper); otherwise the model continues watching.

### Evaluation protocol — the Answer Readiness Score (ARS)
Each ProReady-QA question carries a ground-truth **evidence window** $[t_s, t_e]$ ($t_s$ = when sufficient evidence first appears, $t_e$ = when it stops being valid). Given an answer time $t_a$, ARS combines an **Early Penalty (harsh)** and a **Late Penalty (mild)**, using median evidence duration $\tau$ for scale and $\epsilon$ for stability (Eqs. 10-11; $\sigma$ = sigmoid):
$$\text{EP} = \operatorname{softmin}\!\Big(1,\ 2\,\sigma\big(\gamma_e\,\tfrac{t_a - t_s}{\tau + \epsilon}\big)\Big),\qquad \text{LP} = \operatorname{softmin}\!\Big(1,\ \operatorname{softmax}\big(0,\ 1 - \gamma_\ell\,\tfrac{t_a - t_e}{\tau + \epsilon}\big)\Big).$$
EP $\to 1$ as $t_a \to t_s$ and drops sharply for answers before evidence onset; LP stays near 1 for small delays and decays for prolonged hesitation. Defaults $\gamma_e = 6$ (sharp), $\gamma_\ell = 1$ (gentle) encode that speculating early is worse than answering slightly late. Aggregated over $N$ questions and folded into effective accuracy:
$$\text{ARS} = \frac{1}{N}\sum_i \text{EP}_i\cdot \text{LP}_i,\qquad \text{Acc}_e = \text{Acc}\times \text{ARS}.$$
(For multi-answer questions ARS is computed per valid turn and averaged. A temporal PAUC-style analysis of ARS vs. answer time is also reported in the appendix; its exact definition is n/r in the extracted text.)

### ProReady-QA construction
- **5 task types:** Sequential Steps Recognition (SSR), Repetitive Event Count (REC), Clues Reveal Responding (CRR), Causal Trigger Detection (CTD), Goal-State Detection (GSD).
- **Source video:** 10 Ego-4D (~1 h each) + 22 MovieNet (~30 min each); ~5k proactive QA pairs, 30-60 min videos, multi-turn with local and global dependencies.
- **Generation pipeline:** (1) dense captioning of 30 s segments (8 sampled frames); (2) multi-minute summarization; (3) proactive-QA generation (answers require *future* frames); (4) multi-turn follow-ups referencing earlier turns; (5) frame-wise VLM evidence annotation (earliest/latest supporting frame → $[t_s,t_e]$); (6) human refinement of future-dependency and window accuracy.

## Explicit design choices
- Frozen visual encoder and frozen LLM; only the dual-branch Q-Former **Reasoning Module** and **Readiness Head** are trained (Q-Former initialized from pretrained weights).
- Backbone: **Qwen-2-VL 7B**.
- Three-tier memory: raw FIFO recent frames + EMA-K-means centroids + coarse prototypes — keep detail near the present, compress the past, bound memory for hour-long streams.
- **Adaptive** cluster threshold $\tau_t$ (scene-stability driven) rather than a fixed merge radius.
- Coarse-to-fine query-aware retrieval (Top-K prototypes → Top-m centroids) so long-context reasoning stays cheap.
- Readiness as a **`<RDY>` token inside $\mathcal{Q}_\ell$ + MLP head**, not a separate auxiliary MLLM or heuristic — it shares the reasoning representation with the answer generator.
- **Pseudo-label** readiness supervision from $z_\ell$–centroid similarity, so training needs no ground-truth evidence timestamps.
- LLM answer trigger at readiness threshold **0.35**; ARS penalty sharpness $\gamma_e=6$, $\gamma_\ell=1$.
- Offline-benchmark evaluation disables readiness/contextual reasoning and uses progressively truncated video prefixes (records answer when first correct).

## Key results / what to remember
All verified against the paper's Table 2 (rendered PDF page 6); other tables from the arXiv HTML extraction.

- **ProReady-QA (Table 2, 7B, avg over 5 tasks):** StreamReady **Acc 56.4 / ARS 0.69 / Acc_e 0.53** — best; second-best StreamBridge 53.1 / 0.60 / 0.42; InfiniPot-V 52.0 / 0.47 / 0.36. Per-task Acc/ARS: SSR 74.3/0.78, CRR 63.3/0.73, REC 39.6/0.68, GSD 61.2/0.68, CTD 43.5/0.59. Reported gain ≈ **+3% Acc, +9% ARS** over the best baseline; the ARS/Acc_e gap is much larger than the raw-Acc gap, i.e. the win is mostly in *timing*.
- **Streaming benchmarks (Table 3):** best on proactive subsets of StreamingBench / OVOBench (e.g. proactive StreamingBench 48.2 vs. StreamBridge 32.6, ViSpeak 43.9) and on VStream-QA long-video (RE 64.8 / RM 57.2). (from HTML extraction)
- **Offline long-video (Table 4):** avg **69.8** across VideoMME 65.8 / MLVU 71.3 / MVBench 71.8 / EgoSchema 70.4, above HierarQ (67.0 avg) and the Qwen-2-VL backbone (65.7 avg) — the memory+retrieval stack helps even without the readiness gate. (from HTML extraction)
- **Ablations (Table 5):** query-aware retrieval drives the big accuracy jump (e.g. REC 20.7 → 39.4, GSD 35.1 → 60.9); the readiness mechanism barely moves Acc but sharply raises ARS. The `<RDY>`+Head variant gives the best ARS (e.g. CTD 0.59) vs. MLP-only (0.42), heuristic (0.43), or auxiliary-MLLM (0.46). (from HTML extraction)

No Zotero highlights present.

Takeaways: (1) Timing is a first-class, separately-measurable axis in proactive streaming — ARS/Acc_e expose it where accuracy alone hides it. (2) Coupling the readiness signal to the *same* representation used for answering (the `<RDY>` token in $\mathcal{Q}_\ell$) beats bolt-on readiness predictors. (3) Pseudo-labeled readiness sidesteps expensive evidence-window annotation for training.

## How it connects (evolution)
- [[streambridge]] — strongest proactive-streaming baseline on ProReady-QA; StreamReady's readiness gate is the timing mechanism it lacks.
- [[vispeak]] — proactive/duplex streaming peer; compared on proactive StreamingBench/OVOBench subsets.
- [[dispider]] — earlier "when to respond" decoupled perception/decision streaming model; StreamReady reframes the timing decision as evidence-readiness with an explicit score.
- [[infinipot-v]] — memory-compression baseline; contrast with StreamReady's hierarchical Visual Memory Tree.
- [[proactivevideoqa]] — sibling proactive-QA benchmark under this sub-topic; ProReady-QA adds explicit evidence windows + the ARS timing metric.
- [[streamingbench]] — one of the streaming benchmarks StreamReady evaluates on for generalization.

## Open questions / limitations
- REC and CTD stay hard in absolute terms (REC 39.6, CTD 43.5 Acc) — counting/causality over hour-long streams is far from solved; appendix reports readiness failures specifically on counting.
- Readiness training uses similarity-based pseudo-labels, not the real evidence windows — how well the learned $R_{\text{pred}}$ actually tracks true evidence onset (vs. just visual novelty) is only indirectly validated.
- ARS design bakes in fixed asymmetry ($\gamma_e=6,\gamma_\ell=1$) and a median-duration scale $\tau$; the "right" early-vs-late cost is application-dependent and not learned.
- ProReady-QA is modest scale (~5k QA, 32 videos from Ego-4D + MovieNet); breadth of domains and human-agreement on evidence windows is limited.

*Verification: ProReady-QA Table 2 numbers and the EP/LP/ARS formulas (Eqs. 10-11), γ_e=6, γ_ℓ=1, threshold 0.35 checked against the rendered PDF (page 6); method equations (memory EMA updates, dual-branch retrieval, readiness loss) and Tables 3-5 taken from the arXiv HTML extraction of 2603.08620 — those table numbers not independently re-rendered are marked "(from HTML extraction)".*

---
zotero_key: null
authors: Zicheng Zhao, Kangyu Wang, Shijie Li, Rui Qian, Weiyao Lin, Huabin Liu (corresponding) — Shanghai Jiao Tong University group
year: 2025
arxiv: 2506.10516
pdf: https://arxiv.org/pdf/2506.10516
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# CogStream: Context-guided Streaming Video Question Answering

**Lineage role:** Frames streaming video QA as a *context-selection* problem — instead of feeding all history, the model must *retrieve* the relevant historical dialogue and *compress* the relevant visual events; ships both a densely-annotated benchmark (CogStream) and a retrieval-based baseline (CogReasoner).

## Problem — what was limited before this paper (short)
Streaming video QA models keep accumulating context — every past frame and every past QA turn — as the stream grows. This is expensive and, worse, *distracting*: irrelevant history dilutes attention away from the frames and dialogue turns that actually determine the current answer. Prior streaming benchmarks also rarely test whether an answer *logically depends* on earlier turns (multi-turn co-reference, dialogue recall, answers that update as the video evolves). The paper argues the real skill is picking "the most relevant historical contextual information" at each timestep, not ingesting everything.

## Key idea — the core insight
Reformulate streaming QA so that at each turn the model must actively **select which slice of history matters** — both visually and textually — before reasoning. On the data side they build **CogStream**, a benchmark where questions are deliberately *logically associated* with earlier turns, annotated with which prior QAs are "relevant." On the model side they build **CogReasoner**, which (1) compresses the growing visual stream into a few relevant event tokens via temporal-semantic clustering + question-aware compression, and (2) uses an LLM to retrieve only the relevant historical QA pairs and decide whether the current question even needs vision. This turns "remember everything" into "retrieve the right thing," which is both cheaper and more robust to distractor history.

![[cogreasoner.png]]
> **Crux (Figure 4).** CogReasoner's three modules with a single frozen LLM used in three roles — *Forward* (encode clustered events under a question), *Select* (retrieve relevant history QAs + set the text-only flag δ), *Generate* (answer over time-sorted interleaved visual+text tokens). It IS the paper's thesis: select-then-reason over a compressed stream instead of attending over all history. *Zhao et al. (2025), arXiv:2506.10516. Embedded for personal research reference.*

## Method + math

CogReasoner has three modules driven by one shared (frozen) LLM used in three passes (Forward / Select / Generate), plus two task-specific LoRA adapters.

**Setup.** At timestep $t$: cumulative video $V_t=\{v_1,\dots,v_t\}$ (segments/events), historical dialogue $QA_{t-1}=\{qa_1,\dots,qa_{t-1}\}$, and new question $q_t$. The model must answer $q_t$ using only the *relevant* subset of $(V_t, QA_{t-1})$.

**1. Visual Stream Compression.**
*Temporal-Semantic Clustering* groups frames into coherent events with a composite distance that blends feature similarity and temporal proximity:
$$D=\sqrt{\mathcal{F}_{nf}(d_f)^2+\alpha\,\mathcal{F}_{nt}(d_t)^2}$$
where $d_f$ = feature distance to the cluster centroid, $d_t$ = temporal distance, $\mathcal{F}_{nf},\mathcal{F}_{nt}$ are normalizers, and $\alpha$ weights temporal coherence (implemented as time-based k-means, Algorithm 1). *Question-aware Streaming Compression* then scores each event $j$ for relevance $s_j$ to $q_t$: events with $s_j\ge\theta$ are kept as full token sequences ("Hierarchical Compression" preserves their detail); low-relevance events are collapsed to a single average-pooled summary token so the video's overall temporal skeleton is retained cheaply.

**2. Historic Dialogue Retrieval (HDR).** The LLM reads $QA_{t-1}$ and $q_t$ and returns both the relevant subset and a binary flag:
$$QA_{retrieved},\ \delta=\mathrm{LLM}(QA_{t-1},\,q_t),\qquad \delta\in\{0,1\}$$
where $\delta=1$ marks a *purely linguistic* question (answerable from dialogue alone, e.g. "Dialogue Recalling"), in which case the visual stream is skipped entirely — saving compute and avoiding visual interference.

**3. Video-text Interleave Reasoning.** The compressed visual tokens and retrieved textual QAs are concatenated in **temporal order** (sorted by timestamp, then interleaved) and the answer is generated:
$$a_t=\mathrm{LLM}(V_{compressed},\,QA_{retrieved},\,\delta,\,q_t)$$

**Two-stage training (LoRA).**
- *Stage 1 (retrieval):* text-only fine-tuning of a task-specific LoRA on 100k+ historical-QA + current-question combinations, teaching precise QA selection and δ prediction.
- *Stage 2 (reasoning):* visual encoder frozen; the projection layer and the LLM (distinct LoRA) fine-tuned end-to-end on 48k+ QA pairs over 800+ videos, with ground-truth preceding QAs and their selection status supplied.

### CogStream benchmark (the eval protocol)
**Task taxonomy — three families, 11 capabilities.**
- **Basic QA** (~34.6%): current segment only — actions, objects, attributes, co-reference.
- **Streaming QA** (~54.6%): needs updated visual $V_t$ *and* textual $QA_{t-1}$ — Sequence Perception, Dialogue Recalling, Dynamic Updating, Object Tracking, Causal Reasoning.
- **Global QA** (~10.8%): after the video ends — Global Analysis, Overall Summary.

**Construction pipeline.** Videos are event-segmented (SceneTiling + manual refinement). QA is generated semi-automatically by an MLLM with a *semantic-propagation* strategy: basic $qa_t^{Bas.}=\mathrm{MLLM}(v_t)$; streaming/global $qa_t^{Str./Glo.}=\mathrm{MLLM}(v_t,L_{t-1},S_{t-1})$ where $L_{t-1}$=segment titles, $S_{t-1}$=running summaries. Each candidate historical pair gets a **Helpfulness Score** HS ∈ 0–7 (content relevance + logical supportiveness); only HS > 4 pairs enter the "relevant QA set." Dialogue chains are assembled chronologically with a **Composite Score**
$$SC_i=\max_{qa_j\in Seq}\{\mathrm{RS}_{i,j}+\alpha\cdot\mathrm{len}(qa_j)\},\qquad P(qa_i)=\frac{\exp(SC_i)}{\sum_k \exp(SC_k)}$$
giving probabilistic, diverse dialogue paths.

**Metrics.** Answer quality is LLM-judged on a 0–100 scale, averaged over five axes: **IA** (Information Accuracy), **DC** (Detail Completeness), **CA** (Context Awareness), **TP** (Temporal Precision), **LC** (Logical Consistency). The retrieval sub-task is scored with standard IR **Accuracy / Precision / Recall / F1** against the annotated relevant-QA sets.

**Scale.** 1,088 videos (curated from 6,361 raw), 59,032 QA pairs (→118,058 with combinations), ~5.02 segments/video, 1–7+ min clips. Human validation: 86.72% answer accuracy, 96.35% relevant-QA-set accuracy.

## Explicit design choices
- **Select-then-reason, not attend-over-all:** an explicit retrieval step (LLM `Select`) picks relevant history QAs before generation, rather than concatenating all context.
- **Dual compression policy:** high-relevance events kept at full token resolution; low-relevance events collapsed to one pooled token — preserves temporal skeleton at low cost. Threshold $\theta=0.45$.
- **Text-only shortcut (δ):** purely linguistic questions skip the visual stream entirely, cutting compute and visual distraction.
- **Time-based k-means clustering** (composite feature+temporal distance) to segment the stream into events; reported best K/F ratio ≈ 1/15.
- **Frozen shared LLM in three roles** (Forward/Select/Generate) + **two separate LoRA adapters** (one for retrieval, one for reasoning) — decouples "what to retrieve" from "how to answer."
- **Timestamp-sorted interleaving** of visual and textual tokens so the model sees history in temporal order.
- **Benchmark design:** questions engineered to be *logically associated* with prior turns, with annotated relevant-QA sets — enabling direct evaluation of context selection (the retrieval IR metrics), not just answer quality.

## Key results / what to remember
All scores are 0–100 averaged over the 5 quality metrics on the CogStream test set (Table 3). Basic / Streaming / Global / Avg:
- **CogReasoner (zero-shot, 7B, 1fps):** 74.8 / 68.0 / 70.9 / **67.32**.
- **CogReasoner† (fine-tuned, 7B):** 77.3 / 68.8 / 75.4 / **72.26** — beats all open-source baselines; approaches **GPT-4o** 78.4 / 72.1 / 77.0 / **73.90**.
- Strong open baselines: MiniCPM-V-2.6 (8B) **66.84**, VideoLLaMA3 (7B) **66.52**; streaming-memory baselines lag: ReKV **43.18**, Flash-VStream **40.58**.
- **Context strategy (Table 5, Avg):** No context 66.04 < All context 70.48 < **Retrieved (ours) 72.26** < Ground-truth oracle 77.40 — retrieval beats dumping all history, and there's ~5 pts headroom to perfect retrieval.
- **Noise robustness (Table 6, 30% distractor QAs):** VideoLLaMA3 "all context" drops 69.74→65.02 (−4.72); CogReasoner "retrieved" stays ~71.94 (stable). Selection makes the model robust to injected irrelevant history.
- **Retrieval quality (Appendix, relevant-QA selection):** CogReasoner Acc 0.89 / P 0.73 / R 0.72 / **F1 0.72**, vs GPT-4o F1 0.60 and VideoLLaMA3 F1 0.42.
- **HDR module generalizes (Table 7):** adding CogReasoner's Historic Dialogue Retrieval lifts VideoLLaMA2 50.52→55.16 (+4.64) and MiniCPM-V-2.6 65.46→67.78 (+2.32).
- **Ablation (Table 4, Avg):** full model 72.26 vs "no clustering" 71.48 vs "only selection" 70.52 vs baseline 70.64 — both clustering and question-aware selection contribute.

No Zotero highlights present.

Takeaways: (1) In streaming QA, *what history you feed* matters more than *how much* — an explicit retrieval step both raises accuracy and confers robustness to distractor turns. (2) A frozen LLM reused as retriever + reasoner (two LoRAs) is a cheap, effective recipe. (3) The oracle-retrieval gap (77.40 vs 72.26) says context selection is still the bottleneck. (4) CogStream is the first streaming-video benchmark to *directly* score dialogue-relevance retrieval, not just answer quality.

## How it connects (evolution)
- [[flash-vstream]] and [[rekv]] — the streaming-memory baselines CogReasoner outperforms; they compress/cache all history whereas CogReasoner *retrieves* it.
- [[streaming-memory]] — this note is a core exemplar of the retrieval-over-memory design axis for the sub-topic hub.
- [[streamingbench]] / [[ovo-bench]] — sibling streaming-video benchmarks; CogStream adds the logical-association/context-selection dimension they lack.
- [[videollm-online]] — the online-dialogue framing CogStream's multi-turn streaming QA builds on.
- [[querystream]] — question/query-conditioned streaming processing, closely related to CogReasoner's question-aware compression + retrieval.

## Open questions / limitations
- The δ text-only shortcut and the retrieval step both depend on the LLM's own selection quality; retrieval F1 is only 0.72 and the oracle gap (+5.14 Avg) shows selection errors cap performance.
- Evaluation is LLM-judged on 5 soft axes (IA/DC/CA/TP/LC) — no hard exact-match; judge bias and reproducibility of the 0–100 scores are unquantified here.
- Clustering + relevance scoring add per-turn overhead; the paper emphasizes token savings but does not report end-to-end latency/throughput vs streaming-cache baselines like ReKV.
- Videos are short (1–7+ min, ~5 segments); scaling the retrieval-over-dialogue idea to hour-long streams with hundreds of turns is untested.

*Verification: title, authors, module equations, dataset statistics and metric definitions checked against the arXiv HTML (v3) of 2506.10516; all headline numbers (Tables 3–7 and the relevant-QA-selection F1) cross-checked against the paper's own tables. Figure 4 cropped from the PDF page 5.*

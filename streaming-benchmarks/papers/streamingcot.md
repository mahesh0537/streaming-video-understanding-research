---
zotero_key: null
authors: Yuhang Hu et al. (Zhengzhou U. / Institute of Automation, CAS; Kuaishou Technology)
year: 2025
arxiv: 2510.25332
pdf: https://arxiv.org/pdf/2510.25332
tier: deep
subtopics: [streaming-benchmarks]
tags: [streaming-video-understanding, streaming-benchmarks]
---
# StreamingCoT: A Dataset for Temporal Dynamics and Multimodal Chain-of-Thought Reasoning in Streaming VideoQA

**Lineage role:** ACM MM 2025 dataset that couples *time-evolving* answers with *explicit spatiotemporal chain-of-thought* traces — the first streaming-VideoQA resource whose ground truth shifts as evidence accumulates and whose reasoning is annotated step-by-step with keyframe + bounding-box grounding.

## Problem — what was limited before this paper (short)
Prior VideoQA datasets use **static annotation**: one global answer is stapled to the whole clip. That ignores the defining property of streaming video — the *correct answer changes over time* as events unfold (a count grows, a state transitions, a conclusion only becomes derivable once enough clues arrive). They also lack **explicit reasoning annotations**, so models are trained/judged only on the final answer and drift toward shortcut statistical correlations rather than interpretable step-by-step deduction. There was no dataset that jointly captured (a) temporally-dependent answers and (b) verifiable multimodal reasoning chains.

## Key idea — the core insight
Replace the single global label with a **dynamic hierarchical annotation** in which the video is cut into semantic segments and each segment carries its own (possibly indeterminate → refined) answer, and attach to every segment an **explicit spatiotemporal chain-of-thought (ST-CoT)** whose reasoning steps are grounded to concrete keyframes (temporal anchors) and GroundingDINO bounding boxes (spatial evidence). A five-stage, human-validated construction pipeline makes this reproducible: filter → hierarchical dense captioning + adaptive segmentation → dynamic QA generation with distractor design → multimodal CoT synthesis → iterative human validation.

![[streamingcot.png]]
> **Crux (Figure 1).** The five-stage StreamingCoT construction pipeline: (1) geographically-balanced collection + multimodal filtering, (2) per-second dense captioning fused into semantic segments via Dynamic Semantic Fusion, (3) dynamic QA pairs with distractor-aware options, (4) multimodal CoT synthesis with keyframe + bounding-box grounding, (5) iterative human validation — it is the paper's whole contribution, a standardized recipe for temporally-aware, reasoning-annotated streaming VideoQA. *Hu et al. (2025), arXiv:2510.25332. Embedded for personal research reference.*

## Method + math — the construction / annotation protocol in full
This is a **dataset paper**; the "math" is the construction pipeline, its segmentation/grounding equations, and the human-validation criteria. There are **no model-benchmark tables** in the paper.

**Stage 1 — Collection + multimodal filtering.** Start from 10,288 short-form videos via the YouTube API; stratified geographic sampling and ≤20 videos/channel for diversity. Three-stage filter: (i) *social validation* — keep videos with >5,000 aggregated interactions; (ii) *audio* — Whisper transcripts, discard any exceeding ~2 words per 10 s (removes talk-heavy clips); (iii) *visual* — ≥720p, optical-flow motion check (drop ≥15% high-variance frames or ≥85% static occupancy), and CLIP+MLP aesthetic score ≥7/10. Yields 5,745 high-quality videos.

**Stage 2 — Hierarchical dense captioning + Dynamic Semantic Fusion (DSF).** Generate per-second frame captions $\{C_1,\dots,C_n\}$ with InternVL3 (a fine-grained temporal baseline). Adaptive temporal segmentation merges consecutive seconds with a LIFO stack and cosine-similarity thresholding ($\theta=0.9$). Consecutive-caption similarity:
$$S_{t-1,t} = \frac{\mathbf{E}(C_{t-1})\cdot \mathbf{E}(C_t)}{\lVert\mathbf{E}(C_{t-1})\rVert\,\lVert\mathbf{E}(C_t)\rVert}$$
Segmentation rule — extend the current segment while the running product of similarities stays above threshold, otherwise start a new one:
$$Seg_i = \begin{cases} \mathrm{Merge}(t_{k-n},\dots,t_k) & \text{if } \prod_{m=k-n}^{k-1} S_{m,m+1} \ge \theta \\[4pt] \mathrm{Push}(t_k) & \text{otherwise} \end{cases}$$
Each segment gets a context-aware dense caption conditioned on its per-second captions, a visual window, and the *history* of prior dense captions (for inter-segment coherence):
$$DC_i = \mathrm{VLLM}_{\text{merge}}\big(C_{t_s},\dots,C_{t_e};\ \mathcal{W}_{\text{visual}};\ DC_1,\dots,DC_{i-1}\big)$$
The 0–1 s segment is anchored separately to stop cumulative caption delay. 20 experts validate semantic completeness, narrative coherence, and temporal alignment.

**Stage 3 — Dynamic QA construction.** Six temporally-evolving question types: cumulative counting, periodic pattern recognition, sequential step recognition, state duration, object state recognition, clue-revealing response. InternVL2.5 drafts 5 candidate QA pairs/video from the dense captions; GPT-4o ranks them on visual-grounding consistency, temporal validity, and reasoning-complexity alignment. **Distractor-aware options**: three plausible distractors per question built by systematic temporal perturbation — *temporal misalignment* (elements from adjacent segments), *partial pattern compliance* (satisfies a subset, violates the global constraint), *state-transition fallacy* (physically plausible but temporally inconsistent), and *premature inference* (concluding before enough evidence). Humans verify answer temporal alignment, distractor plausibility, and segment-answer coherence at segment granularity.

**Stage 4 — Multimodal CoT synthesis.** Per segment, pick the **keyframe** as the frame whose visual embedding best matches the segment dense-caption text embedding:
$$Keyframe_i = \Big\{\arg\max_{f_i \in \mathcal{F}_t}\ \mathrm{Sim}\big(\mathbf{E}_{\text{vis}}(f_i),\ \mathbf{E}_{\text{text}}(DC_i)\big)\ \big|\ t \in Seg_i\Big\}$$
Generate an **initial CoT** with InternVL2.5, conditioned on the segment caption, a history window $H$ of prior (caption, CoT) pairs, and the keyframe:
$$CoT_i^{\text{init}} = \mathrm{VLLM}\big(DC_i,\ \{DC_{i-\tau},\,CoT_{i-\tau}\}_{\tau=1}^{H},\ Keyframe_i\big)$$
Answer-consistency verification regenerates any CoT that contradicts the segment ground truth. Extract up to **3 key objects** from the initial CoT, $Obj_i = \mathrm{VLLM}(CoT_i^{\text{init}})$, and spatially ground them in the keyframe with GroundingDINO:
$$BBoxs_i = \big\{(o_i,\ \mathrm{GroundingDINO}(o_i, Keyframe_i))\ \big|\ \forall o_i \in Obj_i\big\}$$
Fuse verified reasoning + temporal anchors + spatial boxes into the final spatiotemporal CoT:
$$CoT_t^{\text{ST}} = \Phi_{\text{fuse}}\big(CoT_t^{\text{init}},\ Keyframe_i,\ BBoxs_i\big)$$
with the grounding constraint that every reasoning step $r_j$ is backed by an object–time–box triple:
$$\forall r_j \in CoT_t^{\text{ST}}\ \exists (o_j,t_j,bbox_j):\ r_j \propto \mathcal{V}(o_j,t_j)\otimes \mathcal{S}(bbox_j)$$
The realized traces use explicit `<think>…</think>`, `<start>…</start>` (frame/time anchor), `<bbox>[x,y,w,h]</bbox>`, and `<answer>…</answer>` tags.

**Stage 5 — Iterative human validation.** 15 certified annotators score each ST-CoT on four axes: spatiotemporal consistency ($\forall(o_j,t_j,bbox_j)\in CoT_t^{\text{ST}},\ \mathcal{V}(o_j,t_j)\otimes\mathcal{S}(bbox_j)\neq\emptyset$), temporal causality (chronological, no anachronistic inference), evidence completeness (each step cites sufficient keyframe/history evidence), and answer-derivation soundness (answer follows without logical gaps). Flagged samples trigger corrective regeneration:
$$CoT_t^{\text{ST*}} = \Phi_{\text{fuse}}\big(\mathrm{VLLM}_{\text{revise}}(CoT_t^{\text{init}}, \mathcal{E}_{\text{annot}}),\ Keyframe_i,\ \mathrm{ReGround}(Obj_i)\big)$$
Maximum 3 iterations; samples still failing go to full manual annotation.

## Explicit design choices
- **Answer-per-segment, not per-clip**: temporally-dependent labels; answers can be *indeterminate early* and only resolve once enough evidence accrues (models the streaming "temporal closure" property).
- **Per-second dense captions first, then fuse** — a fine-grained temporal baseline before semantic segmentation, so grounding is precise.
- **DSF with fixed cosine threshold $\theta=0.9$** and a LIFO stack; running-product-of-similarities merge rule balances coherence vs. over-segmentation.
- **Anchor the 0–1 s segment separately** to prevent propagation of caption-generation delay.
- **Keyframe = max visual↔caption embedding similarity**; grounds abstract reasoning in one representative frame per segment.
- **≤3 key objects per CoT**, box-grounded with GroundingDINO — bounds reasoning complexity while capturing the primary interactions.
- **Distractor engineering by temporal perturbation** (temporal misalignment / partial-pattern / state-transition fallacy / premature inference) — encodes characteristic streaming failure modes into the multiple-choice options.
- **Post-hoc answer-consistency check + regenerate** to prevent hallucinated reasoning chains.
- **Structured CoT token format** (`<think>/<start>/<bbox>/<answer>`) — machine-parseable spatiotemporal reasoning supervision.
- **Two large human panels** (20 for captions, 15 for CoT) with ≤3 refinement iterations then manual fallback — quality vs. scalability trade-off.
- **Sourcing/diversity guards**: >5k social interactions, ≤20 videos/channel, geographic stratification, ≥720p, aesthetic ≥7.

## Key results / what to remember
This is a resource paper; the "results" are the dataset's own composition (verified against the paper's statistics text). **The paper reports no model-evaluation / baseline benchmark table.**

- **5,000** high-quality short-form videos, average duration **25.6 s** (filtered down from 10,288 collected → 5,745 after visual filtering).
- **243,185** time-anchored per-second dense subtitles.
- **68,940** semantic segments, avg **12 segments/video** (via DSF).
- **34,470** dynamic QA pairs (stated as "5 QA pairs per video").
- **68,940** multimodal CoT annotations and **206,820** key-object bounding boxes.
- Six question types with distribution: sequential step recognition 22.4%, object state recognition 19.3%, cumulative counting 18.2%, state duration 16.1%, periodic pattern recognition 15.7%, clue-revealing responses 8.3%.
- Construction constants: DSF threshold **θ=0.9**; **≤3** key objects/CoT; validation panels of **20** (captioning) and **15** (CoT); **≤3** refinement iterations.
- Note: the stated per-video counts don't multiply cleanly against 5,000 (e.g. 68,940 segments / 12 ≈ 5,745, the pre-final video count; 34,470 = 68,940/2). Reported here as the paper writes them.

No Zotero highlights present.

Takeaways: the durable idea is **evolving-answer + grounded-reasoning supervision** for streaming VideoQA — segment-level dynamic answers plus `<think>/<start>/<bbox>/<answer>` ST-CoT traces are directly reusable as a *supervision/eval target* for online video LLMs, distinct from streaming benchmarks that only score a final answer.

## How it connects (evolution)
- [[streaming-benchmarks]] — sub-topic hub; StreamingCoT is a construction/annotation resource within it.
- [[proactivevideoqa]] and [[streamingbench]] — streaming-VideoQA benchmarks that score final answers; StreamingCoT adds *time-evolving answers + reasoning traces* rather than only accuracy.
- [[ovo-bench]] and [[svbench]] — online/streaming eval suites emphasizing temporal, evolving queries; complementary problem framing.
- [[thinkstream]] — explicit reasoning/"thinking" in streaming settings; StreamingCoT is the annotated CoT-supervision counterpart to such reasoning methods.
- [[dispider]] and [[videollm-online]] — online video-LLM models that could consume StreamingCoT's per-segment CoT as training/eval supervision.

## Open questions / limitations
- **Short-form only** (avg 25.6 s): unclear whether the DSF segmentation + CoT recipe scales to minutes/hours-long streams where temporal dependencies are longer-range.
- **No model results in the paper**: it defines the data and format but gives no baseline numbers, so difficulty and headroom for current streaming LLMs are unquantified here.
- **Pipeline is model-conditioned** (InternVL3/2.5, GPT-4o, GroundingDINO): annotation biases/errors from these generators may be baked in despite human validation; fixed θ=0.9 and ≤3-object caps may under-segment/under-ground busy scenes.
- **Sourcing bias**: YouTube + social-interaction + aesthetic filters skew toward polished, popular content and specific cultural contexts despite geographic stratification.

*Verification: equations, pipeline stages, question-type taxonomy, and all dataset statistics checked verbatim against the arXiv HTML (v1) text and the paper's statistics section; Figure 1 cropped from the arXiv PDF page 2 and visually confirmed. Confirmed the paper contains no model-benchmark table (checked HTML twice). Zotero not consulted (offline).*

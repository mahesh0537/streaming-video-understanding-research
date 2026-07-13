---
zotero_key: null
authors: Jiaer Xia, Peixian Chen et al. (Hong Kong Baptist University; Tencent Youtu Lab)
year: 2026
arxiv: 2512.21334
pdf: https://arxiv.org/pdf/2512.21334
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# Streaming Video Instruction Tuning

**Lineage role:** an end-to-end instruction-tuning recipe that folds frame-level "when to speak" decisions (Silence / Standby / Response) directly into next-token prediction, plus a 465K multi-task dataset, converting offline video LLMs into proactive streaming assistants without a separate decision module.

## Problem — what was limited before this paper (short)
Offline video LLMs (Qwen2.5-VL, LLaVA-Video, InternVideo2.5) must ingest a whole clip before emitting a single output, so they cannot run on an unbounded, live stream where the model has to decide *when* the visual context warrants a response and *what* granularity to emit. Two prior families of streaming fixes both have weaknesses: (a) an **auxiliary decision module** that segments the stream into clips before invoking the offline model (Dispider, StreamBridge) — heavy compute, poor multi-turn context retention, and a decision head too small to parse complex instructions; and (b) **special-token timing** trained end-to-end but tied to a single behavior — VideoLLM-Online / StreamingVLM predict an `[EOS]`-style token for real-time narration only and cannot balance silence vs. response across diverse tasks. Existing streaming benchmarks (OVO-Bench, StreamBench, SVBench) also lean on multiple-choice QA, under-testing open-ended instruction following like grounding and captioning.

## Key idea — the core insight, 2-4 sentences
Reformulate a streaming video as an **interleaved multi-turn dialogue** where each one-second segment is a user turn (`<t s-t+1 s><video>`) and the assistant turn is one of three **decision-state tokens** — `<Silence>` (keep watching), `<Standby>` (relevant input detected, wait for completeness), `<Response>` (emit the answer now, followed by the text). Because the decision is just another token in the sequence, the model does frame-level "when to respond" judgement and content generation in **one pass**, trained by ordinary next-token prediction — no external controller. To make this trainable despite `<Silence>` dominating (~12:3:2 Silence:Standby:Response), a **focal + frequency-balanced loss** is applied to the three state tokens. The recipe is fed by **Streamo-Instruct-465K**, where each video carries multiple task annotations (narration, action/event caption, event grounding, time-sensitive QA) with unified temporal boundaries.

![[streamo.png]]
> **Crux (Figure 1).** One video (bartender making a cocktail) annotated for all five streaming tasks at once — each row pairs an instruction with time-boundaried responses interleaved with `<Silence>` gaps; this multi-task, response-timed supervision is exactly what Streamo-Instruct-465K teaches. *Xia, Chen et al. (2026), arXiv:2512.21334. Embedded for personal research reference.*

## Method + math — mechanism then objective in full

**Streaming reformulation.** Offline models take a full video $V=\{v_1,\dots,v_T\}$ and a question $Q$ and emit $A$ in one shot. Streaming instead exposes only partial observations $V_{:t}=v_1,\dots,v_t$ with $t\le T$ (no future frames). Streamo segments a full video into $N$ contiguous one-second pieces
$$V=\{V^{(1)},V^{(2)},\dots,V^{(N)}\} \tag{1}$$
each tagged with explicit boundary markers (e.g. `<2s-3s>`), and builds a multi-turn dialogue
$$D=\{(V^{(1)},R^{(1)}),(V^{(2)},R^{(2)}),\dots,(V^{(N)},R^{(N)})\} \tag{2}$$
where $R^{(i)}$ is the assistant response at turn $i$. Questions/instructions are inserted at task-appropriate turns, so a query can be posed at any point in time.

**Three decision states.** Each $R^{(i)}$ resolves to a special token in $S=\{s_{\text{silence}},s_{\text{standby}},s_{\text{response}}\}$: `<Silence>` = stay quiet and keep consuming frames; `<Standby>` = relevant content detected, wait for the event to complete; `<Response>` = enough evidence, generate the textual answer immediately after the token. This keeps everything inside the standard next-token framework (Table 1 shows a worked "notify me when the light turns green" dialogue: three `<Silence>` turns → `<Standby>` → `<Response>` The light just turned green.).

**Class-imbalance-aware loss.** In streaming data `<Silence>` is often >80% of turns, biasing the model to stay quiet. A **focal weight** emphasizes hard (low-confidence) state predictions:
$$w_{\text{focal}}(x_i)=(1-p_{c_i})^{\gamma} \tag{3}$$
with $p_{c_i}$ the predicted probability of the true state class $c_i$ at position $i$ and $\gamma\ge0$ the focusing parameter ($\gamma=2$ in practice). A per-batch **frequency alpha** up-weights rare states, with $n_k$ the count of special token $k$ in the batch and $|S|=3$:
$$\alpha_k=\frac{1}{|S|}\cdot\frac{\sum_{j\in S} n_j}{n_k} \tag{4}$$
The two are multiplied into cross-entropy only on the state tokens; all other (content) tokens use plain CE:
$$\mathcal{L}_i=\begin{cases}\alpha_{t_i}\,w_{\text{focal}}(i)\,\mathcal{L}_{\text{CE}}(i,t_i), & t_i\in S\\[4pt]\mathcal{L}_{\text{CE}}(i,t_i), & \text{otherwise}\end{cases} \tag{5}$$
with the usual cross-entropy
$$\mathcal{L}_{\text{CE}}(i,t_i)=-\log p_{t_i}=\log\sum_{j=1}^{|V|}e^{z_{i,j}}-z_{i,t_i} \tag{6}$$
and the total averaged over unmasked positions $M$ so sequence length does not distort it:
$$\mathcal{L}_{\text{total}}=\frac{1}{|M|}\sum_{i\in M}\mathcal{L}_i. \tag{7}$$

**Data construction pipeline (Streamo-Instruct-465K).** Re-annotate open-source videos under one protocol, each video getting several task types at different response granularities:
- *Real-time narration* — segment at 1 s; for each adjacent 2 s window, Qwen2.5-VL-72B describes the change; concatenate and post-process with GLM-4.5 to de-duplicate and smooth.
- *Event caption* — ARC-Hunyuan-Video-7B produces segment captions and grounds them; keep only videos whose segment captions have mutually consistent, overlapping spans (filters noise, sharpens event boundaries).
- *Action caption* — same pipeline narrowed to discrete steps with action-oriented prompts/filtering.
- *Event grounding* — sample event captions, rewrite them as advance queries; the model must monitor the ongoing stream and localize the event's time span.
- *Time-sensitive QA* — GLM-4.5V detects change points (attributes, position, actions, counts, scene shifts); build one unified question with diverse, time-specific answers.

Sources include Koala, LLaVA-Video, ActivityNet, QVHighlight, YouCook2, HACS, Ego-TimeQA, DiDeMo, COIN → 135,875 videos, ~400K curated samples + merged LLaVA-Video offline QA = **465.8K** total. Task mix: Time-sensitive QA 34.8%, Event Grounding 26.3%, Offline QA 13.8%, Narration 12.7%, Event Caption 6.7%, Action Caption 5.8%.

**Streamo-Bench eval protocol.** 300 videos, 3,000 instruction tasks; each video paired with tasks of varying temporal scope. Metrics: **grounding = mIoU** (forward vs. backward = whether the query time-point precedes or follows the event); **caption = win rate judged by Qwen2.5-VL-72B** (narration and dense caption); **Time-Sensitive QA = accuracy + recall** (recall specifically probes whether the model updates its answer as conditions change).

## Explicit design choices
- Decision-making and generation are **unified in one autoregressive model** — no separate lightweight decision module; the state token is part of the token stream (one-pass inference).
- **Three** states, not a binary talk/silent flag: `<Standby>` explicitly separates "event detected" from "event complete / answer ready," giving finer response-timing control.
- One-second turn granularity; frames sampled at **1 fps** during training (segment markers encode absolute time boundaries).
- **Focal loss on state tokens only** (γ=2) + per-batch frequency alpha; the empirical state ratio is ≈ 12:3:2 (Silence:Standby:Response).
- **Multi-task-per-video** annotation under one unified protocol — deliberately avoids naively mixing heterogeneous datasets with inconsistent labels.
- Base models: Qwen2.5-VL 3B/7B (main); framework also applied to InternVL3 and Qwen3-VL to show model-agnosticism.
- Training: full-parameter tuning with **vision encoder frozen** (connector + LLM updated), 1 epoch, batch size 512, lr 1e-5.
- **fps generalization**: trained at 1 fps, can be evaluated at 2 fps with no retraining for a further gain.

## Key results / what to remember
No Zotero highlights present.

- **OVO-Bench (online, Table 2, overall avg):** Streamo-7B 55.61 (1 fps); 57.86 (evaluated at 2 fps without retraining, +4.66 over its own 1 fps); Streamo-3B 52.33. Paper's headline claim: Streamo-7B beats prior SOTA **Dispider (41.78) by +13.83%**. (Caveat verified from the table: ViSpeak-7B posts a higher OVO overall of 61.08; Streamo's clearest lead is on Forward Active Responding — CRR 83.33 for Streamo-7B vs 68.52 for ViSpeak.)
- **Dataset ablation (OVO):** Streamo-Instruct-465K beats ET-Instruct-164K by **+7.1% on forward tasks and +11.79% overall**; adding offline LLaVA-Video to ET-Instruct raises real-time perception but hurts streaming — "offline supervision can hinder online learning."
- **Offline benchmarks (Table 3, avg over OVO-RealTime/OVO-Backward/MVBench/TempCompass/VideoMME/LongVideoBench):** Streamo-7B **63.9 (+3.3** over Qwen2.5-VL-7B base 60.6**)**; Streamo-3B 59.2 (+2.6). Surpasses StreamingVLM on every listed benchmark (e.g. MVBench 72.3 vs 69.2, VideoMME 67.9 vs 65.1, LongVideoBench 59.2 vs 59.0). So conversion to online does not cost offline capability.
- **Streamo-Bench (Table 5, average):** Streamo-7B **55.3**, Streamo-3B 44.7 — vs StreamingVLM-7B 24.6, Dispider-7B 14.6, Flash-VStream 15.6, VideoLLM-online 12.6. On open-ended grounding (mIoU), all baselines score **0** on forward grounding while Streamo-7B gets 29.4 forward / 38.3 backward — baselines collapse once multiple-choice options are removed.
- **Loss ablation (Table 4, OVO Forward Active, Qwen2.5-VL-3B):** Focal loss REC 27.94 / SSR 50.72 / CRR 82.5 vs plain CrossEntropy 6.45 / 20.99 / 41.67 and fixed loss-scaling 18.62 / 41.02 / 49.17; same ordering on InternVL3-2B. Fixed inverse-frequency weights (0.3/1.3/2.0) help but focal loss wins by capturing token-level hardness.

Takeaways: (1) putting the "when to respond" decision **inside** the LM as a 3-way state token, trained end-to-end, beats bolt-on decision modules and single-behavior EOS tokens; (2) the win depends heavily on a **class-imbalance-aware loss** on those state tokens; (3) a **multi-task, unified-temporal-annotation** dataset is what unlocks generalized instruction-following (open-ended grounding/captioning), which QA-only training and benchmarks miss.

## How it connects (evolution)
- [[videollm-online]] — earlier end-to-end streaming with a special EOS timing token; Streamo generalizes its single-behavior (narration) timing into a 3-state, multi-task scheme.
- [[streamingvlm]] — the EOS-narration SOTA Streamo compares against and surpasses on both online and offline benchmarks.
- [[dispider]] — the auxiliary decision-module / clip-segmentation approach Streamo critiques and beats (+13.83% OVO overall).
- [[streambridge]] — another separate-module offline→online adapter; Streamo replaces the module with in-model state tokens.
- [[vispeak]] — proactive online peer on OVO-Bench (higher overall there, but weaker on forward active responding).
- [[ovo-bench]] — the primary online benchmark used, whose forward/backward/real-time taxonomy frames the evaluation.

## Open questions / limitations
- **Unbounded context cost:** no long-sequence optimization — memory/latency grow prohibitively with stream length; authors flag KV-cache management, visual-token pruning, sliding-window attention, adaptive frame compression as future work (all deferred).
- **Overall-metric nuance:** the +13.83% headline is measured against Dispider, not the strongest OVO competitor (ViSpeak scores higher overall); Streamo's decisive edge is specifically response-timing / forward-active tasks.
- **1 fps / 1 s granularity** may be too coarse for sub-second events; the 2 fps test-time bump is shown but the fps ceiling and its effect on the state tokens are unexplored.
- **Annotation depends on large teacher models** (Qwen2.5-VL-72B, GLM-4.5/4.5V, ARC-Hunyuan) — dataset quality and any teacher biases are inherited and not independently audited.

*Verification: equations (1)-(7), the three-state design, and dataset construction checked against the extracted PDF text (pages 3-5); headline numbers verified against the paper's own Tables 2-5 (OVO-Bench, offline benchmarks, loss ablation, Streamo-Bench) and Fig. 3 statistics. Crux is Fig. 1 from the same PDF; arXiv HTML was unavailable (404) so all reading was from the downloaded PDF.*

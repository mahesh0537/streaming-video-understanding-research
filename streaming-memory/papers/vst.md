---
zotero_key: null
authors: Yiran Guan, Liang Yin, Dingkang Liang, et al. (Huazhong University of Science and Technology; MiLM Plus, Xiaomi Inc.)
year: 2026
arxiv: 2603.12262
pdf: https://arxiv.org/pdf/2603.12262
tier: deep
subtopics: [streaming-memory]
tags: [streaming-video-understanding, streaming-memory]
---
# Video Streaming Thinking: VideoLLMs Can Watch and Think Simultaneously

**Lineage role:** Reframes streaming video reasoning as *watch-and-think-in-parallel* — the model spends the pre-query playback amortizing chain-of-thought into a compact textual long-term memory, so the answer is near-instant; sets open-source streaming SOTA (79.5 StreamingBench, 59.3 OVO-Bench) while cutting query latency ~15.7×.

## Problem — what was limited before this paper
Prior online VideoLLMs (e.g. [[videollm-online]], [[dispider]], [[timechat-online]], [[flash-vstream]], [[streamforest]]) mostly perceive and buffer frames as they stream, then do all the heavy reasoning *after* the user query arrives — so a hard multi-step question incurs a long Chain-of-Thought (CoT) at exactly the latency-critical moment. Reasoning VideoLLMs (Video-R1) that add CoT are even slower (≈8.8 s per query). Meanwhile fixed context windows force truncation of long streams, and raw visual buffers are an expensive way to carry history. Nothing was using the *idle* pre-query watching time to pre-compute reasoning, and nothing was compressing visual dynamics into a cheap semantic memory suitable for later backward retrieval.

## Key idea — the core insight
Do the thinking *while watching*, not after asking. As frames arrive, the model periodically emits a short natural-language "streaming thought" that summarizes the current clip's essential semantics and folds it into a running long-term textual memory. When the query finally lands, the answer is produced from the accumulated thoughts + the latest short visual buffer — the expensive reasoning was already amortized across playback, so test-time scaling happens *for free* before the query, with no added latency at interaction time. The autoregressive, left-to-right emission of thoughts also naturally matches the temporal causality of a stream.

![[vst.png]]
> **Crux (Fig. 2).** As the video streams through clips 1…N, the model writes an interleaved "Thought" per clip and updates a long-term textual semantic memory; a short-term native visual buffer holds only the most recent frames, so when the query ("total price of all ships shown?") arrives the answer is composed from accumulated thoughts rather than a fresh long CoT. *Guan et al. (2026), arXiv:2603.12262. Embedded for personal research reference.*

## Method + math

**Dual-memory streaming.** Given a video stream $\mathcal{V}$ with per-frame features $\mathbf{v}_i$, incoming features accumulate into discrete clips $\mathbf{c}^k = \{\mathbf{v}_i\}_{i=\tau_{k-1}+1}^{\tau_k}$, where a clip boundary $\tau_k$ is fired once the buffered visual tokens hit a preset capacity $L$. At each interval $k$, conditioned on the current clip $\mathbf{c}^k$ and accumulated memory $\mathbf{m}^{k-1}$, the LLM samples a streaming thought $\mathbf{z}^k \sim p(\mathbf{z}\mid \mathbf{c}^k,\mathbf{m}^{k-1})$. The long-term textual memory is updated by $\mathbf{m}^k = \mathrm{Update}(\mathbf{m}^{k-1}, \mathbf{z}^k)$, a simple **first-in-first-out** eviction of the oldest entries (fixed memory budget). The short-term memory is the *native* visual buffer of the last $L$ tokens.

**Central objective — reasoning decomposition.** Thinking runs for intervals $k=1..K-1$; at step $K$ the query $\mathbf{q}$ arrives and the final answer $\mathbf{y}$ is generated. The joint probability factorizes as

$$
p(\mathbf{y}\mid \mathbf{q},\mathcal{V}) = \underbrace{p(\mathbf{y}\mid \mathbf{q},\mathbf{c}^K,\mathbf{m}^K)}_{\text{Direct Answer}} \; \prod_{k=1}^{K-1}\underbrace{p(\mathbf{z}^k\mid \mathbf{c}^k,\mathbf{m}^{k-1})}_{\text{Streaming Thinking}}. \tag{1}
$$

The product term is the pre-query amortized reasoning; only the first (cheap) term runs at query time.

**Two-stage post-training** on a Qwen2.5-VL base (visual encoder + projector frozen, 2 fps input).

*Stage 1 — VST-SFT.* A training instance is laid out as the interleaved sequence

$$
\mathcal{S} = \big(\mathbf{m}^0,\,(\mathbf{c}^1,\mathbf{z}^1),\dots,(\mathbf{c}^{K-1},\mathbf{z}^{K-1}),\,\mathbf{c}^K,\mathbf{q},\mathbf{y}\big). \tag{2}
$$

A **streaming video attention mask** enforces the inference-time short-term buffer during training. With additive mask $M$, $\mathbb{I}_v(j)\in\{0,1\}$ flagging visual tokens and buffer size $L$:

$$
M_{i,j} = \begin{cases} 0, & j\le i \ \text{and}\ \Big(\mathbb{I}_v(j)=0 \ \text{or}\ \sum_{t=j+1}^{i}\mathbb{I}_v(t) < L\Big)\\[4pt] -\infty, & \text{otherwise} \end{cases} \tag{3}
$$

i.e. token $i$ sees all non-visual (memory/text) tokens causally, but only the most recent $L$ visual tokens — visual history beyond the window is masked out, forcing the model to rely on the textual memory it wrote. To fit long videos in context, $\mathcal{S}$ is sliced into consecutive segments $\{\mathbf{s}_n\}_{n=1}^M$:

$$
\mathbf{s}_n = \begin{cases} \big(\mathbf{m}^{n-1}, \{(\mathbf{c}^k,\mathbf{z}^k)\}_{k=T_{n-1}+1}^{T_n}\big), & n<M \\[4pt] \big(\mathbf{m}^{n-1}, \{(\mathbf{c}^k,\mathbf{z}^k)\}_{k=T_{n-1}+1}^{K-1}, \mathbf{c}^K,\mathbf{q},\mathbf{y}\big), & n=M \end{cases} \tag{4}
$$

with memory carried recursively across segments, $\mathbf{m}^n = \mathrm{Update}(\mathbf{m}^{n-1}, \{\mathbf{z}^k\}_{k=T_{n-1}+1}^{T_n})$. The next-token loss is applied **only** to the thoughts $\{\mathbf{z}^k\}$ and the final answer $\mathbf{y}$; visual tokens and memory are conditioning-only.

*Stage 2 — VST-RL.* An **agentic rollout**: the policy $\pi_{\theta'}$ generates a full trajectory $\mathcal{T}$ (streaming thoughts $\hat{\mathbf{z}}^k$ then answer $\hat{\mathbf{y}}$) following Eq. (1). For a group of $N$ trajectories, GRPO optimizes

$$
\mathcal{J}_{\text{RL}}(\theta) = \mathbb{E}_{q\sim\mathcal{D},\,\{\mathcal{T}_i\}\sim\pi_{\theta'}}\!\left[ \frac{1}{\sum_i |\mathcal{T}_i|}\sum_{i=1}^{N}\sum_{t=1}^{|\mathcal{T}_i|}\Big(\mathcal{L}^{\text{clip}}_{i,t}(\theta) - \beta\, D_{\text{KL}}(\pi_\theta\|\pi_{\text{ref}})\Big)\right], \tag{5}
$$
$$
\mathcal{L}^{\text{clip}}_{i,t}(\theta) = \min\!\Big[\gamma_t(\theta)\hat{A}_i,\ \mathrm{clip}\big(\gamma_t(\theta),\,1-\epsilon_{\text{low}},\,1+\epsilon_{\text{high}}\big)\hat{A}_i\Big]. \tag{6}
$$

The reward $r_i$ is computed **solely from the final answer** (verifiable correctness), but the group-relative advantage $\hat{A}_i$ is assigned to **all** trajectory tokens — including the thoughts — so RL learns to write thoughts that make the eventual answer correct, without any per-thought supervision.

**Data synthesis (knowledge-graph → streaming CoT).** A pipeline builds 100K streaming-thought samples: (1) sliding-window entity extraction with PySceneDetect segmentation maintains a per-video entity bank / knowledge graph; (2) depth-first search over the graph samples multi-hop evidence chains (entity overlap between chains kept <10% for diversity); (3) Gemini 3.0 flash, conditioned on the graph and a sampled evidence chain $\{z^k\}$, synthesizes a streaming CoT rationale, a query $q$, and answer $y$ needing multi-evidence reasoning; (4) a filtering rubric (world-knowledge check, format alignment, logical consistency, repetition check, thought validation) prunes bad samples. Videos drawn from LLaVA-Vid and Video-Marathon.

## Explicit design choices
- **Base model:** Qwen2.5-VL (3B / 7B / 32B ablated); visual encoder + projection layer **frozen** throughout; video sampled at **2 fps**.
- **Dual memory:** short-term = native visual buffer of last $L$ tokens; long-term = accumulated **textual** thoughts with **FIFO** eviction → fixed memory budget over indefinite streams.
- **Clip boundary** fired by a token-count capacity $L$, not a fixed time/frame count.
- **SFT loss** on thoughts + answer only; visual/memory tokens are conditioning-only; streaming attention mask (Eq. 3) enforces the sliding visual window at train time; temporal segmentation (Eq. 4) for long videos, each SFT sample capped at a 128 s time limit.
- **RL = GRPO** with rollout batch 256, group size $N=8$, temperature 1.0, lr 5e-7 (20 warmup steps), KL coeff $\beta=0.001$; reward from final-answer correctness only, advantage broadcast to all tokens.
- **SFT corpus:** 50K open-ended LLaVA-Vid QA + 100K synthesized streaming-thought samples; visual buffer 24K tokens (8K reserved for language), lr 5e-6, 1 epoch, grad-accum 8.
- **RL corpus:** 11K questions (multiple-choice from LLaVA-Vid, Video-Marathon, Onethinker; counting from RepCount).
- **Inference cap** (following StreamingForest): each step (streaming-think or final answer) capped at 8,192 video tokens; **max thinking times capped at 4** for efficient eval.
- **Training compute:** 32 × 80GB GPUs; verl + vLLM + FSDP backend for RL; paralleled encoding to pre-compute video embeddings during rollout.

## Key results / what to remember
No Zotero highlights present.

Verified against the paper's own tables (7B unless noted):
- **StreamingBench (Real-Time, Overall):** **79.5** — beats Streamforest-7B (77.3), TimeChat-Online-7B (75.4), and surpasses GPT-4o (73.3, +6.2) and Gemini 1.5 pro (75.7, +3.8). [Table 1]
- **OVO-Bench (Overall):** **59.3** — beats Streamo-7B (57.9) and Streamforest-7B (55.6); comparable to GPT-4o (59.5). Backward-Tracing avg **56.7** (+4.7 over Streamforest's 52.0), evidencing effective historical retrieval. Real-Time avg 67.2, Forward avg 54.0. [Table 2]
- **VideoMME (w/o subtitles):** Long **55.3** (+6.9 vs TimeChat-Online), Overall **64.9**. **LongVideoBench 58.0** (+2.6 vs TimeChat-Online). **VideoHolmes 41.9** (+5.4 vs Video-R1's 36.5). [Table 3]
- **Latency (Table 6 / text):** VST-7B QA latency ~0.56 s vs Video-R1 with CoT ~8.80 s → ≈**15.7× faster** at query time (reasoning amortized pre-query).
- **Ablation (Table 4):** VST-SFT alone lifts Backward memory (+9.2 on OVO Backward); VST-RL alone lifts Forward prediction (+12.7 on Forward); combined SFT+RL is best (OVO 59.3, VideoMME 64.9). Data mix of 20K LLaVA-Vid + 30K VST beats 50K LLaVA-Vid alone by +6.6 on OVO-Bench.
- **Thinking steps (Fig. 5):** Backward accuracy climbs monotonically 1→16 steps (53.3 → 57.5); Real-Time/Forward plateau around 4 steps — hence the eval cap at 4.
- **Scale (Table 5):** gains consistent across 3B / 7B / 32B bases.

Takeaways: (1) the win is *when* you reason, not just *how* — moving CoT into the pre-query watching window buys test-time scaling with no interaction-time cost; (2) a written textual memory (not a raw visual KV buffer) is what makes long-range Backward Tracing strong under a fixed budget; (3) answer-only RL reward with advantage broadcast to thoughts is enough to shape useful intermediate thoughts without per-thought labels.

## How it connects (evolution)
- [[streamforest]] — the prior open-source streaming SOTA VST directly overtakes (77.3→79.5 StreamingBench); VST borrows its 8,192-token inference cap.
- [[timechat-online]] — main offline-benchmark baseline VST beats on VideoMME-long / LongVideoBench.
- [[dispider]] — decoupled perceive/decide/react streaming model; a key online baseline in Tables 1–2.
- [[videollm-online]] — early online dialogue-over-stream framework this lineage descends from.
- [[flash-vstream]] — memory-buffered streaming VLM; contrasts VST's textual (vs visual) long-term memory.
- [[streamingbench]] / [[ovo-bench]] — the two online benchmarks VST reports SOTA on (OVO-Bench's Backward track is where the textual memory shines).

## Open questions / limitations
- **Thinking is scheduled by token-capacity clips, not content** — a boundary $L$ can split or over-sample events; no learned "when to think" gate (unlike proactive-timing work), so thought placement may be misaligned with salient moments.
- **FIFO eviction is naive** — oldest thoughts are dropped regardless of future relevance; a query needing very old, evicted detail has no recovery path, capping true indefinite-stream recall.
- **Reward is final-answer-only** — no direct check that intermediate thoughts are faithful; thoughts could be persuasive-but-wrong yet still rewarded when the answer happens to be right (credit-assignment noise).
- **Eval cap of 4 thinking steps** is an efficiency compromise; Backward keeps improving to 16 steps, so reported numbers likely understate the paradigm's ceiling, and real-time compute cost of more steps is not fully characterized.

*Verification: all equations (1–6) transcribed from the PDF method text (pages 4–6, PyMuPDF extraction); all headline numbers cross-checked against Tables 1–6 in the arXiv PDF (2603.12262, pages 9–11); StreamingBench 79.5, OVO-Bench 59.3/56.7 Backward, VideoMME-long 55.3, LongVideoBench 58.0, VideoHolmes 41.9, and latency 0.56s vs 8.80s all match the source tables/text.*

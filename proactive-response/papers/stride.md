---
zotero_key: null
authors: Junho Kim, Hosu Lee, James M. Rehg, Minsu Kim, Yong Man Ro (KAIST; UIUC)
year: 2026
arxiv: 2603.27593
pdf: https://arxiv.org/pdf/2603.27593
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# STRIDE: When to Speak Meets Sequence Denoising for Streaming Video Understanding

**Lineage role:** recasts the "when-to-speak" decision in streaming video from per-frame binary classification into a *span-structured sequence-denoising* problem, dropping a lightweight masked-diffusion activation module in front of a frozen Video-LLM.

## Problem — what was limited before this paper (short)
Streaming Video-LLMs must decide not only *what* to say but *when* to say it as frames arrive online. The dominant modular recipe (a lightweight front-end predicts a per-frame binary "trigger now?" signal, and a downstream Video-LLM answers when triggered — the design of [[streambridge]], [[dispider]], [[mmduet]]) reduces activation to isolated point-wise supervision. That ignores the fact that real event triggers are *contiguous spans*, so per-frame heads flicker: activation oscillates 0↔1 near event boundaries, giving unstable triggering and poorly localized onsets/offsets. EOS-token approaches ([[videollm-online]], [[streambridge]]) further conflate the timing decision with language generation.

## Key idea — the core insight, 2-4 sentences
Temporal transitions in a stream form span-structured activation patterns, so activation should be modeled *jointly over a temporal window* rather than one frame at a time. STRIDE keeps the modular front-end/downstream split but replaces the binary head with a **masked diffusion module** that predicts a whole window of activation states and *iteratively denoises* them as new frames arrive. Confident past decisions are retained; uncertain and newly appended positions are selectively re-masked and refined, yielding temporally coherent trigger spans instead of flickering point predictions.

![[stride.png]]
> **Crux (Figure 1).** The two-stage streaming pipeline: a lightweight masked-diffusion **Activation Model** (trainable) maintains a sliding activation region over the frame cache and denoises masked activation tokens to a coherent trigger span; only when an active span is sustained is the accumulated visual context forwarded to a **frozen downstream Video-LLM** that generates the response. Right panel shows training via a masked denoising process over the activation window. *Kim et al. (2026), arXiv:2603.27593. Embedded for personal research reference.*

## Method + math — the mechanism

### Preliminaries: masked diffusion models (MDMs)
For a token sequence $x_0=(x^1_0,\dots,x^L_0)$, the forward process independently replaces each token with a mask token $[M]$ with probability $t\in[0,1]$, producing $x_t$ ($t{=}0$ fully observed, $t{=}1$ fully masked). A mask predictor $p_\theta(\cdot\mid x_t)$ with bidirectional attention predicts all masked positions at once, trained by the masked cross-entropy

$$
\mathcal{L}(\theta) = -\mathbb{E}_{t,x_0,x_t}\left[\frac{1}{t}\sum_{i=1}^{L}\mathbf{1}[x^i_t = M]\,\log p_\theta(x^i_0\mid x_t)\right],\quad t\sim U[0,1].
$$

At inference, start from a fully masked $x_1$ and run $K$ reverse steps $t{=}1\to0$; each step accepts high-confidence predictions and re-masks the rest (predict-and-refine).

### Problem formulation
Stream $\mathcal{V}=\{v_1,v_2,\dots,v_T,\dots\}$ with $v_T$ the frame at step $T$. Under partial observability only $\mathcal{V}_{\le T}=\{v_1,\dots,v_T\}$ and context priors $\mathcal{C}_T$ (user query $q$ + interaction history) are available. At each $T$ the model makes two sequential decisions: (i) whether to respond, (ii) if so, what — decoupled by a two-stage framework.

### Two-stage architecture
A lightweight **Activation Model $\pi$** monitors the stream and decides triggering. When triggered at $T$, the accumulated visual context since the most recent query time $T_q$, i.e. $\mathcal{V}_{[T_q:T]}$, plus $\mathcal{C}_T$, is passed to a **frozen** downstream Video-LLM $f$ that emits $R_T=f(\mathcal{C}_T,\mathcal{V}_{[T_q:T]})$. The response is appended, $\mathcal{C}_{T'}=\mathcal{C}_T\cup R_T$, for dialogue coherence; after each trigger the visual accumulation is cleared and restarted from the current step.

### Span-level activation modeling
Activation is a window-level sequence of size $W$ anchored at $T$: $a_T=[a_{T-W},\dots,a_T]\in\{0,1\}^W$, so the prediction unit aligns with span structure — onset $0\to1$, persistence $1\to1$, offset $1\to0$ — rather than isolated points. Frames arrive at 1 FPS, are encoded into a running visual cache, and the activation region $a_T$ is appended after the cache as the prediction target. Each activation token is drawn from the vocabulary $\{0,1,[M]\}$; $\pi$ conditions on the visual cache and jointly infers the masked states.

### Training — activation as sequence denoising
**Three structured masking strategies** (standard independent MDM masking is inappropriate — isolated unmasked tokens between active positions make denoising trivially solvable by local interpolation). Each training sample is corrupted by one of the three, chosen with equal probability:
- **Boundary-anchored span masking** — masks a contiguous block overlapping ≥1 activation boundary, forcing the model to locate where the active region begins/ends from broader context.
- **Span unmasking** — from a fully masked sequence, reveals a contiguous block while keeping boundary-adjacent positions masked, mimicking the inference-time pattern where confident homogeneous regions are unmasked first.
- **Full masking** — masks the entire sequence (cold-start) so the model learns to estimate the global activation layout from visual context alone.

**Sequence duplication (recovering bidirectional conditioning).** The backbone is AR-pretrained with causal attention (left context only), but MDM needs full-window context. Instead of altering attention masks, STRIDE appends a copy of the activation region to form $[a, a']$: $a'$ produces the diffusion predictions while $a$ (placed entirely before $a'$) serves as a conditioning prefix, so every token in $a'$ can attend to all of $a$ as left-context — full-window visibility without touching the causal mask.

**Objective.** Masked cross-entropy over $a'$, conditioned on query $q$ and visual cache:

$$
\mathcal{L}(\theta) = -\mathbb{E}_{t,\,a'_0,\,a'_t}\left[\frac{1}{t}\sum_{j=1}^{W}\mathbf{1}[a'^{j}_t = M]\,\log p_\theta\!\left(a'^{j}_0\mid q,\mathcal{V}_{\le t},a'_t\right)\right],\quad t\sim U[0,1],
$$

where $a'_t$ is corrupted by the masking mixture above.

### Inference — streaming as progressive unmasking
Each new step $T{+}1$ runs two stages:
1. **Selective re-masking.** Shift the window forward: the out-of-window region is evicted, a new frame appended. A carried-forward decision $a^{j+1}_T$ (now at position $j$) is retained only if confident given new evidence: if $p_\theta(a^j_{T+1}=a^{j+1}_T\mid q,\mathcal{V}_{\le T+1},a_{T+1})>\tau$ it inherits its prior value; otherwise it is re-masked to $[M]$ and re-enters denoising alongside the new slot.
2. **$K$-step progressive denoising.** The masked positions (new + low-confidence re-masked) are resolved over $K$ steps, high-confidence first. For each masked position compute $p_j=p_\theta(a^j{=}1\mid\cdot)$ and confidence $c_j=\max(p_j,1-p_j)$; unmask the top-$k$ by $c_j$ with $k=\lceil N_{\text{init}}/K\rceil$ ($N_{\text{init}}$ = masked count from stage 1), re-mask the rest.

A **trigger** at $T{+}1$ is issued only if an active span is sustained for at least $\gamma$ consecutive positions (span ratio).

## Explicit design choices
- **Modular, frozen downstream.** Activation front-end is trained; downstream Video-LLMs (Gemma3-4B, InternVL3-8B, Qwen3-VL-8B) are kept frozen — preserves their capabilities, keeps the timing decision separate from generation.
- **Activation backbone:** Qwen3-VL-2B (compact, to minimize streaming overhead); a 4B variant is studied for scaling.
- **Frame rate:** 1 FPS sampling into the visual cache.
- **Activation vocabulary** $\{0,1,[M]\}$ — a *binary* per-position state plus a mask symbol, i.e. a very small output space (which is why denoising converges in few steps).
- **Denoising steps** $K=8$ (low-confidence remasking schedule); performance saturates near $K{=}8$ at ~100 ms latency.
- **Retention threshold** $\tau=0.75$; **span/trigger ratio** $\gamma=1$ (per benchmark protocol).
- **Training data:** temporal annotations converted to binary activation sequences at the frame rate (active inside annotated spans) — from dense video captioning (ActivityNet-Captions, ET, VidChapters), temporal activity detection (Charades / Charades-Ego), grounded video QA, sequential step recognition (COIN), moment localization (Charades-STA-style).
- **Compute:** trained on 8×H100 (single node); evaluated on a single H100.
- **Baseline-AR** control: same architecture/training but the masked-diffusion head swapped for an autoregressive binary head with BCE loss (activation formulation of [[streambridge]]) — isolates masked denoising vs point-wise AR prediction.

## Key results / what to remember
Numbers verified against the paper's Tables 1–5.

**ET-Bench online activation accuracy (F1), STRIDE at only 2B params (Table 3):**
- TVG **62.8** (Baseline-AR 35.7 → **+27.1**), SLC 28.5, TAL 24.6, DVC 36.5, EPM 10.7, **Avg 32.6** vs Baseline-AR **24.3** (**+8.3**).
- Beats larger streaming/temporal-localization models on overall average: ETChat (5B) 28.5, Dispider (9B) 26.3, StreamBridge (8B, TVG 34.3), VideoLLM-Online (8B) 12.0.

**OVO-Bench overall (Table 1)**, offline backbones run online with STRIDE:
- Qwen3-VL-8B: 51.77 → **59.07** with STRIDE; Forward Active Responding avg 46.30 → **59.70** (the proactive when-to-speak dimension).
- InternVL3-8B: 51.64 → **56.98**; Gemma3-4B: 42.13 → **50.51**.
- On the same Qwen3-VL-8B backbone, Baseline-AR reaches only 52.81 overall (vs STRIDE 59.07).

**StreamingBench overall (Table 2):**
- Qwen3-VL-8B: 46.84 → **59.29**; Proactive Output (PO) subtask 32.40 → **42.80**.
- InternVL3-8B: 53.97 → **57.58**; Gemma3-4B: 46.03 → **50.14**. Baseline-AR on Qwen3-VL-8B: 57.12.

**Ablations (ET-Bench avg F1, Table 4):**
- Masking: independent-only **7.2** → Span-only 21.1 → Span+Full 23.0 → all three **32.6**.
- Sequence duplication: 22.9 → **32.6** (removing it drops TVG/DVC most).
- Selective re-masking: last-only (≈AR) 22.6 → selective **32.6**.

**Behavior / efficiency:**
- Flickering (Fig. 3): Baseline-AR oscillates heavily near event boundaries; STRIDE gives far fewer transitions / smoother spans.
- Latency (Table 5): STRIDE adds ~113 ms (new frame + $K{=}8$ steps) — only **~7%** over Qwen3-VL-8B's ~1511 ms base response latency; when no trigger is needed it saves compute by not invoking the downstream model.

No Zotero highlights present.

Takeaways: (1) modeling when-to-speak as *span-level sequence denoising* rather than per-frame binary classification is the whole win — biggest gains on the proactive/forward-active subtasks and on boundary-sensitive TVG; (2) two cheap tricks make a causal AR backbone behave like a bidirectional diffusion model — **sequence duplication** (fake bidirectional context) and **selective re-masking** (revise stale carried-forward decisions); (3) because the activation vocabulary is tiny (0/1), $K{=}8$ denoising steps suffice, keeping streaming overhead ~7%.

## How it connects (evolution)
- [[streambridge]] — STRIDE's Baseline-AR and its per-frame binary-activation formulation come directly from StreamBridge; STRIDE keeps the modular split but replaces the AR head.
- [[dispider]] — prior modular front-end/downstream decoupling for proactive streaming; a compared streaming baseline on OVO/StreamingBench/ET-Bench.
- [[mmduet]] — per-frame informative/response-head activation, the point-wise supervision paradigm STRIDE argues against.
- [[videollm-online]] — EOS-token / streaming-EOS timing that conflates triggering with generation; a compared baseline.
- [[proactive-response]] — sub-topic hub: STRIDE is the "activation-as-denoising" node in the when-to-speak lineage.
- [[ovo-bench]] — one of the three evaluation benchmarks (forward active responding measures proactive timing).

## Open questions / limitations
- **Retro-dated arXiv id (2603.27593 → "2026-03"):** the abstract/method/table numbers here are read directly from the fetched PDF and cross-checked internally, but the identifier is future-dated for this vault; treat external citations of exact leaderboard rows with mild caution.
- Frozen downstream means response *quality* is bounded by the backbone — STRIDE only improves *when* context is fed, not the Video-LLM's reasoning; EPM (episodic memory, 10.7 F1) stays weak.
- The span-ratio trigger ($\gamma$) and retention $\tau$ are fixed hyperparameters tuned to benchmark protocol; robustness to varying event durations / frame rates beyond 1 FPS is untested.
- Adds a second model in the loop (2B activation net) — modest ~7% latency but non-zero VRAM/compute, and gains partly depend on the activation backbone scale (2B vs 4B scaling shown in appendix).

*Verification: equations (masked CE loss, span-level activation, selective re-masking rule, confidence score) transcribed from the fetched PDF §3; headline numbers checked against the PDF's Tables 1 (OVO-Bench), 2 (StreamingBench), 3 (ET-Bench), 4 (ablations), 5 (latency); crux is Figure 1 (page 4 of the PDF). Zotero not running; no annotations.*

---
zotero_key: null
authors: Daeun Lee, Subhojyoti Mukherjee, Branislav Kveton, et al. (UNC Chapel Hill; Adobe Research)
year: 2025
arxiv: 2512.01707
pdf: https://arxiv.org/pdf/2512.01707
tier: deep
subtopics: [streaming-benchmarks]
tags: [streaming-video-understanding, streaming-benchmarks]
---
# StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos

**Lineage role:** the first streaming-video benchmark that injects the human **eye-gaze scanpath** as an intention/attention signal, testing whether streaming MLLMs can read where a user is looking to reason about the past, perceive the present, and *proactively* anticipate — the missing perceptual channel in prior streaming benchmarks.

## Problem — what was limited before this paper (short)
Streaming benchmarks (StreamingBench, OVO-Bench, etc.) test temporal reasoning on incoming frames but almost always ignore the **human perceptual signal** that actually drives real AR-glasses / embodied-assistant decisions: where the user is *looking*. They also skew toward past/present recognition and under-cover **proactive** (model-initiated) behavior. Gaze is the most direct indicator of user attention and intent, yet grounding it is hard — raw gaze is noisy and saccadic, and the egocentric camera shakes/rotates with the head, so "what is the user looking at" is a moving spatio-temporal grounding problem, doubly so under the causal (past-and-present-only) streaming constraint.

## Key idea — the core insight
Turn raw egocentric gaze trajectories into a **scanpath** — a temporally ordered sequence of fixated (and non-fixated) objects — and use that scanpath as the conditioning signal for a suite of streaming QA tasks. A semi-automatic pipeline detects stable fixations, extracts objects **inside vs. outside** the field-of-view via region-specific visual prompting, orders them into a scanpath, and has an LLM generate past/present/proactive questions (human-verified). The result is StreamGaze: 8,521 QA pairs over 285 egocentric videos, 10 tasks across three temporal stances, exposing a large human-vs-model gap in gaze-conditioned reasoning.

![[streamgaze.png]]
> **Crux (Figure 3).** The four-stage gaze-guided data construction pipeline: (1) project raw 2D gaze onto egocentric frames, (2) extract fixations by point-wise stability + scene consistency, (3) split each fixation frame into FOV vs. out-of-FOV regions and prompt an MLLM to list attended/unattended objects, (4) assemble the scanpath and generate human-verified past/present/proactive QA. *Lee et al. (2025), arXiv:2512.01707. Embedded for personal research reference.*

## Method + math — the eval protocol as the "math"

**Streaming context.** At query time $t_q$ the model $\mathcal{M}$ answers question $Q$ from a video–gaze context $(\mathcal{V},\mathcal{G})$ restricted to what is causally accessible. The three temporal stances differ only in the visible window (short-term window $\omega = 60$ s):
$$\begin{aligned}
\textbf{Past:}\quad & A = \mathcal{M}\!\left(Q;\ \mathcal{V}_{[0,\,t_q]},\ \mathcal{G}_{[0,\,t_q]}\right)\\
\textbf{Present:}\quad & A = \mathcal{M}\!\left(Q;\ \mathcal{V}_{(t_q-\omega,\,t_q]},\ \mathcal{G}_{(t_q-\omega,\,t_q]}\right)\\
\textbf{Proactive:}\quad & A = \mathcal{M}\!\left(Q;\ \mathcal{V}_{(t_q,\,\infty]},\ \mathcal{G}_{(t_q,\,\infty]}\right)
\end{aligned}$$
Proactive is the streaming-native case: the model watches forward and must *self-trigger* an alert at the right moment.

**Fixation extraction.** From raw gaze trajectory $\mathcal{G}$, a fixation $f_i=(\bar x_i,\bar y_i,t_i^s,t_i^e)$ has spatial centroid $(\bar x_i,\bar y_i)$ and temporal span $[t_i^s,t_i^e]$; the set is $\mathcal{F}=\{f_i\}_{i=1}^N$. Two criteria retain a fixation:
- **Point-wise (spatial) stability** — every gaze point stays within a normalized radius $r_\text{thresh}$ of the centroid, and the interval is long enough:
$$d_t = \lVert (x_t,y_t) - (\bar x_i,\bar y_i)\rVert_2 \le r_\text{thresh},\ \ \forall t\in[t_i^s,t_i^e];\qquad t_i^e - t_i^s \ge \tau_\text{dur}.$$
- **Scene consistency** — guard against camera cuts/motion by requiring visual continuity across the fixation clip $\mathcal{V}_i=\{I_t\}_{t=t_i^s}^{t_i^e}$ via Hue–Saturation histograms $H_t$:
$$S_\text{min} = \min_{t\in[t_i^s,\,t_i^e-1]} \rho(H_t, H_{t+1}) \ge \tau_\text{scene},$$
with $\rho(\cdot)$ the Pearson correlation between normalized histograms.

**FOV vs. out-of-FOV regions.** For frame $I_t$ the foveal region is a disk of radius $\tau_\text{fov}$ around the fixation centroid; the rest is out-of-FOV:
$$\mathcal{R}^\text{fov}_{i,t} = \{(u,v)\in I_t \mid \lVert (u,v)-(\bar x_i,\bar y_i)\rVert_2 \le \tau_\text{fov}\},\qquad \mathcal{R}^\text{out}_{i,t} = I_t \setminus \mathcal{R}^\text{fov}_{i,t}.$$

**Region-specific visual prompting.** An MLLM (InternVL3.5-38B) extracts object sets from each region: the FOV crop gets a red-dot gaze marker so the model reports the *attended* objects $\mathcal{O}^\text{fov}_i=\mathrm{MLLM}(\mathcal{R}^\text{fov}_i)$; the out-of-FOV frame is masked with a solid black disk over the gaze area so the model reports only the *non-attended* background $\mathcal{O}^\text{out}_i=\mathrm{MLLM}(\mathcal{R}^\text{out}_i)$. Each object carries a 1–2 sentence caption for downstream QA.

**Scanpath.** Ordered over the $N$ fixations:
$$\mathcal{S} = \{(\mathcal{O}^\text{fov}_i,\ \mathcal{O}^\text{out}_i)\}_{i=1}^{N},$$
preserving how attention shifts across spatial regions and semantics over time. This is the object queried/perturbed by every task.

**Task taxonomy (10 tasks / 3 stances).**
- *Past (temporal reasoning over gaze):* Non-Fixated Object Identification (NFI — objects visible but never fixated, sampled from $\mathcal{O}^\text{out}$), Object Transition Prediction (OTP — next fixated object), Gaze Sequence Matching (GSM — recover scanpath pattern), Scene Recall (SR — recall earlier background objects).
- *Present (current perceptual state):* Object Identification Easy / Hard (OI-E / OI-H), Object Attribute Recognition (OAR), Future Action Prediction (FAP — intent from recent gaze).
- *Proactive (model-initiated):* Gaze-Triggered Alert (GTA — flag when the user fixates a target), Object Appearance Alert (OAA — flag when a new peripheral object enters the scene).

**Metrics / judging.** Past & present tasks are multiple-choice **accuracy** (exact match with fuzzy semantic-equivalence checking). Proactive tasks use a **multi-triggering precision** protocol that *penalizes premature / excessive triggers* (the primary metric, best reflecting alert reliability); recall reported as a supplement in the appendix. QA is LLM-generated then **human-verified**, with an average correctness of ~83%.

## Explicit design choices
- **Gaze as first-class modality**, not just video: three input encodings compared — textual fixation coordinates, visual overlays (green dot = instantaneous gaze, red circle = FOV), and salience heatmaps.
- **Two-object bookkeeping per fixation** ($\mathcal{O}^\text{fov}$ attended vs. $\mathcal{O}^\text{out}$ non-attended) — enables clean, controllable negatives (e.g. NFI samples from the never-gazed set).
- **Masking-based negative extraction**: black-disk occlusion of the gaze region forces the extractor to name only non-attended objects, avoiding leakage.
- **Fixation filtering by two independent gates** (dispersion radius + HS-histogram scene consistency) to survive egocentric camera shake and cuts.
- **Egocentric sources**: EGTEA+, EgoExoLearn, HoloAssist (assembly / cooking / lab domains); ~285 videos, 815 s average length; 8,521 QA pairs.
- **Streaming windows fixed by stance**: full history (past), 60 s (present), forward-open with self-triggering (proactive).
- **Proactive metric = precision under multi-triggering**, deliberately punishing over-eager alerts.

## Key results / what to remember (verified against Table 2)
Overall accuracy (past/present accuracy + proactive precision, averaged):
- **Human oracle: 0.827** — far above every model.
- **Best model overall: GPT-4o 0.535** (16 frames). Claude Sonnet-4 0.474, Claude Opus-4 0.399.
- Open-source MLLMs: Qwen2.5-VL-7B **0.478**, InternVL3.5-8B 0.444, EgoGPT-7B 0.436, VITA-1.5-7B 0.384, MiniCPM-V-8B 0.365, Kangaroo-7B 0.351.
- Streaming MLLMs: **ViSpeak-7B 0.467** (best proactive GTA 0.635), Dispider 0.323, Flash-VStream-7B 0.243, VideoLLM-online-8B **0.080** (collapses — dialogue-tuned, emits generic descriptions).
- Gaze-specialized fine-tuned model **AssistGaze (26M): 0.223** — fails to generalize to streaming/long-horizon (N/A on both proactive tasks).
- Widest human gaps: **Scene Recall** Human 0.903 vs GPT-4o 0.535; **OTP** Human 0.889 vs GPT-4o 0.449. Proactive is worst overall — **OAA** GPT-4o only 0.149 (vs Human 0.780).
- Gaze-prompting ablation (Qwen2.5-VL-7B): no-gaze 0.446; textual 0.429, visual 0.429 (both hurt), salience map 0.454 (marginal gain) — no single gaze encoding helps uniformly across tasks.

No Zotero highlights present.

Takeaways: (1) even frontier MLLMs cannot turn gaze into long-horizon temporal reasoning — they over-rely on frame-local foreground cues; (2) proactive self-triggering is the hardest regime and dialogue-tuned streaming models are actively miscalibrated there; (3) explicitly gaze-trained small models don't transfer to streaming; (4) the right gaze encoding is task-dependent, arguing for adaptive gaze conditioning rather than a fixed overlay.

## How it connects (evolution)
- [[streaming-benchmarks]] — the sub-topic hub; StreamGaze is the gaze-conditioned entry in the streaming-benchmark lineage.
- [[streamingbench]] / [[ovo-bench]] — prior streaming benchmarks StreamGaze positions against (Table 1); they cover past/present but omit gaze and thin proactive coverage.
- [[proactivevideoqa]] — shares the model-initiated / proactive evaluation axis that StreamGaze extends with gaze triggers.
- [[egopro-bench]] — sibling egocentric proactive benchmark; StreamGaze adds the raw-gaze scanpath signal on top of egocentric video.
- [[vispeak]] — evaluated here as the strongest streaming model on proactive tasks; StreamGaze also fine-tunes ViSpeak variants in its appendix.

## Open questions / limitations
- Object extraction leans on a large MLLM (InternVL3.5-38B) with visual prompting, so benchmark labels inherit that model's biases despite ~83% human-verified correctness.
- Fixation gating uses fixed thresholds ($r_\text{thresh}$, $\tau_\text{dur}$, $\tau_\text{scene}$, $\tau_\text{fov}$) tuned for consistent FOV size; sensitivity across datasets/resolutions is only lightly explored.
- No gaze-encoding strategy helps across all tasks — the benchmark diagnoses the gap but doesn't yet provide a method that closes it.
- Domains are constrained to assembly/cooking/lab egocentric sources; generalization to open-world AR-glasses scenarios is untested.

*Verification: streaming task definitions and fixation/FOV/scanpath equations checked against the rendered method pages (Eqs. 1–8, Fig. 3); all headline numbers (Human 0.827, GPT-4o 0.535, ViSpeak 0.467, VideoLLM-online 0.080, AssistGaze 0.223, SR/OTP/OAA gaps) read directly from the rendered Table 2; dataset stats (8,521 QA / 285 videos / 10 tasks / ~83% verification) from the arXiv HTML abstract+intro.*

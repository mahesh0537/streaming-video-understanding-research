---
zotero_key: null
authors: Junhyeok Kim, Min Soo Kim, Jiwan Chung et al. (Yonsei University; NC Research, NCSOFT)
year: 2025
arxiv: 2502.14892
pdf: https://arxiv.org/pdf/2502.14892
tier: deep
subtopics: [proactive-response]
tags: [streaming-video-understanding, proactive-response]
---
# EgoSpeak: Learning When to Speak for Egocentric Conversational Agents in the Wild

**Lineage role:** Recasts proactive "when to speak" as real-world egocentric turn-taking — predicting the camera-wearer's speech onset from an untrimmed RGB+audio stream, with a scalable YouTube pretraining corpus (YT-Conversation); NAACL 2025.

## Problem — what was limited before this paper
Deciding *when* an agent should start talking is the low-level trigger underneath any proactive assistant, but prior turn-taking work was built for the wrong regime. Earlier models were audio-only, worked on offline/trimmed clips of controlled dyadic conversations, and reacted from a third-person or fixed-microphone view. Two failure modes follow: (1) a pure **detection** approach fires on a silence threshold, giving only ~200 ms of reaction time — too late for natural, sometimes-overlapping human turn-taking; (2) controlled dyadic data does not transfer to messy multi-party, in-the-wild conversations. Nothing combined the four properties a real embodied agent needs at once: **egocentric** (first-person), **RGB** (visual, not just audio), **online** (causal/streaming), and **untrimmed** (continuous stream). Table 1 in the paper positions EgoSpeak as the first to check all four boxes.

## Key idea
Treat the camera-wearer as the "target speaker" and learn to **anticipate** their speech onset from the streaming egocentric video+audio, rather than detect it after a silence. At every timestep the model does a causal three-way classification — background (no speech) / other person speaking / target (wearer) speaking — but predicts these labels for a short future window (anticipation length α = 10 steps ≈ 2 s). This is framed like next-token prediction: the model must commit to "I'm about to speak" *before* the wearer actually does, buying the 600–1500 ms of lead time human turn-taking really uses. A companion contribution, **YT-Conversation**, auto-labels in-the-wild YouTube conversation video via voice-activity detection to give scalable pretraining data.

![[egospeak.png]]
> **Crux (Figure 2).** The EgoSpeak pipeline: at each step an online model ingests the untrimmed egocentric RGB + audio input window and emits three probabilities (agent-speak / person-speak / background); when the agent-speak probability crosses a threshold in the anticipated near future, the agent starts speaking. *Kim et al. (2025), arXiv:2502.14892. Embedded for personal research reference.*

## Method + math
**Streaming task formulation.** Let $X_t = [x_1, \dots, x_t]$ be the online stream observed up to time $t$, where each $x_i$ carries multimodal signals — visual frames $x_i^v$ and audio $x_i^a$. Each modality $x_i^m$ is mapped by an off-the-shelf extractor to a feature $z_i^m$, and the per-step features are concatenated into $z_i$. The model reads a temporal window of length $L$,
$$Z_{t-L+1,\,t} = [z_{t-L+1}, \dots, z_t],$$
and, given anticipation length $\alpha$, produces a **three-way classification for the future range** $t+1$ to $t+\alpha$:
$$\hat{Y} \in \mathbb{R}^{\alpha \times 3}, \qquad \text{classes} = \{\text{background},\ \text{other speaking},\ \text{target (wearer) speaking}\}.$$
Only past/present information is used (causal), so the system is deployable on a live stream. With $\alpha = 10$ steps at 5 FPS this is a 2 s look-ahead.

**Training objective.** Standard per-step, per-class **cross-entropy** against the anticipated frame labels — analogous to next-token prediction, since the model must anticipate the onset before it happens:
$$\mathcal{L} = -\sum_{k=1}^{\alpha}\sum_{c=1}^{3} y_{t+k,c}\,\log \hat{y}_{t+k,c}.$$

**Frame-level label construction (Figure 3).** Per-frame speech labels are expensive, so transcripts with word/utterance timestamps are converted to per-frame one-hot labels: at timestep $t$, label = *target speaker speaking* if the wearer is talking, *other speaking* if someone else is, else *background*. For the YT-Conversation pretraining corpus there are no transcripts, so **pseudo-labels come from voice activity detection** (Pyannote), dropping segments shorter than 200 ms.

**Features.** RGB: ResNet-50 pretrained on Kinetics-400, applied to 4-frame chunks (feature taken at the center frame), extracted at 5 FPS. Audio: wav2vec2 features, temporally aligned/concatenated to the RGB stream. Optional motion: TV-L1 optical flow as an extra modality.

**Backbone variants.** Three interchangeable online sequence models are compared, all fed the same features:
- **Transformer (LSTR-style)** — long-short-term encoder–decoder: long-term encoder over 2048 frames, short-term decoder over 32 frames, 16 heads, 1024 hidden.
- **RNN (GRU)** — 2048-d embeddings, 1024-d hidden.
- **Mamba (SSM)** — state factor 16, local conv width 4, block expansion 2.

**Prediction vs detection (the motivating argument).** Detection = fire on a silence threshold → ~200 ms response. A psycholinguistic estimate (Levinson & Torreira, 2015) puts real human response latency at 600–1500 ms, with turns often overlapping without gaps; anticipation gives the system that budget to prepare a reaction and behave human-like.

**Evaluation metric — per-frame mAP.** The metric is per-frame **mean Average Precision**: frame-level confidence scores are sorted, precision–recall computed at each threshold to get per-class Average Precision (area under P–R), then averaged over the three classes and over anticipation offsets from 0.20 s to 2.00 s. The headline "target speaker AP" isolates the class that matters for onset (the wearer).

## Explicit design choices
- **Target = camera wearer.** The egocentric wearer is defined as the "target speaker," so their own speech onsets are the supervision signal — no external speaker-ID needed.
- **Anticipation, not detection.** Predict future frames $t+1..t+\alpha$ (α = 10 steps ≈ 2 s at 5 FPS) instead of thresholding current silence.
- **Three-class output** (background / other / target) rather than binary speak/not-speak, so the model can condition onset on whether *someone else* is currently holding the floor.
- **Causal / online only** — no future frames at inference; deployable on a live stream.
- **5 FPS** feature rate; RGB via ResNet-50 (Kinetics-400) on 4-frame chunks; audio via wav2vec2; optional TV-L1 optical flow.
- **Backbone-agnostic** — Transformer (LSTR), GRU, and Mamba all plug into the same task; the paper benchmarks all three.
- **Cross-entropy** onset objective framed like next-token prediction.
- **Scalable pretraining data via VAD pseudo-labels**: YT-Conversation = 414 in-the-wild YouTube conversation videos (~41 h), labeled by Pyannote VAD, segments <200 ms removed.
- **Train/test splits.** EasyCom: sessions 1–3 test, 4–12 train (12 sessions, ~5 h 18 min, 3–5 participants). Ego4D (Audio-Visual Diarization subset, 5-min clips): 346 train / 87 test.
- **Optimizer**: Adam, lr $7\times10^{-5}$; 50 epochs / batch 16 (Transformer), 30 epochs / batch 64 (RNN, Mamba).

## Key results / what to remember
No Zotero highlights present.

- **EasyCom, avg mAP (0.20–2.00 s), A+V:** GRU **60.6**, Transformer 58.7, Mamba 57.4; Transformer audio-only 56.9. Multimodal (A+V) beats audio-only here.
- **Ego4D, avg mAP (0.20–2.00 s):** Transformer audio-only **69.2**, Transformer A+V 69.0, GRU A+V 68.2, Mamba A+V 67.5. On Ego4D adding vision did not help (audio dominates).
- **Target-speaker AP vs baselines (Table 3):** EasyCom Transformer A+V **52.7%** vs Random 27.2% / Silence-based 26.6%; Ego4D Transformer A+V **66.8%** vs Random 26.1% / Silence-based 27.7%. The learned model roughly doubles the naive baselines — anticipation is doing real work.
- **Optical flow (Table 4, EasyCom):** consistent gains when added, reported roughly **+0.9 to +9.6 mAP** across architectures.
- **YT-Conversation pretraining:** modest gains, roughly **+0.2 to +1.5** mAP, mainly on the "other speaker" class.
- **Runtime (Table 5):** all backbones run real-time — Transformer ~99.8 FPS (67.2M params), GRU ~13,939 FPS (34.6M), Mamba ~12,009 FPS (83.1M). Even the heaviest is far above real-time at 5 FPS input.
- **Takeaway:** onset prediction from egocentric streaming RGB+audio is feasible and clearly beats silence-thresholding; the best backbone is dataset-dependent (GRU on EasyCom, Transformer on Ego4D), and models underutilize short-term context — longer long-term windows help.

## How it connects (evolution)
- [[proactive-response]] — this is the sub-topic hub; EgoSpeak is the "when to speak" onset primitive underneath proactive assistants.
- [[proassist]] — proactive assistant that must decide *when* to interject; EgoSpeak supplies the onset-timing formulation.
- [[mmduet]] — turns dense video into a stream where the model chooses *when* to respond; shares the "decide the moment to speak" framing at the token level.
- [[vispeak]] — visual-driven proactive speaking on streaming video; closely related "when to speak" objective from vision.
- [[streammind]] — proactive streaming dialogue that gates when to generate; complementary trigger mechanism.
- [[omni-duplexeval]] — evaluates full-duplex/overlap conversational behavior, the same turn-taking regime EgoSpeak targets.

## Open questions / limitations
- **Pseudo-label quality:** YT-Conversation labels come from VAD, and human validation on 100 segments scored only ~2.147/5 alignment — noisy supervision may cap the modest pretraining gains.
- **Vision helps inconsistently:** RGB improves EasyCom but not Ego4D, and models "underutilize short-term information" — the visual/temporal signal is not yet being exploited well.
- **Onset only, no content:** the framework decides *when* to speak, not *what* to say or whether the utterance is appropriate; integrating it with a response generator is left open.
- **Threshold + evaluation gap:** deployment hinges on a speak-probability threshold and per-frame mAP, which don't directly measure downstream conversational quality (naturalness, interruption rate) in a live agent.

*Verification: numbers checked against the paper's own Tables 2 (mAP), 3 (target-speaker AP vs baselines), 4 (optical flow), and 5 (runtime), plus the task/label definitions in Sections 3.1–3.2 and Figures 2–3, via the arXiv HTML (2502.14892) and the rendered PDF pages.*

# When Does Data Selection Pay for Itself? Cost-Charged Evaluation of Online Selection for Language-Model Adaptation

Working draft, 23 November 2026. Numbers come from committed results in this repository; each claim names its source file. Audit status is stated per stage (Section 7). Literature statements marked [snippet] were checked only against search-result summaries and must be verified against the papers before submission.

## Abstract (provisional)

Online data selection chooses, at each step, which examples a model learns from. It is usually compared with random training at equal numbers of updates, which hides the cost of the decision. We charge each policy its measured step cost and decompose every cost-matched comparison exactly into a selection term (value per update) and a throughput term (updates forgone). Across pre-registered experiments on Arabic and English text, from a 35,776-parameter byte-level Transformer trained from scratch to LoRA adaptation of Qwen2.5-0.5B and 1.5B on GPUs, three results recur. (1) On models trained from scratch, gradient-alignment selection improves loss per update by only 0.4–2% of random training's gain but costs 6–11 random steps, and loses in every seed. (2) Per-update value is not stationary: a pre-registered forecast of cost-matched winners from equal-update measurements failed (6/10 correct), because rules such as highest-loss selection change what they pick as training proceeds. (3) On pretrained LLMs, computing per-example LoRA gradients from one batched backward pass lowers the cost of alignment to 1.3–2.1 random steps on a GPU, and alignment then wins exactly when much of the candidate pool is unsuitable for the target: with 90% off-language candidates it beat random in every seed for both model sizes (e.g. Qwen2.5-1.5B on Arabic Wikipedia, −0.090 nats/token, 95% CI [−0.109, −0.073]), whereas with in-domain candidates random won (+0.027 [+0.022, +0.033]). Highest-loss selection, despite being cheap, chose off-target data and was the worst policy whenever contamination was present. The value of a selection rule is therefore a property of the candidate pool and the cost ratio together, and should be reported as such.

## 1 Introduction

The question is not whether a selection rule picks better examples, but whether it picks examples better by more than it costs to pick them, and in which data regimes. Three observations motivate this paper:

1. In this project, every gradient-guided policy lost to uniform random selection under equal wall-clock budgets on small from-scratch models (stages 35, 37, 58, 60, 64, 65), despite small gains at equal update counts (stage 58; stage 64 decomposition).
2. The gap is not noise: the decomposition attributes nearly all of it to throughput (Arabic reuse4, stage 64: selection −0.0123, throughput +0.4923).
3. Recent online selection methods address exactly this cost. GREATS (Wang et al., NeurIPS 2024) computes per-example gradient inner products from activations and output gradients already available in backpropagation ("ghost inner products"), and reports runtime comparable to regular training [snippet]. A later paper reports that the method was memory-intensive and unstable beyond GPT-2 124M in its setting (ESLM, arXiv 2505.19893) [snippet]. Compute-positive selection has also been studied in FLOPs rather than wall-clock (arXiv 2312.05328) [snippet].

Contributions:
- An exact replay decomposition of cost-matched comparisons into selection and throughput terms, with a first-order break-even condition, applicable to any online selection method.
- Evidence that equal-update comparisons at a shared update count can mislead in either direction, because per-update selection value changes over training (pre-registered falsification, stage 67).
- A regime map for LoRA adaptation of pretrained LLMs: cost-charged gradient alignment wins under off-target contamination and loses on in-domain pools, replicated across two model sizes and three stages (68–70).
- Evidence that cheap loss-based rules can have negative per-update value and pick off-target data under contamination.
- Pre-registered protocols committed before outcomes, append-only run journals and audits; Arabic as the primary language throughout.

## 2 The break-even condition

Let L_R(n) be target loss after n random updates and L_G(n) after n guided updates, both from the same state. Within a budget T, random completes N_R updates and the guided rule N_G = N_R / r. Then exactly

L_G(N_G) − L_R(N_R) = [L_G(N_G) − L_R(N_G)] + [L_R(N_G) − L_R(N_R)].

The first bracket (selection term, −s) is measured by replaying the first N_G steps of the random schedule. The second (throughput term) is the loss forgone by fewer updates. If L_R decreases approximately linearly over [N_G, N_R], with a = L_R(0) − L_R(N_G), the throughput term is ≈ (r − 1) a, so the guided rule wins only if s/a > r − 1. Curvature of real learning curves can move the throughput term either way: concavity makes it smaller than (r − 1)a, while a random curve that is flat early and steep later (stage 67, high contamination) makes it larger. This is why the decomposition is measured by replay rather than assumed, and why a single-point estimate of s/a is not a forecast.

## 3 Experimental protocol

Common setting (stages 58–65): ByteTransformer, 35,776 parameters, 256-byte vocabulary, 64-byte context; AdamW (lr 1e-3, wd 0.01); one 65-byte chunk per update; single CPU thread; deterministic Torch. Candidates from official train splits of UD r2.15 Arabic-PADT and English-EWT; probe from dev; target 256 test chunks (stage 64) or 512 previously unscored dev chunks (stage 65). The deadline loop accepts only updates completed within 3 s and charges indexing, cloning, scoring and probe refreshes to each policy. Protocols: `research/ud-reuse-timed-protocol.json`, `research/ud-cost-frontier-protocol.json`.

## 4 Results

### 4.1 Equal-time comparison (stage 64; `results/ud-reuse-timed.json`)

| Arabic | final loss | updates |
| --- | --- | --- |
| alignment, gradient reused 4 steps | 2.19866 | 66.8 |
| alignment, fresh gradient | 2.55321 | 37.9 |
| random | 1.71865 | 412.7 |

Reusing the probe gradient beat fresh alignment in all eight seeds (−0.35455), and random beat both in all seeds.

### 4.2 Varying decision cost (stage 65; `results/ud-cost-frontier.json`)

| Arabic arm | r | s/a | selection term | throughput term | timed loss − random [95% CI] |
| --- | --- | --- | --- | --- | --- |
| highest loss of 4 | 1.22 | −0.0002 | +0.0003 | +0.0133 | +0.0136 [0.0091, 0.0181] |
| highest loss of 16 | 1.56 | −0.0145 | +0.0260 | +0.0378 | +0.0638 [0.0563, 0.0721] |
| highest loss of 64 | 2.84 | −0.0365 | +0.0625 | +0.1281 | +0.1907 [0.1829, 0.1986] |
| alignment, reuse 4 | 6.03 | +0.0044 | −0.0064 | +0.3915 | +0.3850 [0.3658, 0.4082] |

English shows the same ordering with larger harm from highest-loss selection (loss64 selection term +0.4402). The inequality agreed with all 80 state-level outcomes, but none was near the boundary.

### 4.3 Pretrained LLM (stage 66; `results/llm-selection.json`)

Qwen2.5-0.5B, LoRA rank 8 on q/v (540,672 trainable parameters), 64 matched updates, four seeds per language.

| arm | Arabic loss@64 − random | s/a | r | break-even r − 1 |
| --- | --- | --- | --- | --- |
| highest loss of 4 | +0.0034 | −0.073 | 2.12 | 1.12 |
| gradient-dot of 4 | −0.0162 (4/4 seeds) | 0.346 | 4.17 | 3.17 |

English: gradient-dot −0.0232 (4/4), s/a 0.172, r 4.28. Relative to the from-scratch byte model, the alignment gain per update is about 80 times larger, and the gap to break-even narrows from about 1,100× to about 9×. Several factors changed together (pretraining, scale, LoRA, token units, dot-product scoring), so the increase is not attributed to one of them. Replay audit pending.

### 4.4 Can equal-update measurements forecast equal-time outcomes? (stage 67; `results/mixture-crossover-phaseB.json`)

Candidate pools mixed Arabic with a fraction ρ ∈ {0, .5, .8, .9, .95} of English; the target was Arabic. Phase A (64 equal updates, 4 seeds) produced committed predictions; phase B (8 new seeds, 3 s budget, 360 runs) tested them. The pre-registered rule s/a > r − 1 was correct in 6/10 cases, with 4 contradictions whose 95% intervals exclude zero, so it fails its pre-specified criterion. Phase A also exposed a sign error in the ratio form when a < 0 (ρ = .95); the undivided form s > (r − 1)a, added in an addendum before phase B, also scored 6/10.

The failure has a clear mechanism: selection value is nonstationary. Highest-loss selection preferred Arabic during the first 64 updates (English share 0.70 at ρ = .8, below random's 0.82) and switched to English later (0.88 over about 340 updates), because it tracks whichever source currently has higher loss. The linear throughput estimate (r − 1)a from phase A also underestimated the measured throughput term at high ρ (alignment at ρ = .9: 0.27 against 2.38), because random's learning curve on the target was flat early and steep later.

At ρ = .95, highest-loss selection (r = 1.21) beat random under equal time: −0.218 [−0.355, −0.086]. Alignment had the largest selection term at every ρ (down to −1.31) but never overcame its throughput cost (up to +2.50).

Practical consequence: equal-update comparisons should be made at the update count each method attains in the budget, L_G(T/c_G) against L_R(T/c_R), not at a shared n. This requires measuring each curve over its full length.

### 4.5 Cost-charged comparison on the pretrained LLM (stage 68; `results/llm-budget.json`)

Applying the stage-67 lesson, each policy was charged its measured step cost (calibrated once on train candidates before any target evaluation) and evaluated at the update count it attains within a budget of 96 random-step costs: random 96, highest-loss 37, sequential gradient-dot 23, batched gradient-dot 33. Recovering per-example LoRA gradients from one batched backward lowered r from 4.10 to 2.84, and its choices matched the sequential variant on all common steps.

| ρ | arm | arm − random | seeds negative | selection | throughput |
| --- | --- | --- | --- | --- | --- |
| 0 | highest loss | +0.0110 | 0/4 | −0.0025 | +0.0135 |
| 0 | gradient-dot (batched) | +0.0075 | 1/4 | −0.0060 | +0.0135 |
| 0.9 | highest loss | +0.0330 | 0/4 | +0.0385 | −0.0056 |
| 0.9 | gradient-dot (sequential) | −0.0341 | 4/4 | −0.0265 | −0.0076 |
| 0.9 | gradient-dot (batched), primary | −0.0371 | 4/4 | −0.0288 | −0.0084 |

With 90% off-target candidates, random training barely moved the Arabic target (3.3982 to 3.3956 in 96 updates), so extra throughput was worthless (negative throughput term) and alignment's selection term decided the outcome. Highest-loss selection chose English 99% of the time and hurt the target. Four seeds, descriptive; stage 69 replicates on an A100 with six seeds, two model sizes and a new Arabic Wikipedia target.

### 4.6 GPU replication with two model sizes (stage 69; `results/llm-gpu-scale.json`)

Colab, NVIDIA L4 (the protocol named an A100; see `research/llm-gpu-scale-deviations.md`); Qwen2.5-0.5B and 1.5B; budget 400 random-step costs; six seeds; new target of 256 Arabic Wikipedia article openings. On the GPU, batched alignment cost only 1.30 (0.5B) and 1.89 (1.5B) random steps. Batched alignment minus random: 0.5B −0.215 (ρ 0) and −0.192 (ρ 0.9); 1.5B −0.249 (ρ 0) and −0.172 (ρ 0.9, primary, 95% CI [−0.219, −0.126]); 6/6 seeds in every cell. Highest-loss selection was inconclusive or harmful under contamination.

Critical context (post hoc): random LoRA training on the PADT-news/EWT pool raised Wikipedia loss above its starting value (+0.03 to +0.19), and only the alignment policies finished below it (−0.03 to −0.14). In this domain-shifted setting alignment works mainly as a filter against harmful updates; it does not show that alignment beats random when random training already helps the target (stage 68, ρ 0, showed the opposite).

### 4.7 In-domain candidates and a no-update baseline (stage 70; `results/llm-gpu-indomain.json`)

Colab L4; Arabic candidates from Arabic Wikipedia (excluding stage-69 articles), target 256 new Wikipedia articles, English EWT as contamination; six seeds; no-update baseline pre-specified. Random training improved the target by about one nat per token (0.5B −1.129, 1.5B −0.996 at ρ 0). Primary (1.5B, ρ 0): batched alignment minus random +0.0274 [0.0223, 0.0325], 0/6 seeds favouring alignment; selection term −0.0008, throughput term +0.0282. At ρ 0.9 alignment won in every seed (0.5B −0.189 [−0.226, −0.142]; 1.5B −0.090 [−0.109, −0.073]) by choosing English 66–67% of the time against 90% in the pool. Highest-loss selection chose English 99–100% of the time.

Synthesis of stages 68–70: on pretrained LLMs adapted with LoRA, the selection term of gradient alignment is large when a substantial fraction of candidates is unsuitable for the target (off-language or off-domain) and near zero when candidates are in-domain. With batched per-example gradients the cost is 1.3–2.1 random steps on a GPU, so alignment pays off exactly in the first regime. This matches the break-even decomposition: the method's value is a property of the candidate pool, not of the method alone.

### 4.8 Where alignment starts to pay (stage 71; `results/llm-gpu-threshold.json`)

Qwen2.5-1.5B on an L4, Arabic Wikipedia candidates and target, English contamination ρ ∈ {0, .25, .5, .75, .9}, six seeds, r = 1.95. Alignment minus random: +0.025, +0.009, −0.018, −0.038, −0.093; selection term −0.005, −0.026, −0.056, −0.116, −0.230; throughput term +0.030 to +0.137. Pre-registered crossover estimate ρ* = 0.34, 95% CI [0.29, 0.52]; the prediction (crossover in [0, 0.5]) held. Exploratory: the same seeds and schedules on a Tesla T4 (r = 3.24, outside the protocol) gave ρ* ≈ 0.72 [0.43, 0.80], so a 1.7-fold higher decision cost roughly doubled the contamination needed for alignment to pay.

### 4.9 Same-language contamination (stage 72; `results/llm-gpu-noise.json`)

Qwen2.5-1.5B on an L4, r = 1.96, six seeds; clean Arabic Wikipedia candidates mixed at ρ ∈ {0.5, 0.9} with Arabic news (PADT), garbled Arabic Wikipedia (token-permuted chunks) or English. Primary (garbled, ρ 0.5): −0.0049 [−0.021, +0.017], inconclusive. At ρ 0.9: garbled −0.242 [−0.269, −0.217], news −0.091 [−0.143, −0.034], English −0.063 [−0.119, −0.012]. At ρ 0.5 alignment chose 14–16% contaminant chunks against about 50% for random for every type, but its selection term roughly equalled the throughput cost, as expected near the stage-71 crossover. The filtering effect is therefore not specific to a second language.

## 5 Discussion: when alignment pays

| stage | model, hardware | candidate pool → target | alignment r | alignment − random | selection | throughput |
| --- | --- | --- | --- | --- | --- | --- |
| 68 | Qwen2.5-0.5B, CPU | PADT Arabic → PADT dev | 2.84 | +0.0075 (1/4 seeds better) | −0.0060 | +0.0135 |
| 68 | same | 90% English → PADT dev | 2.84 | −0.0371 (4/4) | −0.0288 | −0.0084 |
| 69 | Qwen2.5-1.5B, L4 | PADT Arabic → Wikipedia (domain shift) | 1.89 | −0.249 (6/6) | −0.229 | −0.020 |
| 69 | same | 90% English → Wikipedia | 1.89 | −0.172 (6/6) | −0.186 | +0.014 |
| 70 | Qwen2.5-1.5B, L4 | Wikipedia Arabic → Wikipedia | 2.08 | +0.027 (0/6) | −0.001 | +0.028 |
| 70 | same | 90% English → Wikipedia | 2.08 | −0.090 (6/6) | −0.186 | +0.097 |

Across these cells the throughput term is small or modest once per-example gradients are batched, and the sign of the outcome is set by the selection term. The selection term is near zero when candidates already match the target (stage 70, ρ 0; stage 68, ρ 0) and large when many candidates do not (off-language contamination, or off-domain news for a Wikipedia target). A practical rule follows: use target-aligned selection when a substantial share of the pool is expected to be unsuitable for the target, and random sampling when the pool is curated and in-domain. Stage 71 located the boundary at about one third of the pool (ρ* = 0.34 [0.29, 0.52]) for Qwen2.5-1.5B on an L4 at r = 1.95, and an exploratory T4 run at r = 3.24 moved it to about 0.72, so the threshold must be reported together with the cost ratio of the hardware in use.

Two cautions. In stage 69, random training on the news pool raised the Wikipedia loss above its starting value, so alignment's win there is partly harm avoidance. And the throughput term depends on hardware: the same batched rule cost 1.3–2.1 random steps on an L4, 2.84 on a CPU and 3.24 on a T4.

## 6 Limitations

Small and mid-sized models (35,776 parameters to 1.5 billion), LoRA rather than full fine-tuning, short budgets (96–400 random-step costs), contamination mostly by a second language (stage 72 adds same-language news and one synthetic garbling type), four to six seeds per cell, and one model family. Wall-clock ratios are hardware- and implementation-specific; stage 69 ran on an L4 instead of the pre-registered A100 and a secondary stage-71 run on a T4 (documented deviations). Qwen pretraining data are unknown, so evaluation text may have been seen in pretraining. Corpora were reused across stages (exposure ledger: `data/evaluation_status.json`). The original WAPLV hypothesis (weight–sample features) was not supported in stages 18–55 and is not the subject of this paper.

## 7 Reproducibility and audit status

Every stage has a protocol committed before outcomes and a journal of runs written as they complete. Stages 64, 65 and 67 passed exact replay audits of all trajectories. Stages 69 and 70 passed CPU audits of journals, summaries, budgets, samples, schedules and random choices; their GPU trajectories were not replayed. The stage-68 audit found that batched per-example gradients differ from separately computed ones in 32 of 540,672 entries beyond the pre-registered tolerance (maximum absolute difference 2.9e-6); its full replay and the stage-66 replay were interrupted by machine restarts and are being rerun.

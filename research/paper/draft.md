# Selection Must Pay for Itself: A Break-Even Analysis of Online Data Selection

Working draft, 21 November 2026. Numbers come from committed, audited results in this repository; each claim names its source file. Sections marked TODO depend on stages not yet complete. Literature statements marked [snippet] were checked only against search-result summaries and must be verified against the papers before submission.

## Abstract (provisional)

Online data selection methods choose, at each training step, which examples to learn from. They are usually evaluated at equal numbers of updates or equal FLOPs of the training step, which hides the cost of the decision itself. We make that cost explicit. Writing r for the ratio of the wall-clock cost of a guided step to a random step, a for the loss reduction random training achieves in n updates, and s for the additional reduction a selection rule adds at the same n, a first-order argument gives a necessary condition for selection to beat random training under an equal time budget: s/a > r − 1. We decompose every timed comparison exactly into a selection term and a throughput term by replaying the random schedule to each guided run's update count. In pre-registered experiments on Arabic (UD PADT) and English (UD EWT) text with a 35,776-parameter byte-level Transformer, gradient-alignment selection improves loss per update (s/a ≈ 0.004–0.018) but costs r ≈ 6–11 random steps, and loses to random under equal time in every seed (e.g., Arabic +0.385 nats/byte, 95% CI [0.366, 0.408]). Cheap highest-loss selection (r ≈ 1.2–2.9) is worse still because its selection term is positive: it harms target loss per update, increasingly with the candidate pool. TODO: pretrained-LLM results (stage 66) and a near-boundary test (stage 67).

## 1 Introduction

The question is not whether a selection rule picks better examples, but whether it picks examples better by more than it costs to pick them. Three observations motivate this paper:

1. In this project, every gradient-guided policy lost to uniform random selection under an equal wall-clock budget (stages 35, 37, 58, 60, 64, 65), despite small gains at equal update counts (stage 58; stage 64 decomposition).
2. The gap is not noise: the decomposition attributes nearly all of it to throughput (Arabic reuse4, stage 64: selection −0.0123, throughput +0.4923).
3. Recent online selection methods address exactly this cost. GREATS (Wang et al., NeurIPS 2024) computes per-example gradient inner products from activations and output gradients already available in backpropagation ("ghost inner products"), and reports runtime comparable to regular training [snippet]. A later paper reports that the method was memory-intensive and unstable beyond GPT-2 124M in its setting (ESLM, arXiv 2505.19893) [snippet]. Compute-positive selection has also been studied in FLOPs rather than wall-clock (arXiv 2312.05328) [snippet].

Contributions:
- A break-even inequality and an exact replay decomposition that separate the selection value of a rule from its throughput cost, applicable to any online selection method.
- Pre-registered, audited experiments in which each protocol was pushed before outcomes were measured and every final model was reconstructed exactly from its recorded choices.
- Evidence that cheap loss-based rules can have negative per-update value on held-out text, so lowering r alone is not sufficient.
- Arabic as the primary language throughout.

## 2 The break-even condition

Let L_R(n) be target loss after n random updates and L_G(n) after n guided updates, both from the same state. Within a budget T, random completes N_R updates and the guided rule N_G = N_R / r. Then exactly

L_G(N_G) − L_R(N_R) = [L_G(N_G) − L_R(N_G)] + [L_R(N_G) − L_R(N_R)].

The first bracket (selection term, −s) is measured by replaying the first N_G steps of the random schedule. The second (throughput term) is the loss forgone by fewer updates. If L_R decreases approximately linearly over [N_G, N_R], with a = L_R(0) − L_R(N_G), the throughput term is ≈ (r − 1) a, so the guided rule wins only if s/a > r − 1. Concavity of real learning curves makes the throughput term smaller than (r − 1)a, so the linear condition is conservative in that respect; this is why the decomposition is measured rather than assumed.

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

### 4.3 Pretrained LLM (stage 66) — TODO

Qwen2.5-0.5B, LoRA rank 8 on q/v projections, 64 matched updates; random, highest-loss of 4, gradient-dot alignment of 4. Report s/a and measured r per language.

### 4.4 Near-boundary test (stage 67) — TODO

A rule with alignment-like s and r close to 1, e.g. per-example LoRA gradient inner products from one batched backward pass, so that s/a and r − 1 are comparable and the inequality can be tested for discrimination.

## 5 Limitations

Small models on CPU; wall-clock ratios are hardware- and implementation-specific; corpora reused across stages (exposure ledger: `data/evaluation_status.json`); the original WAPLV hypothesis (weight–sample features) was not supported in stages 18–55 and is not the subject of this paper.

## 6 Reproducibility

Every stage has a protocol committed before outcomes, a journal of runs written as they complete, and an audit that recomputes summaries and replays trajectories (`tools/audit_*.py`).

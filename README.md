# WAPL-Research

Research into **Weight-Aware Predictive Learning Value (WAPLV)**: whether weight–sample interactions improve predictions of training-example utility beyond loss, gradients, and optimizer-aware baselines.

## Reports — التقارير

**[جميع تقارير Word العربية وفهرس مراحل البحث](reports/README.md)**

Each report has a `.docx` download and a readable Markdown source in [`reports/`](reports/). Open a Word file on GitHub and choose **Download raw file**.

## Current evidence

The implemented studies are small CPU diagnostics: logistic regression, character-bigram models, a 9,184-parameter causal Transformer, and a 35,776-parameter byte-level Transformer. They have **not established an independent benefit from the tested weight features or compute savings over simple controls**. Horizon-matched simulations improved utility-value accuracy, and pairwise learning improved selection against a pointwise learner on Bryant; neither establishes general selection superiority or practical efficiency. The independent Blake timing study did not establish a benefit from sparse highest-loss selection over random.

The Arabic synthesis covers stages 1–42 and links 11 distinct contrasts to their results. Later PUD studies found higher Arabic selection regret after adding corrected weight features. In the subsequent policy study, random selection produced lower final loss than update alignment under a three-second CPU budget. Alignment produced slightly lower loss at a fixed 128 updates in both languages. Stage 64 repeated the timed comparison on PADT/EWT: gradient reuse beat fresh alignment, but random still won by a wide margin; a post-hoc decomposition attributes the gap almost entirely to lost update throughput. [Report 64](reports/64_ud_reuse_timed_ar.md) · [Roadmap after 64](research/roadmap-after-64-ar.md) [Stage 42 synthesis](reports/42_scientific_synthesis_ar.md) · [Evidence map](research/evidence-summary.md) · [Frozen rankers on PUD](reports/55_pud_transfer_results_ar.md) · [Timed policy comparison](reports/58_pud_policy_results_ar.md)

Historical Transformer results mentioned in the supplied project document were not reproduced: their original code and artifacts were not present. The executable experiments here are new diagnostics, not reconstructions of that historical run.

## Repository contents

- `reports/`: Arabic Word reports and their Markdown sources.
- `experiments/`: executable diagnostic experiments and policy comparisons.
- `tests/`: numerical, state isolation, data selection, and deadline checks.
- `results/`: metrics, per-run observations, predictions, audit records, and provenance hashes.
- `research/`: reviews, protocols, and external-source retrieval records.
- `data/`: source metadata; downloaded third-party texts are excluded.
- `tools/`: verified data download and Arabic Word report generation.

## Reproduce

Python 3.12 was used. Install the pinned dependencies in a virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-pilot.txt -r requirements-reports.txt -r requirements-transformer.txt
python tools/download_data.py
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
```

The downloader verifies the recorded SHA-256 hashes and refuses to overwrite mismatched local inputs. Network access to `raw.githubusercontent.com` is needed. Dataset edition rights have not been independently audited; review the source metadata and applicable terms before redistribution.

Experiment sequence (run modules from this repository root):

```bash
OPENBLAS_NUM_THREADS=1 python -m experiments.pilot
OPENBLAS_NUM_THREADS=1 python -m experiments.text_pilot
OPENBLAS_NUM_THREADS=1 python -m experiments.invariance
OPENBLAS_NUM_THREADS=1 python -m experiments.centered_pilot
OPENBLAS_NUM_THREADS=1 python -m experiments.policy
OPENBLAS_NUM_THREADS=1 python -m experiments.time_policy
OPENBLAS_NUM_THREADS=1 python -m experiments.state_coverage
OPENBLAS_NUM_THREADS=1 python -m experiments.reuse
OPENBLAS_NUM_THREADS=1 python -m experiments.optimized_time
```

Commands overwrite their corresponding result files; preserve original outputs before rerunning. Wall-clock experiments intentionally have timing-dependent update counts. Stored hashes identify the inputs/code used for the recorded results; regenerating intermediate artifacts can change downstream provenance.

The evaluation partitions used in completed studies are exposed, including the final Emma partition. The final Persuasion 80–100% partition was first evaluated in the calibration study (report 20) and is now exposed. Further method development must use an untouched evaluation set for confirmatory claims.

## Source material

External papers, repository snapshots, model-card copies, and corpus text downloads are kept out of Git. Their URLs and retrieval hashes remain in `research/**/access.json` and `data/*source.json`. The reports distinguish inspected sources, publisher claims, and experiments actually run. Reading a library or model card does not establish successful installation or model evaluation.

## Tiny Transformer pilot

Install `requirements-transformer.txt` for the pinned CPU PyTorch build. Run `OPENBLAS_NUM_THREADS=1 python -m experiments.transformer_pilot`. This is a newly initialized, small character-level model, not a pretrained LLM or reproduction of the historical Transformer experiment. The original pilot left the Persuasion final 80–100% partition unused; report 20 subsequently evaluated it. Both evaluation partitions are now exposed.

## Transformer calibration diagnostic

Run `OPENBLAS_NUM_THREADS=1 python -m experiments.transformer_calibration` to compare first-order update alignment with the measured one-step probe loss change at horizons 1 and 4. See reports 19–20 for the frozen local protocol, limits, and results.

## Horizon-matching diagnostic

Run `OPENBLAS_NUM_THREADS=1 python -m experiments.transformer_horizon`. Reports 21–22 distinguish known continuation schedules from independent alternatives at horizons 1, 4, and 16. The primary horizon-4 comparison was inconclusive; the horizon-16 improvement is secondary and computationally expensive. The Thursday target partition is now exposed.

## Independent horizon-16 replication

Run `OPENBLAS_NUM_THREADS=1 python -m experiments.horizon_replication` for the Alice study in reports 23–24. It averages two independent continuation simulations and fixes probe panel A as primary, panel B as sensitivity analysis. The primary MAE comparison improved, but the simulation is expensive and does not establish useful weight features or compute savings. Alice target data are now exposed.

## Horizon-16 utility student

Run `OPENBLAS_NUM_THREADS=1 python -m experiments.utility_student` for reports 25–26. A five-feature Ridge student trained on Alice probe-simulation labels improves Brown target MAE over affine-calibrated update alignment, but does not improve ranking or top-two selection regret. Adding three weight features has no clear MAE benefit. Timings include feature extraction and teacher-label generation; no end-to-end compute savings are established. Brown target data are now exposed. See `results/student-audit.json` for the repeatability check.

## Exploratory ranking diagnostic

Run `OPENBLAS_NUM_THREADS=1 python -m experiments.ranking_diagnostic` for report 27. On the already exposed Brown target, about 93.6% of the student’s MSE reduction versus calibrated alignment is attributable to reduced state-level mean error. Within-state squared error also improves, but selection superiority is not established. Post-hoc seed-bootstrap intervals for regret and pair accuracy include zero. This diagnostic fits no new model and provides no independent confirmation.

## Pairwise utility transfer

Run `OPENBLAS_NUM_THREADS=1 python -m experiments.pairwise_student` for reports 28–29. Protocol 28 was committed and pushed before the new Bryant evaluation. Pairwise Ridge lowers top-two regret by about 53.7% versus pointwise Ridge with the same regret-based tuning criterion (primary paired seed-bootstrap interval excludes zero). Its descriptive advantage over simple calibrated alignment is small and is not established as a confirmatory result. This is a fixed-state utility evaluation, not an end-to-end training policy or compute-efficiency result. Bryant target data are now exposed. `results/pairwise-audit.json` records refitting and metric checks; full Transformer generation was not repeated.

## Independent simple-control comparison

Run `OPENBLAS_NUM_THREADS=1 python -m experiments.simple_control` for reports 30–31, and `OPENBLAS_NUM_THREADS=1 python -m tools.audit_simple_control` to recompute predictions and statistics. The protocol was pushed before the new Edgeworth evaluation (16 seeds, 256 candidates). Pairwise Ridge did not establish superiority over calibrated update alignment: mean regret difference −0.000009794, paired 95% bootstrap interval [−0.000132040, 0.000123965]. This is not evidence of equivalence. Edgeworth targets are now exposed. Full Transformer trajectories were not rerun; refitting, predictions, metrics, and source hashes were checked. The generator now accepts source/seeds while retaining Bryant defaults; historical hashes in report 29 refer to its original Git version.

## Sequential selection at equal update counts

Run `OPENBLAS_NUM_THREADS=1 python -m experiments.sequential_policy` for reports 32–33; audit stored schedules and statistics with `OPENBLAS_NUM_THREADS=1 python -m tools.audit_sequential_policy`. The Milton protocol was pushed before evaluation. Eight paired seeds share initialization and candidate schedules, then execute 32 selected updates after 12 warmup steps. Pairwise selection did not beat update alignment: final loss difference +0.00004521, paired 95% bootstrap interval [−0.00093120, +0.00119323]. Both guided policies had descriptively lower loss than random at equal updates, but took roughly 26 times as long in this CPU run. This does not establish compute savings. Milton targets are now exposed; complete trajectories were not rerun.

## Transformer policies under equal deadlines

Run `OPENBLAS_NUM_THREADS=1 python -m experiments.deadline_policy` for reports 34–35; audit stored results with `OPENBLAS_NUM_THREADS=1 python -m tools.audit_deadline_policy`. The Whitman protocol was pushed before evaluation. Under one-second acceptance windows across 12 seeds, random selection achieved mean loss 2.92682 versus 4.25955 for pairwise guidance, with about 205 versus 17.4 accepted updates. Pairwise-minus-random loss was +1.33273 (paired 95% bootstrap interval [1.28081, 1.38950]). Late updates were discarded; actual execution can overrun the deadline. Transactional cloning was charged to all policies, including random. This result argues against compute efficiency of the current implementation in this short CPU setting, not against all data selection. Whitman targets are now exposed; timing-dependent trajectories were not rerun.

## Exploratory sparse guidance

Run `OPENBLAS_NUM_THREADS=1 python -m experiments.sparse_policy` for reports 36–37; audit with `OPENBLAS_NUM_THREADS=1 python -m tools.audit_sparse_policy`. On the already exposed Whitman target, guiding every 16 steps improved loss relative to dense guidance (3.07811 versus 4.29949), but remained worse than random (2.92790) under one-second acceptance windows. Sparse-minus-random loss was +0.15021, exploratory paired 95% bootstrap interval [0.13497, 0.16614]. New seeds do not make this an independent corpus confirmation. The deadline helper now accepts a guidance interval, defaulting to 1; historical report-35 hashes refer to the earlier Git version.

## Exploratory batched-loss proxy

Run `OPENBLAS_NUM_THREADS=1 python -m experiments.loss_proxy_policy` for reports 38–39; audit with `OPENBLAS_NUM_THREADS=1 python -m tools.audit_loss_proxy_policy`. Batched forward-only candidate loss avoids trial updates for scoring. On exposed Whitman, selecting the highest-loss candidate every 16 steps achieved mean loss 2.92374 versus random 2.92832, but the exploratory paired 95% bootstrap interval for the difference includes zero [−0.01429, 0.00381]. It improved over sparse pairwise guidance descriptively and in the secondary exploratory interval, without establishing superiority to random. Its slightly higher update count than random is not established as a speedup; timing runs were not repeated. Historical hashes of the scoring helper refer to their original Git versions.

## Independent loss-proxy replication with timing repeats

Run `OPENBLAS_NUM_THREADS=1 python -m experiments.loss_replication` for reports 40–41; audit with `OPENBLAS_NUM_THREADS=1 python -m tools.audit_loss_replication`. The Blake protocol was pushed before target evaluation. Twelve seeds, three arms and two timing repeats produced 72 runs. Repeats are averaged within seed before bootstrap. Sparse highest-loss selection did not establish superiority to random (loss difference +0.000841, 95% CI [−0.004416, +0.006204]) or to a secondary control that computes losses but chooses randomly. Repeated timings changed update counts; this is not evidence of equivalence. Blake targets are now exposed. Historical deadline-helper hashes refer to earlier Git versions; the new sham branch does not change existing arm paths.

## Arabic and English evaluation readiness

Report [43](reports/43_multilingual_readiness_ar.docx) prepares UD Arabic-PADT and English-EWT release r2.15 with official split membership and exact-text deduplication in train/dev/test order. Raw corpora stay outside Git. Declared licenses are CC BY-NC-SA 3.0 for Arabic-PADT and CC BY-SA 4.0 for English-EWT; source-text rights have not been independently cleared. URLs, publisher README hashes, data hashes, and counts are in `data/multilingual_sources.json`. The corpora differ in domain, so comparisons do not isolate language effects.

```bash
python -m tools.prepare_multilingual
OPENBLAS_NUM_THREADS=1 python -m experiments.multilingual_readiness
python -m tools.audit_multilingual_readiness
```

The new 35,776-parameter Transformer uses a 256-byte UTF-8 vocabulary and 64-byte input context. Four training-only smoke runs completed; dev/test were processed for integrity and duplicate checks but have not been scored by a model in this stage. This establishes technical readiness, not policy efficacy, language generalization or pretrained-LLM performance. Byte losses should not be compared across languages as a quality ranking. This was the historical status at stage 43. Protocol 44 subsequently froze the evaluation, and report 45 consumed both test outcomes; see `data/evaluation_status.json` for current exposure. The preparation manifest remains a historical snapshot.

## Arabic-primary multilingual policy evaluation

Run `OPENBLAS_NUM_THREADS=1 python -m experiments.multilingual_policy` for reports 44–45; audit with `python -m tools.audit_multilingual_policy`. The protocol was pushed before test scoring. Ninety-six runs compare random selection with highest loss every 16 steps, using 256 fixed updates or three-second windows with two timing repeats per seed. All 3,668 Arabic and 1,877 English complete test chunks are scored.

The Arabic timed primary comparison did **not** establish superiority: sparse-loss minus random cross-entropy +0.001953, paired 95% bootstrap interval [−0.001528, +0.005375]. A small Arabic fixed-update improvement is secondary and does not establish time efficiency. English results are also inconclusive. Byte losses are compared within language only. Both test splits became exposed in stage 45; dev was not used in that stage. Stage 49 later evaluated a fixed subset of dev, as recorded in `data/evaluation_status.json`. This remains a small-model CPU study, not evidence of weight-feature benefits or pretrained-LLM performance.

## Transformer output-head invariance

Run `OPENBLAS_NUM_THREADS=1 python -m experiments.head_invariance` for reports 46–47. A common shift across output-head classes preserves softmax probabilities and the tested four-step utility, but changes two raw weight features. Projecting out the common output-head component from weights and updates restores feature invariance in 16 float64 training-only comparisons. The full diagnostic repeated byte-identically; see `results/head-invariance-audit.json`. This is a known softmax symmetry and a consistency check, not evidence of predictive value or compute savings. The baseline update norm can also depend on the shift through weight decay, so future ablations must correct the baseline as well. No new dev/test outcomes were evaluated.

## Exploratory corrected-weight feature ablation

Run `OPENBLAS_NUM_THREADS=1 python -m experiments.corrected_features` for reports 48–49; audit with `OPENBLAS_NUM_THREADS=1 python -m tools.audit_corrected_features`. Both the baseline update representation and added weight features are corrected for output-head shifts. Arabic corrected-minus-baseline MAE was inconclusive (−0.00009077, exploratory 95% CI [−0.00034321, +0.00018521]), and the noise-feature control performed better. English corrected features improved MAE over baseline and noise, but not selection ranking/regret; improvement over raw weight features was inconclusive. This is exploratory development on previously used sources, not independent confirmation. The 128 selected dev chunks per language are now exposed; test files were not read in this experiment.

## Noise-control and regularization sensitivity

Run `OPENBLAS_NUM_THREADS=1 python -m experiments.noise_sensitivity` for report 50. This post-hoc audit reuses stage-49 labels with 64 noise-feature draws per language, comparing tuned alpha with alpha fixed at the baseline value. Arabic noise-control performance is sensitive to regularization choice; no fixed-alpha noise draw beat corrected features. English corrected-feature MAE remained below all 64 noise draws in both modes, without establishing independent generalization or selection improvement. Noise-draw percentiles are not confidence intervals over independent datasets. The retrospective alpha sweep is diagnostic and must not replace the original held-out comparison.

## Grouped tuning sensitivity

Run `OPENBLAS_NUM_THREADS=1 python -m experiments.grouped_tuning` for report 51. Leave-one-model-seed-out tuning uses fit rows only, with fold-specific scaling and the same 64-row final fit set as the original holdout rule. On exposed Arabic labels, the baseline improved while corrected weight features did not, producing worse corrected-feature MAE relative to baseline. English corrected-feature MAE improved, while ranking and selection regret worsened. This post-hoc analysis tests sensitivity to tuning, not independent generalization; fold separation does not remove shared-corpus or shared-monitor dependence. Historical holdout metrics matched exactly and the complete analysis repeated byte-identically.

## Grouped tuning by selection regret

Run `OPENBLAS_NUM_THREADS=1 python -m experiments.regret_tuning` for report 52. This post-hoc comparison changes alpha selection from MAE to top-two regret within the same seed-grouped folds; Ridge still fits squared error. It did not improve corrected-feature selection: Arabic corrected features remained worse than baseline, while English selected the same corrected model under both criteria. Direct update alignment had lower regret in both languages in these exposed records. The analysis repeated byte-identically and MAE-arm predictions matched report 51. This does not test a directly optimized ranking loss or establish policy efficiency.

## Direct pairwise logistic ranking

Run `OPENBLAS_NUM_THREADS=1 python -m experiments.pairwise_logistic` for report 53. Pairwise logistic fitting changes the learning objective itself, unlike alpha selection alone. On exposed records, corrected-feature ranking improved Arabic regret relative to the prior corrected Ridge model, but did not beat the ranker using baseline features alone. In English, adding corrected weights increased regret relative to that baseline ranker. Scores are ranking margins, not calibrated utilities; no score MAE is claimed. This exploratory, deterministic rerun does not establish independent weight-feature value or policy efficiency.

## Prospective PUD transfer readiness

Report [54](reports/54_pud_readiness_ar.docx) prepares UD Arabic/English PUD r2.15 without model scoring. Its 1,000 aligned sentence pairs form ten fixed parallel folds; exact-text checks found no matches against the prior prepared PADT/EWT splits, but do not exclude near duplicates or shared documents. The languages are translations, not independent samples. Both sources declare CC BY-SA 3.0, and the publisher recommends ten-fold cross-validation if used for training. Raw text remains excluded from Git.

Restore with `python -m tools.prepare_pud`; freeze the existing rankers with `OPENBLAS_NUM_THREADS=1 python -m tools.freeze_pud_rankers`. The prospective design is in `research/pud-transfer-protocol.json`, the six frozen rankers in `research/pud-frozen-rankers.json`, and preparation checks in `results/pud-readiness-audit.json`. At stage 54 PUD outcomes had not been evaluated. Report 55 subsequently evaluated all ten folds, so PUD is now exposed for future development. The preparation itself is not an efficacy result or an official UD parsing benchmark.

## Frozen-ranker transfer to PUD

Run `OPENBLAS_NUM_THREADS=1 python -m experiments.pud_transfer` for report 55; audit with `OPENBLAS_NUM_THREADS=1 python -m tools.audit_pud_transfer`. The published stage-54 design and six rankers remained fixed. Ten folds, eight seeds and four candidates yielded 640 utility labels across the parallel languages. Fold metrics were averaged within seed before paired bootstrap. The Arabic primary contrast argues against the added corrected features: top-two regret difference +0.000126433, 95% interval [0.000058080, 0.000193542]. The English corrected-versus-baseline contrast was inconclusive. These results do not establish an independent weight-feature benefit or compute savings. PUD folds are now exposed; source languages are translations, not independent samples. Full Transformer generation was not repeated; stored predictions, coverage, metrics and hashes were audited.

## PUD fold and seed sensitivity

Report [56](reports/56_pud_sensitivity_ar.docx) describes a post-hoc analysis of stage-55 records. Run `OPENBLAS_NUM_THREADS=1 python -m experiments.pud_sensitivity`. Corrected-minus-baseline regret remained positive after each single-fold and single-seed omission in both languages. Arabic state counts were 4 improved, 20 worsened and 56 tied; English counts were 9, 12 and 59. This descriptive stability does not change the original inconclusive English interval or establish independent replication. No new Transformer training or hypothesis test was performed. Analysis repeated byte-identically; source hashes, grids, omission means and choice overlap were checked.

## PUD alignment policy protocol

Report [57](reports/57_pud_policy_protocol_ar.docx) fixes an exploratory policy comparison before new runs on already exposed PUD. `research/pud-policy-protocol.json` specifies alignment versus random selection, 3-second budgets with two timing repeats, and a 128-update diagnostic. Across two languages, ten folds and eight new seeds, 960 runs are planned. The policy clock includes probe gradients, candidate trial updates, cloning and selection; late decisions are discarded and wall-time overruns recorded. Arabic timed final loss is primary, with folds and repeats averaged before paired seed bootstrap. No new model evaluation or policy implementation is claimed at this stage; required implementation tests precede execution. This is not independent confirmation or a global compute-savings claim.


## Timing the PUD alignment policy

All 960 runs in the stage 57 protocol completed in 66.28 minutes. The primary Arabic comparison gave final losses of 1.7182 for random selection and 2.6316 for update alignment. The paired difference was +0.9134 nats per byte transition (95% seed bootstrap interval 0.8706 to 0.9618). At 128 updates the Arabic difference was −0.00427, with mean policy times of 0.75 and 11.44 seconds respectively. The results describe the local CPU implementation on previously examined PUD folds. [Arabic Word report](reports/58_pud_policy_results_ar.docx) · [Per-run records](results/pud-policy-records.jsonl) · [Summary](results/pud-policy.json)

With the pinned PUD inputs present, run `OPENBLAS_NUM_THREADS=1 python -m tools.audit_pud_policy` to verify the saved experiment without retraining. For a new timing run, use a separate checkout, move `results/pud-policy-records.jsonl` from that checkout to a separate archive, then run `OPENBLAS_NUM_THREADS=1 python -m experiments.pud_policy`. Record the new machine and timings alongside that attempt.

## Profiling an alignment decision

Report 59 breaks down the time spent choosing among four candidate updates at the first decision after warmup. It uses Arabic and English PUD, folds 0 and 5, and seeds 28000 and 28001. Each of the eight starting states contributes six paired timings of the original and instrumented decision, with the order balanced within state. Preparing and computing the gradient on the 64 probe chunks took 64.27 ms of the 94.87 ms mean instrumented decision (67.75%). Constructing and scoring the four candidate deltas took 1.82 ms (1.92%). Repeats measure variability in runtime at the same state. [Arabic Word report](reports/59_pud_policy_profile_ar.docx) · [Raw timings](results/pud-policy-profile.json)

From the repository root with the pinned PUD inputs available, run `OPENBLAS_NUM_THREADS=1 python -m tools.audit_pud_policy_profile` to check the recorded measurements. To collect another sample, choose a new file, for example `OPENBLAS_NUM_THREADS=1 python -m experiments.pud_policy_profile --output /tmp/pud-policy-profile-repeat.json`. Exact clock readings depend on the host.

## Reusing the probe gradient for four updates

Stage 60 tests the change suggested by the cost profile. The new selector refreshes the probe gradient at steps 0, 4, 8 and so on. It still computes four trial AdamW updates at each step using the current model and optimizer. The pilot compares this selector with fresh alignment and random selection on the same eight PUD starting states. Each state contributes three matched three-second runs per selector. A separate 16-step trajectory per state compares candidate choices with and without gradient reuse at the same weights. [Design published before timing](research/pud-reuse-protocol.json) · [Arabic Word report](reports/60_pud_reuse_ar.docx)

Under the three-second budget, Arabic mean accepted updates rose from 40.92 with fresh alignment to 73.42 with reuse, and mean final loss fell from 2.4230 to 2.1159 nats per byte transition. Random selection reached 1.7180. In English the respective losses were 3.2175, 3.1208 and 2.7016. The cached and freshly recomputed gradients picked different candidates in 11 of the 96 same-state comparisons made between refreshes. These are descriptive observations on two seeds and two folds per language. [Pilot summary](results/pud-reuse.json) · [Audit](results/pud-reuse-audit.json)

Run `OPENBLAS_NUM_THREADS=1 python -m tools.audit_pud_reuse` to check the saved experiment. To repeat the timed pilot with its original inputs, choose an empty output directory and run `OPENBLAS_NUM_THREADS=1 python -m experiments.pud_reuse --output-dir /tmp/pud-reuse-repeat`. The [timed runs](results/pud-reuse-records.jsonl) contain the accepted decisions and measured losses. The [decision comparisons](results/pud-reuse-decisions.jsonl) record which candidate each rule chose on the same state along the separate reuse trajectories.

## Gradient reuse at a fixed update count

Stage 61 follows the same eight PUD starting states for exactly 64 training updates per policy. It pairs the random, fresh-gradient and four-step reuse policies at the same fold and seed, and evaluates all three at 0, 16, 32 and 64 updates. The Arabic loss difference at update 64 (reuse minus fresh) was specified before scoring. [Protocol](research/pud-reuse-matched-protocol.json) · [Arabic Word report](reports/61_pud_reuse_matched_ar.docx) · [Paired loss figure](reports/figures/61_reuse_minus_fresh.png)

At 64 updates, four-step reuse had higher loss than fresh alignment in all eight paired states. The mean difference was +0.00456 nats per byte transition in Arabic and +0.00305 in English. Both alignment policies had lower mean loss than random at this checkpoint. Under the stage-60 time budget, random selection had the lowest loss in all eight states. These descriptive comparisons cover two seeds and two folds per language.

From the repository root run `OPENBLAS_NUM_THREADS=1 python -m tools.audit_pud_reuse_matched` to audit the stored trajectories and `python -m tools.plot_pud_reuse_matched` to regenerate the paired figure (Matplotlib and Pillow with Arabic text shaping). To collect a separate replay, use `OPENBLAS_NUM_THREADS=1 python -m experiments.pud_reuse_matched --output /tmp/pud-reuse-matched-repeat.json` with the pinned inputs. [Individual trajectories and summary](results/pud-reuse-matched.json)

## One-step consequences of a changed candidate

Stage 62 compares the freshly computed and cached probe gradients at each of the first 16 decisions along eight previously studied PUD reuse trajectories. At each decision, both policies see the same weights, AdamW state, probe and four candidates. The experiment evaluates the two resulting models on the held-out fold and continues from the reuse branch. The Arabic endpoint is the mean paired target loss difference over all 48 decisions between refreshes, including agreements. [Published design](research/pud-reuse-one-step-protocol.json) · [Arabic Word report](reports/62_pud_reuse_one_step_ar.docx)

Reusing the gradient increased the immediate target loss on average by 0.00000692 nats per byte transition in Arabic and 0.00001896 in English across 48 stale steps per language. In Arabic the cached choice raised loss in four steps, lowered it in two and tied in 42. In English it raised loss in five steps and tied in 43. Each fresh branch is evaluated from the current cached-policy state, so the signed differences describe one update at a time.

Inspect or rerun the checks with `OPENBLAS_NUM_THREADS=1 python -m tools.audit_pud_reuse_one_step`. To repeat the experiment using the pinned PUD inputs and an unused output path, run `OPENBLAS_NUM_THREADS=1 python -m experiments.pud_reuse_one_step --output /tmp/pud-reuse-one-step-repeat.json`. [Per-decision branch losses](results/pud-reuse-one-step.json) · [Audit record](results/pud-reuse-one-step-audit.json)

## One-step gradient reuse on Arabic PADT and English EWT

Stage 63 repeats the paired branch comparison on two corpora previously studied by this project. Each language uses its official training split for candidate updates, a fixed sample of 64 development chunks for the probe and 256 test chunks for evaluation. Eight new seeds supply the trajectories. The primary contrast averages the 12 stale-gradient steps within each Arabic seed and weights seeds equally. [Published design](research/ud-reuse-transfer-protocol.json) · [Arabic Word report](reports/63_ud_reuse_transfer_ar.docx)

Across the 96 stale decisions per language, cached minus fresh mean next-step loss was +0.00003069 nats per byte transition on PADT and +0.00000405 on EWT. On PADT, the cached choice increased loss 11 times, reduced it once and tied 84 times. The EWT counts were 6, 2 and 88. All eight Arabic seed means favored the fresh gradient. Three English seed means favored the fresh gradient, two favored reuse and three tied. Stage 62 on PUD also recorded positive language means. The transfer changes corpora, probe and target samples, and training seeds together.

Rebuild the processed splits with `python -m tools.prepare_multilingual`. To collect another measurement, use a new path, for example `OPENBLAS_NUM_THREADS=1 python -m experiments.ud_reuse_transfer --output /tmp/ud-reuse-transfer-repeat.json`. Verify the published per-decision results with `OPENBLAS_NUM_THREADS=1 python -m tools.audit_ud_reuse_transfer`. [Source manifest](data/multilingual_sources.json) · [Recorded branches and summaries](results/ud-reuse-transfer.json) · [Audit](results/ud-reuse-transfer-audit.json)

## Timed reuse, fresh and random policies on PADT and EWT

Stage 64 runs the three stage-60 policies under the published three-second deadline loop on the stage-63 PADT/EWT states: eight seeds per language, three rotated timing repeats, 144 runs. [Published design](research/ud-reuse-timed-protocol.json) · [Arabic Word report](reports/64_ud_reuse_timed_ar.docx)

Primary (Arabic, reuse4 minus fresh): −0.35455 nats per byte transition, negative in all eight seeds. Random selection still achieved much lower loss than both guided policies in both languages (Arabic reuse4 minus random +0.48001, English +0.36483; all seeds positive), completing about 411–413 updates versus about 67–69 for reuse4 and 38–40 for fresh. Results are descriptive; no interval or significance test was specified.

An exploratory post-hoc decomposition (not in the protocol) replays the random schedule to each guided run's update count. At matched updates alignment was slightly better than random (Arabic reuse4 −0.01227), but the throughput term was about 40 times larger (+0.49228). In this setting the decision cost, not choice quality, determines the timed outcome. [Decomposition](results/ud-reuse-timed-decomposition.json)

```bash
OPENBLAS_NUM_THREADS=1 python -m experiments.ud_reuse_timed --output-dir /tmp/ud-reuse-timed-repeat
OPENBLAS_NUM_THREADS=1 python -m tools.audit_ud_reuse_timed
```

The audit recomputes the summary, replays all 144 final models from their recorded choices (exact loss match), replays 1,799 guided decisions without a clock, and checks deadline adherence. Stage 63 was first reproduced byte-identically on the stage-64 host (Intel Xeon 2.10 GHz, 4 logical CPUs). Timing outcomes are host-dependent. [Records](results/ud-reuse-timed-records.jsonl) · [Summary](results/ud-reuse-timed.json) · [Audit](results/ud-reuse-timed-audit.json)

## Decision cost against selection gain (stage 65)

Stage 65 tests the stage-64 break-even inequality s/a > r − 1 by varying decision cost: highest-loss selection from 4, 16 or 64 candidates via one batched no-grad forward pass, gradient-reuse alignment and random, under the same three-second deadline loop. Ten new seeds per language, five rotated repeats, 500 runs; the target is 512 previously unscored dev chunks per language. [Protocol](research/ud-cost-frontier-protocol.json) · [Arabic report](reports/65_ud_cost_frontier_ar.md)

Primary (Arabic, loss16 minus random): +0.06381, paired seed-bootstrap 95% CI [+0.05633, +0.07212]; random was better in all ten seeds. No guided arm beat random in any seed or language. The pre-specified decomposition shows highest-loss selection *harms* target loss per update (Arabic loss64 selection term +0.0625; English +0.4402), increasingly with K, while alignment is the only per-update gain (Arabic −0.0064) and the most expensive (r ≈ 6). The inequality agreed with all 80 outcomes, but no case lay near the boundary, so this does not test its discriminating power.

The audit recomputed the summary and exactly replayed all 500 final losses, 17,550 guided decisions and 368 random-curve points. [Records](results/ud-cost-frontier-records.jsonl) · [Summary](results/ud-cost-frontier.json) · [Audit](results/ud-cost-frontier-audit.json)

## Pretrained LLM: Qwen2.5-0.5B with LoRA (stage 66)

First stage on a pretrained model: Qwen2.5-0.5B (revision 060db649, weights SHA-256 pinned, declared Apache-2.0), LoRA rank 8 on q/v projections, 64 matched updates of one 65-token chunk, four seeds per language. Gradient-dot alignment beat random at every checkpoint in all eight seed-language pairs (Arabic −0.01616 nats/token after 64 updates; English −0.02323). Its per-update gain relative to random, s/a = 0.346 (Arabic), is about 80 times the stage-65 value on the small model, but its step cost r = 4.17 leaves it below the first-order break-even s/a > r − 1 = 3.17. [Protocol](research/llm-selection-protocol.json) · [Arabic report](reports/66_llm_selection_ar.md) · [Results](results/llm-selection.json). Weights are downloaded from Hugging Face and kept out of Git (`models/`).

## Forecasting equal-time outcomes under contamination (stage 67)

Candidate pools mixed Arabic with a fraction ρ of off-target English. Phase A (equal updates) produced predictions that were committed before the timed phase B. The pre-registered rule failed its own criterion: 6/10 predictions correct, with 4 contradictions whose intervals exclude zero; a sign-corrected addendum also scored 6/10. Selection value proved nonstationary: highest-loss selection favoured Arabic early and switched to English later. At ρ = 0.95, highest-loss selection beat random under equal time (−0.218, 95% CI [−0.355, −0.086]), the first equal-time win for a guided policy in this project. [Protocol](research/mixture-crossover-protocol.json) · [Predictions](results/mixture-crossover-predictions.json) · [Arabic report](reports/67_mixture_crossover_ar.md) · [Audit](results/mixture-crossover-audit.json)

## Cost-charged LLM selection (stages 68–69)

**Stage 68 (CPU, Qwen2.5-0.5B).** Each policy was charged its measured step cost and evaluated at the update count it attains within 96 random-step costs. Batched per-example LoRA gradients lowered alignment cost from 4.10 to 2.84 random steps with identical choices. With 90% off-target English candidates, batched alignment beat random in all four seeds (−0.0371 nats/token); without contamination random won. [Protocol](research/llm-budget-protocol.json) · [Report](reports/68_llm_budget_ar.md) · [Results](results/llm-budget.json)

**Stage 69 (Colab GPU, Qwen2.5-0.5B and 1.5B, Arabic Wikipedia target).** Run on an NVIDIA L4 rather than the protocol's A100 ([deviations](research/llm-gpu-scale-deviations.md)). Batched alignment cost 1.3–1.9 random steps on the GPU. Primary (1.5B, ρ 0.9): −0.172 nats/token versus random, 95% CI [−0.219, −0.126], 6/6 seeds; alignment beat random in all four model × ρ cells. Interpretation is constrained: random training on the news/English pool *raised* Wikipedia loss above its starting value, and only the alignment policies ended below it (post-hoc). Alignment acted mainly as a filter against harmful updates. [Protocol](research/llm-gpu-scale-protocol.json) · [Notebook](colab/stage69_gpu.ipynb) · [Report](reports/69_llm_gpu_scale_ar.md) · [Audit](results/llm-gpu-scale-audit.json)

**Stage 70 (Colab L4, in-domain candidates, pre-specified no-update baseline).** Arabic candidates were drawn from Arabic Wikipedia itself, so random training now helped the target (about −1.0 nats/token versus no update). Primary (1.5B, ρ 0): batched alignment minus random +0.0274, 95% CI [+0.0223, +0.0325], random better in 6/6 seeds; the alignment selection term was near zero (−0.0008). With 90% English candidates alignment won in every seed (1.5B −0.0896; 0.5B −0.1889). Highest-loss selection chose English 99–100% of the time and barely improved on no update. Across stages 68–70, alignment acts as a data filter: it wins when much of the pool is unsuitable for the target and loses when the pool is already in-domain. [Protocol](research/llm-gpu-indomain-protocol.json) · [Notebook](colab/stage70_gpu.ipynb) · [Report](reports/70_llm_gpu_indomain_ar.md) · [Audit](results/llm-gpu-indomain-audit.json)

**Stage 71 (Colab L4, Qwen2.5-1.5B, contamination threshold).** Five off-target fractions ρ ∈ {0, 0.25, 0.5, 0.75, 0.9}, six seeds, budget 400 random-step costs (batched alignment r = 1.95). Primary: crossover ρ* = 0.34, 95% CI [0.29, 0.52]; the pre-registered prediction (crossover between 0 and 0.5, signs positive at 0 and negative at 0.5–0.9) held, though the point guess of about 0.2 was low. The selection term grew steadily with ρ (−0.005 to −0.230) while alignment picked English far less often than the pool (15% at ρ 0.5). A secondary run on a Tesla T4, outside the protocol (r = 3.24), moved the crossover to about 0.72 [0.43, 0.80] with the same seeds and schedules, consistent with the break-even logic. Practical rule for this setting: alignment pays once roughly a third or more of the candidates are unsuitable for the target, if it costs about twice a normal step. [Protocol](research/llm-gpu-threshold-protocol.json) · [Deviations](research/llm-gpu-threshold-deviations.md) · [Report](reports/71_llm_gpu_threshold_ar.md) · [Audit](results/llm-gpu-threshold-audit.json)

**Stage 72 (Colab L4, Qwen2.5-1.5B, same-language contamination).** Three contaminant types at ρ 0.5 and 0.9: Arabic PADT news (same language, other domain), garbled Arabic Wikipedia (token-permuted chunks), and English. Primary (garbled, ρ 0.5): −0.0049, 95% CI [−0.021, +0.017], inconclusive. At ρ 0.9 alignment beat random for all three types: garbled −0.242 [−0.269, −0.217] (6/6 seeds), news −0.091 [−0.143, −0.034], English −0.063 [−0.119, −0.012]. Alignment chose only 14–16% contaminant chunks at ρ 0.5 against about 50% for random, so it recognised same-language noise as well as English; at ρ 0.5 the selection gain and the throughput cost roughly cancelled, consistent with the stage-71 crossover. [Protocol](research/llm-gpu-noise-protocol.json) · [Report](reports/72_llm_gpu_noise_ar.md) · [Audit](results/llm-gpu-noise-audit.json)

**Stage 73 (Colab L4, Qwen2.5-1.5B, offline filtering of real Arabic web text toward ArabicMMLU).** Pools of 2048 FineWeb-2 Arabic page openings, either all kept by its quality filters or half removed ("raw"); the filter scored every chunk once by gradient alignment with 120 ArabicMMLU dev questions, kept the top quarter, and was charged its scoring time (962 random steps of a 2400-step budget). Primary (raw pool): filter minus random +0.070 nats/token on answer loss of 512 test questions, 95% CI [+0.006, +0.131], random better; the pre-registered prediction failed. Every run raised the task loss above the untrained model, the filter more than random, and the filter kept slightly more removed documents (54%) than the pool contained. A forward-only score cost the same as the exact one. The stage 68–72 rule (alignment pays under heavy contamination) needs a pool that contains data useful to the target. [Protocol](research/llm-web-filter-protocol.json) · [Deviations](research/llm-web-filter-deviations.md) · [Report](reports/73_llm_web_filter_ar.md) · [Audit](results/llm-web-filter-audit.json) · [Notebook](colab/stage73_gpu.ipynb)

**Stage 74 (Colab L4, Qwen2.5-1.5B, web pool mixed with in-format QA, with a repetition control).** Pools of 2048 chunks: ArabicMMLU test items outside every evaluation set, written in the target template (amendment 1, after Aya QA failed the pilot gate), mixed at a share σ of 0.25 or 0.1 with FineWeb-2 kept and removed page openings; near-duplicates of evaluation items removed by four criteria. Arms: random; the stage-73 filter (top quarter by one-pass gradient alignment, scoring charged at 987 random steps of 2400, leaving N = 1412 updates); and random512, a random quarter trained with the filter's schedule so that it repeats data equally. Six seeds, 36 runs. Primary (σ 0.25): filter at N minus random at 2400 +0.063 nats/token, 95% CI [+0.035, +0.091], random better in all six seeds; the pre-registered prediction failed. The filter found the useful data (93% QA among its kept chunks against 25% in the pool; at σ 0.1 nearly all 205 QA chunks) and beat random512 at equal repetition in all 12 seeds (−0.257 at σ 0.25, −0.348 at σ 0.1), but repeating a 512-chunk set cost more (random512 minus random at N: +0.344, +0.397), so the filter lost even at equal updates. σ 0.1 charged: +0.024 [−0.058, +0.087], inconclusive. The QA items come from the target's own distribution, so any gain would measure finding in-distribution data, not general ability. The machine was lost after 14 of 36 runs and the run resumed from the journal on another L4 (deviation 1); the audit passed with one finding: criterion (d) did not test the trained tokenization, and three trained chunks share an 8-gram (a name, a question stem, a year prefix) with evaluation items, none revealing an answer (deviation 2). [Protocol](research/llm-qa-mix-protocol.json) · [Deviations](research/llm-qa-mix-deviations.md) · [Report](reports/74_llm_qa_mix_ar.md) · [Audit](results/llm-qa-mix-audit.json)

**Stage 75 (Colab L4, Qwen2.5-1.5B, one-pass offline filter on the stage-74 pool).** The stage-74 σ 0.25 pool (512 QA, 1536 web chunks), 512 fresh target items (unscored, untrained, decontaminated against everything trained, including the trained tokenization), six new seeds, 30 runs (amendment 1: σ 0.1 left to a later stage). Scoring charged at 915 random steps of 2400, N = 1484. Arms: random; align_once (keep the top N = 72% of the pool and train one pass without replacement, so every kept chunk is seen once by N); random_once (a random N-subset with the same order); and the stage-74 top-quarter filter and random512 as replication. Primary: align_once at N minus random at 2400 −0.033 nats/token, 95% CI [−0.086, +0.019], filter better in 5/6 seeds: the predicted sign, not established. Removing repetition helped the filter itself (align_once − top-quarter filter at N −0.183 [−0.259, −0.107], 6/6), but the larger kept set diluted the QA share to 0.33 (pool 0.25; 492 of the 512 QA chunks kept) and selection at matched repetition was small (−0.021 [−0.063, +0.030]). Stage 74 replicated (top-quarter charged +0.149 [+0.094, +0.204]; matched selection −0.252; repetition +0.380). The audit passed; one journal backup reached GitHub one run late (deviation 1). [Protocol](research/llm-once-filter-protocol.json) · [Deviations](research/llm-once-filter-deviations.md) · [Report](reports/75_llm_once_filter_ar.md) · [Audit](results/llm-once-filter-audit.json)

**Stage 76 (Colab L4, Qwen2.5-1.5B, cheap proxy filter on a pool with enough useful data).** A pool of 8192 chunks at QA share 0.25 (the 2048 stage-74 QA chunks, no new ArabicMMLU item trained, plus 6144 newly drawn FineWeb-2 kept and removed page openings), budget 3000 random steps, 512 fresh decontaminated target items, six new seeds, 24 runs. Scoring used Qwen2.5-0.5B as a proxy (2.5× cheaper than the 1.5B scorer and at least as good at separating QA from web in a smoke test on exposed data), charged at 1366 random steps (46% of the budget), leaving N = 1633. Arms: random; align05_once (keep the top N, one pass without replacement); random_once (random N-subset, same order); qa_once (label oracle, uncharged). Primary: align05_once at N minus random at 3000 −0.198 nats/token, 95% CI [−0.250, −0.138], filter better in 6/6 seeds; all four pre-registered predictions held. Kept sets were 99.4% QA; selection at matched repetition −0.246 [−0.296, −0.202]; the proxy matched the label oracle (+0.008 [−0.038, +0.040]); selection term −0.186, throughput −0.013 (random gained nothing after N). Accuracy +1.7 points [+0.8, +2.6] at equal cost. Several factors changed together relative to stage 75, and the QA items come from the target's own distribution, so the result measures finding in-distribution data at equal cost. The Colab machine was lost twice (after 15 runs and after all 24); both times inputs were re-verified before any outcome was read and nothing completed was rerun. The audit passed, including identical proxy scores and step-50 replays on a third L4. [Protocol](research/llm-cheap-filter-protocol.json) · [Deviations](research/llm-cheap-filter-deviations.md) · [Report](reports/76_llm_cheap_filter_ar.md) · [Audit](results/llm-cheap-filter-audit.json)

**Stage 77 (Colab L4, DeepSeek-LLM-7B-base, self-scoring filter on another model family).** Does stage 76 hold outside the Qwen family? DeepSeek-LLM-7B-base (pretrained from scratch; bfloat16 base, float32 LoRA factors and losses) is both trained and scorer: no smaller DeepSeek model with its tokenizer separated QA from web in a smoke test. Pool of 7968 chunks of 147 DeepSeek tokens at QA share 0.25 (the 4,946 stage-74 training items repacked into 1,992 chunks, no new ArabicMMLU item trained, plus 2,988 kept and 2,988 removed FineWeb-2 page openings decontaminated with Qwen 8-grams), 512 fresh decontaminated target items, budget 7400 random steps, four seeds, 12 runs. Self-scoring was charged at 5,356 random steps (72% of B); N_uncapped = 2044 was capped at the QA supply, N = 1992 (amendment 2, 52 charged steps unused). Arms: random; align_once (keep the top N, one pass, then re-permuted passes); random_once (random N-subset, same order). Primary: align_once at N minus random at 7400 −0.073 nats/token, 95% CI [−0.097, −0.046], filter better in 4/4 seeds (bootstrap over four seeds is coarse). Selection at matched repetition −0.094 [−0.103, −0.083] (4/4); selection term −0.090, throughput +0.017. The kept sets were 84.1% QA (pool 0.25; stage 76: 99.4%), missing the predicted > 0.85, so one of three predictions failed. Accuracy did not follow at equal cost: −0.4 points [−2.9, +2.0] (+0.9 [−1.1, +3.3] at matched repetition). Repetition hurt after the first pass (align_once 1.641 → 1.727 from N to 7400 at 3.7 views), so the gain is measured at N. Many factors differ from stage 76 at once, and the QA items come from the target's own distribution. The Colab machine was lost twice (after 7 runs, resumed with the journaled calibration after 34 checks; after all 12 runs, before the audit); the audit passed on a third L4, including identical seed-37700 self-scores and step-50 replays produced on the first machine. [Protocol](research/llm-deepseek-filter-protocol.json) · [Deviations](research/llm-deepseek-filter-deviations.md) · [Report](reports/77_llm_deepseek_filter_ar.md) · [Audit](results/llm-deepseek-filter-audit.json)

**Stage 78 (Colab L4, useful supply at fixed pool, proxy, budget and N; confirmatory).** Was the stage 75/76 inference right that the useful subset must fill the update budget? Only the number of in-format QA chunks Q in the 8192-chunk pool varies: 1024 (below N), 1633 (= N), 2048 (the stage-76 pool, digest reproduced). Model, proxy, scorer, B = 3000, N = 1633 and schedules are held fixed. The QA chunks are nested subsets of the 2048 stage-74/76 chunks (no new ArabicMMLU item trained), and the web chunks are nested too; the QA share falls with Q (0.125, 0.199, 0.25), so supply and share are one factor. Target: 512 new unexposed All/test items (amendment 1; 4,822 eligible). Six seeds, 54 runs plus one calibration in one session (N_cal 1648, within [1470, 1796]). Primary T = D(1024) − D(2048), with D = align05_once at N minus random at 3000 (scoring charged): **−0.137 [−0.165, −0.108], 6/6 seeds, refuted** (predicted > 0). The filter's equal-cost advantage was larger with less useful supply: D(1024) −0.285 [−0.333, −0.238], D(1633) −0.137 [−0.217, −0.065], D(2048) −0.147 [−0.179, −0.115] (stage 76: −0.198), all 6/6. Kept QA shares were 0.594 / 0.911 / 0.994 (970 of 1024, 1,487 of 1633, 1,624 of 2048 QA chunks). Three of five predictions held; T > 0 and the ordering D(1024) > D(1633) > D(2048) did not. In a descriptive, post hoc split, filter loss barely changed between Q1024 and Q2048 (+0.024 [−0.014, +0.062]), while random at 3000 worsened by +0.161 [+0.149, +0.173] as it drew fewer QA chunks (382 vs 755). The filter at Q1633 was worse than at Q1024 (+0.058); this is unexplained. Accuracy at equal cost was +1.8 / +0.9 / +1.0 points (intervals above 0 at Q1024 and Q2048); T in accuracy was +0.7 [−1.4, +2.2]. The condition "the useful subset must fill the budget" is not supported in this setting. The machine changed after the run; the audit passed on a new L4, with identical seed-37800 proxy scores at Q1024 and Q2048 and identical step-50 replays. [Protocol](research/llm-supply-filter-protocol.json) · [Deviations](research/llm-supply-filter-deviations.md) · [Report](reports/78_llm_supply_filter_ar.md) · [Audit](results/llm-supply-filter-audit.json)

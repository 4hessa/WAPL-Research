# WAPL-Research

Research into **Weight-Aware Predictive Learning Value (WAPLV)**: whether weight–sample interactions improve predictions of training-example utility beyond loss, gradients, and optimizer-aware baselines.

## Reports — التقارير

**[جميع تقارير Word العربية وفهرس مراحل البحث](reports/README.md)**

Each report has a `.docx` download and a readable Markdown source in [`reports/`](reports/). Open a Word file on GitHub and choose **Download raw file**.

## Current evidence

The implemented studies are small CPU diagnostics: logistic regression, character-bigram models, a 9,184-parameter causal Transformer, and a 35,776-parameter byte-level Transformer. They have **not established an independent benefit from the tested weight features or compute savings over simple controls**. Horizon-matched simulations improved utility-value accuracy, and pairwise learning improved selection against a pointwise learner on Bryant; neither establishes general selection superiority or practical efficiency. The independent Blake timing study did not establish a benefit from sparse highest-loss selection over random.

The Arabic synthesis covers stages 1–42 and links 11 distinct contrasts to their results. Later PUD studies found higher Arabic selection regret after adding corrected weight features. In the subsequent policy study, random selection produced lower final loss than update alignment under a three-second CPU budget. Alignment produced slightly lower loss at a fixed 128 updates in both languages. [Stage 42 synthesis](reports/42_scientific_synthesis_ar.md) · [Evidence map](research/evidence-summary.md) · [Frozen rankers on PUD](reports/55_pud_transfer_results_ar.md) · [Timed policy comparison](reports/58_pud_policy_results_ar.md)

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

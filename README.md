# WAPL-Research

Research into **Weight-Aware Predictive Learning Value (WAPLV)**: whether weight–sample interactions improve predictions of training-example utility beyond loss, gradients, and optimizer-aware baselines.

## Reports — التقارير

**[جميع تقارير Word العربية وفهرس مراحل البحث](reports/README.md)**

Each report has a `.docx` download and a readable Markdown source in [`reports/`](reports/). Open a Word file on GitHub and choose **Download raw file**.

## Current evidence

The implemented studies are small CPU diagnostics: logistic regression, character-bigram models, and a tiny causal Transformer. They do **not** establish a benefit from the tested weight features, compute savings, or generalization to Transformers. Shared-softmax reuse accelerated the bigram alignment baseline while preserving its tested decisions and states. See [the latest results](reports/16_optimized_time_results_ar.md) and [the scientific review](research/review-ar.md).

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

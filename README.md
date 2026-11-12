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

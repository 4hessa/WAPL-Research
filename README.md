# WAPL-Research

Research into **Weight-Aware Predictive Learning Value (WAPLV)**: whether weight–sample interactions improve predictions of training-example utility beyond loss, gradients, and optimizer-aware baselines.

## Reports — التقارير

**[جميع تقارير Word العربية الستة عشر وفهرس مراحل البحث](reports/README.md)**

Each report has a `.docx` download and a readable Markdown source in [`reports/`](reports/). Open a Word file on GitHub and choose **Download raw file**.

## Current evidence

The implemented studies are small CPU diagnostics: logistic regression and character-bigram models. They do **not** establish a benefit from the tested weight features, compute savings, or generalization to Transformers. Shared-softmax reuse accelerated the bigram alignment baseline while preserving its tested decisions and states. See [the latest results](reports/16_optimized_time_results_ar.md) and [the scientific review](research/review-ar.md).

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
python -m pip install -r requirements-pilot.txt -r requirements-reports.txt
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

All evaluation partitions used in the final studies are now exposed, including the final Emma partition. Further method development must use a new untouched evaluation set for confirmatory claims.

## Source material

External papers, repository snapshots, model-card copies, and corpus text downloads are kept out of Git. Their URLs and retrieval hashes remain in `research/**/access.json` and `data/*source.json`. The reports distinguish inspected sources, publisher claims, and experiments actually run. Reading a library or model card does not establish successful installation or model evaluation.

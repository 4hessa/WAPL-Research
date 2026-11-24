# Stage 73 deviations from the protocol

## 1. Design amended before any outcome measurement (24 November 2026)

The protocol committed in 2a21c3c fixed a pool of 4096 web chunks, a kept set of 1024, a budget of 1600 random-step costs, 1024 target items and three arms (random, align_grad, align_fd). Before writing the run, a smoke test measured costs on training candidates and the 120 ArabicMMLU dev items only. No test item was evaluated and no training outcome was computed.

Measured on the NVIDIA L4 of this session (float32, deterministic algorithms, TF32 off):

| quantity | value |
| --- | --- |
| random step (median of 24) | 0.110 s |
| align_grad, 512 raw-pool chunks incl. probe gradient | 34.5 s |
| align_fd, 512 raw-pool chunks incl. probe gradient | 33.4 s |
| probe gradient (120 dev items) | about 7 s |
| score of 256 chunks, align_grad batch 16 | 12.5 s |
| score of 256 chunks, align_fd batch 16 / 64 | 13.3 s / 13.6 s (out-of-memory retries at 64) |
| evaluation | 0.086 s per item |

Consequences for the original design: scoring 4096 chunks costs about 1,900 random steps (4096 x 0.049 s + 7 s, over 0.110 s), more than the 1600-step budget, so both filter arms would have been infeasible by construction, and 36 runs with 1024-item evaluations would take about 10 hours.

Changes (all values in `research/llm-web-filter-protocol.json`, field `amendment_before_outcomes`):

- pool M 4096 -> 2048 per pool (raw pool: 1024 kept + 1024 removed chunks), kept set 1024 -> 512 (fraction 0.25 unchanged);
- budget 1600 -> 2400 random-step costs (expected scoring share about 40%);
- target 1024 -> 512 test items (same seed 7302);
- align_fd dropped as a training arm because it costs the same as align_grad here: with LoRA, the backward pass computes no frozen-weight gradients, so an exact per-example gradient costs about as much as two forward passes. Its scores are still computed once in the calibration to report cost and agreement;
- align_fd batch 64 -> 16.

Unchanged: model, numerics, LoRA, optimizer, data files and hashes, decontamination rule, pool construction rule and seeds, task format, six seeds, primary contrast (raw pool, align_grad charged minus random), prediction for align_grad.

## 2. Out-of-memory interruption and resumption (24 November 2026)

The first run session stopped after 8 of 24 runs (all on the kept pool, seeds 37300-37303, both arms) with a CUDA out-of-memory error inside a target evaluation: 9.2 GiB were reserved but unallocated (fragmentation), and models of finished runs were not freed promptly because the per-example gradient hooks form a reference cycle. No outcome was read before resuming. Fix (commit after this note): delete the model, run the garbage collector and empty the CUDA cache after each run, and set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Batch sizes, numerics and the run order are unchanged. As the protocol provides, the restarted process reused the journaled calibration, re-verified the samples and skipped the 8 completed runs; the interrupted run (kept pool, seed 37304) was restarted from its beginning. The interruption is visible in the journal as a second environment entry.

# Stage 77 deviations from the protocol

Amendment 1 (decontamination in Qwen2.5 tokenization) was made before any stage-77 code was committed or any model was run, after a data-only dry run; it is recorded in the protocol and is not a deviation of the run.

## 1. The calibration stopped the run at the N range check; amendment 2 caps N at 1,992 (25 November 2026)

**What happened.** The first launch (commit 96e2a21, NVIDIA L4) built the population exactly as the dry run (pool SHA-256 f239a0dd..., 41,418 and 18,697 eligible web chunks, 5,337 eligible target rows), then calibrated: c_r = 0.16287 s, S = 872.31 s at scoring batch 8 (no out-of-memory), S / c_r = 5,355.8 random steps, N = floor(7400 - 5355.8) = 2044. The protocol required 1000 <= N <= 1992, so the runner exited ("N = 2044 outside [1000, 1992]: design infeasible") before any training run or evaluation. The journal (/content/stage77-journal/journal.jsonl; snapshot f5a1edd on colab/stage77-journal) holds one environment entry and the calibration only.

**Why.** B = 7400 was chosen from the smoke-2 estimate S = 903.5 s, extrapolated from scoring 2048 chunks; the full scoring of 7968 chunks took 3.5% less. The margin left below 1,992 (about 180 steps) was smaller than this error (234 steps).

**What was decided, before any outcome existed.** The owner chose amendment 2 (commit 4089e83): N = min(N_uncapped, 1992) = 1992, so that one pass over a kept set made entirely of QA chunks is possible (QA margin 0). The alternatives were raising the bound (N = 2044, at least 52 web chunks kept, QA share at most 0.975) and lowering B with a new calibration. Code, tests and audit were updated (commit 25edd50).

**Effect.** The filter is charged S (5,355.8 random steps) plus 1,992 updates = 7,347.8 of B = 7400; 52 charged steps (0.7% of B) are unused, while random trains 7400 updates. The equal-cost contrast is therefore slightly conservative against the filter. The calibration is not repeated: the journaled entry (updates = 2044, the uncapped value) is kept, and the resumed runner recomputes N = 1992 from the journaled costs. The resumed process appends a second environment entry (resumed_runs 0).

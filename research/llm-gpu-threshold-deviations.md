# Stage 71 deviations from research/llm-gpu-threshold-protocol.json

Recorded 23 November 2026 while the run was in progress, before the final summary existed.

1. **GPU type.** The protocol allows an NVIDIA L4 or A100. The run's console shows the calibration and runs on a `Tesla T4`. On the T4 the measured cost ratio of batched alignment was r = 0.4720 / 0.1459 = 3.24 (123 alignment updates per 400 random-step budget), against r = 2.08 (191 updates) on the L4 in stage 70. Because the crossover depends on r, the T4 run is not comparable with stage 70 and is **not the protocol-compliant primary analysis**.
2. **Handling.** The T4 run is completed and reported as a secondary, deviating run, without discarding any data. If an L4 or A100 becomes available, the same protocol is run with a new journal (`stage71-journal-l4.jsonl`) and that run is the primary analysis. This rule is fixed here, before the T4 summary is seen, and does not depend on its outcome.
3. **Partial observations already seen.** Per-run losses for rho 0.0, 0.25 and 0.5 were visible in the console when this note was written.

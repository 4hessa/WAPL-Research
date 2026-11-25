# Stage 75 deviations from the protocol

Amendment 1 (sigma 0.25 only, 30 runs) is recorded in `research/llm-once-filter-protocol.json`, not here.

## 1. One journal backup reached GitHub one run late (25 November 2026)

**What happened.** The protocol asks for a journal snapshot on `colab/stage75-journal` after every finished run. After run 23 the runner committed the snapshot in its worktree (commit `1ed1760`, 20261125T010025Z), but `git push` failed; the run log shows `BACKUP FAILED: CalledProcessError`. The runner logs only the exception type, so git's message and the cause are unknown (a transient network or remote error is the likely reading, not verified). The run continued, as designed. The next backup, after run 24 (commit `3fe1461`, 20261125T010644Z), pushed both commits. Every other snapshot (calibration and runs 1-22, 25-30) was pushed right after its run.

**Effect.** For about six minutes the run-23 snapshot existed only on the Colab machine. No journal entry was lost or changed: the final journal (SHA-256 `5dbb40b8…`) is the one audited, and the branch holds 31 snapshots in order. No outcome was read before the run finished.

**For later stages.** Log git's stderr on a failed backup and retry the push once before continuing.

## Not deviations (recorded for completeness)

- My progress monitor matched the word `Error` inside that `CalledProcessError` line, so two waits ended early while the run was still going; the process was never interrupted.
- The machine, device (NVIDIA L4), software and calibration did not change during the run: one environment entry, one calibration (N = 1484).

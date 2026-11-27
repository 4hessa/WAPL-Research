# Stage 78 deviations from the protocol

Amendment 1 (a new unexposed 512-item target, so the stage is confirmatory; six seeds) was made by the owner before any stage-78 code or run. It is recorded in the protocol and is not a deviation of the run.

## 1. New machine after all 54 runs, before the audit; audit run on a new Colab L4 (27 November 2026)

**What happened.** All 54 runs and the calibration finished in one session on the first L4 (one environment entry, `resumed_runs` 0). The runner wrote the run log and the summary, and the journal, run log and summary were committed to `main` (commit `df5ef38`). The audit was not run on that machine. It runs on a new Colab machine. No run was in progress, nothing is rerun, and no outcome was read on the first machine or on this one before the audit passed and its result was committed.

**Journal integrity.** The journal on `main` (`results/llm-supply-filter-journal.jsonl`, SHA-256 `dddd5512...`) is byte-identical to the last snapshot on `colab/stage78-journal` (commit `5a5bbd4`, 20261127T063059Z, "54 finished runs") and equals the `journal_sha256` in `results/llm-supply-filter.json`. It holds 56 lines: one environment entry, the calibration (NVIDIA L4, c_r 0.11482 s, S 155.23 s, N_cal 1648 within [1470, 1796], N = 1633) and 54 runs. The audit relies on these checks and does not use this machine's memory of the run.

**What was re-established on the new machine, before any repository code ran.**

- Device: NVIDIA L4 (the calibration's device type), driver 580.82.07; probably a different physical GPU from the one that produced the runs.
- Software: torch 2.14.1+cu130, CUDA 13.0 and Python 3.13.16, the same as the journaled environment. The image came with transformers 5.22.1, tokenizers 0.23.3 and a 1.x huggingface_hub (version not recorded before the reinstall), so the protocol's pins (transformers 4.46.3, tokenizers 0.20.3, huggingface_hub 0.26.2, as stages 73-77) were reinstalled. The audit records this machine's versions in `gpu.environment`.
- Downloads at the pinned revisions, stored outside the repository (models in `models/`, data in `/content/inputs/`):
  - Qwen/Qwen2.5-1.5B `8faed76`;
  - Qwen/Qwen2.5-0.5B `060db64`;
  - FineWeb-2 `af9c133` (`arb_Arab/test/000_00000.parquet`, `arb_Arab_removed/train/000_00000.parquet`);
  - ArabicMMLU `7aa530e` (All/dev.csv, All/test.csv).
- Hashes checked before the audit: both weight files and the four data files equal the protocol's SHA-256 values (`a961db72...`, `88c14255...`, `a84851f7...`, `2c3b8cf5...`, `3c72bd01...`, `c4ba4d6c...`), and the proxy tokenizer files equal the trained model's. The audit rechecks all of them.

**Audit.** `tools/audit_llm_supply_filter.py` (written after the run and before any outcome was read, commit `ff7df65`; 183 tests pass) implements the protocol's audit plan. It ran on the new L4 against the journal on `main` and passed (`results/llm-supply-filter-audit.json`, commit `fb6582e`). The GPU checks recompute results produced on the first machine (seed 37800). Proxy scores at Q1024 and Q2048 were identical (max relative difference 0.0, same kept sets; audit scoring 152.8 s and 152.9 s, journaled 155.6 s and 155.7 s). The step-50 adapter digests of random and align05_once at Q1024 were identical. Bitwise reproducibility held across the two L4 machines for these checks; the other runs have no replay in the audit plan.

**Push access.** This machine had no GitHub credentials at first, so the audit-tool commit and the audit-result commit were made locally, in the order of the method rules, and are pushed to `main` unchanged when access is available.

**Effect.** None on the data or the results. The audit is a check, not part of the run. Design, journal, summary and code are unchanged.

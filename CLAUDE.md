# WAPL-Research: working rules for Claude Code sessions

Read this before doing anything. The project owner writes in Arabic; answer in Arabic, plainly and without flattery, separating verified facts, inference and speculation. In Arabic text, put every English word or technical term on its own line.

## Where the project stands (stage 75, 25 November 2026)

- The original WAPLV hypothesis (weight-sample features) was not supported (stages 18-55).
- Current line of work: cost-charged evaluation of online data selection for LLM adaptation. Read `research/paper/draft.md` (sections 4-7) and the latest reports in `reports/` (64-75) before planning.
- Main finding (stages 68-72): batched gradient-alignment selection on Qwen2.5 with LoRA beats random selection when much of the candidate pool is unsuitable for the target (English, garbled Arabic, off-domain Arabic news), and loses on in-domain pools. Crossover about 34% contamination on an L4 (r = 1.95); about 72% on a T4 (r = 3.24).
- Stage 73 (offline filtering of FineWeb-2 Arabic toward ArabicMMLU, L4): negative. Scoring the pool once cost 40% of the budget; all training raised the task loss; filter minus random +0.070 [+0.006, +0.131] on the raw pool. The rule above needs a pool with data useful to the target. Filter repetition (4.7x vs 1.2x) is an unseparated confound.
- Stage 74 (web pool mixed with in-format ArabicMMLU QA items outside all evaluation sets, plus a random512 repetition control, L4): negative primary. The filter found the QA data (93% of kept vs 25% of pool) and beat random512 at equal repetition in all 12 seeds (-0.257 at sigma 0.25, -0.348 at 0.1), but repetition of a 512-chunk set cost more (+0.344, +0.397), so filter minus random at equal cost was +0.063 [+0.035, +0.091] at sigma 0.25; +0.024 [-0.058, +0.087] at 0.1. For offline filtering, the repetition implied by the kept fraction is part of the cost. Deviations: lost machine after 14 runs (resumed, audited); criterion (d) missed templated-token overlap (3 trained chunks; decontaminate on the trained tokenization in future stages).
- Stage 75 (same sigma 0.25 pool, 512 fresh target items, 30 runs, L4, N = 1484): keeping the top N chunks (72% of the pool) and training one pass without replacement removed the repetition loss (-0.183 vs the quarter filter, 6/6), but diluted the QA share to 0.33; filter minus random at equal cost -0.033 [-0.086, +0.019] (5/6 seeds), predicted sign, not established. Stage 74 replicated (+0.149, -0.252, +0.380). Offline filtering also needs a useful subset large enough to fill the update budget. Sigma 0.1 deferred (amendment 1). Deviation: one journal backup pushed one run late.
- Open items: replay audits of stages 66 and 68 (`tools/audit_llm_selection.py`, `tools/audit_llm_budget.py`, resumable via `results/*-audit-progress.jsonl`). No stage 76 is specified yet (candidates: sigma 0.1 with the stage-75 design; intermediate kept sizes or score-weighted sampling). 4,946 ArabicMMLU test items are training data and 1,792 test items are exposed as targets or pilot (`data/evaluation_status.json`).

## Method rules (do not skip)

1. **Protocol first.** Write `research/<name>-protocol.json` with question, design, primary endpoint, prediction and audit plan; commit and push it before any code that measures outcomes.
2. **Code and tests next**, committed before running. Run `python -m unittest discover -s tests`.
3. **Run** with an append-only JSON-lines journal; resumable; record device and software versions. Never rerun selectively because of outcomes. Do not read evaluation results before the run finishes.
4. **Audit** with a `tools/audit_*.py` that recomputes the summary and checks data, schedules and samples; commit the audit result.
5. **Report** in Arabic: `reports/NN_<name>_ar.md`, convert with `python tools/arabic_docx.py reports/NN_<name>_ar.md reports/NN_<name>_ar.docx`, add a row to `reports/README.md`.
6. **Update** `README.md` and `research/paper/draft.md`.
7. **One commit per step**, pushed to `main` (the owner asked for this), in the order above. Document any deviation from a protocol in `research/<name>-deviations.md` before writing the report.
8. Do not use or regenerate exposed evaluation data for confirmatory claims; see `data/evaluation_status.json` and the exclusion logic in `experiments/llm_gpu_noise.py`.

## Environment notes

- GPU code: `experiments/llm_gpu.py` (step loop, numerics, schedules), `llm_gpu_indomain.py`, `llm_gpu_threshold.py`, `llm_gpu_noise.py`. Model weights in `models/<name>` (not in Git); Arabic Wikipedia shard pinned by SHA-256 in the protocols.
- Data: `python -m tools.prepare_multilingual` rebuilds the UD files (hashes checked).
- Never print, commit or echo GitHub tokens or other credentials.

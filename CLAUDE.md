# WAPL-Research: working rules for Claude Code sessions

Read this before doing anything. The project owner writes in Arabic; answer in Arabic, plainly and without flattery, separating verified facts, inference and speculation. In Arabic text, put every English word or technical term on its own line.

## Where the project stands (stage 73, 24 November 2026)

- The original WAPLV hypothesis (weight-sample features) was not supported (stages 18-55).
- Current line of work: cost-charged evaluation of online data selection for LLM adaptation. Read `research/paper/draft.md` (sections 4-7) and the latest reports in `reports/` (64-73) before planning.
- Main finding (stages 68-72): batched gradient-alignment selection on Qwen2.5 with LoRA beats random selection when much of the candidate pool is unsuitable for the target (English, garbled Arabic, off-domain Arabic news), and loses on in-domain pools. Crossover about 34% contamination on an L4 (r = 1.95); about 72% on a T4 (r = 3.24).
- Stage 73 (offline filtering of FineWeb-2 Arabic toward ArabicMMLU, L4): negative. Scoring the pool once cost 40% of the budget; all training raised the task loss; filter minus random +0.070 [+0.006, +0.131] on the raw pool. The rule above needs a pool with data useful to the target. Filter repetition (4.7x vs 1.2x) is an unseparated confound.
- Open items: replay audits of stages 66 and 68 (`tools/audit_llm_selection.py`, `tools/audit_llm_budget.py`, resumable via `results/*-audit-progress.jsonl`). No stage 74 is specified yet.

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

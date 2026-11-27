# WAPL-Research: working rules for Claude Code sessions

Read this before doing anything. The project owner writes in Arabic; answer in Arabic, plainly and without flattery, separating verified facts, inference and speculation. In Arabic text, put every English word or technical term on its own line.

## Where the project stands (stage 77, 26 November 2026)

- The original WAPLV hypothesis (weight-sample features) was not supported (stages 18-55).
- Current line of work: cost-charged evaluation of online data selection for LLM adaptation. Read `research/paper/draft.md` (sections 4-7) and the latest reports in `reports/` (64-77) before planning.
- Main finding (stages 68-72): batched gradient-alignment selection on Qwen2.5 with LoRA beats random selection when much of the candidate pool is unsuitable for the target (English, garbled Arabic, off-domain Arabic news), and loses on in-domain pools. Crossover about 34% contamination on an L4 (r = 1.95); about 72% on a T4 (r = 3.24).
- Stage 73 (offline filtering of FineWeb-2 Arabic toward ArabicMMLU, L4): negative. Scoring the pool once cost 40% of the budget; all training raised the task loss; filter minus random +0.070 [+0.006, +0.131] on the raw pool. The rule above needs a pool with data useful to the target. Filter repetition (4.7x vs 1.2x) is an unseparated confound.
- Stage 74 (web pool mixed with in-format ArabicMMLU QA items outside all evaluation sets, plus a random512 repetition control, L4): negative primary. The filter found the QA data (93% of kept vs 25% of pool) and beat random512 at equal repetition in all 12 seeds (-0.257 at sigma 0.25, -0.348 at 0.1), but repetition of a 512-chunk set cost more (+0.344, +0.397), so filter minus random at equal cost was +0.063 [+0.035, +0.091] at sigma 0.25; +0.024 [-0.058, +0.087] at 0.1. For offline filtering, the repetition implied by the kept fraction is part of the cost. Deviations: lost machine after 14 runs (resumed, audited); criterion (d) missed templated-token overlap (3 trained chunks; decontaminate on the trained tokenization in future stages).
- Stage 75 (same sigma 0.25 pool, 512 fresh target items, 30 runs, L4, N = 1484): keeping the top N chunks (72% of the pool) and training one pass without replacement removed the repetition loss (-0.183 vs the quarter filter, 6/6), but diluted the QA share to 0.33; filter minus random at equal cost -0.033 [-0.086, +0.019] (5/6 seeds), predicted sign, not established. Stage 74 replicated (+0.149, -0.252, +0.380). Offline filtering also needs a useful subset large enough to fill the update budget. Sigma 0.1 deferred (amendment 1). Deviation: one journal backup pushed one run late.
- Stage 76 (8192-chunk pool at QA share 0.25 with the 2048 stage-74 QA chunks, B = 3000, Qwen2.5-0.5B proxy scorer charged at 46% of B, N = 1633, 24 runs, L4): positive primary. Kept sets 99.4% QA; one-pass filter minus random at equal cost -0.198 [-0.250, -0.138] (6/6); matched selection -0.246; proxy vs label oracle +0.008 [-0.038, +0.040]; throughput term -0.013; accuracy +1.7 points [+0.8, +2.6]. All four predictions held. First offline-filtering stage where all three conditions hold (unsuitable bulk, useful data present, useful supply fills the budget). Several factors changed together vs stage 75, so the cause is not isolated. Deviations: machine lost twice (after 15 runs; after all 24 before commit), inputs re-verified before outcomes, nothing rerun; summary rebuilt by `tools/summarize_llm_cheap_filter.py`.
- Stage 77 (another family: DeepSeek-LLM-7B-base, bf16 base, trained and self-scoring; 7968-chunk pool of 147-token chunks at QA share 0.25, the 4,946 stage-74 items repacked into 1,992 QA chunks; B = 7400, scoring 72% of B, N = 1992 capped at the QA supply by amendment 2; 4 seeds, 12 runs, L4): positive primary in loss. One-pass filter minus random at equal cost -0.073 [-0.097, -0.046] (4/4); matched selection -0.094; throughput +0.017; kept sets 84.1% QA, missing the predicted > 0.85 (2 of 3 predictions held); accuracy at equal cost -0.4 points [-2.9, +2.0], not improved. Repetition hurt after the first pass. Many factors changed together vs stage 76. Deviations: calibration N above the bound (amendment 2); machine lost after 7 runs (resumed, 34 checks) and after all 12 before the audit; audit passed on a third L4.
- Stage-78 pre-protocol smokes (exposed data only, commits `26f973c`, `14726f9`; rule `research/llm-wiki-probe-smoke-rule.json`): no natural, non-benchmark useful source works with this scorer.
  - EXAMS Arabic: overlaps ArabicMMLU (534/562 identical).
  - FineWeb2-HQ: raised target loss (+0.055).
  - Translated FineWeb-Edu: weak, inconsistent (-0.043).
  - Long-article Wikipedia: lowered loss consistently but little (-0.097 vs -0.833 for QA chunks; web +0.146), no accuracy gain. 0.5B proxy AUC vs web 0.654 (cloze probe) / 0.599 (plain-text probe), below the rule's 0.80, so the decision was stop.
  - This is a recorded limitation: stages 74-77 show the filter finds in-format target-distribution data, not useful natural text.
- Open items: replay audits of stages 66 and 68 (`tools/audit_llm_selection.py`, `tools/audit_llm_budget.py`, resumable via `results/*-audit-progress.jsonl`). No stage 78 is specified yet (candidates: vary one factor at a time between stages 75 and 76, e.g. useful supply at fixed pool and scorer; a kept size below the useful supply for DeepSeek; sigma 0.1; a useful source not drawn from the target benchmark). 4,946 ArabicMMLU test items are training data and 2,816 test items are exposed as targets or pilot (`data/evaluation_status.json`).

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

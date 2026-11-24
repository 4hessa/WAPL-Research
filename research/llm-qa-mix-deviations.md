# Stage 74 deviations from the protocol

Phase A's first outcome (Aya, gate failed) and amendment 1 are recorded in `research/llm-qa-mix-protocol.json`, not here.

## 1. Lost machine during phase B; resumed on a new Colab machine (24 November 2026)

**What happened.** The Colab machine that ran phase B was lost mid-run. Its last journal snapshot on branch `colab/stage74-journal` is commit `e42a7db` (20261124T062455Z). Anything the machine wrote after that snapshot is lost.

**How many runs had finished.** The snapshot commit messages say "16 finished runs", but the snapshot script counted every journal line. The journal holds 16 lines: one environment entry, one calibration entry and **14 finished runs**. That count matches `results/stage74-interim/runB.log` (last line "14/36 runs complete"). The finished runs are sigma 0.25, seeds 37400-37403 (all three arms) and seed 37404 (align_grad, random512). The run in progress at the last snapshot (sigma 0.25, seed 37404, random) is restarted from step 0, as in stage 73 deviation 2. If a further run finished on the old machine but was never snapshotted, its result is lost too, and it is rerun with the same seed and schedule. Whether a run is rerun depends only on what the journal contains, never on outcomes.

**What was re-established on the new machine, before any outcome was read.**

- Device: NVIDIA L4 (the same type as the calibration; the runner refuses any other type), driver 580.82.07. This is a different physical GPU from the one that produced the calibration.
- Software: torch 2.14.1+cu130, CUDA 13.0, Python 3.13.16, the same as the journaled environment. The new image came with transformers 5.22.1, tokenizers 0.23.3 and huggingface_hub 1.33.0, so the protocol's pins (4.46.3, 0.20.3, 0.26.2) were reinstalled before any code ran.
- Downloads at the pinned revisions: Qwen/Qwen2.5-1.5B `8faed76`, FineWeb-2 `af9c133` (kept and removed files), ArabicMMLU `7aa530e` (All/dev.csv, All/test.csv). All five SHA-256 values match the protocols.
- `tools/verify_llm_qa_mix_resume.py` (log: `results/llm-qa-mix-resume-verification.log`) recomputed the full population and found it equal to the journaled calibration. The population covers item samples, the four decontamination criteria and their counts, QA item IDs, web document indices and pool digests. The tool also confirmed that the pilot2 journal is complete on the same population; its SHA-256 `c6430ac3…` equals the one in `results/llm-qa-mix-pilot2.json`. Finally, it found the 14 finished runs unique and in the protocol's run order, with their schedule digests, random512 subset digests and align_grad kept sets (recomputed from the journaled scores) all matching. The tool prints pass/fail only.

**What was not changed.** The calibration (c_r, S and N = 1412) is reused from the journal, as the protocol's resumption rule requires; it is not recalibrated on the new GPU. The design, run order, seeds, schedules, evaluation and code are unchanged. A second environment entry in the journal marks the restart.

**Possible effects.** Per-step and scoring times on the new GPU may differ from the calibration machine. They are journaled per run and reported, but N stays as calibrated. Bitwise equality of training across two physical L4 GPUs is not guaranteed. The planned audit replays seed 37400's first 50 random steps to its step-50 adapter digest, which was produced on the old machine, so on the new machine that replay also tests cross-machine reproducibility. Its outcome will be reported either way.

**Snapshot procedure from here on.** The working journal is kept outside the repository (`/content/stage74-journal/journal.jsonl`). After every finished run, a copy is pushed to `colab/stage74-journal` as `results/stage74-interim/journal.jsonl`, and commit messages count only entries of kind `run`.

## 2. Post-run audit finding: decontamination criterion (d) did not test the trained (templated) tokenization (24 November 2026)

**Nature.** This is not a departure from the written protocol: the run applied amendment 1 exactly. It is a weakness of criterion (d) found by the audit after the run and before any outcome was read. It is recorded here so that the report can state it.

**Cause.** Criterion (d) compares token 8-grams of each field tokenized alone. Qwen's byte-level BPE joins a leading space to the next word, so a training answer is tokenized differently in the template (`'الجواب: جعفر…'` gives `' ج'`) than alone (`'جعفر…'` gives `'ج'`). The same applies to an evaluation phrase that starts a field, compared with the same phrase inside a sentence. An 8-gram that starts at such a word was therefore never tested.

**Check added** (`tools/audit_llm_qa_mix.py`, `templated_overlap`, commit 3b2f7f6). The check tokenizes each pooled training item in its trained template and compares it with the content fields of every evaluation item (dev probe, pilot, target), each field tokenized both alone and with a leading space. Result in `results/llm-qa-mix-audit.json` (`data.templated_item_overlap`, status `passed_with_findings`):

- Over the full 2048-chunk QA list: 30 training items, 38 matches.
- Phase-B pools (the first 512 QA chunks at sigma 0.25 and the first 205 at sigma 0.1): 4 of these items, of which 3 have the matching 8-gram inside the 65 tokens actually trained. For item 7085 the match falls in the part of the item cut off by chunking. The remaining 26 items occur only in the phase-A pilot pool (sigma 1.0).

| QA chunk | Pools | Training item | Evaluation item | Shared tokens | Does the shared text carry the evaluation answer? |
| --- | --- | --- | --- | --- | --- |
| 84 | 0.25, 0.1 | 333 (Islamic Studies), answer "جعفر بن أبي طالب" | target 3958, question "أين ولد عبدالله بن جعفر بن أبي طالب؟" | the name "جعفر بن أبي طالب" | no (answer "الحبشة") |
| 493 | 0.25 | 6252 (Arabic Language), question "الكلمة التي تحتوي على حرف (ف) بآخر الكلمة" | target 6276, question "الكلمة التي تحتوي على حرف ميم في أول الكلمة" | the question stem "الكلمة التي تحتوي على حرف" | no (answer "مريم"; the training answer is "سيف") |
| 498 | 0.25 | 5463, answer "١٩٤٨ م" | dev (probe) 14530, "…عام ١٩٤٦…" | the year prefix "١٩٤" | no (answer "لا") |

**Assessment (inference).** The matches are a shared name, a shared question stem of the same exercise type, and a numeral prefix. None reveals an evaluation answer. Two of 512 target items are touched at sigma 0.25 and one at sigma 0.1. The journal stores only aggregate target loss and accuracy, so a sensitivity analysis without these items is not possible without new evaluation, and none was run. The finding does not change the scope statement of amendment 1: the training items are already from the target's own distribution.

**For future stages.** Decontaminate on the tokenization actually trained (templated items and leading-space variants), or on characters, in addition to fields tokenized alone.

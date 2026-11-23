"""Audit stage 66 by rerunning every arm without a clock and comparing exactly.

Step times are not replayable. Choices and every checkpoint loss are recomputed
from the pinned model, data and schedule, and the summary is rebuilt.
"""
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments import llm_common as llm
from experiments import llm_selection as stage
from tools.audit_pud_reuse import digest, read_json, read_rows, require

RESULT = Path('results/llm-selection.json')
RECORDS = Path('results/llm-selection-records.jsonl')
PROGRESS = Path('results/llm-selection-audit-progress.jsonl')  # resumable: one verified run per line
OUTPUT = Path('results/llm-selection-audit.json')


def verified():
    if not PROGRESS.exists():
        return set()
    return {tuple(json.loads(l)['key']) for l in PROGRESS.read_text().splitlines() if l.strip()}


def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    result = read_json(RESULT)
    plan = result['protocol']
    require(plan == read_json(stage.PROTOCOL), 'protocol changed after the run')
    require(digest(RECORDS) == result['output_sha256'][RECORDS.name], 'record journal digest')
    require(llm.file_sha256(llm.MODEL_DIR / 'model.safetensors') == plan['model']['safetensors_sha256'], 'weights')
    for path, expected in result['provenance']['sha256'].items():
        if not path.endswith('model.safetensors'):
            require(digest(Path(path)) == expected, f'source digest {path}')
    rows = read_rows(RECORDS)
    require(len(rows) == result['runs'] == 24, 'run count')
    require(json.dumps(stage.summarize(rows, plan), sort_keys=True) ==
            json.dumps(result['summary'], sort_keys=True), 'summary recomputation')
    tokenizer = AutoTokenizer.from_pretrained(llm.MODEL_DIR)
    manifest = read_json(stage.MANIFEST)
    checks = dict(runs_replayed=0, choices_compared=0, losses_compared=0)
    done = verified()
    for lang in plan['languages']:
        train, probe, target, population = stage.prepare_language(manifest['languages'][lang]['splits'],
                                                                  tokenizer, plan, {})
        require(population == result['populations'][lang], 'population')
        require(not set(population['probe_ids']) & set(population['target_ids']), 'probe/target overlap')
        for row in [r for r in rows if r['language'] == lang]:
            key = (lang, row['seed'], row['arm'])
            if list(key) in [list(k) for k in done]:
                checks['runs_replayed'] += 1
                checks['choices_compared'] += len(row['choices'])
                checks['losses_compared'] += len(row['losses'])
                continue
            ids, choices = stage.schedule(row['seed'], len(train), plan)
            require(ids.tolist() == row['candidate_ids'], 'schedule')
            ticks = iter(range(10 ** 6))
            replay = stage.run_arm(row['seed'], train, probe, target, ids, choices, row['arm'], plan,
                                   clock=lambda: float(next(ticks)))
            require(replay['choices'] == row['choices'], f'{lang} {row["seed"]} {row["arm"]}: choices')
            require(replay['losses'] == row['losses'], f'{lang} {row["seed"]} {row["arm"]}: losses')
            if row['arm'] == 'random':
                require(row['choices'] == choices.tolist(), 'random choices')
            with PROGRESS.open('a') as fh:
                fh.write(json.dumps(dict(key=list(key))) + '\n')
            checks['runs_replayed'] += 1
            checks['choices_compared'] += len(row['choices'])
            checks['losses_compared'] += len(row['losses'])
            print(lang, row['seed'], row['arm'], 'replayed', flush=True)
    audit = dict(status='passed', stage=66, result_sha256=digest(RESULT), records_sha256=digest(RECORDS),
                 checks=checks, notes='Step times are not replayable; all choices and checkpoint losses '
                                      'were recomputed exactly on the audit host.')
    with OUTPUT.open('x') as out:
        json.dump(audit, out, indent=2)
        out.write('\n')


if __name__ == '__main__':
    main()

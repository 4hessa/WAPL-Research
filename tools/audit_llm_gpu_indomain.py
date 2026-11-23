"""Audit stage 70 on CPU: journal, summary, budgets, samples, schedules and random choices.

GPU trajectories of guided arms are not replayed (different device).
"""
import hashlib
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from experiments import llm_common as llm
from experiments import llm_gpu as s69
from experiments import llm_gpu_indomain as stage
from tools.audit_pud_reuse import require

RESULT = Path('results/llm-gpu-indomain.json')
JOURNAL = Path('results/llm-gpu-indomain-journal.jsonl')
WIKI = Path('data/wiki/20231101.ar/train-00004-of-00007.parquet')
OUTPUT = Path('results/llm-gpu-indomain-audit.json')


def main():
    result = json.loads(RESULT.read_text())
    plan = json.loads(stage.PROTOCOL.read_text())
    require(result['protocol'] == plan, 'protocol changed')
    require(s69.sha256(JOURNAL) == result['journal_sha256'], 'journal digest')
    journal = s69.read_journal(JOURNAL)
    rows = [r for r in journal if r['kind'] == 'run']
    cals = {r['model']: r for r in journal if r['kind'] == 'calibration'}
    require(len([r for r in journal if r['kind'] == 'calibration']) == 2, 'one calibration per model')
    require(len(rows) == 72 and len({(r['model'], r['rho'], r['seed'], r['arm']) for r in rows}) == 72, 'run grid')
    require(all(r['device'] == cals[r['model']]['device'] for r in rows), 'device consistency')
    require(json.dumps(stage.summarize(rows, cals, plan), sort_keys=True) ==
            json.dumps(result['summary'], sort_keys=True), 'summary recomputation')
    require(s69.sha256(WIKI) == stage.WIKI_SHA256, 'wiki shard')
    texts = pd.read_parquet(WIKI, columns=['text'])['text'].tolist()
    manifest = json.loads(s69.MANIFEST.read_text())
    checks = dict(runs=0, schedules=0, random_choice_sequences=0)
    for name in ('qwen2.5-0.5b', 'qwen2.5-1.5b'):
        cal = cals[name]
        expected = {a: s69.BUDGET if a == 'random' else int(s69.BUDGET * cal['costs']['random'] / cal['costs'][a] + 1e-9)
                    for a in stage.ARMS}
        require(cal['updates'] == expected, f'{name}: budgets')
        tok = AutoTokenizer.from_pretrained(llm.MODEL_DIR.parent / name)
        eligible, chunks = stage.eligible_chunks(texts, tok)
        _, _, ar, population = stage.split(eligible, chunks)
        require(population == cal['population'], f'{name}: samples')
        n_en = len(llm.token_chunks(manifest['languages']['en']['splits']['train']['path'], tok, 65))
        for k, rho in enumerate(plan['rho']):
            for seed in plan['seeds']:
                ids, choices, digest = s69.schedule(k, seed, rho, len(ar), n_en)
                checks['schedules'] += 1
                for r in [x for x in rows if (x['model'], x['rho'], x['seed']) == (name, rho, seed)]:
                    require(r['schedule_sha256'] == digest, f'{name} {rho} {seed}: schedule')
                    require(len(r['choices']) == r['updates'] == cal['updates'][r['arm']], 'update count')
                    share = sum(float(ids[t][c] >= len(ar)) for t, c in enumerate(r['choices'])) / len(r['choices'])
                    require(abs(share - r['english_share']) < 1e-12, 'english share')
                    if r['arm'] == 'random':
                        require(r['choices'] == choices[:r['updates']].tolist(), 'random choices')
                        checks['random_choice_sequences'] += 1
                    checks['runs'] += 1
        print(name, 'audited', flush=True)
    OUTPUT.write_text(json.dumps(dict(
        status='passed', stage=70, journal_sha256=s69.sha256(JOURNAL),
        result_sha256=hashlib.sha256(RESULT.read_bytes()).hexdigest(), checks=checks,
        devices=sorted({r['device'] for r in rows}),
        notes='CPU audit of journal, summary, budgets, samples, schedules, English shares and random choices. '
              'Guided GPU trajectories were not replayed.'), indent=2) + '\n')


if __name__ == '__main__':
    main()

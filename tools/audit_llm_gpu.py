"""Audit stage 69 on CPU: journal integrity, summary, schedules, budgets and evaluation samples.

GPU trajectories are not replayed here (different device, bitwise equality not expected).
"""
import hashlib
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from experiments import llm_common as llm
from experiments import llm_gpu as stage
from tools.audit_pud_reuse import require

RESULT = Path('results/llm-gpu-scale.json')
JOURNAL = Path('results/llm-gpu-scale-journal.jsonl')
WIKI = Path('data/wiki/20231101.ar/train-00004-of-00007.parquet')
OUTPUT = Path('results/llm-gpu-scale-audit.json')


def main():
    result = json.loads(RESULT.read_text())
    plan = json.loads(stage.PROTOCOL.read_text())
    require(result['protocol'] == plan, 'protocol changed')
    require(stage.sha256(JOURNAL) == result['journal_sha256'], 'journal digest')
    journal = stage.read_journal(JOURNAL)
    rows = [r for r in journal if r['kind'] == 'run']
    cals = {r['model']: r for r in journal if r['kind'] == 'calibration'}
    envs = [r for r in journal if r['kind'] == 'environment']
    require(len(rows) == 96 and len({(r['model'], r['rho'], r['seed'], r['arm']) for r in rows}) == 96, 'run grid')
    require(json.dumps(stage.summarize(rows, cals, plan), sort_keys=True) ==
            json.dumps(result['summary'], sort_keys=True), 'summary recomputation')
    require(stage.sha256(WIKI) == 'f035e9118745d465529d48b0b1893824a4f266af2474694dcd35823e767aad17', 'wiki shard')
    texts = pd.read_parquet(WIKI, columns=['text'])['text'].tolist()
    manifest = json.loads(stage.MANIFEST.read_text())
    checks = dict(runs=0, schedules=0, random_choice_sequences=0)
    for name in plan['models']:
        cal = cals[name]
        require(cal['updates'] == stage.budgets(cal['costs']), f'{name}: budgets')
        tok = AutoTokenizer.from_pretrained(llm.MODEL_DIR.parent / name)
        _, _, population = stage.select_articles(texts, tok, plan)
        require(population == cal['population'], f'{name}: evaluation sample')
        n_ar = len(llm.token_chunks(manifest['languages']['ar']['splits']['train']['path'], tok, 65))
        n_en = len(llm.token_chunks(manifest['languages']['en']['splits']['train']['path'], tok, 65))
        for k, rho in enumerate(plan['rho']):
            for seed in plan['seeds']:
                ids, choices, digest = stage.schedule(k, seed, rho, n_ar, n_en)
                checks['schedules'] += 1
                for r in [x for x in rows if (x['model'], x['rho'], x['seed']) == (name, rho, seed)]:
                    require(r['schedule_sha256'] == digest, f'{name} {rho} {seed}: schedule')
                    require(len(r['choices']) == r['updates'] == cal['updates'][r['arm']], 'update count')
                    share = sum(float(ids[t][c] >= n_ar) for t, c in enumerate(r['choices'])) / len(r['choices'])
                    require(abs(share - r['english_share']) < 1e-12, 'english share')
                    if r['arm'] == 'random':
                        require(r['choices'] == choices[:r['updates']].tolist(), 'random choices')
                        checks['random_choice_sequences'] += 1
                    checks['runs'] += 1
        print(name, 'audited', flush=True)
    audit = dict(status='passed', stage=69, journal_sha256=stage.sha256(JOURNAL),
                 result_sha256=hashlib.sha256(RESULT.read_bytes()).hexdigest(), checks=checks,
                 devices=sorted({e['device'] for e in envs}),
                 notes='CPU audit of journal, summary, budgets, evaluation samples, schedules and random choices. '
                       'Guided GPU trajectories were not replayed.')
    OUTPUT.write_text(json.dumps(audit, indent=2) + '\n')


if __name__ == '__main__':
    main()

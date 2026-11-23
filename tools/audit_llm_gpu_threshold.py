"""Audit stage 71 on CPU and summarise the deviating T4 run as a secondary result.

Checks both journals: summary (primary), budget, samples, schedules, random choices
and English shares. GPU trajectories of the alignment arm are not replayed.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from experiments import llm_common as llm
from experiments import llm_gpu as s69
from experiments import llm_gpu_indomain as s70
from experiments import llm_gpu_threshold as stage
from tools.audit_pud_reuse import require

RESULT = Path('results/llm-gpu-threshold.json')
JOURNALS = dict(primary=Path('results/llm-gpu-threshold-journal.jsonl'),
                t4_secondary=Path('results/llm-gpu-threshold-t4-partial-journal.jsonl'))
WIKI = Path('data/wiki/20231101.ar/train-00004-of-00007.parquet')
OUTPUT = Path('results/llm-gpu-threshold-audit.json')
T4_SUMMARY = Path('results/llm-gpu-threshold-t4.json')


def main():
    plan = json.loads(stage.PROTOCOL.read_text())
    result = json.loads(RESULT.read_text())
    require(result['protocol'] == plan, 'protocol changed')
    require(s69.sha256(JOURNALS['primary']) == result['journal_sha256'], 'primary journal digest')
    require(s69.sha256(WIKI) == s70.WIKI_SHA256, 'wiki shard')
    tok = AutoTokenizer.from_pretrained(llm.MODEL_DIR.parent / stage.MODEL)
    texts = pd.read_parquet(WIKI, columns=['text'])['text'].tolist()
    eligible, chunks = s70.eligible_chunks(texts, tok)
    _, _, ar, population = stage.split(eligible, chunks)
    en_path = json.loads(s69.MANIFEST.read_text())['languages']['en']['splits']['train']['path']
    n_en = len(llm.token_chunks(en_path, tok, 65))
    out = dict(status='passed', stage=71, checks={}, devices={})
    for label, path in JOURNALS.items():
        journal = s69.read_journal(path)
        rows = [r for r in journal if r['kind'] == 'run']
        cals = [r for r in journal if r['kind'] == 'calibration']
        require(len(cals) == 1, f'{label}: one calibration')
        cal = cals[0]
        require(len(rows) == 60 and len({(r['rho'], r['seed'], r['arm']) for r in rows}) == 60, f'{label}: grid')
        require(all(r['device'] == cal['device'] for r in rows), f'{label}: device consistency')
        require(cal['population'] == population, f'{label}: samples')
        require(cal['updates'] == dict(random=s69.BUDGET, graddot4_batched=int(
            s69.BUDGET * cal['costs']['random'] / cal['costs']['graddot4_batched'] + 1e-9)), f'{label}: budget')
        summary = stage.summarize(rows, cal, plan)
        if label == 'primary':
            require(json.dumps(summary, sort_keys=True) == json.dumps(result['summary'], sort_keys=True), 'summary')
        else:
            T4_SUMMARY.write_text(json.dumps(dict(
                stage=71, status='secondary run on Tesla T4, outside the protocol (see research/llm-gpu-threshold-deviations.md)',
                journal_sha256=s69.sha256(path), calibration=cal, summary=summary), indent=2) + '\n')
        checks = dict(runs=0, schedules=0, random_choice_sequences=0)
        for k, rho in enumerate(plan['rho']):
            for seed in plan['seeds']:
                ids, choices, digest = s69.schedule(k, seed, rho, len(ar), n_en)
                checks['schedules'] += 1
                for r in [x for x in rows if (x['rho'], x['seed']) == (rho, seed)]:
                    require(r['schedule_sha256'] == digest, f'{label} {rho} {seed}: schedule')
                    require(len(r['choices']) == r['updates'] == cal['updates'][r['arm']], 'update count')
                    share = sum(float(ids[t][c] >= len(ar)) for t, c in enumerate(r['choices'])) / len(r['choices'])
                    require(abs(share - r['english_share']) < 1e-12, 'english share')
                    if r['arm'] == 'random':
                        require(r['choices'] == choices[:r['updates']].tolist(), 'random choices')
                        checks['random_choice_sequences'] += 1
                    checks['runs'] += 1
        out['checks'][label] = checks
        out['devices'][label] = cal['device']
        print(label, 'audited', flush=True)
    out.update(result_sha256=hashlib.sha256(RESULT.read_bytes()).hexdigest(),
               notes='CPU audit of both journals; the T4 journal is summarised separately as a secondary result. '
                     'Alignment-arm GPU trajectories were not replayed.')
    OUTPUT.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()

"""Audit stage 72 on CPU: journal, summary, budget, samples, garbling, schedules,
random choices and contaminant shares. Alignment GPU trajectories are not replayed."""
import hashlib
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from experiments import llm_common as llm
from experiments import llm_gpu as s69
from experiments import llm_gpu_indomain as s70
from experiments import llm_gpu_noise as stage
from tools.audit_pud_reuse import require

RESULT = Path('results/llm-gpu-noise.json')
JOURNAL = Path('results/llm-gpu-noise-journal.jsonl')
WIKI = Path('data/wiki/20231101.ar/train-00004-of-00007.parquet')
OUTPUT = Path('results/llm-gpu-noise-audit.json')


def main():
    plan = json.loads(stage.PROTOCOL.read_text())
    result = json.loads(RESULT.read_text())
    require(result['protocol'] == plan, 'protocol changed')
    require(s69.sha256(JOURNAL) == result['journal_sha256'], 'journal digest')
    journal = s69.read_journal(JOURNAL)
    rows = [r for r in journal if r['kind'] == 'run']
    cals = [r for r in journal if r['kind'] == 'calibration']
    require(len(cals) == 1, 'one calibration')
    cal = cals[0]
    require(len(rows) == 72 and len({(r['type'], r['rho'], r['seed'], r['arm']) for r in rows}) == 72, 'grid')
    require(all(r['device'] == cal['device'] for r in rows), 'device consistency')
    require(cal['updates'] == dict(random=s69.BUDGET, graddot4_batched=int(
        s69.BUDGET * cal['costs']['random'] / cal['costs']['graddot4_batched'] + 1e-9)), 'budget')
    require(json.dumps(stage.summarize(rows, cal, plan), sort_keys=True) ==
            json.dumps(result['summary'], sort_keys=True), 'summary recomputation')
    require(s69.sha256(WIKI) == s70.WIKI_SHA256, 'wiki shard')
    tok = AutoTokenizer.from_pretrained(llm.MODEL_DIR.parent / stage.MODEL)
    texts = pd.read_parquet(WIKI, columns=['text'])['text'].tolist()
    eligible, chunks = s70.eligible_chunks(texts, tok)
    _, _, clean, garbled, population = stage.split(eligible, chunks)
    require(population == cal['population'], 'samples')
    splits = json.loads(s69.MANIFEST.read_text())['languages']
    sizes = dict(news=len(llm.token_chunks(splits['ar']['splits']['train']['path'], tok, 65)), garbled=len(garbled),
                 english=len(llm.token_chunks(splits['en']['splits']['train']['path'], tok, 65)))
    checks = dict(runs=0, schedules=0, random_choice_sequences=0)
    for t, kind in enumerate(stage.TYPES):
        for k, rho in enumerate(plan['rho']):
            for seed in plan['seeds']:
                ids, choices, digest = s69.schedule(3 * t + k, seed, rho, len(clean), sizes[kind])
                checks['schedules'] += 1
                for r in [x for x in rows if (x['type'], x['rho'], x['seed']) == (kind, rho, seed)]:
                    require(r['schedule_sha256'] == digest, f'{kind} {rho} {seed}: schedule')
                    require(len(r['choices']) == r['updates'] == cal['updates'][r['arm']], 'update count')
                    share = sum(float(ids[s][c] >= len(clean)) for s, c in enumerate(r['choices'])) / len(r['choices'])
                    require(abs(share - r['contaminant_share']) < 1e-12, 'contaminant share')
                    if r['arm'] == 'random':
                        require(r['choices'] == choices[:r['updates']].tolist(), 'random choices')
                        checks['random_choice_sequences'] += 1
                    checks['runs'] += 1
    OUTPUT.write_text(json.dumps(dict(
        status='passed', stage=72, journal_sha256=s69.sha256(JOURNAL),
        result_sha256=hashlib.sha256(RESULT.read_bytes()).hexdigest(), checks=checks, device=cal['device'],
        notes='CPU audit of journal, summary, budget, samples (including garbling), schedules, contaminant shares '
              'and random choices. Alignment GPU trajectories were not replayed.'), indent=2) + '\n')


if __name__ == '__main__':
    main()

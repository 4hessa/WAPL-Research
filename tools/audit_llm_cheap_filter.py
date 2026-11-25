"""Audit stage 76 (research/llm-cheap-filter-protocol.json, audit field).

CPU: journal digest, summary recomputation, environments, calibration and N, run grid and order, schedules,
update counts, checkpoints, kept sets from journaled scores, subset digests, QA shares, random-arm QA draws, views
per chunk, data and model hashes (both models), pool rebuilt with its digest, QA chunks equal the stage-74 sigma 1.0
pool QA, target eligibility and sample recomputed (criteria (a)-(e) and counts), target disjoint from every exposed
or trained ArabicMMLU ID.
GPU: recompute the proxy scores of seed 37600; replay the first 50 steps of seed 37600 for random and align05_once
to their step-50 adapter digests.

python -m tools.audit_llm_cheap_filter --journal J --kept K --removed R --mmlu M [--no-gpu]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import llm_cheap_filter as stage
from experiments import llm_gpu as s69
from experiments import llm_web_filter as s73
from tools.audit_pud_reuse import require

RESULT = Path('results/llm-cheap-filter.json')
OUTPUT = Path('results/llm-cheap-filter-audit.json')
STAGE74_JOURNAL = Path('results/llm-qa-mix-journal.jsonl')
AUDIT_SEED, SCORE_TOL = 37600, 1e-4
ENV_KEYS = ('device', 'torch', 'transformers', 'cuda', 'python')


def check_run(row, n):
    tag = f"seed {row['seed']} {row['arm']}"
    sch = stage.schedule(row['seed'], n)
    require(row['schedule_sha256'] == sch['sha256'], f'{tag}: schedule')
    require(len(row['step_seconds']) == stage.BUDGET, f'{tag}: update count')
    require(set(row['losses']) == {'0', str(n), str(stage.BUDGET)} == set(row['accuracy']), f'{tag}: checkpoints')
    require(isinstance(row['adapter_digest_50'], str) and len(row['adapter_digest_50']) == 64, f'{tag}: step-50 digest')
    kept = None
    if row['arm'] == 'align05_once':
        require(len(row['scores']) == stage.POOL_SIZE, f'{tag}: score count')
        kept = s73.keep_top(row['scores'], n)
        require(kept.tolist() == row['kept'], f'{tag}: kept set from scores')
        require(hashlib.sha256(kept.tobytes()).hexdigest() == row['kept_sha256'], f'{tag}: kept digest')
        require(row['scoring_seconds'] > 0, f'{tag}: scoring time')
    else:
        require(not {'kept', 'scores'} & set(row), f'{tag}: control carries filter fields')
    order, members = stage.training_order(row['arm'], sch, kept)
    if row['arm'] in ('random_once', 'qa_once'):
        require(row['subset_sha256'] == hashlib.sha256(np.asarray(members).tobytes()).hexdigest(), f'{tag}: subset')
    if row['arm'] == 'random':
        require(row['random_qa'] == {str(t): stage.random_qa(sch['rand'], t) for t in (n, stage.BUDGET)},
                f'{tag}: random QA draws')
    else:
        require(row['qa_share'] == stage.qa_share(members), f'{tag}: QA share')
        require(row['views'][str(n)]['max'] == 1, f'{tag}: one pass by N')
    if row['arm'] == 'qa_once':
        require(row['qa_share'] == 1.0, f'{tag}: oracle subset is all QA')
    require(row['views'] == {str(t): stage.views(order, members, t) for t in (n, stage.BUDGET)}, f'{tag}: views')
    return order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('--journal', '--kept', '--removed', '--mmlu'):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--no-gpu', action='store_true')
    args = parser.parse_args()
    plan = json.loads(stage.PROTOCOL.read_text())
    result = json.loads(RESULT.read_text())
    journal = s69.read_journal(args.journal)
    require(s69.sha256(args.journal) == result['journal_sha256'], 'journal digest')
    envs = [r for r in journal if r['kind'] == 'environment']
    require(len({tuple(e[k] for k in ENV_KEYS) for e in envs}) == 1, 'environments differ')
    cals = [r for r in journal if r['kind'] == 'calibration']
    require(len(cals) == 1, 'one calibration')
    cal = cals[0]
    n = cal['updates']
    require(n == int(np.floor(stage.BUDGET - cal['costs']['proxy_align_grad'] / cal['costs']['random'] + 1e-9)), 'N')
    require(stage.N_RANGE[0] <= n <= stage.N_RANGE[1], 'N in range')
    rows = [r for r in journal if r['kind'] == 'run']
    require([(r['seed'], r['arm']) for r in rows] == stage.run_grid(plan['design']['seeds']), 'run grid and order')
    for r in rows:
        check_run(r, n)
    summary = stage.summarize(rows, cal, plan)
    require(json.loads(json.dumps(summary)) == result['summary'], 'summary recomputation')

    from transformers import AutoTokenizer
    specs = json.loads(s69.PROTOCOL.read_text())['models']
    model_dir, proxy_dir = args.models_dir / stage.MODEL, args.models_dir / stage.PROXY
    for name, d in ((stage.MODEL, model_dir), (stage.PROXY, proxy_dir)):
        require(s69.sha256(d / 'model.safetensors') == specs[name]['safetensors_sha256'], f'{name} hash')
    tok = AutoTokenizer.from_pretrained(model_dir)
    probe, target, pool, population = stage.load_inputs(args, tok)  # data hashes, QA digest, criteria (a)-(e)
    require(population == cal['population'], 'population (pool, target eligibility counts, target IDs)')
    import pandas as pd
    test = pd.read_csv(args.mmlu / 'test.csv').to_dict('records')
    cal74 = next(r for r in s69.read_journal(STAGE74_JOURNAL) if r.get('kind') == 'calibration')['population']
    cal75 = next(r for r in s69.read_journal(stage.STAGE75_JOURNAL) if r.get('kind') == 'calibration')['population']
    ledger = json.loads(stage.s75.LEDGER.read_text())['additional_sources']['ArabicMMLU']['stage_74']
    exposed = ({int(test[i]['ID']) for i in s73.sample_target(len(test))} | set(cal74['pilot_items'])
               | set(cal74['target_items']) | set(cal75['target_items']) | set(ledger['used_for_training']['ids']))
    target_ids = set(population['target_items'])
    require(len(target_ids) == stage.TARGET_ITEMS and not target_ids & exposed, 'target disjoint from exposed IDs')

    gpu = None
    if not args.no_gpu:
        import torch
        s69.setup_numerics()
        device = torch.device('cuda')
        require(torch.cuda.get_device_name(0) == cal['device'], 'audit device type differs from the run')
        data = pool.to(device)
        row = next(r for r in rows if (r['seed'], r['arm']) == (AUDIT_SEED, 'align05_once'))
        proxy = s69.load(proxy_dir, AUDIT_SEED, device)
        scores, seconds = s73.score(proxy, 'align_grad', data, probe, device)
        del proxy
        torch.cuda.empty_cache()
        journaled = np.array(row['scores'])
        rel = float(np.abs(scores - journaled).max() / np.abs(journaled).max())
        require(rel <= SCORE_TOL, 'proxy score recomputation')
        gpu = dict(score_recomputation=dict(max_rel_diff=rel, audit_scoring_seconds=seconds,
                                            identical_kept_set=s73.keep_top(scores, n).tolist() == row['kept']),
                   replay_step50={})
        for arm in ('random', 'align05_once'):
            ref = next(r for r in rows if (r['seed'], r['arm']) == (AUDIT_SEED, arm))
            order = check_run(ref, n)
            rep = s73.train(s69.load(model_dir, AUDIT_SEED, device), data, order, s73.DIGEST_STEP, set(), None, device)
            same = rep['adapter_digest_50'] == ref['adapter_digest_50']
            require(same, f'{arm} step-50 replay')
            gpu['replay_step50'][arm] = dict(identical_adapter_digest=same)
            torch.cuda.empty_cache()
    out = dict(stage=76, status='passed', journal_sha256=result['journal_sha256'], runs=len(rows), updates=n,
               environments=len(envs), summary_recomputed=True,
               target=dict(items=len(target_ids), disjoint_from_exposed=len(exposed), eligibility=population['target_pool']),
               step_seconds_median={arm: float(np.median([np.median(r['step_seconds']) for r in rows if r['arm'] == arm]))
                                    for arm in stage.ARMS},
               gpu=gpu)
    OUTPUT.write_text(json.dumps(out, indent=2) + '\n')
    print('audit passed', flush=True)


if __name__ == '__main__':
    main()

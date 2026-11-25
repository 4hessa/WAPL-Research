"""Audit stage 75 (research/llm-once-filter-protocol.json, audit field).

CPU: journal digest, summary recomputation, environments, calibration and N, run grid and order, schedules,
update counts, checkpoints, kept sets from journaled scores, identical scores of align_q and align_once per seed,
subset digests, QA shares, views per chunk, data hashes, pool digests equal stage 74, target eligibility and
sample recomputed (criteria (a)-(e) and counts), target disjoint from every exposed or trained ArabicMMLU ID.
GPU: recompute align_grad scores of seed 37500; replay the first 50 steps of seed 37500 for the random and
align_once arms to their step-50 adapter digests.

python -m tools.audit_llm_once_filter --journal J --kept K --removed R --mmlu M [--no-gpu]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import llm_gpu as s69
from experiments import llm_once_filter as stage
from experiments import llm_qa_mix as s74
from experiments import llm_web_filter as s73
from tools.audit_pud_reuse import require

RESULT = Path('results/llm-once-filter.json')
OUTPUT = Path('results/llm-once-filter-audit.json')
STAGE74_JOURNAL = Path('results/llm-qa-mix-journal.jsonl')
AUDIT_SEED, SCORE_TOL = 37500, 1e-4
ENV_KEYS = ('device', 'torch', 'transformers', 'cuda', 'python')


def check_run(row, n):
    tag = f"seed {row['seed']} {row['arm']}"
    sch = stage.schedule(stage.SIGMAS.index(row['sigma']), row['seed'], n)
    require(row['schedule_sha256'] == sch['sha256'], f'{tag}: schedule')
    require(len(row['step_seconds']) == stage.BUDGET, f'{tag}: update count')
    require(set(row['losses']) == {'0', str(n), str(stage.BUDGET)} == set(row['accuracy']), f'{tag}: checkpoints')
    require(isinstance(row['adapter_digest_50'], str) and len(row['adapter_digest_50']) == 64, f'{tag}: step-50 digest')
    kept = None
    if row['arm'] in stage.FILTER_ARMS:
        require(len(row['scores']) == stage.POOL_SIZE, f'{tag}: score count')
        kept = s73.keep_top(row['scores'], stage.KEEP_Q if row['arm'] == 'align_q' else n)
        require(kept.tolist() == row['kept'], f'{tag}: kept set from scores')
        require(hashlib.sha256(kept.tobytes()).hexdigest() == row['kept_sha256'], f'{tag}: kept digest')
        require(row['scoring_seconds'] > 0, f'{tag}: scoring time')
    else:
        require(not {'kept', 'scores'} & set(row), f'{tag}: control carries filter fields')
    order, members = stage.training_order(row['arm'], sch, kept)
    if row['arm'] in ('random512', 'random_once'):
        require(row['subset_sha256'] == hashlib.sha256(np.asarray(members).tobytes()).hexdigest(), f'{tag}: subset')
    if row['arm'] != 'random':
        require(row['qa_share'] == s74.qa_share(members, row['sigma']), f'{tag}: QA share')
    require(row['views'] == {str(t): stage.views(order, members, t) for t in (n, stage.BUDGET)}, f'{tag}: views')
    if row['arm'] in ('align_once', 'random_once'):
        require(row['views'][str(n)]['max'] == 1, f'{tag}: one pass by N')
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
    require(n == int(np.floor(stage.BUDGET - cal['costs']['align_grad'] / cal['costs']['random'] + 1e-9)), 'N')
    rows = [r for r in journal if r['kind'] == 'run']
    grid = stage.run_grid(plan['design']['seeds'])
    require([(r['sigma'], r['seed'], r['arm']) for r in rows] == grid, 'run grid and order')
    for r in rows:
        check_run(r, n)
    score_identity = {}
    for seed in plan['design']['seeds']:
        a, b = (next(r for r in rows if (r['seed'], r['arm']) == (seed, arm)) for arm in stage.FILTER_ARMS)
        score_identity[str(seed)] = float(np.abs(np.array(a['scores']) - np.array(b['scores'])).max())
    summary = stage.summarize(rows, cal, plan)
    require(json.loads(json.dumps(summary)) == result['summary'], 'summary recomputation')

    from transformers import AutoTokenizer
    spec = json.loads(s69.PROTOCOL.read_text())['models'][stage.MODEL]
    model_dir = args.models_dir / stage.MODEL
    require(s69.sha256(model_dir / 'model.safetensors') == spec['safetensors_sha256'], 'model hash')
    tok = AutoTokenizer.from_pretrained(model_dir)
    probe, target, pools, population = stage.load_inputs(args, tok)  # data hashes, stage-74 pools, criteria (a)-(e)
    require(population == cal['population'], 'population (pools, target eligibility counts, target IDs)')
    cal74 = next(r for r in s69.read_journal(STAGE74_JOURNAL) if r.get('kind') == 'calibration')['population']
    ledger = json.loads(stage.LEDGER.read_text())['additional_sources']['ArabicMMLU']['stage_74']
    import pandas as pd
    test = pd.read_csv(args.mmlu / 'test.csv').to_dict('records')
    stage73_ids = {int(test[i]['ID']) for i in s73.sample_target(len(test))}
    exposed = stage73_ids | set(cal74['pilot_items']) | set(cal74['target_items']) | set(ledger['used_for_training']['ids'])
    target_ids = set(population['target_items'])
    require(len(target_ids) == stage.TARGET_ITEMS and not target_ids & exposed, 'target disjoint from exposed IDs')

    gpu = None
    if not args.no_gpu:
        import torch
        s69.setup_numerics()
        device = torch.device('cuda')
        require(torch.cuda.get_device_name(0) == cal['device'], 'audit device type differs from the run')
        data = pools[0.25].to(device)
        row = next(r for r in rows if (r['seed'], r['arm']) == (AUDIT_SEED, 'align_once'))
        scores, seconds = s73.score(s69.load(model_dir, AUDIT_SEED, device), 'align_grad', data, probe, device)
        journaled = np.array(row['scores'])
        rel = float(np.abs(scores - journaled).max() / np.abs(journaled).max())
        require(rel <= SCORE_TOL, 'score recomputation')
        gpu = dict(score_recomputation=dict(max_rel_diff=rel, audit_scoring_seconds=seconds,
                                            identical_kept_set=s73.keep_top(scores, n).tolist() == row['kept']),
                   replay_step50={})
        for arm in ('random', 'align_once'):
            ref = next(r for r in rows if (r['seed'], r['arm']) == (AUDIT_SEED, arm))
            order = check_run(ref, n)
            rep = s73.train(s69.load(model_dir, AUDIT_SEED, device), data, order, s73.DIGEST_STEP, set(), None, device)
            same = rep['adapter_digest_50'] == ref['adapter_digest_50']
            require(same, f'{arm} step-50 replay')
            gpu['replay_step50'][arm] = dict(identical_adapter_digest=same)
            torch.cuda.empty_cache()
    out = dict(stage=75, status='passed', journal_sha256=result['journal_sha256'], runs=len(rows), updates=n,
               environments=len(envs), summary_recomputed=True, score_identity_align_q_vs_once=score_identity,
               target=dict(items=len(target_ids), disjoint_from_exposed=len(exposed), eligibility=population['target_pool']),
               step_seconds_median={arm: float(np.median([np.median(r['step_seconds']) for r in rows if r['arm'] == arm]))
                                    for arm in stage.ARMS},
               gpu=gpu)
    OUTPUT.write_text(json.dumps(out, indent=2) + '\n')
    print('audit passed', flush=True)


if __name__ == '__main__':
    main()

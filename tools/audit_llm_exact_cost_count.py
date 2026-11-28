"""Audit stage 80 (research/llm-exact-cost-count-protocol.json, audit field).

CPU: journal digest; summary recomputation; environments (one or more, all NVIDIA L4) and block fields; retimings;
calibration, N_8192 and N_4096 recomputed and within their bounds, cost-point update counts; schedule digests per seed
and pool size; 192-run grid, block membership and order; update counts and checkpoints; step 0 present in each seed's
first random run, otherwise only in the first random run after a new machine, and identical everywhere; kept sets
recomputed from journaled scores and their digests; QA shares and counts; random-arm QA draws; views per chunk; data
hashes and model hashes (both models, proxy tokenizer); the four pools rebuilt with records equal to stage 79; target
eligibility recomputed (criteria (a)-(e), counts) and the 512 target IDs re-drawn; target disjoint from all 3,840
exposed and 4,946 trained IDs (computed independently from the stage 73-79 journals and the ledger).
GPU (NVIDIA L4): recompute the proxy scores of seed 38000 in cells a and d; replay the first 50 steps of random and
align05_once in cell a to their step-50 adapter digests for seeds 38000 (block 1) and 38012 (block 2).

Prints pass/fail only; no outcome is printed.

python -m tools.audit_llm_exact_cost_count --journal J --kept K --removed R --mmlu M [--no-gpu]
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from experiments import llm_exact_cost_count as stage
from experiments import llm_gpu as s69
from experiments import llm_web_filter as s73
from tools import audit_llm_count_share_filter as a79
from tools.audit_pud_reuse import require

RESULT = Path('results/llm-exact-cost-count.json')
OUTPUT = Path('results/llm-exact-cost-count-audit.json')
SCORE_SEED, REPLAY_SEEDS, SCORE_TOL, TRAINED_IDS, RUNS = 38000, (38000, 38012), 1e-4, 4946, 192
SOFTWARE = ('torch', 'transformers', 'tokenizers', 'cuda', 'python')


def check_calibration(cal):
    require(cal['model'] == stage.MODEL and cal['proxy'] == stage.PROXY, 'calibration models')
    require(cal['device'].startswith('NVIDIA L4'), 'calibration on an L4')
    c = cal['costs']
    require(c['random'] > 0 and c['proxy_align_grad_8192'] > 0 and c['proxy_align_grad_4096'] > 0, 'calibration costs')
    n = {m: int(np.floor(stage.BUDGET - c[f'proxy_align_grad_{m}'] / c['random'] + 1e-9)) for m in stage.SIZES}
    require(stage.n_of(cal) == n, 'N_M = floor(B - S(M) / c_r)')
    for m, (lo, hi) in stage.N_BOUNDS.items():
        require(lo <= n[m] <= hi, f'N_{m} within [{lo}, {hi}]')
    require(n[4096] > n[8192], 'N_4096 above N_8192')
    require(cal['schedules'] == stage.schedule_digests(stage.SEEDS, n), 'schedule digests per seed and pool size')
    return n


def step0_expected(journal):
    """(seed, cell, arm) of the runs that must carry step 0: each seed's first random run in the grid, and the first
    random run journaled after an environment on a GPU not seen before, when runs were already journaled."""
    grid = stage.run_grid()
    expected = {stage.first_random(s, grid) for s in stage.SEEDS}
    seen, runs, pending = set(), 0, False
    for r in journal:
        if r['kind'] == 'environment':
            if r.get('gpu_uuid') not in seen and runs:
                pending = True
            seen.add(r.get('gpu_uuid'))
        elif r['kind'] == 'run':
            runs += 1
            if pending and r['arm'] == 'random':
                expected.add((r['seed'], r['cell'], r['arm']))
                pending = False
    return expected


def check_run(row, n, step0):
    tag = f"seed {row['seed']} cell {row['cell']} {row['arm']}"
    arm, cell = row['arm'], row['cell']
    m, _, q = stage.CELLS[cell]
    n_m = n[m]
    require((row['M'], row['Q'], row['N_M']) == (m, q, n_m), f'{tag}: cell M, Q and N_M')
    require(row['block'] == stage.block_of(row['seed']), f'{tag}: block')
    require(row['device'].startswith('NVIDIA L4'), f'{tag}: device')
    sch = stage.schedule(row['seed'], m, n_m)
    require(row['model'] == stage.MODEL, f'{tag}: model')
    require(row['schedule_sha256'] == sch['sha256'], f'{tag}: schedule')
    require(len(row['step_seconds']) == stage.updates(arm, n_m), f'{tag}: update count')
    ckpts = stage.checkpoints(arm, n_m, step0)
    keys = {str(t) for t in ckpts}
    require(set(row['losses']) == keys == set(row['accuracy']), f'{tag}: checkpoints')
    require(isinstance(row['adapter_digest_50'], str) and len(row['adapter_digest_50']) == 64, f'{tag}: step-50 digest')
    kept = None
    if arm == 'align05_once':
        require(len(row['scores']) == m, f'{tag}: score count')
        kept = s73.keep_top(row['scores'], n_m)
        require(kept.tolist() == row['kept'], f'{tag}: kept set from scores')
        require(hashlib.sha256(kept.tobytes()).hexdigest() == row['kept_sha256'], f'{tag}: kept digest')
        require(row['scoring_seconds'] > 0, f'{tag}: scoring time')
    else:
        require(not {'kept', 'scores', 'kept_sha256', 'scoring_seconds'} & set(row), f'{tag}: random carries filter fields')
    order, members = stage.training_order(arm, sch, m, kept)
    if arm == 'random':
        require(row['random_qa'] == {str(t): stage.random_qa(sch['rand'], t, q) for t in stage.COST_POINTS},
                f'{tag}: random QA draws')
        require('qa_share' not in row and 'qa_count' not in row, f'{tag}: random carries a QA share')
    else:
        require(row['qa_share'] == stage.qa_share(members, q), f'{tag}: QA share')
        require(row['qa_count'] == stage.qa_count(members, q), f'{tag}: QA count')
        require(row['views'][str(n_m)]['max'] == 1, f'{tag}: one pass by N_M')
    require(row['views'] == {str(t): stage.views(order, members, t) for t in sorted(ckpts - {0})}, f'{tag}: views')
    return order


def check_environments(envs, rows):
    require(len(envs) >= 1 and all(e['device'].startswith('NVIDIA L4') for e in envs), 'environments all on an L4')
    require(all(e.get('block') in (1, 2) for e in envs), 'environment block fields')
    blocks = [e['block'] for e in envs]
    require(blocks == sorted(blocks), 'block 2 environments follow block 1')
    uuids = {r['gpu_uuid'] for r in rows}
    require(uuids <= {e.get('gpu_uuid') for e in envs}, 'run GPUs journaled in environments')
    return dict(count=len(envs), gpus=len({e.get('gpu_uuid') for e in envs}),
                software_identical=len({tuple(e[k] for k in SOFTWARE) for e in envs}) == 1,
                drivers=sorted({str(e.get('driver')) for e in envs}))


def exposed_and_trained(test):
    """Evaluated IDs of stages 73-79 and the 4,946 trained IDs."""
    evaluated, trained = a79.exposed_and_trained(test)
    return evaluated | set(stage.stage79_population()['target_items']), trained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('--journal', '--kept', '--removed', '--mmlu'):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--no-gpu', action='store_true')
    args = parser.parse_args()
    plan = json.loads(stage.PROTOCOL.read_text())
    stage.check_protocol(plan)
    result = json.loads(RESULT.read_text())
    journal = s69.read_journal(args.journal)
    require(s69.sha256(args.journal) == result['journal_sha256'], 'journal digest')
    require(result['runs'] == RUNS and result['stage'] == 80, 'summary header')
    envs = [r for r in journal if r['kind'] == 'environment']
    rows = [r for r in journal if r['kind'] == 'run']
    env_info = check_environments(envs, rows)
    require(result['environments'] == envs, 'environments in the summary')
    retimings = [r for r in journal if r['kind'] == 'retiming']
    require(result['retimings'] == retimings and all(r['random'] > 0 for r in retimings), 'retimings')
    cals = [r for r in journal if r['kind'] == 'calibration']
    require(len(cals) == 1, 'one calibration')
    cal = cals[0]
    n = check_calibration(cal)
    require(journal.index(cal) == 1 and journal[0]['kind'] == 'environment' and journal[0]['block'] == 1,
            'calibration first, in block 1, before any run')
    require(result['calibration'] == {k: v for k, v in cal.items() if k not in ('population', 'schedules')},
            'calibration in the summary')
    require([(r['seed'], r['cell'], r['arm']) for r in rows] == stage.run_grid(), 'run grid and order')
    require(len({(r['seed'], r['cell'], r['arm']) for r in rows}) == RUNS, '192 distinct runs')
    expected0 = step0_expected(journal)
    for r in rows:
        check_run(r, n, (r['seed'], r['cell'], r['arm']) in expected0)
    with0 = [r for r in rows if '0' in r['losses']]
    require({(r['seed'], r['cell'], r['arm']) for r in with0} == expected0, 'step-0 runs')
    require(len({r['losses']['0'] for r in with0}) == 1 and len({r['accuracy']['0'] for r in with0}) == 1,
            'identical step-0 loss and accuracy across seeds and machines')
    summary = stage.summarize(rows, cal)
    require(json.loads(json.dumps(summary)) == result['summary'], 'summary recomputation')

    from transformers import AutoTokenizer
    specs = json.loads(s69.PROTOCOL.read_text())['models']
    model_dir, proxy_dir = args.models_dir / stage.MODEL, args.models_dir / stage.PROXY
    for name, d in ((stage.MODEL, model_dir), (stage.PROXY, proxy_dir)):
        require(s69.sha256(d / 'model.safetensors') == specs[name]['safetensors_sha256'], f'{name} hash')
    for f in ('tokenizer.json', 'vocab.json', 'merges.txt'):
        require(s69.sha256(proxy_dir / f) == s69.sha256(model_dir / f), 'proxy tokenizer equals the trained model')
    data_hashes = {str(p): s69.sha256(p) for p in (args.kept, args.removed, args.mmlu / 'dev.csv', args.mmlu / 'test.csv')}
    require(list(data_hashes.values()) == [s73.KEPT_SHA256, s73.REMOVED_SHA256, s73.DEV_SHA256, s73.TEST_SHA256],
            'data hashes')
    tok = AutoTokenizer.from_pretrained(model_dir)
    probe, target, pools, population = stage.load_inputs(args, tok)  # pools = stage 79, (a)-(e), target draw
    require(population == cal['population'], 'population (QA, web, pools, target eligibility, target IDs)')
    s79pop = stage.stage79_population()
    require({c: population['cells'][c]['pool_sha256'] for c in stage.CELLS}
            == {c: s79pop['cells'][c]['pool_sha256'] for c in stage.CELLS}, 'pool digests equal stage 79')
    import pandas as pd
    test = pd.read_csv(args.mmlu / 'test.csv').to_dict('records')
    evaluated, trained = exposed_and_trained(test)
    require(len(evaluated) == stage.EXPOSED_ROWS and len(trained) == TRAINED_IDS, 'exposed and trained ID counts')
    require({int(test[i]['ID']) for i in stage.excluded_rows(test)} == evaluated, 'excluded rows equal the exposed IDs')
    target_ids = set(population['target_items'])
    require(len(target_ids) == stage.TARGET_ITEMS == len(target), 'target size')
    require(not target_ids & evaluated and not target_ids & trained, 'target disjoint from exposed and trained IDs')
    require([it['id'] for it in target] == population['target_items'], 'target items re-drawn')

    gpu = None
    if not args.no_gpu:
        import torch
        s69.setup_numerics()
        device = torch.device('cuda')
        require(torch.cuda.get_device_name(0).startswith('NVIDIA L4'), 'audit on an L4')
        data = {c: pools[c].to(device) for c in ('a', 'd')}
        gpu = dict(score_recomputation={}, replay_step50={})
        for c in ('a', 'd'):
            m = stage.CELLS[c][0]
            row = next(r for r in rows if (r['seed'], r['cell'], r['arm']) == (SCORE_SEED, c, 'align05_once'))
            proxy = s69.load(proxy_dir, SCORE_SEED, device)
            scores, seconds = s73.score(proxy, 'align_grad', data[c], probe, device)
            del proxy
            torch.cuda.empty_cache()
            journaled = np.array(row['scores'])
            rel = float(np.abs(scores - journaled).max() / np.abs(journaled).max())
            require(rel <= SCORE_TOL, f'cell {c} proxy score recomputation')
            gpu['score_recomputation'][c] = dict(
                max_rel_diff=rel, audit_scoring_seconds=seconds, journaled_scoring_seconds=row['scoring_seconds'],
                identical_kept_set=s73.keep_top(scores, n[m]).tolist() == row['kept'])
        for seed in REPLAY_SEEDS:
            for arm in stage.ARMS:
                ref = next(r for r in rows if (r['seed'], r['cell'], r['arm']) == (seed, 'a', arm))
                order = check_run(ref, n, (seed, 'a', arm) in expected0)
                rep = s73.train(s69.load(model_dir, seed, device), data['a'], order, s73.DIGEST_STEP, set(), None,
                                device)
                same = rep['adapter_digest_50'] == ref['adapter_digest_50']
                require(same, f'seed {seed} cell a {arm} step-50 replay')
                gpu['replay_step50'][f'{seed}/a/{arm}'] = dict(identical_adapter_digest=same, run_gpu=ref['gpu_uuid'])
                torch.cuda.empty_cache()
        import tokenizers
        import transformers
        gpu['environment'] = dict(device=torch.cuda.get_device_name(0), gpu_uuid=stage.gpu_uuid(torch),
                                  torch=torch.__version__, transformers=transformers.__version__,
                                  tokenizers=tokenizers.__version__, cuda=torch.version.cuda,
                                  python=platform.python_version())
    out = dict(stage=80, status='passed', journal_sha256=result['journal_sha256'], runs=len(rows),
               N={str(m): v for m, v in n.items()}, environments=env_info, retimings=len(retimings),
               summary_recomputed=True, step0_identical=True, step0_runs=len(with0), data_hashes=data_hashes,
               population=dict(web_eligible=population['web_eligible'],
                               pools={c: population['cells'][c]['pool_sha256'] for c in population['cells']},
                               equal_to_stage79=True),
               target=dict(items=len(target_ids), disjoint_from_exposed=len(evaluated), disjoint_from_trained=len(trained),
                           eligibility=population['target_pool']),
               step_seconds_median={arm: float(np.median([np.median(r['step_seconds']) for r in rows if r['arm'] == arm]))
                                    for arm in stage.ARMS},
               scoring_seconds_median={str(m): float(np.median([r['scoring_seconds'] for r in rows
                                                                if r['arm'] == 'align05_once' and r['M'] == m]))
                                       for m in stage.SIZES},
               gpu=gpu)
    OUTPUT.write_text(json.dumps(out, indent=2) + '\n')
    print('audit passed', flush=True)


if __name__ == '__main__':
    main()

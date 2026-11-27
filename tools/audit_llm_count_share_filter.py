"""Audit stage 79 (research/llm-count-share-filter-protocol.json, audit field).

CPU: journal digest; summary recomputation; environments; calibration, the N_cal bound and B_eq(4096); 72-run grid and
order; schedules per seed and pool size; update counts and checkpoints; kept sets recomputed from journaled scores and
their digests; subset digests; QA shares and counts; random-arm QA draws; views per chunk; data hashes and model hashes
(both models, proxy tokenizer); QA subsets (Q512, Q256), web lists and the four pools rebuilt with their digests (cell
d equal to the stage-78 Q1024 pool); nesting of QA and web chunks across cells (index sets, document lists and pool
tensors); web extras recomputed by an independent implementation; target eligibility recomputed (criteria (a)-(e) on
all four pools, counts) and the 512 target IDs re-drawn; target disjoint from all 3,328 exposed and 4,946 trained IDs
(computed independently from the stage 73-78 journals and the ledger); identical step-0 losses.
GPU (same device type): recompute the proxy scores of seed 37900 in cells a and d; replay the first 50 steps of seed
37900 for random and align05_once in cell a to their step-50 adapter digests.

Prints pass/fail only; no outcome is printed.

python -m tools.audit_llm_count_share_filter --journal J --kept K --removed R --mmlu M [--no-gpu]
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np

from experiments import llm_cheap_filter as s76
from experiments import llm_count_share_filter as stage
from experiments import llm_gpu as s69
from experiments import llm_web_filter as s73
from tools import audit_llm_supply_filter as a78
from tools.audit_pud_reuse import require

RESULT = Path('results/llm-count-share-filter.json')
OUTPUT = Path('results/llm-count-share-filter-audit.json')
AUDIT_SEED, SCORE_TOL, TRAINED_IDS, RUNS = 37900, 1e-4, 4946, 72
ENV_KEYS = ('device', 'torch', 'transformers', 'tokenizers', 'cuda', 'python')


def check_calibration(cal):
    require(cal['model'] == stage.MODEL and cal['proxy'] == stage.PROXY, 'calibration models')
    c = cal['costs']
    require(c['random'] > 0 and c['proxy_align_grad_8192'] > 0 and c['proxy_align_grad_4096'] > 0, 'calibration costs')
    n_cal = int(np.floor(stage.BUDGET - c['proxy_align_grad_8192'] / c['random'] + 1e-9))
    b_eq = stage.N + math.ceil(c['proxy_align_grad_4096'] / c['random'])
    require(cal['N_cal'] == n_cal, 'N_cal = floor(B - S(8192) / c_r)')
    require(cal['B_eq_4096'] == b_eq, 'B_eq(4096) = N + ceil(S(4096) / c_r)')
    require(stage.N_BOUNDS[0] <= n_cal <= stage.N_BOUNDS[1], 'N_cal within [1470, 1796]')
    require(b_eq < stage.BUDGET, 'B_eq(4096) below 3000')
    require(cal['updates'] == stage.N == 1633, 'N fixed at 1633')
    return n_cal, b_eq


def check_run(row, b_eq):
    tag = f"seed {row['seed']} cell {row['cell']} {row['arm']}"
    arm, cell = row['arm'], row['cell']
    m, _, q = stage.CELLS[cell]
    require((row['M'], row['Q']) == (m, q), f'{tag}: cell M and Q')
    sch = stage.schedule(row['seed'], m)
    require(row['model'] == stage.MODEL, f'{tag}: model')
    require(row['schedule_sha256'] == sch['sha256'], f'{tag}: schedule')
    require(len(row['step_seconds']) == stage.updates(arm), f'{tag}: update count')
    ckpts = stage.checkpoints(arm, cell, b_eq)
    keys = {str(t) for t in ckpts}
    require(set(row['losses']) == keys == set(row['accuracy']), f'{tag}: checkpoints')
    require(isinstance(row['adapter_digest_50'], str) and len(row['adapter_digest_50']) == 64, f'{tag}: step-50 digest')
    kept = None
    if arm == 'align05_once':
        require(len(row['scores']) == m, f'{tag}: score count')
        kept = s73.keep_top(row['scores'], stage.N)
        require(kept.tolist() == row['kept'], f'{tag}: kept set from scores')
        require(hashlib.sha256(kept.tobytes()).hexdigest() == row['kept_sha256'], f'{tag}: kept digest')
        require(row['scoring_seconds'] > 0, f'{tag}: scoring time')
    else:
        require(not {'kept', 'scores', 'kept_sha256', 'scoring_seconds'} & set(row), f'{tag}: control carries filter fields')
    order, members = stage.training_order(arm, sch, m, kept)
    if arm == 'random_once':
        require(row['subset_sha256'] == hashlib.sha256(np.asarray(members).tobytes()).hexdigest(), f'{tag}: subset')
    if arm == 'random':
        require(row['random_qa'] == {str(t): stage.random_qa(sch['rand'], t, q) for t in sorted(ckpts - {0})},
                f'{tag}: random QA draws')
        require('qa_share' not in row and 'qa_count' not in row, f'{tag}: random carries a QA share')
    else:
        require(row['qa_share'] == stage.qa_share(members, q), f'{tag}: QA share')
        require(row['qa_count'] == stage.qa_count(members, q), f'{tag}: QA count')
        require(row['views'][str(stage.N)]['max'] == 1, f'{tag}: one pass by N')
    require(row['views'] == {str(t): stage.views(order, members, t) for t in sorted(ckpts - {0})}, f'{tag}: views')
    return order


def check_nesting(population, pools):
    """QA subsets, web document lists and pool tensors are nested across cells as the protocol specifies."""
    subsets = stage.qa_subsets()
    s = {q: set(subsets[q].tolist()) for q in subsets}
    require({q: len(v) for q, v in s.items()} == {256: 256, 512: 512, 1024: 1024}, 'QA subset sizes')
    require(s[256] <= s[512] <= s[1024], 'QA subsets nested')
    q1024 = stage.s78.qa_subsets()[1024]
    q512 = np.sort(q1024[np.random.default_rng(7901).choice(1024, 512, replace=False)])
    require(np.array_equal(subsets[512], q512), 'Q512 draw')
    require(np.array_equal(subsets[256], np.sort(q512[np.random.default_rng(7902).choice(512, 256, replace=False)])),
            'Q256 draw')
    cells = population['cells']
    for c, (m, share, q) in stage.CELLS.items():
        rec = cells[c]
        nk, nr = stage.web_counts(c)
        require((rec['M'], rec['share'], rec['qa'], rec['kept_web'], rec['removed_web']) == (m, share, q, nk, nr)
                and q + nk + nr == m, f'cell {c} counts')
        require(rec['qa_subset_sha256'] == hashlib.sha256(subsets[q].tobytes()).hexdigest(), f'cell {c} QA digest')
        require(len(set(rec['kept_documents'])) == nk and len(set(rec['removed_documents'])) == nr,
                f'cell {c} documents distinct')
        require(rec['pool_sha256'] == s73.tensor_sha256(pools[c]), f'cell {c} pool digest')
    for small, large in (('b', 'a'), ('a', 'd'), ('d', 'c')):
        for key in ('kept_documents', 'removed_documents'):
            a, b = cells[small][key], cells[large][key]
            require(b[:len(a)] == a, f'web {key} of cell {small} a prefix of cell {large}')
    qa1024 = pools['d'][:1024]
    big = pools['c']
    nkc, _ = stage.web_counts('c')
    for c, (m, _, q) in stage.CELLS.items():
        nk, nr = stage.web_counts(c)
        pos = np.searchsorted(q1024, subsets[q])
        require(bool((pools[c][:q] == qa1024[pos]).all()), f'cell {c} QA chunks are the subset of Q1024')
        require(bool((pools[c][q:q + nk] == big[512:512 + nk]).all()), f'cell {c} kept web chunks nested')
        require(bool((pools[c][q + nk:] == big[512 + nkc:512 + nkc + nr]).all()), f'cell {c} removed web chunks nested')
    require(bool((pools['b'][:512] == pools['c'][:512]).all()), 'cells b and c share the 512 QA chunks')
    n_k, n_r = population['web_eligible']
    k76, r76 = s76.web_draws(n_k, n_r)
    kl, rl, nk_x, nr_x = stage.web_lists(k76, r76, n_k, n_r)
    for drawn, n, seed, lst, k, j in ((k76, n_k, 7811, kl, nk_x, 0), (r76, n_r, 7812, rl, nr_x, 1)):
        mine = a78.independent_extras(n, drawn, seed, k)
        require(np.array_equal(mine, lst[len(drawn):]), 'web extras (independent recomputation)')
        require(population['web_extras'][j] == k == 768 and not set(mine.tolist()) & set(np.asarray(drawn).tolist()),
                'web extras count and disjointness from the stage-76 draws')
    p78 = stage.stage78_population()
    require(cells['d']['pool_sha256'] == p78['levels']['1024']['pool_sha256'], 'cell d pool equals stage-78 Q1024')
    require(cells['d']['kept_documents'] == p78['levels']['1024']['kept_documents']
            and cells['d']['removed_documents'] == p78['levels']['1024']['removed_documents'],
            'cell d web documents equal stage-78 Q1024')
    require(population['web_eligible'] == p78['web_eligible'], 'web eligibility equals stage 78')


def exposed_and_trained(test):
    """Evaluated IDs of stages 73-78 and the 4,946 trained IDs."""
    evaluated, trained = a78.exposed_and_trained(test)
    evaluated = evaluated | set(stage.stage78_population()['target_items'])
    return evaluated, trained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('--journal', '--kept', '--removed', '--mmlu'):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--no-gpu', action='store_true')
    args = parser.parse_args()
    plan = json.loads(stage.PROTOCOL.read_text())
    stage.check_protocol(plan)
    seeds = plan['design']['seeds']
    result = json.loads(RESULT.read_text())
    journal = s69.read_journal(args.journal)
    require(s69.sha256(args.journal) == result['journal_sha256'], 'journal digest')
    require(result['runs'] == RUNS and result['stage'] == 79, 'summary header')
    envs = [r for r in journal if r['kind'] == 'environment']
    require(len({tuple(e[k] for k in ENV_KEYS) for e in envs}) == 1, 'environments differ')
    require(result['environments'] == envs, 'environments in the summary')
    cals = [r for r in journal if r['kind'] == 'calibration']
    require(len(cals) == 1, 'one calibration')
    cal = cals[0]
    n_cal, b_eq = check_calibration(cal)
    require(envs[0]['device'] == cal['device'], 'environment device equals the calibration')
    require(result['calibration'] == {k: v for k, v in cal.items() if k != 'population'}, 'calibration in the summary')
    require(cal['population']['schedules'] == stage.schedule_digests(seeds), 'schedule digests per seed and pool size')
    rows = [r for r in journal if r['kind'] == 'run']
    require([(r['seed'], r['cell'], r['arm']) for r in rows] == stage.run_grid(seeds), 'run grid and order')
    require(len(rows) == RUNS and len({(r['seed'], r['cell'], r['arm']) for r in rows}) == RUNS, '72 distinct runs')
    for r in rows:
        require(r['device'] == cal['device'], 'run device equals the calibration')
        check_run(r, b_eq)
    zero = [r for r in rows if r['arm'] == 'random']
    require(len({r['losses']['0'] for r in zero}) == 1 and len({r['accuracy']['0'] for r in zero}) == 1,
            'identical step-0 loss and accuracy in all random runs')
    summary = stage.summarize(rows, cal, plan)
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
    probe, target, pools, population = stage.load_inputs(args, tok)  # QA digest, pools, cell-d digest, (a)-(e), draw
    population['schedules'] = stage.schedule_digests(seeds)
    require(population == cal['population'], 'population (QA, web, pools, target eligibility, target IDs, schedules)')
    check_nesting(population, pools)
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
        require(torch.cuda.get_device_name(0) == cal['device'], 'audit device type differs from the run')
        data = {c: pools[c].to(device) for c in ('a', 'd')}
        gpu = dict(score_recomputation={}, replay_step50={})
        for c in ('a', 'd'):
            row = next(r for r in rows if (r['seed'], r['cell'], r['arm']) == (AUDIT_SEED, c, 'align05_once'))
            proxy = s69.load(proxy_dir, AUDIT_SEED, device)
            scores, seconds = s73.score(proxy, 'align_grad', data[c], probe, device)
            del proxy
            torch.cuda.empty_cache()
            journaled = np.array(row['scores'])
            rel = float(np.abs(scores - journaled).max() / np.abs(journaled).max())
            require(rel <= SCORE_TOL, f'cell {c} proxy score recomputation')
            gpu['score_recomputation'][c] = dict(
                max_rel_diff=rel, audit_scoring_seconds=seconds, journaled_scoring_seconds=row['scoring_seconds'],
                identical_kept_set=s73.keep_top(scores, stage.N).tolist() == row['kept'])
        for arm in ('random', 'align05_once'):
            ref = next(r for r in rows if (r['seed'], r['cell'], r['arm']) == (AUDIT_SEED, 'a', arm))
            order = check_run(ref, b_eq)
            rep = s73.train(s69.load(model_dir, AUDIT_SEED, device), data['a'], order, s73.DIGEST_STEP, set(), None,
                            device)
            same = rep['adapter_digest_50'] == ref['adapter_digest_50']
            require(same, f'cell a {arm} step-50 replay')
            gpu['replay_step50'][f'a/{arm}'] = dict(identical_adapter_digest=same)
            torch.cuda.empty_cache()
        import tokenizers
        import transformers
        gpu['environment'] = dict(device=torch.cuda.get_device_name(0), torch=torch.__version__,
                                  transformers=transformers.__version__, tokenizers=tokenizers.__version__,
                                  cuda=torch.version.cuda, python=platform.python_version())
    out = dict(stage=79, status='passed', journal_sha256=result['journal_sha256'], runs=len(rows), N=stage.N,
               N_cal=n_cal, B_eq_4096=b_eq, environments=len(envs), summary_recomputed=True, step0_identical=True,
               data_hashes=data_hashes,
               population=dict(web_eligible=population['web_eligible'], web_extras=population['web_extras'],
                               pools={c: population['cells'][c]['pool_sha256'] for c in population['cells']},
                               nesting_checked=True),
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

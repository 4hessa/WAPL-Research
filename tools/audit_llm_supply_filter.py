"""Audit stage 78 (research/llm-supply-filter-protocol.json, audit field, with amendment 1).

CPU: journal digest; summary recomputation; environments; calibration and the N_cal bound; 54-run grid and order;
schedules; update counts and checkpoints; kept sets recomputed from journaled scores and their digests; subset digests;
QA shares; random-arm QA draws; views per chunk; data hashes (in load_inputs) and model hashes (both models, proxy
tokenizer); QA subsets, web extras and the three pools rebuilt with their digests (Q2048 equal to stage 76, checked in
load_inputs and against the stage-76 journal); nesting of QA and web chunks across levels (index sets, document lists
and pool tensors); web extras recomputed by an independent implementation; target eligibility recomputed (criteria
(a)-(e) on all three pools, counts) and the 512 target IDs re-drawn; target disjoint from all 2,816 exposed and 4,946
trained IDs (computed independently from the stage 73-77 journals and the ledger); identical step-0 losses.
GPU (same device type): recompute the proxy scores of seed 37800 at Q1024 and Q2048; replay the first 50 steps of seed
37800 for random and align05_once at Q1024 to their step-50 adapter digests.

Prints pass/fail only; no outcome is printed.

python -m tools.audit_llm_supply_filter --journal J --kept K --removed R --mmlu M [--no-gpu]
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from experiments import llm_cheap_filter as s76
from experiments import llm_gpu as s69
from experiments import llm_qa_mix as s74
from experiments import llm_supply_filter as stage
from experiments import llm_web_filter as s73
from tools.audit_pud_reuse import require

RESULT = Path('results/llm-supply-filter.json')
OUTPUT = Path('results/llm-supply-filter-audit.json')
STAGE74_JOURNAL = Path('results/llm-qa-mix-journal.jsonl')
EARLIER_TARGET_JOURNALS = (Path('results/llm-once-filter-journal.jsonl'), Path('results/llm-cheap-filter-journal.jsonl'),
                           Path('results/llm-deepseek-filter-journal.jsonl'))
AUDIT_SEED, SCORE_TOL, TRAINED_IDS = 37800, 1e-4, 4946
ENV_KEYS = ('device', 'torch', 'transformers', 'cuda', 'python')


def check_calibration(cal):
    require(cal['model'] == stage.MODEL and cal['proxy'] == stage.PROXY, 'calibration models')
    costs = cal['costs']
    require(costs['random'] > 0 and costs['proxy_align_grad'] > 0, 'calibration costs')
    n_cal = int(np.floor(stage.BUDGET - costs['proxy_align_grad'] / costs['random'] + 1e-9))
    require(cal['N_cal'] == n_cal, 'N_cal = floor(B - S / c_r)')
    require(stage.N_BOUNDS[0] <= n_cal <= stage.N_BOUNDS[1], 'N_cal within [1470, 1796]')
    require(cal['updates'] == stage.N == 1633, 'N fixed at 1633')
    return n_cal


def check_run(row):
    tag = f"seed {row['seed']} Q{row['level']} {row['arm']}"
    arm, q = row['arm'], row['level']
    sch = stage.schedule(row['seed'])
    require(row['model'] == stage.MODEL, f'{tag}: model')
    require(row['schedule_sha256'] == sch['sha256'], f'{tag}: schedule')
    require(len(row['step_seconds']) == stage.updates(arm), f'{tag}: update count')
    keys = {str(t) for t in stage.checkpoints(arm)}
    require(set(row['losses']) == keys == set(row['accuracy']), f'{tag}: checkpoints')
    require(isinstance(row['adapter_digest_50'], str) and len(row['adapter_digest_50']) == 64, f'{tag}: step-50 digest')
    kept = None
    if arm == 'align05_once':
        require(len(row['scores']) == stage.POOL_SIZE, f'{tag}: score count')
        kept = s73.keep_top(row['scores'], stage.N)
        require(kept.tolist() == row['kept'], f'{tag}: kept set from scores')
        require(hashlib.sha256(kept.tobytes()).hexdigest() == row['kept_sha256'], f'{tag}: kept digest')
        require(row['scoring_seconds'] > 0, f'{tag}: scoring time')
    else:
        require(not {'kept', 'scores', 'kept_sha256', 'scoring_seconds'} & set(row), f'{tag}: control carries filter fields')
    order, members = stage.training_order(arm, sch, kept)
    if arm == 'random_once':
        require(row['subset_sha256'] == hashlib.sha256(np.asarray(members).tobytes()).hexdigest(), f'{tag}: subset')
    if arm == 'random':
        require(row['random_qa'] == {str(t): stage.random_qa(sch['rand'], t, q) for t in (stage.N, stage.BUDGET)},
                f'{tag}: random QA draws')
        require('qa_share' not in row, f'{tag}: random carries a QA share')
    else:
        require(row['qa_share'] == stage.qa_share(members, q), f'{tag}: QA share')
        require(row['views'][str(stage.N)]['max'] == 1, f'{tag}: one pass by N')
    require(row['views'] == {str(t): stage.views(order, members, t) for t in sorted(stage.checkpoints(arm) - {0})},
            f'{tag}: views')
    return order


def independent_extras(n_eligible, drawn, seed, k):
    """Web extras by a second implementation (set difference, then the seeded permutation)."""
    rest = np.setdiff1d(np.arange(n_eligible), np.asarray(drawn))
    return rest[np.random.default_rng(seed).permutation(len(rest))[:k]]


def check_nesting(population, pools):
    """QA subsets, web document lists and pool tensors are nested across levels as the protocol specifies."""
    subsets = stage.qa_subsets()
    s = {q: set(subsets[q].tolist()) for q in stage.LEVELS}
    require(len(s[1024]) == 1024 and len(s[1633]) == 1633 and len(s[2048]) == 2048, 'QA subset sizes')
    require(s[1024] <= s[1633] <= s[2048], 'QA subsets nested')
    require(np.array_equal(subsets[1633], np.sort(np.random.default_rng(7801).choice(2048, 1633, replace=False))),
            'Q1633 draw')
    lv = population['levels']
    for q in stage.LEVELS:
        rec = lv[str(q)]
        nk, nr = stage.web_counts(q)
        require((rec['qa'], rec['kept_web'], rec['removed_web']) == (q, nk, nr) and q + nk + nr == stage.POOL_SIZE,
                f'Q{q} counts')
        require(rec['qa_subset_sha256'] == hashlib.sha256(subsets[q].tobytes()).hexdigest(), f'Q{q} QA subset digest')
        require(len(rec['kept_documents']) == nk and len(rec['removed_documents']) == nr, f'Q{q} document counts')
        require(len(set(rec['kept_documents'])) == nk and len(set(rec['removed_documents'])) == nr,
                f'Q{q} documents distinct')
        require(rec['pool_sha256'] == s73.tensor_sha256(pools[q]), f'Q{q} pool digest')
    for small, large in ((2048, 1633), (1633, 1024)):  # fewer web chunks at higher Q: a prefix of the lower level
        for key in ('kept_documents', 'removed_documents'):
            a, b = lv[str(small)][key], lv[str(large)][key]
            require(b[:len(a)] == a, f'web {key} of Q{small} a prefix of Q{large}')
    qa_all = pools[2048][:2048]
    for q in stage.LEVELS:
        nk, nr = stage.web_counts(q)
        require(bool((pools[q][:q] == qa_all[subsets[q]]).all()), f'Q{q} QA chunks are the subset of the 2048')
        low = pools[1024]
        nk0, _ = stage.web_counts(1024)
        require(bool((pools[q][q:q + nk] == low[1024:1024 + nk]).all()), f'Q{q} kept web chunks nested')
        require(bool((pools[q][q + nk:] == low[1024 + nk0:1024 + nk0 + nr]).all()), f'Q{q} removed web chunks nested')
    # web extras: second implementation over the stage-76 draws
    n_k, n_r = population['web_eligible']
    k76, r76 = s76.web_draws(n_k, n_r)
    nk_max, nr_max = stage.web_counts(1024)
    for drawn, n, seed, k, j in ((k76, n_k, 7811, nk_max - len(k76), 0), (r76, n_r, 7812, nr_max - len(r76), 1)):
        mine = independent_extras(n, drawn, seed, k)
        require(np.array_equal(mine, stage.extra_draws(n, drawn, seed, k)), 'web extras (independent recomputation)')
        require(population['web_extras'][j] == k and not set(mine.tolist()) & set(np.asarray(drawn).tolist()),
                'web extras count and disjointness from the stage-76 draws')
    stage76 = next(r for r in s69.read_journal(EARLIER_TARGET_JOURNALS[1]) if r.get('kind') == 'calibration')['population']
    require(lv['2048']['pool_sha256'] == stage.STAGE76_POOL_SHA256, 'Q2048 pool equals stage 76')
    require(population['web_eligible'] == stage76['web_eligible'], 'web eligibility equals stage 76')


def exposed_and_trained(test):
    """Evaluated IDs of stages 73-77 and the 4,946 trained IDs, from the stage functions, journals and ledger."""
    pilot, target74 = s74.item_splits(len(test))
    evaluated = {int(test[i]['ID']) for i in s73.sample_target(len(test))}
    evaluated |= {int(test[i]['ID']) for i in np.r_[pilot, target74]}
    cal74 = next(r for r in s69.read_journal(STAGE74_JOURNAL) if r.get('kind') == 'calibration')['population']
    require(set(cal74['pilot_items']) | set(cal74['target_items']) <= evaluated, 'stage-74 splits match the journal')
    for path in EARLIER_TARGET_JOURNALS:
        evaluated |= set(next(r for r in s69.read_journal(path) if r.get('kind') == 'calibration')['population']
                         ['target_items'])
    ledger = json.loads(stage.s75.LEDGER.read_text())['additional_sources']['ArabicMMLU']['stage_74']
    trained = set(ledger['used_for_training']['ids'])
    return evaluated, trained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('--journal', '--kept', '--removed', '--mmlu'):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--no-gpu', action='store_true')
    args = parser.parse_args()
    plan = json.loads(stage.PROTOCOL.read_text())
    require(plan['stage'] == 78 and tuple(plan['design']['levels']) == stage.LEVELS, 'protocol')
    result = json.loads(RESULT.read_text())
    journal = s69.read_journal(args.journal)
    require(s69.sha256(args.journal) == result['journal_sha256'], 'journal digest')
    require(result['runs'] == 54 and result['stage'] == 78, 'summary header')
    envs = [r for r in journal if r['kind'] == 'environment']
    require(len({tuple(e[k] for k in ENV_KEYS) for e in envs}) == 1, 'environments differ')
    require(result['environments'] == envs, 'environments in the summary')
    cals = [r for r in journal if r['kind'] == 'calibration']
    require(len(cals) == 1, 'one calibration')
    cal = cals[0]
    n_cal = check_calibration(cal)
    require(envs[0]['device'] == cal['device'], 'environment device equals the calibration')
    require(result['calibration'] == {k: v for k, v in cal.items() if k != 'population'}, 'calibration in the summary')
    rows = [r for r in journal if r['kind'] == 'run']
    require([(r['seed'], r['level'], r['arm']) for r in rows] == stage.run_grid(plan['design']['seeds']),
            'run grid and order')
    require(len(rows) == 54 and len({(r['seed'], r['level'], r['arm']) for r in rows}) == 54, '54 distinct runs')
    for r in rows:
        require(r['device'] == cal['device'], 'run device equals the calibration')
        check_run(r)
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
    probe, target, pools, population = stage.load_inputs(args, tok)  # QA digest, pools, Q2048 digest, (a)-(e), draw
    require(population == cal['population'], 'population (QA, web, pools, target eligibility, target IDs)')
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
        data = {q: pools[q].to(device) for q in (1024, 2048)}
        gpu = dict(score_recomputation={}, replay_step50={})
        for q in (1024, 2048):
            row = next(r for r in rows if (r['seed'], r['level'], r['arm']) == (AUDIT_SEED, q, 'align05_once'))
            proxy = s69.load(proxy_dir, AUDIT_SEED, device)
            scores, seconds = s73.score(proxy, 'align_grad', data[q], probe, device)
            del proxy
            torch.cuda.empty_cache()
            journaled = np.array(row['scores'])
            rel = float(np.abs(scores - journaled).max() / np.abs(journaled).max())
            require(rel <= SCORE_TOL, f'Q{q} proxy score recomputation')
            gpu['score_recomputation'][str(q)] = dict(
                max_rel_diff=rel, audit_scoring_seconds=seconds, journaled_scoring_seconds=row['scoring_seconds'],
                identical_kept_set=s73.keep_top(scores, stage.N).tolist() == row['kept'])
        for arm in ('random', 'align05_once'):
            ref = next(r for r in rows if (r['seed'], r['level'], r['arm']) == (AUDIT_SEED, 1024, arm))
            order = check_run(ref)
            rep = s73.train(s69.load(model_dir, AUDIT_SEED, device), data[1024], order, s73.DIGEST_STEP, set(), None,
                            device)
            same = rep['adapter_digest_50'] == ref['adapter_digest_50']
            require(same, f'Q1024 {arm} step-50 replay')
            gpu['replay_step50'][f'1024/{arm}'] = dict(identical_adapter_digest=same)
            torch.cuda.empty_cache()
        import transformers
        gpu['environment'] = dict(device=torch.cuda.get_device_name(0), torch=torch.__version__,
                                  transformers=transformers.__version__, cuda=torch.version.cuda,
                                  python=platform.python_version())
    out = dict(stage=78, status='passed', journal_sha256=result['journal_sha256'], runs=len(rows), N=stage.N,
               N_cal=n_cal, environments=len(envs), summary_recomputed=True, step0_identical=True,
               data_hashes=data_hashes,
               population=dict(web_eligible=population['web_eligible'], web_extras=population['web_extras'],
                               pools={q: population['levels'][q]['pool_sha256'] for q in population['levels']},
                               nesting_checked=True),
               target=dict(items=len(target_ids), disjoint_from_exposed=len(evaluated), disjoint_from_trained=len(trained),
                           eligibility=population['target_pool']),
               step_seconds_median={arm: float(np.median([np.median(r['step_seconds']) for r in rows if r['arm'] == arm]))
                                    for arm in stage.ARMS},
               scoring_seconds_median=float(np.median([r['scoring_seconds'] for r in rows if r['arm'] == 'align05_once'])),
               gpu=gpu)
    OUTPUT.write_text(json.dumps(out, indent=2) + '\n')
    print('audit passed', flush=True)


if __name__ == '__main__':
    main()

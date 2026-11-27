"""Stage 79: count vs share of useful (in-format QA) chunks. Pool size M in {4096, 8192} crossed with QA share
s in {1/16, 1/8} (cells a-d, Q = 256, 512, 512, 1024), at the stage-78 trained model (Qwen2.5-1.5B), proxy scorer
(Qwen2.5-0.5B), budget (B = 3000 random-step costs) and number of filter updates (N = 1633) in every cell; one-pass
offline filter vs random and a repetition-matched random subset, on a new unexposed 512-item target.

Implements research/llm-count-share-filter-protocol.json with the stage 73-78 building blocks.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
from statistics import mean, median
import time

import numpy as np

from experiments import llm_cheap_filter as s76
from experiments import llm_gpu as s69
from experiments import llm_once_filter as s75
from experiments import llm_qa_mix as s74
from experiments import llm_supply_filter as s78
from experiments import llm_web_filter as s73

PROTOCOL = Path('research/llm-count-share-filter-protocol.json')
MODEL, PROXY = 'qwen2.5-1.5b', 'qwen2.5-0.5b'
ARMS = ('random', 'align05_once', 'random_once')
CELLS = {'a': (4096, 0.0625, 256), 'b': (4096, 0.125, 512), 'c': (8192, 0.0625, 512), 'd': (8192, 0.125, 1024)}
SIZES = (4096, 8192)
BUDGET, N, N_QA = 3000, 1633, 2048
N_BOUNDS = (1470, 1796)
CAL_SEED, TARGET_SEED, TARGET_ITEMS = 79999, 7922, 512
QA_SEEDS = (7901, 7902)
BOOT_SEED = 7903
STAGE78_JOURNAL = Path('results/llm-supply-filter-journal.jsonl')
EXPOSED_ROWS = 3328
INTERIM = 'results/stage79-interim'
PRIMARY_Q, SECONDARY_Q = (.0125, .9875), (.025, .975)
STAGE78_D1024 = dict(mean=-0.285, ci95=[-0.333, -0.238])  # protocol, secondary 10 (descriptive reference)


# ---------------------------------------------------------------- data

def stage78_population():
    return next(r for r in s69.read_journal(STAGE78_JOURNAL) if r.get('kind') == 'calibration')['population']


def qa_subsets():
    """Nested QA-chunk index sets (positions in the 2048 stage-74/76 QA chunks), sorted: Q1024 = stage-78 Q1024."""
    q1024 = s78.qa_subsets()[1024]
    q512 = np.sort(q1024[np.random.default_rng(QA_SEEDS[0]).choice(1024, 512, replace=False)])
    q256 = np.sort(q512[np.random.default_rng(QA_SEEDS[1]).choice(512, 256, replace=False)])
    return {256: q256, 512: q512, 1024: q1024}


def web_counts(cell):
    m, _, q = CELLS[cell]
    w = m - q
    return math.ceil(w / 2), w // 2


def web_lists(k76, r76, n_kept, n_removed):
    """Kept-web and removed-web index lists: the stage-76 draws, then the extras in stage-78 permutation order
    (default_rng(7811) / (7812)); long enough for every cell."""
    nk = max(web_counts(c)[0] for c in CELLS)
    nr = max(web_counts(c)[1] for c in CELLS)
    k_extra = s78.extra_draws(n_kept, k76, s78.EXTRA_SEEDS[0], nk - len(k76))
    r_extra = s78.extra_draws(n_removed, r76, s78.EXTRA_SEEDS[1], nr - len(r76))
    return (np.concatenate([k76, k_extra]).astype(int), np.concatenate([r76, r_extra]).astype(int),
            len(k_extra), len(r_extra))


def cell_members(cell, kept_list, removed_list):
    nk, nr = web_counts(cell)
    if nk > len(kept_list) or nr > len(removed_list):
        raise ValueError('Web lists too short')
    return kept_list[:nk], removed_list[:nr]


def excluded_rows(test):
    """Every earlier evaluation set: the 2,816 rows of stage 78 plus the stage-78 target."""
    ids78 = set(stage78_population()['target_items'])
    return s78.excluded_rows(test) | {i for i, r in enumerate(test) if int(r['ID']) in ids78}


def load_inputs(args, tokenizer):
    """Probe, target items, the four pools (QA 0..Q-1, then kept web, then removed web) and the population record."""
    import pandas as pd
    import pyarrow.parquet as pq
    import torch
    for path, digest in ((args.kept, s73.KEPT_SHA256), (args.removed, s73.REMOVED_SHA256),
                         (args.mmlu / 'dev.csv', s73.DEV_SHA256), (args.mmlu / 'test.csv', s73.TEST_SHA256)):
        if s69.sha256(path) != digest:
            raise ValueError(f'{path} differs from the protocol')
    dev = pd.read_csv(args.mmlu / 'dev.csv').to_dict('records')
    test = pd.read_csv(args.mmlu / 'test.csv').to_dict('records')
    probe = s73.build_items(dev, tokenizer)
    # QA chunks exactly as stages 74, 76 and 78
    banned = s74.content_banned(dev + test, tokenizer)
    pilot_rows, target74_rows = s74.item_splits(len(test))
    excl74 = set(s73.sample_target(len(test)).tolist()) | set(pilot_rows.tolist()) | set(target74_rows.tolist())
    cand_rows = [i for i in range(len(test)) if i not in excl74]
    evals = dev + [test[i] for i in pilot_rows] + [test[i] for i in target74_rows]
    keep, _ = s74.near_duplicates([test[i] for i in cand_rows], evals, tokenizer)
    train_rows = [cand_rows[k] for k in keep]
    items = [tokenizer(s74.mmlu_qa_text(test[i]), add_special_tokens=False)['input_ids'] for i in train_rows]
    qa, _, members = s74.qa_list(items)
    if s73.tensor_sha256(qa) != s76.QA_POOL_SHA256:
        raise ValueError('QA chunks differ from stage 74')
    qa_ids = sorted({int(test[train_rows[m]]['ID']) for chunk in members for m in chunk})
    trained_ids = set(json.loads(s75.LEDGER.read_text())['additional_sources']['ArabicMMLU']['stage_74']
                      ['used_for_training']['ids'])
    if not set(qa_ids) <= trained_ids:
        raise ValueError('QA chunks contain items not recorded as trained')
    # web chunks: stage-74 decontamination, stage-76 draws, then the stage-78 permutation extras
    kept_texts = pq.read_table(args.kept, columns=['text'])['text'].to_pylist()
    removed_texts = pq.ParquetFile(args.removed).read_row_groups(list(range(s73.REMOVED_ROW_GROUPS)),
                                                                  columns=['text'])['text'].to_pylist()
    kr, kc, k_drop = s73.decontaminate(*s73.first_tokens(kept_texts, tokenizer), banned)
    rr, rc, r_drop = s73.decontaminate(*s73.first_tokens(removed_texts, tokenizer), banned)
    k76, r76 = s76.web_draws(len(kc), len(rc))
    kept_list, removed_list, nk_extra, nr_extra = web_lists(k76, r76, len(kc), len(rc))
    subsets, pools, record = qa_subsets(), {}, {}
    for cell, (m, share, q) in CELLS.items():
        km, rm = cell_members(cell, kept_list, removed_list)
        pools[cell] = torch.cat([qa[subsets[q]], torch.tensor([kc[i] for i in km], dtype=torch.long),
                                 torch.tensor([rc[i] for i in rm], dtype=torch.long)])
        if len(pools[cell]) != m or round(q / m, 6) != share:
            raise ValueError('Pool size differs from the protocol')
        record[cell] = dict(M=m, share=share, qa=len(subsets[q]), kept_web=len(km), removed_web=len(rm),
                            qa_subset_sha256=hashlib.sha256(subsets[q].tobytes()).hexdigest(),
                            kept_documents=[int(kr[i]) for i in km], removed_documents=[int(rr[i]) for i in rm],
                            pool_sha256=s73.tensor_sha256(pools[cell]))
    if record['d']['pool_sha256'] != stage78_population()['levels']['1024']['pool_sha256']:
        raise ValueError('Cell d pool differs from the stage-78 Q1024 pool')
    # target: new rows outside every exposed or trained item, clean under (a)-(e) on all four pools
    excluded = excluded_rows(test)
    if len(excluded) != EXPOSED_ROWS:
        raise ValueError(f'{len(excluded)} excluded rows, expected {EXPOSED_ROWS}')
    counts, eligible = s75.target_candidates(test, tokenizer, [pools[c] for c in CELLS], trained_ids, excluded)
    if len(eligible) < TARGET_ITEMS:
        raise SystemExit('Fewer than 512 eligible target rows')
    rows = np.asarray(eligible)
    rows = np.sort(rows[np.random.default_rng(TARGET_SEED).choice(len(rows), TARGET_ITEMS, replace=False)])
    target = s73.build_items([test[i] for i in rows], tokenizer)
    population = dict(qa_pool_sha256=s76.QA_POOL_SHA256, qa_item_ids=len(qa_ids), web_eligible=[len(kc), len(rc)],
                      web_decontaminated=[k_drop, r_drop], web_extras=[nk_extra, nr_extra], cells=record,
                      trained_ids=len(trained_ids), excluded_evaluated_rows=len(excluded), target_pool=counts,
                      target_items=[it['id'] for it in target])
    return probe, target, pools, population


# ---------------------------------------------------------------- schedules

def schedule(seed, m):
    """Per seed and pool size: rand, subsetN, once, in this order from default_rng(seed)."""
    rng = np.random.default_rng(seed)
    rand = rng.integers(m, size=BUDGET)
    subset_n = np.sort(rng.choice(m, N, replace=False))
    once = rng.permutation(N)
    digest = hashlib.sha256(b''.join(a.tobytes() for a in (rand, subset_n, once))).hexdigest()
    return dict(rand=rand, subset_n=subset_n, once=once, sha256=digest)


def schedule_digests(seeds):
    return {str(seed): {str(m): schedule(seed, m)['sha256'] for m in SIZES} for seed in seeds}


def rotate(seq, i):
    return tuple(seq[i % len(seq):]) + tuple(seq[:i % len(seq)])


def run_grid(seeds):
    return [(seed, cell, arm) for i, seed in enumerate(seeds) for cell in rotate(tuple(CELLS), i)
            for arm in rotate(ARMS, i)]


def training_order(arm, sch, m, kept=None):
    if arm == 'random':
        return sch['rand'], np.arange(m)
    members = dict(align05_once=kept, random_once=sch['subset_n'])[arm]
    return members[sch['once']], members


def updates(arm):
    return BUDGET if arm == 'random' else N


def checkpoints(arm, cell, b_eq):
    """Random: 0, N and 3000, and B_eq(4096) in the M = 4096 cells; the once arms: N."""
    if arm != 'random':
        return {N}
    return {0, N, BUDGET} | ({b_eq} if CELLS[cell][0] == 4096 else set())


def views(order, members, steps):
    counts = np.bincount(np.asarray(order[:steps]), minlength=int(np.max(members)) + 1)[np.asarray(members)]
    return dict(max=int(counts.max()), mean=steps / len(members))


def qa_share(indices, q):
    return float(np.mean(np.asarray(indices) < q))


def qa_count(indices, q):
    return int((np.asarray(indices) < q).sum())


def random_qa(rand, steps, q):
    first = np.asarray(rand[:steps])
    return dict(drawn=int((first < q).sum()), distinct=int(len(set(first[first < q].tolist()))))


def calibrate(c_r, s8192, s4096):
    """N_cal from S(8192) and B_eq(4096) = N + ceil(S(4096) / c_r); abort conditions are checked by the caller."""
    return int(np.floor(BUDGET - s8192 / c_r + 1e-9)), int(N + math.ceil(s4096 / c_r))


def check_calibration_bounds(n_cal, b_eq):
    if not N_BOUNDS[0] <= n_cal <= N_BOUNDS[1]:
        raise SystemExit(f'N_cal = {n_cal} outside {N_BOUNDS}: the fixed N no longer matches the hardware; '
                         'abort before any run')
    if not b_eq < BUDGET:
        raise SystemExit(f'B_eq(4096) = {b_eq} not below {BUDGET}: abort before any run')


# ---------------------------------------------------------------- summary

def bootstrap(values, q=SECONDARY_Q):
    draws = np.random.default_rng(BOOT_SEED).choice(values, (10000, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, list(q))]


def contrast(values, q=SECONDARY_Q):
    values = [float(v) for v in values]
    key = 'ci975' if q == PRIMARY_Q else 'ci95'
    return {'mean': mean(values), key: bootstrap(values, q), 'per_seed': values,
            'seeds_negative': sum(v < 0 for v in values), 'seeds_positive': sum(v > 0 for v in values)}


def e_count(x):
    """1/2 [(x(c) - x(a)) + (x(d) - x(b))], per seed."""
    return [.5 * ((c - a) + (d - b)) for a, b, c, d in zip(x['a'], x['b'], x['c'], x['d'])]


def e_share(x):
    """x(c) - x(b), per seed."""
    return [c - b for b, c in zip(x['b'], x['c'])]


def e_share_fixed_m(x):
    """1/2 [(x(b) - x(a)) + (x(d) - x(c))], per seed (secondary 9, descriptive)."""
    return [.5 * ((b - a) + (d - c)) for a, b, c, d in zip(x['a'], x['b'], x['c'], x['d'])]


def verdict(ci):
    return 'supported' if ci[1] < 0 else 'refuted' if ci[0] > 0 else 'not established'


def summarize(rows, calibration, plan):
    n, B = str(N), str(BUDGET)
    beq = str(calibration['B_eq_4096'])
    seeds = plan['design']['seeds']
    g = {(c, arm): sorted([r for r in rows if r['cell'] == c and r['arm'] == arm], key=lambda r: r['seed'])
         for c in CELLS for arm in ARMS}
    for key, rs in g.items():
        if [r['seed'] for r in rs] != seeds:
            raise ValueError(f'Incomplete grid for {key}')
    zero = {r['losses']['0'] for c in CELLS for r in g[(c, 'random')]}
    if len(zero) != 1:
        raise ValueError('Random runs do not share the initial state')
    val = lambda rs, k, key='losses': [r[key][k] for r in rs]
    sub = lambda a, b: [x - y for x, y in zip(a, b)]
    per = {}
    for c, (m, _, q) in CELLS.items():
        R, A, RO = (g[(c, a)] for a in ARMS)
        p = dict(A=val(A, n), R_B=val(R, B), R_N=val(R, n), R_0=val(R, '0'), RO=val(RO, n),
                 A_acc=val(A, n, 'accuracy'), R_B_acc=val(R, B, 'accuracy'), RO_acc=val(RO, n, 'accuracy'))
        p['D'] = sub(p['A'], p['R_B'])
        p['S'] = sub(p['A'], p['RO'])
        p['sel'] = sub(p['A'], p['R_N'])
        p['D_acc'] = sub(p['A_acc'], p['R_B_acc'])
        p['S_acc'] = sub(p['A_acc'], p['RO_acc'])
        if m == 4096:
            p['R_eq'] = val(R, beq)
            p['R_eq_acc'] = val(R, beq, 'accuracy')
            p['D_eq'] = sub(p['A'], p['R_eq'])
            p['D_eq_acc'] = sub(p['A_acc'], p['R_eq_acc'])
        else:
            p['D_eq'], p['D_eq_acc'] = p['D'], p['D_acc']  # B_eq(8192) is taken as B = 3000
        p['negR_B'] = [-v for v in p['R_B']]
        ks = [n, B] + ([beq] if m == 4096 else [])
        p['info'] = dict(
            M=m, Q=q,
            means={'random': {k: mean(val(R, k)) for k in ['0'] + ks}, 'align05_once': mean(p['A']),
                   'random_once': mean(p['RO'])},
            accuracy_means={'random': {k: mean(val(R, k, 'accuracy')) for k in ['0'] + ks},
                            'align05_once': mean(p['A_acc']), 'random_once': mean(p['RO_acc'])},
            random_late=mean(sub(p['R_B'], p['R_N'])), random_minus_no_update=contrast(sub(p['R_B'], p['R_0'])),
            kept_qa={'share': mean(r['qa_share'] for r in A), 'count': mean(r['qa_count'] for r in A),
                     'count_range': [min(r['qa_count'] for r in A), max(r['qa_count'] for r in A)],
                     'count_over_Q': mean(r['qa_count'] for r in A) / q},
            random_once_qa={'share': mean(r['qa_share'] for r in RO), 'count': mean(r['qa_count'] for r in RO)},
            random_qa={k: {f: mean(r['random_qa'][k][f] for r in R) for f in ('drawn', 'distinct')} for k in ks},
            scoring_seconds=mean(r['scoring_seconds'] for r in A))
        per[c] = p
    by = lambda key: {c: per[c][key] for c in CELLS}
    primary = {'E_count': contrast(e_count(by('D')), PRIMARY_Q), 'E_share': contrast(e_share(by('D')), PRIMARY_Q)}
    for k in primary:
        primary[k]['verdict'] = verdict(primary[k]['ci975'])
    cells = {c: dict(charged=contrast(p['D']), matched=contrast(p['S']), selection_term=mean(p['sel']),
                     throughput_term=mean(p['D']) - mean(p['sel']), accuracy_charged=contrast(p['D_acc']),
                     accuracy_matched=contrast(p['S_acc']),
                     **({'charged_exact_equal_cost': contrast(p['D_eq']),
                         'accuracy_charged_exact_equal_cost': contrast(p['D_eq_acc'])} if CELLS[c][0] == 4096 else {}),
                     **p['info'])
             for c, p in per.items()}
    both = lambda f, key: f(by(key))
    secondary = dict(
        exact_equal_cost={'E_count': contrast(both(e_count, 'D_eq')), 'E_share': contrast(both(e_share, 'D_eq'))},
        decomposition={'E_count': {'filter_part': contrast(both(e_count, 'A')),
                                   'random_part': contrast(both(e_count, 'negR_B'))},
                       'E_share': {'filter_part': contrast(both(e_share, 'A')),
                                   'random_part': contrast(both(e_share, 'negR_B'))}},
        matched={'E_count': contrast(both(e_count, 'S')), 'E_share': contrast(both(e_share, 'S'))},
        accuracy={'E_count': contrast(both(e_count, 'D_acc')), 'E_share': contrast(both(e_share, 'D_acc'))},
        share_effect_fixed_pool_size=contrast(both(e_share_fixed_m, 'D')),
        replication_d_vs_stage78={'D_d': contrast(per['d']['D']), 'stage78_D1024': STAGE78_D1024})
    m = {c: cells[c]['charged'] for c in CELLS}
    dec = secondary['decomposition']['E_share']
    holds = {
        'E_count < 0 with 97.5% interval below 0': primary['E_count']['ci975'][1] < 0,
        'E_share < 0 with 97.5% interval below 0': primary['E_share']['ci975'][1] < 0,
        'D(cell) < 0 in all four cells, each 95% interval below 0': all(m[c]['ci95'][1] < 0 for c in CELLS),
        'kept QA count > 0.85 x Q in every cell': all(cells[c]['kept_qa']['count_over_Q'] > .85 for c in CELLS),
        'E_share decomposition: |random part| > |filter part|':
            abs(dec['random_part']['mean']) > abs(dec['filter_part']['mean']),
        'D(d) < 0': m['d']['mean'] < 0}
    return dict(primary={k: dict(mean=v['mean'], ci975=v['ci975'], verdict=v['verdict']) for k, v in primary.items()},
                prediction_holds=holds, costs=calibration['costs'], N=N, N_cal=calibration['N_cal'],
                B_eq_4096=calibration['B_eq_4096'], device=calibration['device'],
                results=dict(loss0=zero.pop(), primary=primary, cells=cells, secondary=secondary))


# ---------------------------------------------------------------- backup

def backup(journal_path, log_path, worktree, branch, run=subprocess.run):
    """Copy the journal (and run log) to the journal branch, commit and push; a failed push logs git's stderr and is
    retried once. Failures are reported, not fatal (as stages 76-78)."""
    if worktree is None:
        return
    try:
        dest = worktree / INTERIM
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(journal_path, dest / 'journal.jsonl')
        if log_path is not None and log_path.exists():
            shutil.copy2(log_path, dest / 'run.log')
        runs = sum(r.get('kind') == 'run' for r in s69.read_journal(journal_path))
        stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        git = lambda *a: run(['git', '-C', str(worktree), *a], check=True, capture_output=True, text=True)
        git('add', INTERIM)
        if run(['git', '-C', str(worktree), 'diff', '--cached', '--quiet']).returncode == 0:
            return
        git('commit', '-q', '-m', f'Stage 79 interim journal snapshot: {runs} finished runs ({stamp})')
    except Exception as exc:
        print(f'BACKUP FAILED (commit): {type(exc).__name__}: {getattr(exc, "stderr", "") or exc}', flush=True)
        return
    for attempt in (1, 2):
        try:
            git('push', '-q', 'origin', f'HEAD:{branch}')
            print(f'backup pushed: {runs} finished runs' + (' (retry)' if attempt == 2 else ''), flush=True)
            return
        except Exception as exc:
            print(f'BACKUP FAILED (push attempt {attempt}): {type(exc).__name__}: '
                  f'{(getattr(exc, "stderr", "") or str(exc)).strip()}', flush=True)


# ---------------------------------------------------------------- main

def check_protocol(plan):
    d = plan['design']
    cells = {c: (v['M'], v['share'], v['Q']) for c, v in d['cells'].items()}
    if (plan['stage'] != 79 or tuple(d['arms']) != ARMS or cells != CELLS or d['N'] != N or d['B'] != BUDGET):
        raise ValueError('Protocol differs from the implementation')


def main():
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--kept', type=Path, required=True)
    parser.add_argument('--removed', type=Path, required=True)
    parser.add_argument('--mmlu', type=Path, required=True)
    parser.add_argument('--run-log', type=Path, default=None)
    parser.add_argument('--backup-worktree', type=Path, default=None)
    parser.add_argument('--backup-branch', default='colab/stage79-journal')
    parser.add_argument('--summary', type=Path, default=Path('results/llm-count-share-filter.json'))
    args = parser.parse_args()
    plan = json.loads(PROTOCOL.read_text())
    check_protocol(plan)
    import torch
    import transformers
    from transformers import AutoTokenizer
    s69.setup_numerics()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'
    specs = json.loads(s69.PROTOCOL.read_text())['models']
    model_dir, proxy_dir = args.models_dir / MODEL, args.models_dir / PROXY
    for name, d in ((MODEL, model_dir), (PROXY, proxy_dir)):
        if s69.sha256(d / 'model.safetensors') != specs[name]['safetensors_sha256']:
            raise ValueError(f'{name} weights differ from the protocol')
    for f in ('tokenizer.json', 'vocab.json', 'merges.txt'):
        if s69.sha256(proxy_dir / f) != s69.sha256(model_dir / f):
            raise ValueError('Proxy tokenizer differs')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    seeds = plan['design']['seeds']
    probe, target, pools, population = load_inputs(args, tokenizer)
    population['schedules'] = schedule_digests(seeds)
    pools = {c: p.to(device) for c, p in pools.items()}
    env = dict(kind='environment', time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), torch=torch.__version__,
               transformers=transformers.__version__, tokenizers=__import__('tokenizers').__version__,
               cuda=torch.version.cuda, python=platform.python_version(), device=device_name)
    journal = s69.read_journal(args.journal)
    args.journal.parent.mkdir(parents=True, exist_ok=True)

    def free():
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    def proxy_score(seed, pool):
        proxy = s69.load(proxy_dir, seed, device)
        scores, seconds = s73.score(proxy, 'align_grad', pool, probe, device)
        del proxy
        free()
        return scores, seconds

    with args.journal.open('a') as out:
        def log(row):
            out.write(json.dumps(row, allow_nan=False) + '\n')
            out.flush()
            os.fsync(out.fileno())
        log(dict(env, resumed_runs=sum(r.get('kind') == 'run' for r in journal)))
        done = {(r['seed'], r['cell'], r['arm']) for r in journal if r.get('kind') == 'run'}
        calibration = next((r for r in journal if r.get('kind') == 'calibration'), None)
        if calibration is None:
            rand = np.random.default_rng(CAL_SEED).integers(CELLS['d'][0], size=24)
            row = s73.train(s69.load(model_dir, CAL_SEED, device), pools['d'], rand, 24, set(), None, device)
            costs = dict(random=median(row['step_seconds'][4:]))
            free()
            _, costs['proxy_align_grad_8192'] = proxy_score(CAL_SEED, pools['d'])
            _, costs['proxy_align_grad_4096'] = proxy_score(CAL_SEED, pools['b'])
            n_cal, b_eq = calibrate(costs['random'], costs['proxy_align_grad_8192'], costs['proxy_align_grad_4096'])
            calibration = dict(kind='calibration', model=MODEL, proxy=PROXY, device=device_name, costs=costs,
                               N_cal=n_cal, B_eq_4096=b_eq, updates=N, population=population)
            log(calibration)
            backup(args.journal, args.run_log, args.backup_worktree, args.backup_branch)
        elif not calibration['device'].startswith('NVIDIA L4') or calibration['device'] != device_name:
            raise ValueError('Resumed on another device type')
        elif calibration['population'] != population:
            raise ValueError('Resumed with different samples, pools, target or schedules')
        check_calibration_bounds(calibration['N_cal'], calibration['B_eq_4096'])
        b_eq = calibration['B_eq_4096']
        print(MODEL, PROXY, device_name, 'N', N, 'N_cal', calibration['N_cal'], 'B_eq(4096)', b_eq,
              flush=True)  # costs only
        ev = lambda mdl: s73.evaluate(mdl, target, device)
        grid = run_grid(seeds)
        for seed, cell, arm in grid:
            if (seed, cell, arm) in done:
                continue
            m, _, q = CELLS[cell]
            sch = schedule(seed, m)
            extra, kept = {}, None
            if arm == 'align05_once':
                scores, seconds = proxy_score(seed, pools[cell])
                kept = s73.keep_top(scores, N)
                extra = dict(scores=scores.tolist(), kept=kept.tolist(), scoring_seconds=seconds,
                             kept_sha256=hashlib.sha256(kept.tobytes()).hexdigest())
            order, members = training_order(arm, sch, m, kept)
            ckpts = checkpoints(arm, cell, b_eq)
            if arm == 'random_once':
                extra['subset_sha256'] = hashlib.sha256(np.asarray(members).tobytes()).hexdigest()
            if arm == 'random':
                extra['random_qa'] = {str(t): random_qa(sch['rand'], t, q) for t in sorted(ckpts - {0})}
            else:
                extra['qa_share'] = qa_share(members, q)
                extra['qa_count'] = qa_count(members, q)
            extra['views'] = {str(t): views(order, members, t) for t in sorted(ckpts - {0})}
            model = s69.load(model_dir, seed, device)
            row = s73.train(model, pools[cell], order, updates(arm), ckpts, ev, device)
            row.update(extra, kind='run', model=MODEL, seed=seed, cell=cell, M=m, Q=q, arm=arm, device=device_name,
                       schedule_sha256=sch['sha256'])
            log(row)
            done.add((seed, cell, arm))
            del model
            free()
            print(f'{len(done)}/{len(grid)} runs complete', flush=True)  # outcomes are not printed
            backup(args.journal, args.run_log, args.backup_worktree, args.backup_branch)
    journal = s69.read_journal(args.journal)
    rows = [r for r in journal if r.get('kind') == 'run']
    result = dict(stage=79, protocol=plan, journal_sha256=s69.sha256(args.journal),
                  environments=[r for r in journal if r.get('kind') == 'environment'],
                  calibration={k: v for k, v in calibration.items() if k != 'population'},
                  summary=summarize(rows, calibration, plan), runs=len(rows))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('summary written to', args.summary)  # path only; outcomes are not printed


if __name__ == '__main__':
    main()

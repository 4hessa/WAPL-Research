"""Stage 80: count effect of useful (in-format QA) chunks at exact equal cost. The four stage-79 cells (pool size M in
{4096, 8192} x QA share s in {1/16, 1/8}; Q = 256, 512, 512, 1024), the stage-79 trained model (Qwen2.5-1.5B), proxy
scorer (Qwen2.5-0.5B) and budget (B = 3000 random-step costs); the one-pass filter's update count N_M is set per pool
size from this run's own calibration so that scoring plus training equals B in every cell; random vs align05_once only;
checkpoints at equal charged cost 2400, 2700, 3000; 24 seeds in two pre-registered blocks of 12; a new unexposed
512-item target.

Implements research/llm-exact-cost-count-protocol.json with the stage 73-79 building blocks.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from statistics import mean, median
import time

import numpy as np

from experiments import llm_cheap_filter as s76
from experiments import llm_count_share_filter as s79
from experiments import llm_gpu as s69
from experiments import llm_once_filter as s75
from experiments import llm_qa_mix as s74
from experiments import llm_web_filter as s73

PROTOCOL = Path('research/llm-exact-cost-count-protocol.json')
MODEL, PROXY = s79.MODEL, s79.PROXY
ARMS = ('random', 'align05_once')
CELLS = s79.CELLS
SIZES = s79.SIZES
BUDGET = 3000
COST_POINTS = (2400, 2700, 3000)
N_BOUNDS = {8192: (1417, 1731), 4096: (2041, 2495)}
SEEDS = tuple(range(38000, 38024))
BLOCKS = {1: SEEDS[:12], 2: SEEDS[12:]}
CAL_SEED, TARGET_SEED, TARGET_ITEMS, BOOT_SEED = 80999, 8022, 512, 8003
STAGE79_JOURNAL = Path('results/llm-count-share-filter-journal.jsonl')
EXPOSED_ROWS = 3840
INTERIM = 'results/stage80-interim'
CI_Q = (.025, .975)
SPIKE = 0.1
# protocol, secondary 11 (descriptive references from results/llm-count-share-filter.json)
STAGE79 = dict(D_c=-0.229, D_d=-0.279, E_count_fixed_N=-0.083, E_count_exact_equal_cost=-0.047)


# ---------------------------------------------------------------- data

def stage79_population():
    return next(r for r in s69.read_journal(STAGE79_JOURNAL) if r.get('kind') == 'calibration')['population']


def excluded_rows(test):
    """Every earlier evaluation set: the 3,328 rows excluded in stage 79 plus the stage-79 target."""
    ids79 = set(stage79_population()['target_items'])
    return s79.excluded_rows(test) | {i for i, r in enumerate(test) if int(r['ID']) in ids79}


def build_pools(qa, kc, rc, kr, rr, k76, r76):
    """The four stage-79 pools (QA 0..Q-1, then kept web, then removed web) and their record, as stage 79."""
    import torch
    kept_list, removed_list, nk_extra, nr_extra = s79.web_lists(k76, r76, len(kc), len(rc))
    subsets, pools, record = s79.qa_subsets(), {}, {}
    for cell, (m, share, q) in CELLS.items():
        km, rm = s79.cell_members(cell, kept_list, removed_list)
        pools[cell] = torch.cat([qa[subsets[q]], torch.tensor([kc[i] for i in km], dtype=torch.long),
                                 torch.tensor([rc[i] for i in rm], dtype=torch.long)])
        if len(pools[cell]) != m or round(q / m, 6) != share:
            raise ValueError('Pool size differs from the protocol')
        record[cell] = dict(M=m, share=share, qa=len(subsets[q]), kept_web=len(km), removed_web=len(rm),
                            qa_subset_sha256=hashlib.sha256(subsets[q].tobytes()).hexdigest(),
                            kept_documents=[int(kr[i]) for i in km], removed_documents=[int(rr[i]) for i in rm],
                            pool_sha256=s73.tensor_sha256(pools[cell]))
    return pools, record, (nk_extra, nr_extra)


def check_pools_equal_stage79(record, stage79_cells):
    for cell in CELLS:
        if record[cell] != stage79_cells[cell]:
            raise ValueError(f'Cell {cell} pool differs from stage 79')


def draw_target(eligible):
    rows = np.asarray(eligible)
    return np.sort(rows[np.random.default_rng(TARGET_SEED).choice(len(rows), TARGET_ITEMS, replace=False)])


def load_inputs(args, tokenizer):
    """Probe, target items, the four stage-79 pools and the population record."""
    import pandas as pd
    import pyarrow.parquet as pq
    for path, digest in ((args.kept, s73.KEPT_SHA256), (args.removed, s73.REMOVED_SHA256),
                         (args.mmlu / 'dev.csv', s73.DEV_SHA256), (args.mmlu / 'test.csv', s73.TEST_SHA256)):
        if s69.sha256(path) != digest:
            raise ValueError(f'{path} differs from the protocol')
    dev = pd.read_csv(args.mmlu / 'dev.csv').to_dict('records')
    test = pd.read_csv(args.mmlu / 'test.csv').to_dict('records')
    probe = s73.build_items(dev, tokenizer)
    # QA chunks exactly as stages 74, 76, 78 and 79
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
    kept_texts = pq.read_table(args.kept, columns=['text'])['text'].to_pylist()
    removed_texts = pq.ParquetFile(args.removed).read_row_groups(list(range(s73.REMOVED_ROW_GROUPS)),
                                                                  columns=['text'])['text'].to_pylist()
    kr, kc, k_drop = s73.decontaminate(*s73.first_tokens(kept_texts, tokenizer), banned)
    rr, rc, r_drop = s73.decontaminate(*s73.first_tokens(removed_texts, tokenizer), banned)
    k76, r76 = s76.web_draws(len(kc), len(rc))
    pools, record, extras = build_pools(qa, kc, rc, kr, rr, k76, r76)
    check_pools_equal_stage79(record, stage79_population()['cells'])
    excluded = excluded_rows(test)
    if len(excluded) != EXPOSED_ROWS:
        raise ValueError(f'{len(excluded)} excluded rows, expected {EXPOSED_ROWS}')
    counts, eligible = s75.target_candidates(test, tokenizer, [pools[c] for c in CELLS], trained_ids, excluded)
    if len(eligible) < TARGET_ITEMS:
        raise SystemExit('Fewer than 512 eligible target rows')
    rows = draw_target(eligible)
    target = s73.build_items([test[i] for i in rows], tokenizer)
    population = dict(qa_pool_sha256=s76.QA_POOL_SHA256, qa_item_ids=len(qa_ids), web_eligible=[len(kc), len(rc)],
                      web_decontaminated=[k_drop, r_drop], web_extras=list(extras),
                      cells={c: {k: v for k, v in r.items() if not k.endswith('_documents')} for c, r in record.items()},
                      trained_ids=len(trained_ids), excluded_evaluated_rows=len(excluded), target_pool=counts,
                      target_items=[it['id'] for it in target])
    return probe, target, pools, population


# ---------------------------------------------------------------- calibration and schedules

def calibrate(c_r, s8192, s4096):
    """N_M = floor(B - S(M) / c_r + 1e-9) per pool size."""
    return {8192: int(np.floor(BUDGET - s8192 / c_r + 1e-9)), 4096: int(np.floor(BUDGET - s4096 / c_r + 1e-9))}


def check_calibration_bounds(n):
    for m, (lo, hi) in N_BOUNDS.items():
        if not lo <= n[m] <= hi:
            raise SystemExit(f'N_{m} = {n[m]} outside [{lo}, {hi}]: abort before any run')
    if not n[4096] > n[8192]:
        raise SystemExit(f'N_4096 = {n[4096]} not above N_8192 = {n[8192]}: abort before any run')


def n_of(calibration):
    return {int(m): v for m, v in calibration['N'].items()}


def schedule(seed, m, n_m):
    """Per seed and pool size: rand, once, in this order from default_rng(seed)."""
    rng = np.random.default_rng(seed)
    rand = rng.integers(m, size=BUDGET)
    once = rng.permutation(n_m)
    digest = hashlib.sha256(b''.join(a.tobytes() for a in (rand, once))).hexdigest()
    return dict(rand=rand, once=once, sha256=digest)


def schedule_digests(seeds, n):
    return {str(seed): {str(m): schedule(seed, m, n[m])['sha256'] for m in SIZES} for seed in seeds}


def rotate(seq, i):
    return tuple(seq[i % len(seq):]) + tuple(seq[:i % len(seq)])


def run_grid(seeds=SEEDS):
    """Seed index i over the full seed list (0 to 23): cells rotated by i mod 4, arms by i mod 2."""
    return [(seed, cell, arm) for i, seed in enumerate(seeds) for cell in rotate(tuple(CELLS), i)
            for arm in rotate(ARMS, i)]


def block_of(seed):
    return next(b for b, s in BLOCKS.items() if seed in s)


def first_random(seed, grid=None):
    """The seed's first random run in the grid, where step 0 is evaluated."""
    return next((s, c, a) for s, c, a in (grid or run_grid()) if s == seed and a == 'random')


def training_order(arm, sch, m, kept=None):
    if arm == 'random':
        return sch['rand'], np.arange(m)
    return kept[sch['once']], kept


def updates(arm, n_m):
    return BUDGET if arm == 'random' else n_m


def cost_steps(arm, n_m):
    """Update count at each charged cost point: C for random, N_M - (B - C) for the filter."""
    return {c: (c if arm == 'random' else n_m - (BUDGET - c)) for c in COST_POINTS}


def checkpoints(arm, n_m, step0=False):
    return set(cost_steps(arm, n_m).values()) | ({0} if step0 and arm == 'random' else set())


views, qa_share, qa_count, random_qa = s79.views, s79.qa_share, s79.qa_count, s79.random_qa


# ---------------------------------------------------------------- summary

def bootstrap(values, q=CI_Q):
    draws = np.random.default_rng(BOOT_SEED).choice(values, (10000, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, list(q))]


def contrast(values):
    values = [float(v) for v in values]
    return {'mean': mean(values), 'ci95': bootstrap(values), 'per_seed': values,
            'seeds_negative': sum(v < 0 for v in values), 'seeds_positive': sum(v > 0 for v in values)}


e_count, e_share, verdict = s79.e_count, s79.e_share, s79.verdict


def at_cost(row, arm, n_m, c, key='losses'):
    return row[key][str(cost_steps(arm, n_m)[c])]


def spike(row, arm, n_m):
    lo, mid, hi = (at_cost(row, arm, n_m, c) for c in COST_POINTS)
    return mid - lo > SPIKE and mid - hi > SPIKE


def summarize(rows, calibration, seeds=SEEDS):
    n = n_of(calibration)
    g = {(c, arm): sorted([r for r in rows if r['cell'] == c and r['arm'] == arm], key=lambda r: r['seed'])
         for c in CELLS for arm in ARMS}
    for key, rs in g.items():
        if [r['seed'] for r in rs] != list(seeds):
            raise ValueError(f'Incomplete grid for {key}')
    zero = sorted({r['losses']['0'] for r in rows if '0' in r['losses']})
    zero_seeds = sorted({r['seed'] for r in rows if '0' in r['losses']})
    if not set(seeds) <= set(zero_seeds):
        raise ValueError('Step 0 not evaluated for every seed')
    sub = lambda a, b: [x - y for x, y in zip(a, b)]
    per, info = {}, {}
    for c, (m, _, q) in CELLS.items():
        R, A = g[(c, 'random')], g[(c, 'align05_once')]
        p = {}
        for cp in COST_POINTS:
            for key, suffix in (('losses', ''), ('accuracy', '_acc')):
                p[f'A{cp}{suffix}'] = [at_cost(r, 'align05_once', n[m], cp, key) for r in A]
                p[f'R{cp}{suffix}'] = [at_cost(r, 'random', n[m], cp, key) for r in R]
            p[f'D{cp}'] = sub(p[f'A{cp}'], p[f'R{cp}'])
        p['A'], p['R'], p['D'] = p['A3000'], p['R3000'], p['D3000']
        p['D_acc'] = sub(p['A3000_acc'], p['R3000_acc'])
        p['A_avg'] = [mean(v) for v in zip(*(p[f'A{cp}'] for cp in COST_POINTS))]
        p['R_avg'] = [mean(v) for v in zip(*(p[f'R{cp}'] for cp in COST_POINTS))]
        p['D_avg'] = sub(p['A_avg'], p['R_avg'])
        p['negR'] = [-v for v in p['R']]
        p['R_0'] = [r['losses']['0'] for r in R if '0' in r['losses']]
        realized = [r['scoring_seconds'] + sum(r['step_seconds']) for r in A]
        info[c] = dict(
            M=m, Q=q, N_M=n[m], filter_steps={str(k): v for k, v in cost_steps('align05_once', n[m]).items()},
            means={arm: {str(cp): mean(p[f'{arm[0].upper()}{cp}']) for cp in COST_POINTS}
                   for arm in ('align05_once', 'random')},
            accuracy_means={arm: {str(cp): mean(p[f'{arm[0].upper()}{cp}_acc']) for cp in COST_POINTS}
                            for arm in ('align05_once', 'random')},
            random_minus_no_update=contrast([r['losses']['3000'] - zero[0] for r in R]) if len(zero) == 1 else None,
            kept_qa={'share': mean(r['qa_share'] for r in A), 'count': mean(r['qa_count'] for r in A),
                     'count_range': [min(r['qa_count'] for r in A), max(r['qa_count'] for r in A)],
                     'count_over_Q': mean(r['qa_count'] for r in A) / q},
            random_qa={str(cp): {f: mean(r['random_qa'][str(cp)][f] for r in R) for f in ('drawn', 'distinct')}
                       for cp in COST_POINTS},
            scoring_seconds=mean(r['scoring_seconds'] for r in A),
            realized_cost_seconds=dict(filter=mean(realized), random=mean(sum(r['step_seconds']) for r in R),
                                       filter_minus_random=mean(sub(realized, [sum(r['step_seconds']) for r in R]))),
            spikes={arm: sum(spike(r, arm, n[m]) for r in g[(c, arm)]) for arm in ARMS})
        per[c] = p
    by = lambda key: {c: per[c][key] for c in CELLS}
    primary = contrast(e_count(by('D')))
    primary['verdict'] = verdict(primary['ci95'])
    cells = {c: dict(charged=contrast(p['D']), accuracy_charged=contrast(p['D_acc']),
                     charged_at={str(cp): contrast(p[f'D{cp}']) for cp in COST_POINTS[:-1]},
                     charged_mean_of_cost_points=contrast(p['D_avg']), **info[c])
             for c, p in per.items()}
    both = lambda f, key: contrast(f(by(key)))
    idx = {b: [i for i, s in enumerate(seeds) if s in BLOCKS[b]] for b in BLOCKS}
    per_seed = e_count(by('D'))
    secondary = dict(
        E_share=both(e_share, 'D'),
        decomposition={'E_count': {'filter_part': both(e_count, 'A'), 'random_part': both(e_count, 'negR')},
                       'E_share': {'filter_part': both(e_share, 'A'), 'random_part': both(e_share, 'negR')}},
        checkpoint_robust={'mean_of_cost_points': {'E_count': both(e_count, 'D_avg'),
                                                   'E_share': both(e_share, 'D_avg')},
                           **{str(cp): {'E_count': both(e_count, f'D{cp}'), 'E_share': both(e_share, f'D{cp}')}
                              for cp in COST_POINTS[:-1]}},
        accuracy={'E_count': both(e_count, 'D_acc'), 'E_share': both(e_share, 'D_acc')},
        blocks={str(b): mean(per_seed[i] for i in ix) for b, ix in idx.items() if ix},
        stage79_comparison=dict(D_c=mean(per['c']['D']), D_d=mean(per['d']['D']), E_count=primary['mean'],
                                stage79=STAGE79))
    dec = secondary['decomposition']['E_count']
    holds = {
        'E_count < 0 with 95% interval below 0': primary['ci95'][1] < 0,
        'D(cell) < 0 in all four cells, each 95% interval below 0': all(cells[c]['charged']['ci95'][1] < 0
                                                                       for c in CELLS),
        'kept QA count > 0.85 x Q in every cell': all(cells[c]['kept_qa']['count_over_Q'] > .85 for c in CELLS),
        'E_count decomposition: |filter part| > |random part|':
            abs(dec['filter_part']['mean']) > abs(dec['random_part']['mean'])}
    return dict(primary=dict(E_count=dict(mean=primary['mean'], ci95=primary['ci95'], verdict=primary['verdict'])),
                prediction_holds=holds, costs=calibration['costs'], N=calibration['N'], device=calibration['device'],
                results=dict(loss0=dict(values=zero, identical=len(zero) == 1, seeds=zero_seeds),
                             primary=dict(E_count=primary), cells=cells, secondary=secondary))


# ---------------------------------------------------------------- backup

def backup(journal_path, log_path, worktree, branch, run=subprocess.run):
    """Copy the journal (and run log) to the journal branch, commit and push; a failed push is retried once.
    Failures are reported, not fatal (as stages 76-79)."""
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
        git('commit', '-q', '-m', f'Stage 80 interim journal snapshot: {runs} finished runs ({stamp})')
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
    if (plan['stage'] != 80 or tuple(d['arms']) != ARMS or cells != CELLS or d['B'] != BUDGET
            or '38000 to 38023' not in d['seeds'] or 'seeds 38000-38011' not in plan['blocks']['definition']
            or 'seeds 38012-38023' not in plan['blocks']['definition'] or '{2400, 2700, 3000}' not in d['cost_points']
            or f'seed {CAL_SEED}' not in d['calibration'] or f'default_rng({TARGET_SEED})' not in plan['target']['items']
            or f'seed {BOOT_SEED}' not in plan['primary']['definition']):
        raise ValueError('Protocol differs from the implementation')


def check_block(block, done):
    """Block 2 starts only after every block-1 run is in the journal."""
    if block == 2:
        missing = [k for k in run_grid() if block_of(k[0]) == 1 and k not in done]
        if missing:
            raise SystemExit(f'Block 1 incomplete ({len(missing)} runs missing): block 2 cannot start')


def gpu_uuid(torch):
    try:
        return str(torch.cuda.get_device_properties(0).uuid)
    except Exception:
        return None


def driver_version():
    try:
        return subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def main():
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block', type=int, choices=(1, 2), required=True)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--kept', type=Path, required=True)
    parser.add_argument('--removed', type=Path, required=True)
    parser.add_argument('--mmlu', type=Path, required=True)
    parser.add_argument('--run-log', type=Path, default=None)
    parser.add_argument('--backup-worktree', type=Path, default=None)
    parser.add_argument('--backup-branch', default='colab/stage80-journal')
    parser.add_argument('--summary', type=Path, default=Path('results/llm-exact-cost-count.json'))
    args = parser.parse_args()
    plan = json.loads(PROTOCOL.read_text())
    check_protocol(plan)
    import torch
    import transformers
    from transformers import AutoTokenizer
    s69.setup_numerics()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'
    if not device_name.startswith('NVIDIA L4'):
        raise SystemExit(f'{device_name} is not an NVIDIA L4: abort')
    specs = json.loads(s69.PROTOCOL.read_text())['models']
    model_dir, proxy_dir = args.models_dir / MODEL, args.models_dir / PROXY
    for name, d in ((MODEL, model_dir), (PROXY, proxy_dir)):
        if s69.sha256(d / 'model.safetensors') != specs[name]['safetensors_sha256']:
            raise ValueError(f'{name} weights differ from the protocol')
    for f in ('tokenizer.json', 'vocab.json', 'merges.txt'):
        if s69.sha256(proxy_dir / f) != s69.sha256(model_dir / f):
            raise ValueError('Proxy tokenizer differs')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    probe, target, pools, population = load_inputs(args, tokenizer)
    pools = {c: p.to(device) for c, p in pools.items()}
    uuid = gpu_uuid(torch)
    env = dict(kind='environment', time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), torch=torch.__version__,
               transformers=transformers.__version__, tokenizers=__import__('tokenizers').__version__,
               cuda=torch.version.cuda, python=platform.python_version(), device=device_name, gpu_uuid=uuid,
               driver=driver_version(),
               block=args.block)
    journal = s69.read_journal(args.journal)
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    grid = run_grid()

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

    def time_random(seed):
        rand = np.random.default_rng(seed).integers(CELLS['d'][0], size=24)
        row = s73.train(s69.load(model_dir, seed, device), pools['d'], rand, 24, set(), None, device)
        free()
        return median(row['step_seconds'][4:])

    with args.journal.open('a') as out:
        def log(row):
            out.write(json.dumps(row, allow_nan=False) + '\n')
            out.flush()
            os.fsync(out.fileno())
        new_machine = uuid not in {r.get('gpu_uuid') for r in journal if r.get('kind') == 'environment'}
        log(dict(env, resumed_runs=sum(r.get('kind') == 'run' for r in journal)))
        done = {(r['seed'], r['cell'], r['arm']) for r in journal if r.get('kind') == 'run'}
        check_block(args.block, done)
        calibration = next((r for r in journal if r.get('kind') == 'calibration'), None)
        if calibration is None:
            if args.block != 1:
                raise SystemExit('No calibration in the journal: block 1 runs first')
            costs = dict(random=time_random(CAL_SEED))
            _, costs['proxy_align_grad_8192'] = proxy_score(CAL_SEED, pools['d'])
            _, costs['proxy_align_grad_4096'] = proxy_score(CAL_SEED, pools['b'])
            n = calibrate(costs['random'], costs['proxy_align_grad_8192'], costs['proxy_align_grad_4096'])
            calibration = dict(kind='calibration', model=MODEL, proxy=PROXY, device=device_name, costs=costs,
                               N={str(m): v for m, v in n.items()}, population=population,
                               schedules=schedule_digests(SEEDS, n))
            log(calibration)  # journaled before the bounds check, so an abort is recorded
            backup(args.journal, args.run_log, args.backup_worktree, args.backup_branch)
            check_calibration_bounds(n)
        else:
            if calibration['population'] != population:
                raise ValueError('Resumed with different samples, pools or target')
            n = n_of(calibration)
            check_calibration_bounds(n)
            if calibration['schedules'] != schedule_digests(SEEDS, n):
                raise ValueError('Schedules differ from the journal')
            if new_machine:  # descriptive only; never changes N
                log(dict(kind='retiming', gpu_uuid=uuid, device=device_name, random=time_random(CAL_SEED)))
        print(MODEL, PROXY, device_name, 'block', args.block, 'N', calibration['N'], flush=True)  # costs only
        ev = lambda mdl: s73.evaluate(mdl, target, device)
        session_check = new_machine and bool(done)  # step 0 again for the first seed run on a new machine
        todo = [k for k in grid if block_of(k[0]) == args.block and k not in done]
        for seed, cell, arm in todo:
            m, _, q = CELLS[cell]
            sch = schedule(seed, m, n[m])
            extra, kept = {}, None
            if arm == 'align05_once':
                scores, seconds = proxy_score(seed, pools[cell])
                kept = s73.keep_top(scores, n[m])
                extra = dict(scores=scores.tolist(), kept=kept.tolist(), scoring_seconds=seconds,
                             kept_sha256=hashlib.sha256(kept.tobytes()).hexdigest())
            order, members = training_order(arm, sch, m, kept)
            step0 = arm == 'random' and ((seed, cell, arm) == first_random(seed, grid) or session_check)
            ckpts = checkpoints(arm, n[m], step0)
            if arm == 'random':
                extra['random_qa'] = {str(t): random_qa(sch['rand'], t, q) for t in COST_POINTS}
            else:
                extra['qa_share'] = qa_share(members, q)
                extra['qa_count'] = qa_count(members, q)
            extra['views'] = {str(t): views(order, members, t) for t in sorted(ckpts - {0})}
            model = s69.load(model_dir, seed, device)
            row = s73.train(model, pools[cell], order, updates(arm, n[m]), ckpts, ev, device)
            row.update(extra, kind='run', model=MODEL, seed=seed, cell=cell, M=m, Q=q, arm=arm, N_M=n[m],
                       block=args.block, device=device_name, gpu_uuid=uuid, schedule_sha256=sch['sha256'])
            log(row)
            done.add((seed, cell, arm))
            if step0:
                session_check = False
                zeros = {r['losses']['0'] for r in s69.read_journal(args.journal)
                         if r.get('kind') == 'run' and '0' in r['losses']}
                print('step-0 loss identical across journaled runs:', len(zeros) == 1, flush=True)  # check only
            del model
            free()
            print(f'{len(done)}/{len(grid)} runs complete', flush=True)  # outcomes are not printed
            backup(args.journal, args.run_log, args.backup_worktree, args.backup_branch)
    if len(done) < len(grid):
        print(f'block {args.block} complete; no summary until all {len(grid)} runs are in the journal')
        return
    journal = s69.read_journal(args.journal)
    rows = [r for r in journal if r.get('kind') == 'run']
    result = dict(stage=80, protocol=plan, journal_sha256=s69.sha256(args.journal),
                  environments=[r for r in journal if r.get('kind') == 'environment'],
                  retimings=[r for r in journal if r.get('kind') == 'retiming'],
                  calibration={k: v for k, v in calibration.items() if k not in ('population', 'schedules')},
                  summary=summarize(rows, calibration), runs=len(rows))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('summary written to', args.summary)  # path only; outcomes are not printed


if __name__ == '__main__':
    main()

"""Stage 78: useful QA supply Q in {1024, 1633, 2048} at fixed pool size (8192), proxy scorer (Qwen2.5-0.5B), budget
(3000 random-step costs) and number of filter updates (N = 1633); one-pass offline filter vs random at equal cost and
a repetition-matched random subset, on a new unexposed 512-item target (amendment 1).

Implements research/llm-supply-filter-protocol.json with the stage 73-77 building blocks.
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
from experiments import llm_deepseek_filter as s77
from experiments import llm_gpu as s69
from experiments import llm_once_filter as s75
from experiments import llm_qa_mix as s74
from experiments import llm_web_filter as s73

PROTOCOL = Path('research/llm-supply-filter-protocol.json')
MODEL, PROXY = 'qwen2.5-1.5b', 'qwen2.5-0.5b'
ARMS = ('random', 'align05_once', 'random_once')
LEVELS = (1024, 1633, 2048)
BUDGET, POOL_SIZE, N, N_QA = 3000, 8192, 1633, 2048
N_BOUNDS = (1470, 1796)
CAL_SEED, TARGET_SEED, TARGET_ITEMS = 78999, 7822, 512
QA_SEEDS, EXTRA_SEEDS = (7801, 7802), (7811, 7812)
STAGE76_POOL_SHA256 = 'f4d4285c96a3b32fdb5123cc3393ce9cae47b8dd20135729a580bbd24c932a83'
STAGE77_JOURNAL = Path('results/llm-deepseek-filter-journal.jsonl')
EXPOSED_ROWS = 2816
INTERIM = 'results/stage78-interim'


# ---------------------------------------------------------------- data

def qa_subsets():
    """Nested QA-chunk index sets (positions in the 2048 stage-74/76 QA chunks), sorted."""
    q1633 = np.sort(np.random.default_rng(QA_SEEDS[0]).choice(N_QA, 1633, replace=False))
    q1024 = np.sort(q1633[np.random.default_rng(QA_SEEDS[1]).choice(1633, 1024, replace=False)])
    return {1024: q1024, 1633: q1633, 2048: np.arange(N_QA)}


def web_counts(q):
    w = POOL_SIZE - q
    return math.ceil(w / 2), w // 2


def extra_draws(n_eligible, drawn, seed, k):
    """First k of a seeded permutation of the eligible indices not already drawn (in list order)."""
    taken = set(int(i) for i in drawn)
    rest = np.array([i for i in range(n_eligible) if i not in taken])
    return rest[np.random.default_rng(seed).permutation(len(rest))[:k]]


def level_members(q, k76, r76, k_extra, r_extra):
    """Kept-web and removed-web index lists of level q: the stage-76 draws first, then the extras."""
    nk, nr = web_counts(q)
    return (np.concatenate([k76, k_extra[:nk - len(k76)]]).astype(int),
            np.concatenate([r76, r_extra[:nr - len(r76)]]).astype(int))


def excluded_rows(test):
    """Every earlier evaluation set: stages 73-76 (experiments.llm_deepseek_filter.excluded_rows) and the stage-77
    target."""
    ids77 = set(next(r for r in s69.read_journal(STAGE77_JOURNAL) if r.get('kind') == 'calibration')
                ['population']['target_items'])
    return s77.excluded_rows(test) | {i for i, r in enumerate(test) if int(r['ID']) in ids77}


def load_inputs(args, tokenizer):
    """Probe, target items, the three pools (QA 0..Q-1, then kept web, then removed web) and the population record."""
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
    # QA chunks exactly as stages 74 and 76
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
    # web chunks: stage-74 decontamination, stage-76 draws, then the extras
    kept_texts = pq.read_table(args.kept, columns=['text'])['text'].to_pylist()
    removed_texts = pq.ParquetFile(args.removed).read_row_groups(list(range(s73.REMOVED_ROW_GROUPS)),
                                                                  columns=['text'])['text'].to_pylist()
    kr, kc, k_drop = s73.decontaminate(*s73.first_tokens(kept_texts, tokenizer), banned)
    rr, rc, r_drop = s73.decontaminate(*s73.first_tokens(removed_texts, tokenizer), banned)
    k76, r76 = s76.web_draws(len(kc), len(rc))
    nk_max, nr_max = web_counts(min(LEVELS))
    k_extra = extra_draws(len(kc), k76, EXTRA_SEEDS[0], nk_max - len(k76))
    r_extra = extra_draws(len(rc), r76, EXTRA_SEEDS[1], nr_max - len(r76))
    subsets, pools, record = qa_subsets(), {}, {}
    for q in LEVELS:
        km, rm = level_members(q, k76, r76, k_extra, r_extra)
        pools[q] = torch.cat([qa[subsets[q]], torch.tensor([kc[i] for i in km], dtype=torch.long),
                              torch.tensor([rc[i] for i in rm], dtype=torch.long)])
        if len(pools[q]) != POOL_SIZE:
            raise ValueError('Pool size differs from the protocol')
        record[str(q)] = dict(qa=len(subsets[q]), kept_web=len(km), removed_web=len(rm),
                              qa_subset_sha256=hashlib.sha256(subsets[q].tobytes()).hexdigest(),
                              kept_documents=[int(kr[i]) for i in km], removed_documents=[int(rr[i]) for i in rm],
                              pool_sha256=s73.tensor_sha256(pools[q]))
    if record['2048']['pool_sha256'] != STAGE76_POOL_SHA256:
        raise ValueError('Q2048 pool differs from stage 76')
    # target (amendment 1): new rows outside every exposed or trained item, clean under (a)-(e) on all three pools
    excluded = excluded_rows(test)
    if len(excluded) != EXPOSED_ROWS:
        raise ValueError(f'{len(excluded)} excluded rows, expected {EXPOSED_ROWS}')
    counts, eligible = s75.target_candidates(test, tokenizer, [pools[q] for q in LEVELS], trained_ids, excluded)
    if len(eligible) < TARGET_ITEMS:
        raise SystemExit('Fewer than 512 eligible target rows')
    rows = np.asarray(eligible)
    rows = np.sort(rows[np.random.default_rng(TARGET_SEED).choice(len(rows), TARGET_ITEMS, replace=False)])
    target = s73.build_items([test[i] for i in rows], tokenizer)
    population = dict(qa_pool_sha256=s76.QA_POOL_SHA256, qa_item_ids=len(qa_ids), web_eligible=[len(kc), len(rc)],
                      web_decontaminated=[k_drop, r_drop], web_extras=[len(k_extra), len(r_extra)], levels=record,
                      trained_ids=len(trained_ids), excluded_evaluated_rows=len(excluded), target_pool=counts,
                      target_items=[it['id'] for it in target])
    return probe, target, pools, population


# ---------------------------------------------------------------- schedules

def schedule(seed):
    rng = np.random.default_rng(seed)
    rand = rng.integers(POOL_SIZE, size=BUDGET)
    subset_n = np.sort(rng.choice(POOL_SIZE, N, replace=False))
    once = rng.permutation(N)
    digest = hashlib.sha256(b''.join(a.tobytes() for a in (rand, subset_n, once))).hexdigest()
    return dict(rand=rand, subset_n=subset_n, once=once, sha256=digest)


def rotate(seq, i):
    return tuple(seq[i % len(seq):]) + tuple(seq[:i % len(seq)])


def run_grid(seeds):
    return [(seed, q, arm) for i, seed in enumerate(seeds) for q in rotate(LEVELS, i) for arm in rotate(ARMS, i)]


def training_order(arm, sch, kept=None):
    if arm == 'random':
        return sch['rand'], np.arange(POOL_SIZE)
    members = dict(align05_once=kept, random_once=sch['subset_n'])[arm]
    return members[sch['once']], members


def updates(arm):
    return BUDGET if arm == 'random' else N


def checkpoints(arm):
    return {0, N, BUDGET} if arm == 'random' else {N}


def views(order, members, steps):
    counts = np.bincount(np.asarray(order[:steps]), minlength=POOL_SIZE)[np.asarray(members)]
    return dict(max=int(counts.max()), mean=steps / len(members))


def qa_share(indices, q):
    return float(np.mean(np.asarray(indices) < q))


def random_qa(rand, steps, q):
    first = np.asarray(rand[:steps])
    return dict(drawn=int((first < q).sum()), distinct=int(len(set(first[first < q].tolist()))))


# ---------------------------------------------------------------- summary

def bootstrap(values):
    draws = np.random.default_rng(7803).choice(values, (10000, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [.025, .975])]


def contrast(values):
    return dict(mean=mean(values), ci95=bootstrap(values), per_seed=values, seeds_negative=sum(v < 0 for v in values),
                seeds_positive=sum(v > 0 for v in values))


def summarize(rows, calibration, plan):
    n, B = str(N), str(BUDGET)
    seeds = plan['design']['seeds']
    g = {(q, arm): sorted([r for r in rows if r['level'] == q and r['arm'] == arm], key=lambda r: r['seed'])
         for q in LEVELS for arm in ARMS}
    for key, rs in g.items():
        if [r['seed'] for r in rs] != seeds:
            raise ValueError(f'Incomplete grid for {key}')
    zero = {r['losses']['0'] for q in LEVELS for r in g[(q, 'random')]}
    if len(zero) != 1:
        raise ValueError('Random runs do not share the initial state')
    diff = lambda x, kx, y, ky, key='losses': [a[key][kx] - b[key][ky] for a, b in zip(x, y)]
    lv = {}
    for q in LEVELS:
        R, A, RO = (g[(q, a)] for a in ARMS)
        D, sel = diff(A, n, R, B), diff(A, n, R, n)
        lv[q] = dict(
            D=D, S=diff(A, n, RO, n), sel=sel, D_acc=diff(A, n, R, B, 'accuracy'),
            means={'random': {k: mean(r['losses'][k] for r in R) for k in ('0', n, B)},
                   'align05_once': mean(r['losses'][n] for r in A), 'random_once': mean(r['losses'][n] for r in RO)},
            accuracy_means={'random': {k: mean(r['accuracy'][k] for r in R) for k in ('0', n, B)},
                            'align05_once': mean(r['accuracy'][n] for r in A),
                            'random_once': mean(r['accuracy'][n] for r in RO)},
            random_late=mean(diff(R, B, R, n)), random_minus_no_update=mean(diff(R, B, R, '0')),
            qa_share={'align05_once': mean(r['qa_share'] for r in A), 'random_once': mean(r['qa_share'] for r in RO)},
            qa_share_range=[min(r['qa_share'] for r in A), max(r['qa_share'] for r in A)],
            random_qa={k: {f: mean(r['random_qa'][k][f] for r in R) for f in ('drawn', 'distinct')} for k in (n, B)},
            scoring_seconds=mean(r['scoring_seconds'] for r in A))
    sub = lambda a, b: [x - y for x, y in zip(a, b)]
    T = sub(lv[1024]['D'], lv[2048]['D'])
    out = dict(
        loss0=zero.pop(), primary=contrast(T),
        levels={str(q): dict(charged=contrast(v['D']), matched=contrast(v['S']), selection_term=mean(v['sel']),
                             throughput_term=mean(v['D']) - mean(v['sel']), accuracy_charged=contrast(v['D_acc']),
                             **{k: v[k] for k in ('means', 'accuracy_means', 'random_late', 'random_minus_no_update',
                                                  'qa_share', 'qa_share_range', 'random_qa', 'scoring_seconds')})
                for q, v in lv.items()},
        D1633_minus_D2048=contrast(sub(lv[1633]['D'], lv[2048]['D'])),
        D1024_minus_D1633=contrast(sub(lv[1024]['D'], lv[1633]['D'])),
        S1024_minus_S2048=contrast(sub(lv[1024]['S'], lv[2048]['S'])),
        T_accuracy=contrast(sub(lv[1024]['D_acc'], lv[2048]['D_acc'])))
    m = {q: mean(lv[q]['D']) for q in LEVELS}
    ci = out['primary']['ci95']
    share = {q: lv[q]['qa_share']['align05_once'] for q in LEVELS}
    holds = {'primary T > 0 with interval above 0': ci[0] > 0,
             'D(2048) < 0': m[2048] < 0, 'D(1633) < 0': m[1633] < 0,
             'kept QA share > 0.95 / 0.80 / 0.55 at 2048 / 1633 / 1024':
                 share[2048] > .95 and share[1633] > .80 and share[1024] > .55,
             'D(1024) > D(1633) > D(2048)': m[1024] > m[1633] > m[2048]}
    verdict = 'supported' if ci[0] > 0 else 'refuted' if ci[1] < 0 else 'not established'
    return dict(primary=dict(contrast='D(1024) - D(2048), D = align05_once at N minus random at 3000',
                             mean=out['primary']['mean'], ci95=ci, verdict=verdict),
                prediction_holds=holds, costs=calibration['costs'], N=N, N_cal=calibration['N_cal'],
                device=calibration['device'], results=out)


# ---------------------------------------------------------------- backup

def backup(journal_path, log_path, worktree, branch, run=subprocess.run):
    """Copy the journal (and run log) to the journal branch, commit and push; a failed push logs git's stderr and is
    retried once. Failures are reported, not fatal (as stage 76)."""
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
        git('commit', '-q', '-m', f'Stage 78 interim journal snapshot: {runs} finished runs ({stamp})')
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
    parser.add_argument('--backup-branch', default='colab/stage78-journal')
    parser.add_argument('--summary', type=Path, default=Path('results/llm-supply-filter.json'))
    args = parser.parse_args()
    plan = json.loads(PROTOCOL.read_text())
    if plan['stage'] != 78 or tuple(plan['design']['arms']) != ARMS or tuple(plan['design']['levels']) != LEVELS:
        raise ValueError('Protocol differs from the implementation')
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
    probe, target, pools, population = load_inputs(args, tokenizer)
    pools = {q: p.to(device) for q, p in pools.items()}
    env = dict(kind='environment', time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), torch=torch.__version__,
               transformers=transformers.__version__, cuda=torch.version.cuda, python=platform.python_version(),
               device=device_name)
    journal = s69.read_journal(args.journal)
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    seeds = plan['design']['seeds']

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
        done = {(r['seed'], r['level'], r['arm']) for r in journal if r.get('kind') == 'run'}
        calibration = next((r for r in journal if r.get('kind') == 'calibration'), None)
        if calibration is None:
            rand = np.random.default_rng(CAL_SEED).integers(POOL_SIZE, size=24)
            row = s73.train(s69.load(model_dir, CAL_SEED, device), pools[2048], rand, 24, set(), None, device)
            costs = dict(random=median(row['step_seconds'][4:]))
            free()
            _, costs['proxy_align_grad'] = proxy_score(CAL_SEED, pools[2048])
            calibration = dict(kind='calibration', model=MODEL, proxy=PROXY, device=device_name, costs=costs,
                               N_cal=int(np.floor(BUDGET - costs['proxy_align_grad'] / costs['random'] + 1e-9)),
                               updates=N, population=population)
            log(calibration)
            backup(args.journal, args.run_log, args.backup_worktree, args.backup_branch)
        elif calibration['device'] != device_name or calibration['population'] != population:
            raise ValueError('Resumed on another device type or with different samples')
        if not N_BOUNDS[0] <= calibration['N_cal'] <= N_BOUNDS[1]:
            raise SystemExit(f"N_cal = {calibration['N_cal']} outside {N_BOUNDS}: the fixed N no longer matches the "
                             'hardware; abort before any run')
        print(MODEL, PROXY, device_name, 'N', N, 'N_cal', calibration['N_cal'], flush=True)  # costs only
        ev = lambda m: s73.evaluate(m, target, device)
        grid = run_grid(seeds)
        for seed, q, arm in grid:
            if (seed, q, arm) in done:
                continue
            sch = schedule(seed)
            extra, kept = {}, None
            if arm == 'align05_once':
                scores, seconds = proxy_score(seed, pools[q])
                kept = s73.keep_top(scores, N)
                extra = dict(scores=scores.tolist(), kept=kept.tolist(), scoring_seconds=seconds,
                             kept_sha256=hashlib.sha256(kept.tobytes()).hexdigest())
            order, members = training_order(arm, sch, kept)
            steps = updates(arm)
            if arm == 'random_once':
                extra['subset_sha256'] = hashlib.sha256(np.asarray(members).tobytes()).hexdigest()
            if arm == 'random':
                extra['random_qa'] = {str(t): random_qa(sch['rand'], t, q) for t in (N, BUDGET)}
            else:
                extra['qa_share'] = qa_share(members, q)
            extra['views'] = {str(t): views(order, members, t) for t in sorted(checkpoints(arm) - {0})}
            model = s69.load(model_dir, seed, device)
            row = s73.train(model, pools[q], order, steps, checkpoints(arm), ev, device)
            row.update(extra, kind='run', model=MODEL, seed=seed, level=q, arm=arm, device=device_name,
                       schedule_sha256=sch['sha256'])
            log(row)
            done.add((seed, q, arm))
            del model
            free()
            print(f'{len(done)}/{len(grid)} runs complete', flush=True)  # outcomes are not printed
            backup(args.journal, args.run_log, args.backup_worktree, args.backup_branch)
    journal = s69.read_journal(args.journal)
    rows = [r for r in journal if r.get('kind') == 'run']
    result = dict(stage=78, protocol=plan, journal_sha256=s69.sha256(args.journal),
                  environments=[r for r in journal if r.get('kind') == 'environment'],
                  calibration={k: v for k, v in calibration.items() if k != 'population'},
                  summary=summarize(rows, calibration, plan), runs=len(rows))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('summary written to', args.summary)


if __name__ == '__main__':
    main()

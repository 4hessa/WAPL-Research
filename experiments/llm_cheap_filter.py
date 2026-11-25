"""Stage 76: one-pass offline filter scored by a cheap proxy (Qwen2.5-0.5B gradient alignment) on an 8192-chunk pool
at QA share 0.25, against random sampling at equal cost, a repetition-matched random subset and a QA label oracle.

Implements research/llm-cheap-filter-protocol.json with the stage 73-75 building blocks.
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

from experiments import llm_gpu as s69
from experiments import llm_once_filter as s75
from experiments import llm_qa_mix as s74
from experiments import llm_web_filter as s73

PROTOCOL = Path('research/llm-cheap-filter-protocol.json')
MODEL, PROXY = 'qwen2.5-1.5b', 'qwen2.5-0.5b'
ARMS = ('random', 'align05_once', 'random_once', 'qa_once')
BUDGET, POOL_SIZE, N_QA, N_WEB = 3000, 8192, 2048, 3072
CAL_SEED, TARGET_SEED, TARGET_ITEMS = 76999, 7622, 512
WEB_SEEDS = (7611, 7612)
WEB_ELIGIBLE = (41443, 18831)
N_RANGE = (1000, N_QA)
QA_POOL_SHA256 = '12b3a1f89b0b1c6a15eb67222ea3be6b6c39aed0c1604979373d85c20d44c24d'  # stage-74 sigma 1.0 pool
STAGE75_JOURNAL = Path('results/llm-once-filter-journal.jsonl')
INTERIM = 'results/stage76-interim'


# ---------------------------------------------------------------- data

def web_draws(n_kept, n_removed):
    if (n_kept, n_removed) != WEB_ELIGIBLE:
        raise ValueError(f'Eligible web chunks {n_kept}, {n_removed} differ from the protocol {WEB_ELIGIBLE}')
    return tuple(np.random.default_rng(s).choice(n, N_WEB, replace=False) for s, n in zip(WEB_SEEDS, WEB_ELIGIBLE))


def qa_share(indices):
    return float(np.mean(np.asarray(indices) < N_QA))


def load_inputs(args, tokenizer):
    """Probe, target items, pool (QA 0-2047, kept web 2048-5119, removed web 5120-8191) and the population record."""
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
    # QA chunks exactly as stage 74 (experiments.llm_qa_mix.load_inputs)
    banned = s74.content_banned(dev + test, tokenizer)
    pilot_rows, target74_rows = s74.item_splits(len(test))
    excl74 = set(s73.sample_target(len(test)).tolist()) | set(pilot_rows.tolist()) | set(target74_rows.tolist())
    cand_rows = [i for i in range(len(test)) if i not in excl74]
    evals = dev + [test[i] for i in pilot_rows] + [test[i] for i in target74_rows]
    keep, _ = s74.near_duplicates([test[i] for i in cand_rows], evals, tokenizer)
    train_rows = [cand_rows[k] for k in keep]
    items = [tokenizer(s74.mmlu_qa_text(test[i]), add_special_tokens=False)['input_ids'] for i in train_rows]
    qa, _, members = s74.qa_list(items)
    if s73.tensor_sha256(qa) != QA_POOL_SHA256:
        raise ValueError('QA chunks differ from stage 74')
    qa_ids = sorted({int(test[train_rows[m]]['ID']) for chunk in members for m in chunk})
    trained_ids = set(json.loads(s75.LEDGER.read_text())['additional_sources']['ArabicMMLU']['stage_74']
                      ['used_for_training']['ids'])
    if not set(qa_ids) <= trained_ids:
        raise ValueError('QA chunks contain items not recorded as trained')
    # web chunks: stage-74 decontamination, new draws
    kept_texts = pq.read_table(args.kept, columns=['text'])['text'].to_pylist()
    removed_texts = pq.ParquetFile(args.removed).read_row_groups(list(range(s73.REMOVED_ROW_GROUPS)),
                                                                  columns=['text'])['text'].to_pylist()
    kr, kc, k_drop = s73.decontaminate(*s73.first_tokens(kept_texts, tokenizer), banned)
    rr, rc, r_drop = s73.decontaminate(*s73.first_tokens(removed_texts, tokenizer), banned)
    k_idx, r_idx = web_draws(len(kc), len(rc))
    pool = torch.cat([qa, torch.tensor([kc[i] for i in k_idx], dtype=torch.long),
                      torch.tensor([rc[i] for i in r_idx], dtype=torch.long)])
    # target: stage-75 eligibility with the stage-75 target excluded and criterion (e) on this pool
    cal75 = next(r for r in s69.read_journal(STAGE75_JOURNAL) if r.get('kind') == 'calibration')
    target75 = set(cal75['population']['target_items'])
    excluded = excl74 | {i for i, r in enumerate(test) if int(r['ID']) in target75}
    counts, eligible = s75.target_candidates(test, tokenizer, [pool], trained_ids, excluded)
    if len(eligible) < TARGET_ITEMS:
        raise SystemExit('Fewer than 512 eligible target rows')
    rows = np.asarray(eligible)
    rows = np.sort(rows[np.random.default_rng(TARGET_SEED).choice(len(rows), TARGET_ITEMS, replace=False)])
    target = s73.build_items([test[i] for i in rows], tokenizer)
    population = dict(qa_pool_sha256=QA_POOL_SHA256, qa_item_ids=len(qa_ids), web_eligible=[len(kc), len(rc)],
                      web_decontaminated=[k_drop, r_drop], kept_documents=[int(kr[i]) for i in k_idx],
                      removed_documents=[int(rr[i]) for i in r_idx], pool_sha256=s73.tensor_sha256(pool),
                      trained_ids=len(trained_ids), excluded_evaluated_rows=len(excluded), target_pool=counts,
                      target_items=[it['id'] for it in target])
    return probe, target, pool, population


# ---------------------------------------------------------------- schedules

def schedule(seed, n):
    rng = np.random.default_rng(seed)
    rand = rng.integers(POOL_SIZE, size=BUDGET)
    subset_n = np.sort(rng.choice(POOL_SIZE, n, replace=False))
    qa_n = np.sort(rng.choice(N_QA, n, replace=False))
    once = np.concatenate([rng.permutation(n) for _ in range(math.ceil(BUDGET / n))])[:BUDGET]
    digest = hashlib.sha256(b''.join(a.tobytes() for a in (rand, subset_n, qa_n, once))).hexdigest()
    return dict(rand=rand, subset_n=subset_n, qa_n=qa_n, once=once, sha256=digest)


def arm_order(i):
    return ARMS[i % len(ARMS):] + ARMS[:i % len(ARMS)]


def run_grid(seeds):
    return [(seed, arm) for i, seed in enumerate(seeds) for arm in arm_order(i)]


def views(order, members, steps):
    counts = np.bincount(np.asarray(order[:steps]), minlength=POOL_SIZE)[np.asarray(members)]
    return dict(max=int(counts.max()), mean=steps / len(members))


def training_order(arm, sch, kept=None):
    if arm == 'random':
        return sch['rand'], np.arange(POOL_SIZE)
    members = dict(align05_once=kept, random_once=sch['subset_n'], qa_once=sch['qa_n'])[arm]
    return members[sch['once']], members


def random_qa(rand, steps):
    first = np.asarray(rand[:steps])
    return dict(drawn=int((first < N_QA).sum()), distinct=int(len(set(first[first < N_QA].tolist()))))


# ---------------------------------------------------------------- summary

def bootstrap(values):
    draws = np.random.default_rng(7603).choice(values, (10000, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [.025, .975])]


def contrast(values):
    return dict(mean=mean(values), ci95=bootstrap(values), per_seed=values, seeds_negative=sum(v < 0 for v in values))


def summarize(rows, calibration, plan):
    n, B = str(calibration['updates']), str(BUDGET)
    seeds = plan['design']['seeds']
    g = {arm: sorted([r for r in rows if r['arm'] == arm], key=lambda r: r['seed']) for arm in ARMS}
    for arm in ARMS:
        if [r['seed'] for r in g[arm]] != seeds:
            raise ValueError(f'Incomplete grid for {arm}')
        if [r['losses']['0'] for r in g[arm]] != [r['losses']['0'] for r in g['random']]:
            raise ValueError('Arms do not share the initial state')
    R, A, RO, Q = (g[a] for a in ARMS)
    d = lambda x, kx, y, ky, key='losses': [a[key][kx] - b[key][ky] for a, b in zip(x, y)]
    primary, sel = d(A, n, R, B), d(A, n, R, n)
    out = dict(
        loss0=mean(r['losses']['0'] for r in R), accuracy0=mean(r['accuracy']['0'] for r in R),
        means={arm: {k: mean(r['losses'][k] for r in g[arm]) for k in ('0', n, B)} for arm in ARMS},
        accuracy_means={arm: {k: mean(r['accuracy'][k] for r in g[arm]) for k in ('0', n, B)} for arm in ARMS},
        charged=contrast(primary), matched=contrast(d(A, n, RO, n)), ranking_vs_oracle=contrast(d(A, n, Q, n)),
        oracle_charged=contrast(d(Q, n, R, B)), one_view=contrast(d(RO, n, R, n)),
        selection_term=mean(sel), throughput_term=mean(primary) - mean(sel),
        amortized=contrast(d(A, B, R, B)),
        accuracy_charged=contrast(d(A, n, R, B, 'accuracy')), accuracy_matched=contrast(d(A, n, RO, n, 'accuracy')),
        qa_share={arm: mean(r['qa_share'] for r in g[arm]) for arm in ARMS if arm != 'random'},
        random_qa={k: {f: mean(r['random_qa'][k][f] for r in R) for f in ('drawn', 'distinct')} for k in (n, B)},
        random_minus_no_update=mean(d(R, B, R, '0')),
        scoring_seconds=mean(r['scoring_seconds'] for r in A))
    return dict(primary=dict(contrast='align05_once at N minus random at 3000', mean=out['charged']['mean'],
                             ci95=out['charged']['ci95']),
                prediction_holds={'qa share of align05_once kept > 0.85': out['qa_share']['align05_once'] > .85,
                                  'primary negative': out['charged']['mean'] < 0,
                                  'align05_once minus random_once at N negative': out['matched']['mean'] < 0,
                                  'align05_once minus qa_once at N non-negative': out['ranking_vs_oracle']['mean'] >= 0},
                costs=calibration['costs'], updates=calibration['updates'], device=calibration['device'],
                scoring_in_random_steps=calibration['costs']['proxy_align_grad'] / calibration['costs']['random'],
                arms=out)


# ---------------------------------------------------------------- backup

def backup(journal_path, log_path, worktree, branch, run=subprocess.run):
    """Copy the journal (and run log) to the journal branch, commit and push; a failed push logs git's stderr and is
    retried once. Failures are reported, not fatal."""
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
        git('commit', '-q', '-m', f'Stage 76 interim journal snapshot: {runs} finished runs ({stamp})')
    except Exception as exc:
        print(f'BACKUP FAILED (commit): {type(exc).__name__}: {getattr(exc, "stderr", "") or exc}', flush=True)
        return
    for attempt in (1, 2):
        try:
            git('push', '-q', 'origin', f'HEAD:{branch}')
            print(f'backup pushed: {runs} finished runs' + (' (retry)' if attempt == 2 else ''), flush=True)
            return
        except Exception as exc:  # the next backup retries with the whole journal
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
    parser.add_argument('--backup-branch', default='colab/stage76-journal')
    parser.add_argument('--summary', type=Path, default=Path('results/llm-cheap-filter.json'))
    args = parser.parse_args()
    plan = json.loads(PROTOCOL.read_text())
    if plan['stage'] != 76 or tuple(plan['design']['arms']) != ARMS:
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
    probe, target, pool, population = load_inputs(args, tokenizer)
    pool = pool.to(device)
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

    def proxy_score(seed):
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
        done = {(r['seed'], r['arm']) for r in journal if r.get('kind') == 'run'}
        calibration = next((r for r in journal if r.get('kind') == 'calibration'), None)
        if calibration is None:
            rand = np.random.default_rng(CAL_SEED).integers(POOL_SIZE, size=24)
            row = s73.train(s69.load(model_dir, CAL_SEED, device), pool, rand, 24, set(), None, device)
            costs = dict(random=median(row['step_seconds'][4:]))
            free()
            _, costs['proxy_align_grad'] = proxy_score(CAL_SEED)
            calibration = dict(kind='calibration', model=MODEL, proxy=PROXY, device=device_name, costs=costs,
                               updates=int(np.floor(BUDGET - costs['proxy_align_grad'] / costs['random'] + 1e-9)),
                               population=population)
            log(calibration)
            backup(args.journal, args.run_log, args.backup_worktree, args.backup_branch)
        elif calibration['device'] != device_name or calibration['population'] != population:
            raise ValueError('Resumed on another device type or with different samples')
        n = calibration['updates']
        if not N_RANGE[0] <= n <= N_RANGE[1]:
            raise SystemExit(f'N = {n} outside [{N_RANGE[0]}, {N_RANGE[1]}]: design infeasible')
        print(MODEL, PROXY, device_name, 'updates', n, flush=True)  # costs only; no outcome
        ev = lambda m: s73.evaluate(m, target, device)
        grid = run_grid(seeds)
        for seed, arm in grid:
            if (seed, arm) in done:
                continue
            sch = schedule(seed, n)
            extra, kept = {}, None
            if arm == 'align05_once':
                scores, seconds = proxy_score(seed)
                kept = s73.keep_top(scores, n)
                extra = dict(scores=scores.tolist(), kept=kept.tolist(), scoring_seconds=seconds,
                             kept_sha256=hashlib.sha256(kept.tobytes()).hexdigest())
            order, members = training_order(arm, sch, kept)
            if arm in ('random_once', 'qa_once'):
                extra['subset_sha256'] = hashlib.sha256(np.asarray(members).tobytes()).hexdigest()
            if arm == 'random':
                extra['random_qa'] = {str(t): random_qa(sch['rand'], t) for t in (n, BUDGET)}
            else:
                extra['qa_share'] = qa_share(members)
            extra['views'] = {str(t): views(order, members, t) for t in (n, BUDGET)}
            model = s69.load(model_dir, seed, device)
            row = s73.train(model, pool, order, BUDGET, {0, n, BUDGET}, ev, device)
            row.update(extra, kind='run', model=MODEL, seed=seed, arm=arm, device=device_name,
                       schedule_sha256=sch['sha256'])
            log(row)
            done.add((seed, arm))
            del model
            free()
            print(f'{len(done)}/{len(grid)} runs complete', flush=True)  # outcomes are not printed
            backup(args.journal, args.run_log, args.backup_worktree, args.backup_branch)
    journal = s69.read_journal(args.journal)
    rows = [r for r in journal if r.get('kind') == 'run']
    result = dict(stage=76, protocol=plan, journal_sha256=s69.sha256(args.journal),
                  environments=[r for r in journal if r.get('kind') == 'environment'],
                  calibration={k: v for k, v in calibration.items() if k != 'population'},
                  summary=summarize(rows, calibration, plan), runs=len(rows))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('summary written to', args.summary)


if __name__ == '__main__':
    main()

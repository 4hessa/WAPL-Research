"""Stage 75: one-pass offline alignment filter (kept size N, each kept chunk seen once by step N) vs random on the
stage-74 pools, with the stage-74 kept fraction 0.25 as replication and two repetition-matched random controls.

Implements research/llm-once-filter-protocol.json (amendment 1: sigma 0.25 only) with the stage-73/74 building blocks.
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

from experiments import llm_gpu as s69
from experiments import llm_qa_mix as s74
from experiments import llm_web_filter as s73

PROTOCOL = Path('research/llm-once-filter-protocol.json')
MODEL = s74.MODEL
ARMS = ('random', 'align_q', 'random512', 'align_once', 'random_once')
FILTER_ARMS = ('align_q', 'align_once')
SIGMAS = (0.25,)
CAL_SEED, TARGET_SEED, TARGET_ITEMS = 75999, 7522, 512
BUDGET, POOL_SIZE, KEEP_Q = s74.BUDGET, s74.POOL_SIZE, s74.KEEP
STAGE74_POOL_SHA256 = {'0.25': 'bf12ca51827e6924e6df086a9e18c9161e30ef419a64ac439ca84d066d210775',
                       '0.1': 'dba2e7898ff8622371131ff1c861418fccebf881bca46ed2f2c54d5f901fd29d'}
LEDGER = Path('data/evaluation_status.json')
INTERIM = 'results/stage75-interim'


# ---------------------------------------------------------------- target (moved unchanged from tools/smoke_llm_once_filter.py)

def eval_forms(r, tokenizer):
    """Token sequences of each content field of a candidate item (Question, Context, every option), alone and with a
    leading space."""
    seqs = []
    fields = [r['Question']] + ([r['Context']] if s73._present(r.get('Context')) else [])
    fields += [r[f'Option {j}'] for j in range(1, 6) if s73._present(r.get(f'Option {j}'))]
    for f in fields:
        for text in (str(f).strip(), ' ' + str(f).strip()):
            seqs.append(tokenizer(text, add_special_tokens=False)['input_ids'])
    return seqs


def target_candidates(test, tokenizer, pools, trained_ids, excluded_rows):
    """Eligible target rows: outside evaluated rows and trained IDs, and clean under criteria (a)-(e)."""
    trained = [r for r in test if int(r['ID']) in trained_ids]
    cand_rows = [i for i, r in enumerate(test) if i not in excluded_rows and int(r['ID']) not in trained_ids]
    keep, counts = s74.near_duplicates([test[i] for i in cand_rows], trained, tokenizer)  # criteria (a)-(d)
    banned = set()
    for p in pools:
        for c in p.tolist():
            banned |= s73.ngrams(c)
    for r in trained:
        banned |= s73.ngrams(tokenizer(s74.mmlu_qa_text(r), add_special_tokens=False)['input_ids'])
    e_hits, survivors = 0, []
    for k in keep:
        if any(s73.ngrams(s) & banned for s in eval_forms(test[cand_rows[k]], tokenizer)):
            e_hits += 1
        else:
            survivors.append(cand_rows[k])
    counts['trained_tokenization_8gram_after_a_to_d'] = e_hits
    return dict(candidates=len(cand_rows), decontamination=counts, eligible=len(survivors)), survivors


def sample_target(eligible_rows):
    rows = np.asarray(eligible_rows)
    return np.sort(rows[np.random.default_rng(TARGET_SEED).choice(len(rows), TARGET_ITEMS, replace=False)])


# ---------------------------------------------------------------- schedules

def schedule(k, seed, n):
    """rand, filt, subset512 exactly as stage 74, then subsetN and the one-pass order (two fresh permutations)."""
    rng = np.random.default_rng(seed + 1000 * k)
    rand = rng.integers(POOL_SIZE, size=BUDGET)
    filt = rng.integers(KEEP_Q, size=BUDGET)
    subset512 = np.sort(rng.choice(POOL_SIZE, KEEP_Q, replace=False))
    subset_n = np.sort(rng.choice(POOL_SIZE, n, replace=False))
    once = np.concatenate([rng.permutation(n), rng.permutation(n)])[:BUDGET]
    digest = hashlib.sha256(b''.join(a.tobytes() for a in (rand, filt, subset512, subset_n, once))).hexdigest()
    return dict(rand=rand, filt=filt, subset512=subset512, subset_n=subset_n, once=once, sha256=digest)


def arm_order(i):
    return ARMS[i % len(ARMS):] + ARMS[:i % len(ARMS)]


def run_grid(seeds):
    return [(sigma, seed, arm) for sigma in SIGMAS for i, seed in enumerate(seeds) for arm in arm_order(i)]


def views(order, members, steps):
    """Max and mean number of times a member of the training set is seen in the first `steps` updates."""
    counts = np.bincount(np.asarray(order[:steps]), minlength=POOL_SIZE)[np.asarray(members)]
    return dict(max=int(counts.max()), mean=steps / len(members))


def training_order(arm, sch, kept=None):
    """(pool indices per update, training set) of a non-random arm given its kept set (filters) or subset (controls)."""
    if arm == 'random':
        return sch['rand'], np.arange(POOL_SIZE)
    if arm == 'random512':
        return sch['subset512'][sch['filt']], sch['subset512']
    if arm == 'random_once':
        return sch['subset_n'][sch['once']], sch['subset_n']
    if arm == 'align_q':
        return kept[sch['filt']], kept
    if arm == 'align_once':
        return kept[sch['once']], kept
    raise ValueError(arm)


# ---------------------------------------------------------------- summary

def bootstrap(values):
    draws = np.random.default_rng(7503).choice(values, (10000, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [.025, .975])]


def contrast(values):
    return dict(mean=mean(values), ci95=bootstrap(values), per_seed=values, seeds_negative=sum(v < 0 for v in values))


def summarize(rows, calibration, plan):
    n, B = str(calibration['updates']), str(BUDGET)
    seeds = plan['design']['seeds']
    out = {}
    for sigma in plan['design']['sigma']:
        g = {arm: sorted([r for r in rows if (r['sigma'], r['arm']) == (sigma, arm)], key=lambda r: r['seed'])
             for arm in ARMS}
        for arm in ARMS:
            if [r['seed'] for r in g[arm]] != seeds:
                raise ValueError(f'Incomplete grid for sigma {sigma} {arm}')
            if [r['losses']['0'] for r in g[arm]] != [r['losses']['0'] for r in g['random']]:
                raise ValueError('Arms do not share the initial state')
        R, Q, C, O, RO = (g[a] for a in ARMS)
        d = lambda x, kx, y, ky, key='losses': [a[key][kx] - b[key][ky] for a, b in zip(x, y)]
        sel_o, sel_q = d(O, n, R, n), d(Q, n, R, n)
        out[str(sigma)] = dict(
            loss0=mean(r['losses']['0'] for r in R), accuracy0=mean(r['accuracy']['0'] for r in R),
            means={arm: {k: mean(r['losses'][k] for r in g[arm]) for k in ('0', n, B)} for arm in ARMS},
            accuracy_means={arm: {k: mean(r['accuracy'][k] for r in g[arm]) for k in ('0', n, B)} for arm in ARMS},
            charged_once=contrast(d(O, n, R, B)), matched_once=contrast(d(O, n, RO, n)),
            fixing_repetition=contrast(d(O, n, Q, n)), repetition_once=contrast(d(RO, n, R, n)),
            charged_q=contrast(d(Q, n, R, B)), matched_q=contrast(d(Q, n, C, n)), repetition_q=contrast(d(C, n, R, n)),
            selection_term_once=mean(sel_o), throughput_term_once=mean(d(O, n, R, B)) - mean(sel_o),
            selection_term_q=mean(sel_q), throughput_term_q=mean(d(Q, n, R, B)) - mean(sel_q),
            amortized_once=contrast(d(O, B, R, B)), amortized_q=contrast(d(Q, B, R, B)),
            accuracy_charged_once=contrast(d(O, n, R, B, 'accuracy')),
            accuracy_matched_once=contrast(d(O, n, RO, n, 'accuracy')),
            accuracy_charged_q=contrast(d(Q, n, R, B, 'accuracy')),
            accuracy_matched_q=contrast(d(Q, n, C, n, 'accuracy')),
            qa_share={arm: mean(r['qa_share'] for r in g[arm]) for arm in ARMS if arm != 'random'},
            random_minus_no_update=mean(d(R, B, R, '0')),
            scoring_seconds={arm: mean(r['scoring_seconds'] for r in g[arm]) for arm in FILTER_ARMS})
    p = out['0.25']
    return dict(primary=dict(sigma=0.25, contrast='align_once at N minus random at 2400',
                             mean=p['charged_once']['mean'], ci95=p['charged_once']['ci95']),
                prediction_holds={
                    'sigma0.25 align_once qa share > 0.30': p['qa_share']['align_once'] > .30,
                    'sigma0.25 primary negative': p['charged_once']['mean'] < 0,
                    'sigma0.25 align_once minus random_once at N negative': p['matched_once']['mean'] < 0,
                    'sigma0.25 align_once minus align_q at N negative': p['fixing_repetition']['mean'] < 0,
                    'replication: align_q charged positive': p['charged_q']['mean'] > 0,
                    'replication: align_q minus random512 negative': p['matched_q']['mean'] < 0,
                    'replication: random512 minus random positive': p['repetition_q']['mean'] > 0},
                not_tested='sigma 0.1 prediction item (amendment 1)',
                costs=calibration['costs'], updates=calibration['updates'], device=calibration['device'],
                scoring_in_random_steps=calibration['costs']['align_grad'] / calibration['costs']['random'],
                sigmas=out)


# ---------------------------------------------------------------- inputs

def load_inputs(args, tokenizer):
    import pandas as pd
    probe, _, _, pools, pop74, _ = s74.load_inputs(args, tokenizer)  # checks the four data-file hashes
    for s in STAGE74_POOL_SHA256:
        if pop74['pool_sha256'][s] != STAGE74_POOL_SHA256[s]:
            raise ValueError(f'Pool {s} differs from stage 74')
    test = pd.read_csv(args.mmlu / 'test.csv').to_dict('records')
    trained_ids = set(json.loads(LEDGER.read_text())['additional_sources']['ArabicMMLU']['stage_74']
                      ['used_for_training']['ids'])
    pilot_rows, target74_rows = s74.item_splits(len(test))
    excluded = set(s73.sample_target(len(test)).tolist()) | set(pilot_rows.tolist()) | set(target74_rows.tolist())
    counts, eligible = target_candidates(test, tokenizer, [pools[0.25], pools[0.1]], trained_ids, excluded)
    rows = sample_target(eligible)
    target = s73.build_items([test[i] for i in rows], tokenizer)
    population = dict(pool_sha256={s: STAGE74_POOL_SHA256[s] for s in ('0.25', '0.1')}, trained_ids=len(trained_ids),
                      excluded_evaluated_rows=len(excluded), target_pool=counts,
                      target_items=[it['id'] for it in target])
    return probe, target, {s: pools[s] for s in SIGMAS}, population


# ---------------------------------------------------------------- backup

def backup(journal_path, log_path, worktree, branch):
    """Copy the journal (and run log) to the journal branch, commit and push; failures are reported, not fatal."""
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
        git = lambda *a: subprocess.run(['git', '-C', str(worktree), *a], check=True, capture_output=True, text=True)
        git('add', INTERIM)
        if subprocess.run(['git', '-C', str(worktree), 'diff', '--cached', '--quiet']).returncode == 0:
            return
        git('commit', '-q', '-m', f'Stage 75 interim journal snapshot: {runs} finished runs ({stamp})')
        git('push', '-q', 'origin', f'HEAD:{branch}')
        print(f'backup pushed: {runs} finished runs', flush=True)
    except Exception as exc:  # the run continues; the next backup retries with the whole journal
        print(f'BACKUP FAILED: {type(exc).__name__}', flush=True)


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
    parser.add_argument('--backup-branch', default='colab/stage75-journal')
    parser.add_argument('--summary', type=Path, default=Path('results/llm-once-filter.json'))
    args = parser.parse_args()
    plan = json.loads(PROTOCOL.read_text())
    if plan['stage'] != 75 or tuple(plan['design']['sigma']) != SIGMAS or tuple(plan['design']['arms']) != ARMS:
        raise ValueError('Protocol differs from the implementation')
    import torch
    import transformers
    from transformers import AutoTokenizer
    s69.setup_numerics()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'
    spec = json.loads(s69.PROTOCOL.read_text())['models'][MODEL]
    model_dir = args.models_dir / MODEL
    if s69.sha256(model_dir / 'model.safetensors') != spec['safetensors_sha256']:
        raise ValueError('Model weights differ from the protocol')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    probe, target, pools, population = load_inputs(args, tokenizer)
    pools = {k: v.to(device) for k, v in pools.items()}
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

    with args.journal.open('a') as out:
        def log(row):
            out.write(json.dumps(row, allow_nan=False) + '\n')
            out.flush()
            os.fsync(out.fileno())
        log(dict(env, resumed_runs=sum(r.get('kind') == 'run' for r in journal)))
        done = {(r['sigma'], r['seed'], r['arm']) for r in journal if r.get('kind') == 'run'}
        calibration = next((r for r in journal if r.get('kind') == 'calibration'), None)
        if calibration is None:
            sch = schedule(0, CAL_SEED, 1)
            row = s73.train(s69.load(model_dir, CAL_SEED, device), pools[0.25], sch['rand'], 24, set(), None, device)
            costs = dict(random=median(row['step_seconds'][4:]))
            free()
            _, costs['align_grad'] = s73.score(s69.load(model_dir, CAL_SEED, device), 'align_grad', pools[0.25], probe,
                                               device)
            free()
            calibration = dict(kind='calibration', model=MODEL, device=device_name, costs=costs,
                               updates=int(np.floor(BUDGET - costs['align_grad'] / costs['random'] + 1e-9)),
                               population=population)
            log(calibration)
            backup(args.journal, args.run_log, args.backup_worktree, args.backup_branch)
        elif calibration['device'] != device_name or calibration['population'] != population:
            raise ValueError('Resumed on another device type or with different samples')
        n = calibration['updates']
        if not KEEP_Q < n <= POOL_SIZE:
            raise SystemExit('N outside (512, 2048]: design infeasible')
        print(MODEL, device_name, 'updates', n, flush=True)  # costs only; no outcome
        ev = lambda m: s73.evaluate(m, target, device)
        grid = run_grid(seeds)
        for sigma, seed, arm in grid:
            if (sigma, seed, arm) in done:
                continue
            k = SIGMAS.index(sigma)
            data = pools[sigma]
            sch = schedule(k, seed, n)
            model = s69.load(model_dir, seed, device)
            extra = {}
            kept = None
            if arm in FILTER_ARMS:
                scores, seconds = s73.score(model, 'align_grad', data, probe, device)
                kept = s73.keep_top(scores, KEEP_Q if arm == 'align_q' else n)
                extra = dict(scores=scores.tolist(), kept=kept.tolist(), scoring_seconds=seconds,
                             kept_sha256=hashlib.sha256(kept.tobytes()).hexdigest())
            order, members = training_order(arm, sch, kept)
            if arm in ('random512', 'random_once'):
                extra['subset_sha256'] = hashlib.sha256(np.asarray(members).tobytes()).hexdigest()
            if arm != 'random':
                extra['qa_share'] = s74.qa_share(members, sigma)
            extra['views'] = {str(t): views(order, members, t) for t in (n, BUDGET)}
            row = s73.train(model, data, order, BUDGET, {0, n, BUDGET}, ev, device)
            row.update(extra, kind='run', model=MODEL, sigma=sigma, seed=seed, arm=arm, device=device_name,
                       schedule_sha256=sch['sha256'])
            log(row)
            done.add((sigma, seed, arm))
            del model
            free()
            print(f'{len(done)}/{len(grid)} runs complete', flush=True)  # outcomes are not printed
            backup(args.journal, args.run_log, args.backup_worktree, args.backup_branch)
    journal = s69.read_journal(args.journal)
    rows = [r for r in journal if r.get('kind') == 'run']
    result = dict(stage=75, protocol=plan, journal_sha256=s69.sha256(args.journal),
                  environments=[r for r in journal if r.get('kind') == 'environment'],
                  calibration={k: v for k, v in calibration.items() if k != 'population'},
                  summary=summarize(rows, calibration, plan), runs=len(rows))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('summary written to', args.summary)


if __name__ == '__main__':
    main()

"""Audit stage 74 (research/llm-qa-mix-protocol.json, audit field, and deviation 1).

CPU: journal and summary digests, summary and gate recomputation, environments, budget, the 14-vs-16
reconciliation of finished runs before the machine loss, run grid and order, schedules, checkpoints,
kept sets, QA shares, random512 subsets, data hashes, samples, pools, decontamination of web chunks and
of the ArabicMMLU training items, ID disjointness. Informational: per-session step and scoring times.
GPU: recompute align_grad scores of seed 37400 on both pools; replay the first 50 random steps of seed
37400 at sigma 0.25 (a run made on the lost machine) and at sigma 0.1 (made on the resumed machine).

python -m tools.audit_llm_qa_mix --kept K --removed R --mmlu M [--no-gpu]
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import re
from statistics import median
import subprocess

import numpy as np

from experiments import llm_gpu as s69
from experiments import llm_qa_mix as stage
from experiments import llm_web_filter as s73
from tools.audit_pud_reuse import require

RESULT = Path('results/llm-qa-mix.json')
JOURNAL = Path('results/llm-qa-mix-journal.jsonl')
PILOT_JOURNAL = Path('results/llm-qa-mix-pilot2-journal.jsonl')
OUTPUT = Path('results/llm-qa-mix-audit.json')
SNAPSHOT_COMMIT = 'e42a7db'  # last snapshot pushed by the lost machine (deviation 1)
SNAPSHOT_JOURNAL = 'results/stage74-interim/journal.jsonl'
SNAPSHOT_LOG = 'results/stage74-interim/runB.log'
AUDIT_SEED, SCORE_TOL = 37400, 1e-4
ENV_KEYS = ('device', 'torch', 'transformers', 'cuda', 'python', 'phase')


def expected_order(plan):
    seeds = plan['phase_B_main']['seeds']
    return [(sigma, seed, arm) for sigma in stage.SIGMAS for i, seed in enumerate(seeds)
            for arm in stage.ARMS[i % 3:] + stage.ARMS[:i % 3]]


def progress(log_text):
    return [int(m) for m in re.findall(r'^(\d+)/\d+ runs complete$', log_text, flags=re.M)]


def parse(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def reconcile_resumption(final_text, snapshot_text, snapshot_message, old_log_text, plan):
    """How many runs had finished before the machine loss, and why the snapshot said 16."""
    snap = parse(snapshot_text)
    final = parse(final_text)
    kinds = Counter(r['kind'] for r in snap)
    before = kinds.get('run', 0)
    claimed = int(re.search(r'(\d+) finished runs', snapshot_message).group(1))
    require(final_text.startswith(snapshot_text), 'snapshot journal is not a byte prefix of the final journal')
    require(claimed == len(snap), 'snapshot message count is not the number of journal lines')
    require(kinds.get('environment') == 1 and kinds.get('calibration') == 1, 'snapshot header lines')
    envs = [i for i, r in enumerate(final) if r['kind'] == 'environment']
    require(len(envs) == 2, 'exactly two environment entries (original and resumption)')
    require(envs[1] == len(snap), 'resumption environment directly follows the snapshot')
    require(final[envs[0]]['resumed_runs'] == 0, 'first session resumed nothing')
    require(final[envs[1]]['resumed_runs'] == before, 'resumption skipped exactly the snapshot runs')
    log_counts = progress(old_log_text)
    require(log_counts and log_counts == list(range(1, before + 1)), 'lost-machine log progress matches the snapshot')
    order = expected_order(plan)
    first_after = next(r for r in final[envs[1] + 1:] if r['kind'] == 'run')
    restarted = (first_after['sigma'], first_after['seed'], first_after['arm'])
    require(restarted == order[before], 'first run after resumption is the next run in protocol order')
    return dict(
        snapshot_commit=SNAPSHOT_COMMIT, snapshot_journal_sha256=hashlib.sha256(snapshot_text.encode()).hexdigest(),
        snapshot_commit_message_count=claimed, snapshot_journal_lines=len(snap), snapshot_line_kinds=dict(kinds),
        finished_runs_before_loss=before, lost_machine_log_last_progress=log_counts[-1],
        resumed_runs_in_second_environment=final[envs[1]]['resumed_runs'],
        restarted_run=dict(sigma=restarted[0], seed=restarted[1], arm=restarted[2]),
        explanation=(f'The snapshot commit message said {claimed} finished runs because the lost machine\'s snapshot '
                     f'script counted every journal line. The snapshot holds {len(snap)} lines: '
                     f'{kinds["environment"]} environment, {kinds["calibration"]} calibration and {before} run entries. '
                     f'Finished runs before the loss = {before}, confirmed three ways: the run entries in the snapshot, '
                     f'the lost machine\'s log (last line {log_counts[-1]}/36), and the resumed process '
                     f'(resumed_runs = {final[envs[1]]["resumed_runs"]}). The snapshot is a byte-for-byte prefix of '
                     f'the final journal, so no finished run was altered or dropped; the run in progress at the loss '
                     f'({restarted[0]}, {restarted[1]}, {restarted[2]}) was restarted from step 0 and is the first '
                     f'entry after the resumption environment.'))


def check_run(row, k, n):
    """Schedule, checkpoints and selection digests of one run; raises on any mismatch."""
    tag = f"sigma {row['sigma']} seed {row['seed']} {row['arm']}"
    _, _, subset, digest = stage.schedule(k, row['seed'])
    require(row['schedule_sha256'] == digest, f'{tag}: schedule')
    require(len(row['step_seconds']) == stage.BUDGET, f'{tag}: update count')
    expected = {'0', str(n), str(stage.BUDGET)}
    require(set(row['losses']) == expected == set(row['accuracy']), f'{tag}: checkpoints')
    require(isinstance(row['adapter_digest_50'], str) and len(row['adapter_digest_50']) == 64, f'{tag}: step-50 digest')
    if row['arm'] == 'align_grad':
        require(len(row['scores']) == stage.POOL_SIZE, f'{tag}: score count')
        kept = s73.keep_top(row['scores'], stage.KEEP)
        require(kept.tolist() == row['kept'], f'{tag}: kept set from scores')
        require(hashlib.sha256(kept.tobytes()).hexdigest() == row['kept_sha256'], f'{tag}: kept digest')
        require(row['qa_share'] == stage.qa_share(kept, row['sigma']), f'{tag}: QA share')
        require(row['scoring_seconds'] > 0, f'{tag}: scoring time')
    elif row['arm'] == 'random512':
        require(row['subset_sha256'] == hashlib.sha256(subset.tobytes()).hexdigest(), f'{tag}: subset digest')
        require(row['qa_share'] == stage.qa_share(subset, row['sigma']), f'{tag}: QA share')
    else:
        require(not {'kept', 'scores', 'subset_sha256'} & set(row), f'{tag}: random arm carries selection fields')


def session_costs(journal, cal):
    """Median random-step seconds and align_grad scoring seconds per session (cost, not outcome)."""
    out, session = [], None
    for r in journal:
        if r['kind'] == 'environment':
            session = dict(started=r['time'], runs=0, random_step_median=[], scoring_seconds=[])
            out.append(session)
        elif r['kind'] == 'run':
            session['runs'] += 1
            if r['arm'] == 'random':
                session['random_step_median'].append(median(r['step_seconds'][4:]))
            if r['arm'] == 'align_grad':
                session['scoring_seconds'].append(r['scoring_seconds'])
    for s in out:
        steps, scores = s.pop('random_step_median'), s.pop('scoring_seconds')
        s['random_step_seconds_median'] = median(steps) if steps else None
        s['align_grad_scoring_seconds_median'] = median(scores) if scores else None
        s['scoring_in_random_steps'] = (s['align_grad_scoring_seconds_median'] / s['random_step_seconds_median']
                                        if steps and scores else None)
    return dict(calibration=dict(random=cal['costs']['random'], align_grad=cal['costs']['align_grad'],
                                 scoring_in_random_steps=cal['costs']['align_grad'] / cal['costs']['random']),
                sessions=out)


def git_show(path):
    return subprocess.run(['git', 'show', f'{SNAPSHOT_COMMIT}:{path}'], check=True, capture_output=True, text=True).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kept', type=Path, required=True)
    parser.add_argument('--removed', type=Path, required=True)
    parser.add_argument('--mmlu', type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--no-gpu', action='store_true', help='Skip the GPU recomputation and replays')
    args = parser.parse_args()
    plan = json.loads(stage.PROTOCOL.read_text())
    result = json.loads(RESULT.read_text())
    require(result['protocol'] == plan, 'protocol changed since the run')
    require(s69.sha256(JOURNAL) == result['journal_sha256'], 'journal digest')
    require(s69.sha256(PILOT_JOURNAL) == result['pilot_journal_sha256'], 'pilot journal digest')
    pilot2 = json.loads(Path('results/llm-qa-mix-pilot2.json').read_text())
    require(pilot2['journal_sha256'] == result['pilot_journal_sha256'], 'pilot journal is the committed phase A journal')
    journal, pilot_journal = s69.read_journal(JOURNAL), s69.read_journal(PILOT_JOURNAL)
    rows = [r for r in journal if r['kind'] == 'run']
    cals = [r for r in journal if r['kind'] == 'calibration']
    require(len(cals) == 1, 'one calibration')
    cal = cals[0]
    pilot_runs = [r for r in pilot_journal if r['kind'] == 'pilot']
    require(stage.gate(pilot_runs) and result['gate_passed'], 'gate recomputation')

    # resumption (deviation 1): the 14 vs 16 reconciliation
    snapshot_message = subprocess.run(['git', 'log', '-1', '--format=%s', SNAPSHOT_COMMIT], check=True,
                                      capture_output=True, text=True).stdout
    resumption = reconcile_resumption(JOURNAL.read_text(), git_show(SNAPSHOT_JOURNAL), snapshot_message,
                                      git_show(SNAPSHOT_LOG), plan)
    envs = [r for r in journal if r['kind'] == 'environment']
    require(all(e[k] == envs[0][k] for e in envs + [r for r in pilot_journal if r['kind'] == 'environment']
                for k in ENV_KEYS if k != 'phase'), 'software and device identical across sessions and phase A')
    require(envs[0]['transformers'] == '4.46.3', 'pinned transformers')

    # grid, order, budget, summary
    order = expected_order(plan)
    require([(r['sigma'], r['seed'], r['arm']) for r in rows] == order, 'complete grid in protocol order')
    require(all(r['device'] == cal['device'] and r['model'] == stage.MODEL for r in rows), 'device and model')
    n = cal['updates']
    require(n == int(np.floor(stage.BUDGET - cal['costs']['align_grad'] / cal['costs']['random'] + 1e-9)), 'budget N')
    require(json.dumps(stage.summarize(rows, cal, plan), sort_keys=True) ==
            json.dumps(result['summary'], sort_keys=True), 'summary recomputation')
    require(result['runs'] == len(rows) == len(order), 'run count in summary')
    for r in rows:
        check_run(r, stage.SIGMAS.index(r['sigma']), n)
    checks = dict(runs=len(rows), schedules=len({(r['sigma'], r['seed']) for r in rows}),
                  kept_sets=sum(r['arm'] == 'align_grad' for r in rows),
                  random512_subsets=sum(r['arm'] == 'random512' for r in rows))

    # data, samples, pools, decontamination
    import pandas as pd
    from transformers import AutoTokenizer
    model_dir = args.models_dir / stage.MODEL
    spec = json.loads(s69.PROTOCOL.read_text())['models'][stage.MODEL]
    require(s69.sha256(model_dir / 'model.safetensors') == spec['safetensors_sha256'], 'model weights')
    tok = AutoTokenizer.from_pretrained(model_dir)
    probe, _, _, pools, population, banned_all = stage.load_inputs(args, tok)  # checks the four data-file hashes
    require(population == cal['population'], 'samples, decontamination counts, pools (calibration)')
    require(all(r['population'] == population for r in pilot_runs), 'phase A on the same population')
    test = pd.read_csv(args.mmlu / 'test.csv').to_dict('records')
    dev = pd.read_csv(args.mmlu / 'dev.csv').to_dict('records')
    pilot_rows, target_rows = stage.item_splits(len(test))
    s73_rows = s73.sample_target(len(test))
    require(len(pilot_rows) == 256 and len(target_rows) == 512, 'sample sizes')
    require(not set(pilot_rows) & set(target_rows) and not (set(pilot_rows) | set(target_rows)) & set(s73_rows),
            'pilot, target and stage-73 target disjoint')
    ids = lambda idx: {int(test[i]['ID']) for i in idx}
    eval_ids = ids(pilot_rows) | ids(target_rows) | {int(r['ID']) for r in dev}
    used = set(population['qa_item_ids_in_pools'])
    require(not used & (eval_ids | ids(s73_rows)), 'training items disjoint from all evaluation and stage-73 items')
    by_id = {int(r['ID']): r for r in test}
    used_records = [by_id[i] for i in sorted(used)]
    evals = dev + [test[i] for i in pilot_rows] + [test[i] for i in target_rows]
    keep, counts = stage.near_duplicates(used_records, evals, tok)
    require(len(keep) == len(used_records) and counts['dropped_total'] == 0,
            'no pooled training item matches an evaluation item under criteria (a)-(d)')
    for s, pool in pools.items():
        require(tuple(pool.shape) == (stage.POOL_SIZE, stage.CHUNK), f'pool {s} shape')
        n_qa = stage.composition(s)[0]
        require(not any(s73.ngrams(c) & banned_all for c in pool[n_qa:].tolist()), f'pool {s} web decontamination')
    require(bool((pools[0.1][:205] == pools[0.25][:205]).all()) and bool((pools[1.0][:512] == pools[0.25][:512]).all()),
            'QA prefixes nested across pools')
    eval_banned = stage.content_banned(evals, tok)
    qa_chunk_hits = {str(s): sum(bool(s73.ngrams(c) & eval_banned) for c in pool[:stage.composition(s)[0]].tolist())
                     for s, pool in pools.items()}
    data_checks = dict(qa_items_in_pools=len(used), near_duplicate_recheck=counts, web_chunks_checked={
        str(s): stage.POOL_SIZE - stage.composition(s)[0] for s in pools},
        qa_chunks_sharing_an_eval_8gram_after_templating=qa_chunk_hits,
        qa_chunk_note='Informational. Decontamination is defined per item on fields tokenized alone (amendment 1); '
                      'chunk-level 8-grams span template tokens and item boundaries, so they are reported, not required.')

    gpu = None
    if not args.no_gpu:
        import torch
        s69.setup_numerics()
        device = torch.device('cuda')
        require(torch.cuda.get_device_name(0) == cal['device'], 'audit device type differs from the run')
        gpu = dict(score_recomputation={}, random_replay_step50={})
        for k, sigma in enumerate(stage.SIGMAS):
            row = next(r for r in rows if (r['sigma'], r['seed'], r['arm']) == (sigma, AUDIT_SEED, 'align_grad'))
            scores, seconds = s73.score(s69.load(model_dir, AUDIT_SEED, device), 'align_grad', pools[sigma].to(device),
                                        probe, device)
            journaled = np.array(row['scores'])
            rel = float(np.abs(scores - journaled).max() / np.abs(journaled).max())
            same_kept = s73.keep_top(scores, stage.KEEP).tolist() == row['kept']
            require(rel <= SCORE_TOL, f'sigma {sigma} score recomputation')
            gpu['score_recomputation'][str(sigma)] = dict(max_rel_diff=rel, identical_kept_set=same_kept,
                                                          audit_scoring_seconds=seconds)
            rand, _, _, _ = stage.schedule(k, AUDIT_SEED)
            ref = next(r for r in rows if (r['sigma'], r['seed'], r['arm']) == (sigma, AUDIT_SEED, 'random'))
            replay = s73.train(s69.load(model_dir, AUDIT_SEED, device), pools[sigma].to(device), rand,
                               s73.DIGEST_STEP, set(), None, device)
            session = 'lost machine' if order.index((sigma, AUDIT_SEED, 'random')) < resumption['finished_runs_before_loss'] \
                else 'resumed machine'
            gpu['random_replay_step50'][str(sigma)] = dict(
                original_run_on=session, identical_adapter_digest=replay['adapter_digest_50'] == ref['adapter_digest_50'])
            if session == 'resumed machine':
                require(gpu['random_replay_step50'][str(sigma)]['identical_adapter_digest'],
                        f'same-machine replay sigma {sigma}')
            del scores, replay
            gc.collect()
            torch.cuda.empty_cache()
    status = 'passed'
    findings = []
    if gpu and not gpu['random_replay_step50']['0.25']['identical_adapter_digest']:
        status = 'passed_with_findings'
        findings.append('The step-50 replay of a run made on the lost machine did not reproduce its adapter digest on '
                        'the resumed machine (same device type, different physical GPU); the same-machine replay did.')
    out = dict(status=status, stage=74, journal_sha256=s69.sha256(JOURNAL),
               pilot_journal_sha256=s69.sha256(PILOT_JOURNAL), result_sha256=hashlib.sha256(RESULT.read_bytes()).hexdigest(),
               resumption=resumption, checks=checks, data=data_checks, costs_by_session=session_costs(journal, cal),
               environments=[{k: e[k] for k in ENV_KEYS + ('time', 'resumed_runs')} for e in envs],
               device=cal['device'], gpu=gpu, findings=findings,
               notes='CPU: digests, summary and gate recomputation, environments, budget, resumption reconciliation, '
                     'grid order, schedules, checkpoints, kept sets, QA shares, random512 subsets, data hashes, samples, '
                     'pools, decontamination and ID disjointness. GPU: align_grad scores of seed 37400 on both pools and '
                     'the first 50 random steps of seed 37400 on both pools. Full trajectories were not replayed.')
    OUTPUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: out[k] for k in ('status', 'resumption', 'checks', 'gpu', 'findings')}, indent=2,
                     ensure_ascii=False))


if __name__ == '__main__':
    main()

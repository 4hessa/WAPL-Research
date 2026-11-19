"""Independently verify stage 62 using recorded branch outcomes and older journals."""
import hashlib
import json
import math
import re
import statistics
from itertools import product
from pathlib import Path

import numpy as np

from tools.audit_pud_reuse import close, digest, read_json, read_rows, require

STATE = ('language', 'fold', 'seed')
BRANCHES = ('fresh', 'reuse')


def state_key(row):
    return tuple(row[k] for k in STATE)


def archived_results(stage60, stage61):
    for stage, path in [(stage60, 'results/pud-reuse'), (stage61, 'results/pud-reuse-matched')]:
        audit = read_json(path + '-audit.json')
        require(audit['status'] == 'passed' and digest(path+'.json') == audit['result_sha256'],
                f'historical result digest mismatch: {path}')
    journal = Path('results/pud-reuse-decisions.jsonl')
    require(digest(journal) == stage60['output_sha256'][journal.name], 'historical diagnostic journal')
    stage60_choices = {(*state_key(r), r['step']): r for r in read_rows(journal)}
    require(len(stage60_choices) == 128, 'incomplete stage-60 decisions')
    stage61_runs = {state_key(r): r for r in stage61['rows'] if r['arm'] == 'reuse4'}
    require(len(stage61_runs) == 8, 'incomplete stage-61 trajectories')
    return stage60_choices, stage61_runs


def rebuild_schedule(seed, pool_size):
    rng = np.random.default_rng(seed)
    for _ in range(32):
        rng.integers(pool_size, size=4)
    ids = rng.integers(pool_size, size=(20000, 4))
    uniform = rng.integers(4, size=20000)
    return ids, hashlib.sha256(ids.tobytes()+uniform.tobytes()).hexdigest()


def verify_branches(row):
    for name in ['parent', *BRANCHES]:
        for field in ('model_sha256', 'optimizer_sha256'):
            require(re.fullmatch('[0-9a-f]{64}', row[name+'_'+field]), 'invalid branch fingerprint')
    for branch in BRANCHES:
        require(type(row[branch+'_choice']) is int and 0 <= row[branch+'_choice'] < 4,
                'invalid winner')
        loss = row[branch+'_loss']
        require(math.isfinite(loss) and loss >= 0, 'nonfinite or negative target loss')
    agree = row['fresh_choice'] == row['reuse_choice']
    require(agree is row['agreement'], 'invalid choice agreement')
    if agree:
        for field in ('loss', 'model_sha256', 'optimizer_sha256'):
            require(row['fresh_'+field] == row['reuse_'+field], 'different states or loss for identical choice')
    require(row['delta'] == row['reuse_loss']-row['fresh_loss'], 'branch difference mismatch')


def check_paths(result, protocol, stage60, stage61):
    history, endpoints = archived_results(stage60, stage61)
    states = list(product(protocol['languages'], protocol['folds'], protocol['seeds']))
    coverage = {state_key(r): r for r in result['coverage']}
    prior_coverage = {state_key(r): r for r in stage61['coverage']}
    rows = result['rows']
    require(len(rows) == 128 and len(result['coverage']) == len(coverage) == 8, 'coverage size')
    require(set(states) == set(coverage) == set(prior_coverage), 'state coverage mismatch')
    errors, start_errors, initial_hash_changes = [], [], []
    for i, state in enumerate(states):
        entry = coverage[state]
        past = prior_coverage[state]
        seed = state[-1]
        ids, schedule_hash = rebuild_schedule(seed, entry['pool_chunks'])
        for field in ('pool_chunks', 'probe_chunks', 'target_chunks', 'schedule_sha256'):
            require(entry[field] == past[field], f'{state}: historical coverage {field}')
        require(entry['probe_chunks'] == 64 and schedule_hash == entry['schedule_sha256'],
                f'{state}: probe or schedule hash')
        sequence = rows[16*i:16*(i+1)]
        initial_hash_changes.append(any(entry['initial_'+k] != endpoints[state]['initial_'+k]
                                        for k in ['model_sha256', 'optimizer_sha256']))
        first_loss = endpoints[state]['losses']['0']
        require(entry['stage61_initial_loss_difference'] == entry['initial_loss'] - first_loss,
                'misreported initial loss discrepancy')
        start_errors.append(abs(entry['initial_loss'] - first_loss))
        require(start_errors[-1] <= np.spacing(np.float32(first_loss)), 'historical initial checkpoint')
        previous = None
        for step, row in enumerate(sequence):
            key = (*state, step)
            old = history[key]
            require((*state_key(row), row['step']) == key, 'out-of-order branch')
            require(row['age'] == old['age'] == step % 4, 'invalid branch age')
            require(row['candidate_ids'] == ids[step].tolist(), 'candidate index mismatch')
            require(row['reuse_choice'] == old['reuse_choice'] == endpoints[state]['choices'][step],
                    'stage-60 or stage-61 cached choice mismatch')
            require(row['fresh_choice'] == old['fresh_choice'], 'stage-60 fresh choice mismatch')
            verify_branches(row)
            for field in ('model_sha256', 'optimizer_sha256'):
                parent = entry['initial_'+field] if previous is None else previous['reuse_'+field]
                require(row['parent_'+field] == parent, 'broken cached-branch trajectory')
            if row['age'] == 0:
                require(row['agreement'] is True, 'age-zero refresh mismatch')
            previous = row
        expected_loss = endpoints[state]['losses']['16']
        delta = sequence[-1]['reuse_loss'] - expected_loss
        errors.append(delta)
        require(entry['stage61_step16_difference'] == delta, 'misreported endpoint discrepancy')
        require(abs(delta) <= np.spacing(np.float32(expected_loss)), 'historical endpoint >1 float32 ULP')
    return {'max_stage61_endpoint_step16_abs_difference': max(map(abs, errors)),
            'max_stage61_initial_loss_abs_difference': max(start_errors),
            'stage61_initial_state_hashes_differing_states': sum(initial_hash_changes),
            'stage61_endpoint_differences_by_state': errors}


def sign_counts(deltas):
    return dict(worse=sum(d > 0 for d in deltas), better=sum(d < 0 for d in deltas),
                tied=sum(d == 0 for d in deltas))


def check_metrics(block, group, count_key):
    deltas = [r['reuse_loss'] - r['fresh_loss'] for r in group]
    require(len(deltas) == block[count_key], 'observation count')
    for field, count in sign_counts(deltas).items():
        require(count == block[field], f'wrong sign count: {field}')
    for field, values in [('mean_delta', deltas), ('mean_abs_delta', list(map(abs, deltas)))]:
        if deltas:
            close(statistics.fmean(values), block[field], f'incorrect {field}')
        else:
            require(block[field] is None, 'empty conditional mean must be null')


def check_group(block, rows):
    changed = [row for row in rows if row['reuse_choice'] != row['fresh_choice']]
    require(len(changed) == block['changed'], 'changed decisions count')
    check_metrics(block, rows, 'n')
    check_metrics(block['changed_only'], changed, 'n')


def check_aggregate(result):
    summary = result['summary']
    rows = result['rows']
    groups = [(s, [r for r in rows if state_key(r) == state_key(s)]) for s in summary['states']]
    require(len(groups) == len({state_key(g[0]) for g in groups}) == 8, 'summary state count')
    require(set(summary['languages']) == {'ar', 'en'}, 'summary languages')
    groups.extend((block, [r for r in rows if r['language'] == l])
                  for l, block in summary['languages'].items())
    for block, group_records in groups:
        require(group_records, 'unknown state summary')
        check_group(block['all'], group_records)
        check_group(block['stale'], [r for r in group_records if r['age'] != 0])
        require(set(block['by_age']) == {'0', '1', '2', '3'}, 'age summary grid')
        for age in range(4):
            check_group(block['by_age'][str(age)], [r for r in group_records if r['age'] == age])
    for language, data in summary['languages'].items():
        state_means = []
        for fold, seed in product([0, 5], [28000, 28001]):
            stale = [r['reuse_loss'] - r['fresh_loss'] for r in rows
                     if state_key(r) == (language, fold, seed) and r['age'] > 0]
            require(len(stale) == 12, 'stale steps per state')
            state_means.append(statistics.mean(stale))
        close(data['stale']['mean_delta'], statistics.fmean(state_means), 'state weighting')


def main():
    path = Path('results/pud-reuse-one-step.json')
    plan_path = Path('research/pud-reuse-one-step-protocol.json')
    result, plan = read_json(path), read_json(plan_path)
    require(digest(plan_path) == '7611abdad40fad8c8643564319c95de78e640fea085c1dc98e90d8e0e93537da',
            'published stage-62 protocol changed')
    require(result['stage'] == 62 and result['protocol'] == plan, 'protocol mismatch')
    stage60 = read_json('results/pud-reuse.json')
    stage61 = read_json('results/pud-reuse-matched.json')
    metadata = result['provenance']
    require(metadata['deterministic'] is True and metadata['torch_threads'] == 1 and
            metadata['openblas_num_threads'] == '1', 'execution configuration')
    necessary = {str(plan_path), 'results/pud-reuse-audit.json', 'results/pud-reuse-matched-audit.json',
                 'results/pud-reuse-matched.json', 'results/pud-reuse.json', 'results/pud-reuse-decisions.jsonl',
                 'experiments/pud_reuse.py', 'experiments/pud_reuse_matched.py', 'experiments/pud_reuse_one_step.py',
                 'tests/test_pud_reuse_one_step.py'}
    require(necessary <= metadata['sha256'].keys(), 'missing provenance')
    for hashes in [stage61['provenance']['sha256'], metadata['sha256']]:
        for name, sha in hashes.items():
            require(digest(name) == sha, f'input hash differs: {name}')
    historical = check_paths(result, plan, stage60, stage61)
    require(historical['stage61_initial_state_hashes_differing_states'] == 0,
            'historical warmup model or optimizer differs')
    historical['stage61_cpu'] = stage61['provenance']['cpu']
    historical['stage62_cpu'] = metadata['cpu']
    check_aggregate(result)
    rows = result['rows']
    changed = [r for r in rows if not r['agreement']]
    n_refresh = sum(r['age'] == 0 for r in rows)
    require(len(changed) == 11 and n_refresh == 32, 'unequal coverage relative to prior decisions')
    per_language = {}
    for lang in plan['languages']:
        stale = [r for r in rows if r['language'] == lang and r['age'] > 0]
        deltas = [r['delta'] for r in stale]
        per_language[lang] = dict(stale_count=len(stale), changed=sum(not r['agreement'] for r in stale),
                                  signs=sign_counts(deltas), mean_signed_stale=statistics.fmean(deltas),
                                  mean_absolute_stale=statistics.fmean(map(abs, deltas)))
    audit = dict(stage=62, status='passed',
                 published_protocol_commit='5785ae929fa2f5db3488602ee293416d089c938c',
                 paired_decisions=len(rows), branch_loss_observations=2*len(rows),
                 stale_steps=len(rows)-n_refresh, changed_stale_steps=len(changed),
                 age_zero_exact_controls=n_refresh, agreeing_branch_losses_and_hashes_equal=len(rows)-len(changed),
                 cached_parent_state_continuity_verified=True, candidate_schedule_verified=True,
                 historical_candidate_positions_matched=True, input_and_protocol_hashes_verified=True,
                 state_language_age_sign_and_absolute_summaries_recomputed=True,
                 conditional_observations_and_empty_groups_verified=True,
                 historical_comparison=historical, stale_results_by_language=per_language,
                 training_runs_in_audit=0, result_sha256=digest(path), audit_script_sha256=digest(__file__))
    Path('results/pud-reuse-one-step-audit.json').write_text(json.dumps(audit, indent=2)+'\n')


if __name__ == '__main__':
    main()

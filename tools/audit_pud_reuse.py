"""Recompute stage-60 timing and decision diagnostics from persisted observations."""
import hashlib
import json
import math
import statistics
from itertools import product
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, field):
    require(math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12), field)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def check_timing(record, budget):
    key = (record['language'], record['fold'], record['seed'], record['arm'], record['repeat'])
    accepted = record['updates']
    attempted = record['attempted']
    require(type(accepted) is int and type(attempted) is int and
            accepted == len(record['choices']) and attempted in (accepted, accepted + 1),
            f'{key}: step counts')
    require(all(type(choice) is int and choice in range(4) for choice in record['choices']),
            f'{key}: invalid choice')
    for field in ['last_accepted_seconds', 'actual_seconds', 'overrun_seconds',
                  'initial_loss', 'final_loss']:
        require(math.isfinite(record[field]) and record[field] >= 0, f'{key}: {field}')
    require(record['actual_seconds'] >= budget, f'{key}: early exit before budget')
    require(record['last_accepted_seconds'] <= min(budget, record['actual_seconds']),
            f'{key}: deadline')
    close(record['overrun_seconds'], max(0., record['actual_seconds'] - budget), f'{key}: overrun')
    for field, decisions in [('accepted_refreshes', accepted), ('attempted_refreshes', attempted)]:
        expected = {'random': 0, 'fresh': decisions, 'reuse4': (decisions+3)//4}[record['arm']]
        require(type(record[field]) is int and record[field] == expected, f'{key}: {field}')


def historic():
    summary = read_json('results/pud-policy.json')
    journal_path = Path('results/pud-policy-records.jsonl')
    require(digest(journal_path) == summary['records_sha256'], 'stage-58 journal digest')
    fixed = {(r['language'], r['fold'], r['seed'], r['arm']): r for r in read_rows(journal_path)
             if r['mode'] == 'updates' and r['repeat'] == 0}
    require(len(fixed) == 320, 'stage-58 reference grid')
    return summary, fixed


def schedule(seed, size):
    rng = np.random.default_rng(seed)
    for _ in range(32):
        rng.integers(size, size=4)
    ids = rng.integers(size, size=(20000, 4))
    choices = rng.integers(4, size=20000)
    return hashlib.sha256(ids.tobytes() + choices.tobytes()).hexdigest(), choices


def check_observations(timed, diag, protocol):
    previous, fixed = historic()
    states = list(product(protocol['languages'], protocol['folds'], protocol['seeds']))
    arms = protocol['arms']
    nr = protocol['timing_repeats']
    n_steps = protocol['drift_steps']
    interval = protocol['refresh_interval']
    require(len(timed) == len(states)*nr*len(arms) == 72, 'timed grid size')
    require(len(diag) == len(states)*n_steps == 128, 'diagnostic grid size')
    by_age = {age: [] for age in range(interval)}
    offset = 0
    historical_errors = []
    for i, (lang, fold, seed) in enumerate(states):
        state = lang, fold, seed
        previous_cov = next(c for c in previous['coverage'][lang] if c['fold'] == fold)
        schedule_hash, uniform = schedule(seed, previous_cov['candidate_chunks'])
        historical = fixed[(*state, 'alignment')]
        require(schedule_hash == historical['schedule_sha256'], f'{state}: schedule')
        track = diag[i*n_steps:(i+1)*n_steps]
        for step, obs in enumerate(track):
            require((obs['language'], obs['fold'], obs['seed'], obs['step']) == (*state, step),
                    'unexpected diagnostic row')
            age = step % interval
            require(obs['age'] == age, 'wrong gradient age')
            require(all(type(obs[k]) is int and obs[k] in range(4)
                        for k in ['fresh_choice', 'reuse_choice']), 'invalid diagnostic choice')
            equal = obs['fresh_choice'] == obs['reuse_choice']
            by_age[age].append(equal)
            if age == 0:
                require(equal and obs['refresh_parity'] is True, 'age-zero parity')
            else:
                require(obs['refresh_parity'] is None, 'unexpected refresh parity check')
        require(track[0]['fresh_choice'] == historical['choices'][0], 'fresh diagnostic at step zero')
        group_rows = []
        for repeat in range(nr):
            start = (i + repeat) % len(arms)
            rotated = arms[start:] + arms[:start]
            for position, arm in enumerate(rotated):
                obs = timed[offset]
                offset += 1
                key = (*state, repeat, position, arm)
                require(tuple(obs[k] for k in ['language', 'fold', 'seed', 'repeat', 'position', 'arm']) == key,
                        f'out-of-order observation; expected {key}')
                require(obs['schedule_sha256'] == schedule_hash, 'schedule hash mismatch')
                # The stage-58 and stage-60 CPUs differ. Both models use float32.
                error = abs(obs['initial_loss'] - historical['initial_loss'])
                historical_errors.append(error)
                require(error <= np.spacing(np.float32(historical['initial_loss'])),
                        f'{key}: initial loss differs by more than one float32 ULP')
                check_timing(obs, protocol['budget_seconds'])
                if arm == 'random':
                    reference = uniform[:obs['updates']].tolist()
                elif arm == 'fresh':
                    reference = historical['choices']
                else:
                    reference = [r['reuse_choice'] for r in track]
                prefix = min(len(reference), obs['updates'])
                require(obs['choices'][:prefix] == reference[:prefix], f'{key}: inconsistent choices')
                group_rows.append(obs)
        require(len({r['initial_loss'] for r in group_rows}) == 1, 'unequal initial losses within a stage-60 state')
        for arm in arms:
            runs = [r for r in group_rows if r['arm'] == arm]
            longest = max(runs, key=lambda r: r['updates'])
            require(sorted(r['position'] for r in runs) == list(range(len(arms))),
                    f'{state}: position balance')
            for r in runs:
                require(r['choices'] == longest['choices'][:r['updates']], 'nonreproducible prefix')
                for comparison in runs:
                    if r['updates'] == comparison['updates']:
                        require(r['final_loss'] == comparison['final_loss'], 'loss for matching trajectory')
    return by_age, max(historical_errors)


def check_summary_group(observations, group):
    for arm in ('reuse4', 'fresh', 'random'):
        same_arm = [r for r in observations if r['arm'] == arm]
        expected_metrics = {'final_loss', 'updates', 'actual_seconds', 'overrun_seconds',
                            'accepted_refreshes', 'attempted_refreshes'}
        require(set(group['arms'][arm]) == expected_metrics, f'{arm}: incomplete metrics')
        for key, saved in group['arms'][arm].items():
            close(saved, statistics.fmean(r[key] for r in same_arm), f'{arm} {key} mean')
    for control in ('fresh', 'random'):
        contrast = group['contrasts']['reuse4_minus_' + control]
        for key in ('updates', 'final_loss'):
            paired = {}
            for r in observations:
                if r['arm'] in ('reuse4', control):
                    pair_key = (r['language'], r['fold'], r['seed'], r['repeat'])
                    paired.setdefault(pair_key, {})[r['arm']] = r[key]
            expected = statistics.fmean(v['reuse4'] - v[control] for v in paired.values())
            close(contrast[key], expected, f'paired reuse4 minus {control}, {key}')


def check_coverage(result, protocol, previous):
    keys = ('language', 'fold', 'seed')
    expected = list(product(protocol['languages'], protocol['folds'], protocol['seeds']))
    require(len(result['coverage']) == len(expected), 'coverage count')
    prior_profile = read_json('results/pud-policy-profile.json')
    for state, cov in zip(expected, result['coverage']):
        lang, fold, seed = state
        require(tuple(cov[k] for k in keys) == state, 'coverage order')
        old = next(r for r in prior_profile['coverage'] if tuple(r[k] for k in keys) == state)
        old_fold = next(r for r in previous['coverage'][lang] if r['fold'] == fold)
        require(cov['pool_chunks'] == old['pool_chunks'] == old_fold['candidate_chunks'] and
                cov['target_chunks'] == old_fold['target_chunks'] and
                cov['probe_chunks'] == old['probe_chunks'] == old_fold['probe_chunks'] == 64,
                f'{state}: partition sizes')
        require(cov['schedule_sha256'] == old['schedule_sha256'], f'{state}: coverage hash')


def main():
    result_path = Path('results/pud-reuse.json')
    protocol_path = Path('research/pud-reuse-protocol.json')
    timed_path = Path('results/pud-reuse-records.jsonl')
    decisions_path = Path('results/pud-reuse-decisions.jsonl')
    result = read_json(result_path)
    protocol = read_json(protocol_path)
    fixed_protocol_digest = 'a967caf9ba18ea51ba54a2933238796bc94031e17bc20c41db359ac4197ee495'
    require(digest(protocol_path) == fixed_protocol_digest, 'published stage-60 protocol changed')
    require(protocol == result['protocol'] and result['stage'] == protocol['stage'] == 60,
            'run protocol mismatch')
    require(result['runs'] == 72 and result['diagnostic_decisions'] == 128, 'result counts')
    meta = result['provenance']
    require(meta['torch_threads'] == 1 and meta['openblas_num_threads'] == '1' and
            meta['deterministic'] is True, 'thread or determinism metadata')
    require(math.isfinite(result['wall_seconds']) and result['wall_seconds'] > 0, 'wall clock')
    baseline = read_json('results/pud-policy-profile.json')['provenance']['sha256']
    expected_paths = {'research/pud-reuse-protocol.json', 'experiments/pud_reuse.py',
                      'tests/test_pud_reuse.py', 'results/pud-policy-profile.json'}
    require(set(meta['sha256']) == set(baseline) | expected_paths and
            baseline.items() <= meta['sha256'].items(), 'input hash coverage')
    for path, sha in meta['sha256'].items():
        require(digest(path) == sha, f'changed input {path}')
    require(result['output_sha256'] == {p.name: digest(p) for p in (timed_path, decisions_path)},
            'output digests')
    timed = read_rows(timed_path)
    decisions = read_rows(decisions_path)
    by_age, max_initial_error = check_observations(timed, decisions, protocol)
    check_coverage(result, protocol, read_json('results/pud-policy.json'))
    summary = result['summary']
    states = list(product(protocol['languages'], protocol['folds'], protocol['seeds']))
    require(len(summary['states']) == len(states), 'summary state count')
    for state, metrics in zip(states, summary['states']):
        fields = ('language', 'fold', 'seed')
        same_state = [r for r in timed if tuple(r[k] for k in fields) == state]
        require(tuple(metrics[k] for k in fields) == state, 'state summary keys')
        check_summary_group(same_state, metrics)
    for lang in protocol['languages']:
        check_summary_group([r for r in timed if r['language'] == lang], summary['languages'][lang])
    drift = [dict(age=a, decisions=len(hits), agreement=sum(hits)) for a, hits in by_age.items()]
    require(all(d['decisions'] == 32 for d in drift) and drift == summary['drift_by_age'],
            'diagnostic counts per cache age')
    audit = dict(stage=60, status='passed', published_protocol_commit='e9a03225160e06c66ba8c8746465d688a307ce9d',
                 measured_runs=72, states=8, decision_comparisons=128, drift_by_age=drift,
                 within_stage_initial_losses_exactly_paired=True,
                 historical_initial_loss_tolerance='one float32 ULP at historical initial loss',
                 max_historical_initial_loss_absolute_difference=max_initial_error,
                 schedule_hashes_matched_stage58=True, coverage_matched_stages58_59=True,
                 all_three_orders_balanced_at_each_state=True, accepted_and_attempted_refresh_counts_verified=True,
                 paired_contrasts_and_arm_means_verified=True, deadlines_and_overruns_verified=True,
                 random_sequences_match_schedules=True, fresh_prefixes_match_stage58=True,
                 reuse4_prefixes_match_drift_trajectory=True, within_arm_choice_prefixes_verified=True,
                 age_zero_checks='32 matching winners; model and optimizer parity recorded online',
                 source_hashes_verified=True, record_hashes_verified=True, audit_training_runs=0,
                 result_sha256=digest(result_path), audit_script_sha256=digest(__file__))
    Path('results/pud-reuse-audit.json').write_text(json.dumps(audit, indent=2)+'\n')


if __name__ == '__main__':
    main()

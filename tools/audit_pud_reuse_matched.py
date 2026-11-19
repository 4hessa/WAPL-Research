"""Independent audit of a matched-update pilot using recorded data only."""
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import numpy as np

from tools.audit_pud_reuse import close, digest, historic, read_json, read_rows, require, schedule

ARMS = ('reuse4', 'fresh', 'random')
CHECKPOINTS = (0, 16, 32, 64)
PAIRS = (('reuse4', 'fresh'), ('reuse4', 'random'), ('fresh', 'random'))
STATE_FIELDS = ('language', 'fold', 'seed')


def state_key(row):
    return tuple(row[k] for k in STATE_FIELDS)


def input_checks(previous, timed):
    by_run = defaultdict(list)
    for row in timed:
        by_run[(*state_key(row), row['arm'])].append(row)
    for _, repeats in by_run.items():
        require(sorted(r['repeat'] for r in repeats) == [0, 1, 2], 'unpaired stage-60 input')
    require(len(timed) == 72 and len(by_run) == 24, 'incomplete stage-60 input')
    reference_coverage = {}
    for language, folds in previous['coverage'].items():
        for row in folds:
            reference_coverage[language, row['fold']] = row
    return by_run, reference_coverage


def check_runs(result, old60, timed, plan):
    old58, fixed = historic()
    historical_runs, fold_reference = input_checks(old58, timed)
    coverage = {state_key(r): r for r in result['coverage']}
    expected_states = list(product(plan['languages'], plan['folds'], plan['seeds']))
    require(len(coverage) == len(result['coverage']) == 8 and set(coverage) == set(expected_states),
            'coverage grid mismatch')
    rows = result['rows']
    indexed = {(*state_key(r), r['arm']): r for r in rows}
    require(len(rows) == len(indexed) == 24 and
            set(indexed) == {(*s, a) for s in expected_states for a in ARMS}, 'trajectory grid mismatch')
    previous_diagnostics = read_rows('results/pud-reuse-decisions.jsonl')
    require(digest('results/pud-reuse-decisions.jsonl') ==
            old60['output_sha256']['pud-reuse-decisions.jsonl'], 'historical diagnostic digest')
    errors = defaultdict(list)
    overlaps = Counter()
    ref_cpu = old58['metadata']['cpu'].split(':')[-1].strip()
    stage60_cpu = old60['provenance']['cpu']
    current_cpu = result['provenance']['cpu']
    for state in expected_states:
        lang, f, seed = state
        cov = coverage[state]
        expected_cov = fold_reference[lang, f]
        require(cov['pool_chunks'] == expected_cov['candidate_chunks'] and
                cov['probe_chunks'] == expected_cov['probe_chunks'] == 64 and
                cov['target_chunks'] == expected_cov['target_chunks'], 'partition coverage')
        schedule_hash, uniform = schedule(seed, cov['pool_chunks'])
        require(schedule_hash == cov['schedule_sha256'] == fixed[(*state, 'alignment')]['schedule_sha256'],
                'historical schedule digest mismatch')
        state_rows = [indexed[(*state, a)] for a in ARMS]
        for field in ('initial_model_sha256', 'initial_optimizer_sha256'):
            hashes = {r[field] for r in state_rows}
            require(len(hashes) == 1 and re.fullmatch('[0-9a-f]{64}', hashes.pop()), 'unpaired initial state')
        require(len({r['losses']['0'] for r in state_rows}) == 1, 'unpaired initial loss')
        for arm, row in zip(ARMS, state_rows):
            require(row['schedule_sha256'] == schedule_hash and
                    row['refreshes'] == {'reuse4': 16, 'fresh': 64, 'random': 0}[arm], 'refreshes or schedule')
            require(set(row['losses']) == set(map(str, CHECKPOINTS)) and
                    all(math.isfinite(v) and v >= 0 for v in row['losses'].values()), 'invalid checkpoints')
            selections = row['choices']
            require(len(selections) == 64 and all(type(v) is int and v in range(4) for v in selections),
                    'invalid 64 choices')
            if arm == 'reuse4':
                trace = sorted((r for r in previous_diagnostics if state_key(r) == state),
                               key=lambda r: r['step'])
                require([r['step'] for r in trace] == list(range(16)) and
                        selections[:16] == [r['reuse_choice'] for r in trace],
                        'stage-60 cached diagnostic choices differ')
            if arm == 'random':
                require(selections == uniform[:64].tolist(), 'random sequence disagrees with schedule')
            if arm in ('random', 'fresh'):
                old_arm = 'alignment' if arm == 'fresh' else 'random'
                require(selections == fixed[(*state, old_arm)]['choices'][:64], 'stage-58 choices differ')
            historical_deltas = row['stage60_matched_prefix_lengths']
            require(len(historical_deltas) == 3, 'archived prefix-length count')
            for past in historical_runs[(*state, arm)]:
                shared = min(len(selections), past['updates'])
                require(selections[:shared] == past['choices'][:shared] and
                        past['schedule_sha256'] == schedule_hash, f'{state} {arm}: stage-60 prefix differs')
                require(historical_deltas[past['repeat']] == shared, 'reported overlap mismatch')
                overlaps[arm] += shared
                if past['updates'] in CHECKPOINTS:
                    error = abs(row['losses'][str(past['updates'])] - past['final_loss'])
                    errors['overlapping_final_stage60'].append(error)
                    require(error <= np.spacing(np.float32(past['final_loss'])),
                            f'{state} {arm}: different final loss at matched update count')
            initial_loss = row['losses']['0']
            old_input_loss = historical_runs[(*state, arm)][0]['initial_loss']
            require(row['stage60_initial_loss_difference'] == initial_loss - old_input_loss,
                    'recorded historical initial-loss difference')
            for stage, old_loss in [('stage58', fixed[(*state, 'alignment')]['initial_loss']),
                                    ('stage60', historical_runs[(*state, arm)][0]['initial_loss'])]:
                delta = abs(initial_loss - old_loss)
                errors[stage].append(delta)
                require(delta <= float(np.spacing(np.float32(old_loss))),
                        f'{stage} {state}: initial loss differs by >1 float32 ULP')
    return dict(cpu_by_stage={'58': ref_cpu, '60': stage60_cpu, '61': current_cpu},
                stage60_overlapping_choice_positions_across_repeats=dict(overlaps),
                max_abs_historical_initial_loss_difference={k: max(errors[k]) for k in ['stage58','stage60']},
                stage60_matched_checkpoint_final_loss_comparisons=len(errors['overlapping_final_stage60']),
                stage60_max_abs_matched_checkpoint_final_loss_difference=max(errors['overlapping_final_stage60'], default=None))


def recompute(result):
    rows = result['rows']
    by_state = defaultdict(dict)
    for row in rows:
        by_state[state_key(row)][row['arm']] = row['losses']
    state_summary = {state_key(s): s for s in result['summary']['states']}
    require(len(state_summary) == len(result['summary']['states']) == 8 and
            set(state_summary) == set(by_state), 'state summary grid')
    require(set(result['summary']['languages']) == {'ar', 'en'}, 'language summary grid')
    means_by_language = {}
    for state, losses in by_state.items():
        for a, b in PAIRS:
            field = f'{a}_minus_{b}'
            close(state_summary[state]['contrasts'][field], losses[a]['64']-losses[b]['64'],
                  'state final contrast')
    for lang, group in result['summary']['languages'].items():
        observations = [losses for state, losses in by_state.items() if state[0] == lang]
        require(len(observations) == 4 and set(group['arms']) == set(ARMS), 'invalid language grid')
        for a in ARMS:
            for checkpoint in map(str, CHECKPOINTS):
                mean = statistics.fmean(losses[a][checkpoint] for losses in observations)
                close(mean, group['arms'][a]['losses'][checkpoint], 'language checkpoint mean')
        means = {}
        for a, b in PAIRS:
            key = f'{a}_minus_{b}'
            means[key] = {k: statistics.fmean(loss[a][k] - loss[b][k] for loss in observations)
                          for k in map(str, CHECKPOINTS)}
            for checkpoint, difference in means[key].items():
                close(group['contrasts_by_checkpoint'][key][checkpoint], difference,
                      'language paired checkpoint contrast')
            close(means[key]['64'], group['contrasts'][key], 'language final contrast')
        means_by_language[lang] = means
    return means_by_language


def main():
    path = Path('results/pud-reuse-matched.json')
    result = read_json(path)
    plan = read_json('research/pud-reuse-matched-protocol.json')
    require(digest('research/pud-reuse-matched-protocol.json') ==
            '08462023952551e422e4df030159add2152385e4eaded49ebd4597da39b78f33',
            'published stage-61 design changed')
    require(result['protocol'] == plan and result['stage'] == plan['stage'] == 61,
            'incorrect stage or protocol')
    require(math.isfinite(result['wall_seconds']) and result['wall_seconds'] > 0, 'total clock')
    old60 = read_json('results/pud-reuse.json')
    previous_audit = read_json('results/pud-reuse-audit.json')
    require(previous_audit['status'] == 'passed' and previous_audit['result_sha256'] ==
            digest('results/pud-reuse.json'), 'prior audited summary changed')
    history_path = Path('results/pud-reuse-records.jsonl')
    require(digest(history_path) == old60['output_sha256'][history_path.name], 'timing journal digest')
    provenance = result['provenance']
    require(provenance['torch_threads'] == 1 and provenance['openblas_num_threads'] == '1' and
            provenance['deterministic'] is True, 'execution configuration')
    required_paths = {'research/pud-reuse-matched-protocol.json', 'experiments/pud_reuse_matched.py',
                      'tests/test_pud_reuse_matched.py', 'experiments/pud_reuse.py', 'results/pud-reuse.json',
                      'results/pud-reuse-audit.json', 'results/pud-reuse-records.jsonl'}
    require(required_paths <= set(provenance['sha256']), 'missing input hash')
    for hashes in [old60['provenance']['sha256'], provenance['sha256']]:
        for filename, sha in hashes.items():
            require(digest(filename) == sha, f'input digest mismatch for {filename}')
    recorded = check_runs(result, old60, read_rows(history_path), plan)
    means = recompute(result)
    audit = dict(stage=61, status='passed', design_commit='2c28ea5ee11fee85248c8e74ec0869309f96af58',
                 audited_trajectories=24, completed_updates_per_trajectory=64, states=8,
                 target_loss_observations=96, checkpoints=list(CHECKPOINTS), training_runs_in_audit=0,
                 within_stage_initial_loss_model_optimizer_exactly_paired=True,
                 historical_initial_loss_tolerance='one float32 ULP at historical initial loss',
                 stage58_fixed_alignment_and_random_first64_choices_verified_per_state=True,
                 stage60_reuse4_diagnostic_first16_choices_verified_per_state=True,
                 stage60_all_available_timed_prefixes_verified=True, schedules_reconstructed=True,
                 expected_refresh_counts_verified=True, input_hashes_verified=True,
                 per_state_final_contrasts_and_paired_language_checkpoint_means_verified=True,
                 language_paired_loss_differences=means, historical_comparison=recorded,
                 result_sha256=digest(path), audit_script_sha256=digest(__file__))
    Path('results/pud-reuse-matched-audit.json').write_text(json.dumps(audit, indent=2)+'\n')


if __name__ == '__main__':
    main()

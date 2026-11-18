"""Verify stage-59 timing records against the completed stage-58 journal."""
import hashlib
import json
import math
import statistics
from itertools import product

import numpy as np
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def stage58_fixed_runs():
    summary = load_json('results/pud-policy.json')
    journal = Path('results/pud-policy-records.jsonl')
    require(summary['records_sha256'] == sha256(journal), 'stage-58 journal checksum mismatch')
    choices = {}
    seeds = set(range(28000, 28008))
    for line in journal.read_text().splitlines():
        run = json.loads(line)
        if run['mode'] == 'updates':
            key = (run['language'], run['fold'], run['seed'], run['arm'])
            require(key not in choices and run['repeat'] == 0, f'duplicate reference {key}')
            require(len(run['choices']) == run['updates'] == 128, f'truncated reference {key}')
            choices[key] = run
    expected = {(lang, fold, seed, arm) for lang in ['ar', 'en'] for fold in range(10)
                for seed in seeds for arm in ['alignment', 'random']}
    require(choices.keys() == expected, 'incomplete stage-58 fixed-update grid')
    return choices


COMPONENTS = ('probe_clone', 'probe_gradient', 'candidate_clones', 'candidate_updates',
              'candidate_delta_scores', 'residual')


def verify_summaries(rows, summary):
    plain = [r['plain_ns'] for r in rows]
    hooked = [r['profiled_ns'] for r in rows]
    differences = [h - p for p, h in zip(plain, hooked)]
    n = len(rows)

    def close(actual, expected, field):
        require(math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12),
                f'summary mismatch in {field}')

    require(summary['pairs'] == n, 'wrong pair count')
    for name, expected in {
            'plain_mean_ms': sum(plain) / (1e6*n),
            'profiled_mean_ms': sum(hooked) / (1e6*n),
            'paired_difference_mean_ms': sum(differences) / (1e6*n),
            'paired_difference_median_ms': statistics.median(differences) / 1e6,
            'overhead_pct': 100 * sum(differences) / sum(plain)}.items():
        close(summary[name], expected, name)
    for k in COMPONENTS:
        total = sum(r['component_ns'][k] for r in rows)
        close(summary['component_mean_ms'][k], total / (n * 1e6), k)
        close(summary['component_share_pct'][k], 100 * total / sum(hooked), k)
    close(sum(summary['component_share_pct'].values()), 100., 'sum of shares')


def main():
    result_path = Path('results/pud-policy-profile.json')
    prof = load_json(result_path)
    provenance = prof['provenance']
    stage58 = load_json('results/pud-policy.json')
    expected_hashes = stage58['metadata']['sha256']
    extras = {'experiments/pud_policy_profile.py', 'tests/test_pud_policy_profile.py',
              'results/pud-policy-environment.json', 'results/pud-policy-records.jsonl'}
    hashes = provenance['sha256']
    require(expected_hashes.items() <= hashes.items(), 'historical input hash mismatch')
    require(set(expected_hashes) | extras == set(hashes), 'missing provenance')
    for name, digest in hashes.items():
        require(sha256(name) == digest, f'file hash mismatch: {name}')
    fixed = stage58_fixed_runs()
    states = list(product(('ar', 'en'), (0, 5), (28000, 28001)))
    rows = prof['rows']
    design = prof['design']
    require(prof['stage'] == 59 and len(rows) == 48 and len(prof['coverage']) == len(states),
            'unexpected run grid')
    require((design['languages'], design['folds'], design['seeds']) ==
            (['ar', 'en'], [0, 5], [28000, 28001]), 'unexpected state grid')
    require(design['repeats_per_state'] == 6 and design['warmup_updates'] == 32
            and design['warmup_batch'] == 4 and design['target_evaluations'] == 0,
            'unexpected design')
    require(provenance['torch_threads'] == 1 and provenance['openblas_num_threads'] == '1'
            and provenance['deterministic'] is True, 'thread/determinism mismatch')

    for i, (language, fold, seed) in enumerate(states):
        cov = prof['coverage'][i]
        require((cov['language'], cov['fold'], cov['seed']) == states[i], 'coverage order')
        old = fixed[language, fold, seed, 'alignment']
        old_cov = next(c for c in stage58['coverage'][language] if c['fold'] == fold)
        pool = old_cov['candidate_chunks']
        require(cov['pool_chunks'] == pool and cov['probe_chunks'] == old_cov['probe_chunks'] == 64,
                'different pool or probe size')
        rng = np.random.default_rng(seed)
        for _ in range(32):
            rng.integers(pool, size=4)
        schedule = rng.integers(pool, size=(20000, 4))
        uniform_choices = rng.integers(4, size=20000)
        digest = hashlib.sha256(schedule.tobytes() + uniform_choices.tobytes()).hexdigest()
        require(digest == cov['schedule_sha256'] == old['schedule_sha256'], 'different schedule')
        require(cov['candidate_ids'] == schedule[0].tolist(), 'different first candidates')
        state_rows = rows[6*i:6*i+6]
        for repeat, row in enumerate(state_rows):
            require((row['language'], row['fold'], row['seed'], row['repeat']) ==
                    (language, fold, seed, repeat), 'unexpected observation order')
            require(row['profiled_first'] is bool((i+repeat) % 2) and row['parity'] is True,
                    'order or recorded parity disagrees')
            require(row['winner'] == old['choices'][0] == cov['historical_winner'],
                    'winner differs from stage 58 fixed-budget decision')
            components = row['component_ns']
            require(set(components) == set(COMPONENTS), 'component mismatch')
            require(all(type(v) is int and v >= 0 for v in components.values()),
                    'invalid component duration')
            require(all(type(row[k]) is int and row[k] > 0 for k in ('plain_ns', 'profiled_ns')),
                    'invalid whole decision timing')
            require(sum(components.values()) == row['profiled_ns'], 'non-additive timings')
        require(sum(r['profiled_first'] for r in state_rows) == 3, 'unbalanced order')

    verify_summaries(rows, prof['summary'])
    output = dict(stage=59, status='passed', states=8, measured_pairs=48,
                  winning_positions_matching_stage58=48, schedules_verified=8,
                  stage58_fixed_alignment_first_choices=[{
                      'language': l, 'fold': f, 'seed': s, 'winner': fixed[l, f, s, 'alignment']['choices'][0]
                  } for l, f, s in states],
                  balanced_order_per_state=True, all_component_sums_match_ns=True,
                  aggregate_statistics_recomputed=True, provenance_verified=True,
                  state_parity_basis='recorded per-pair check of winners, model parameters and optimizer state',
                  result_sha256=sha256(result_path), audit_script_sha256=sha256(__file__))
    Path('results/pud-policy-profile-audit.json').write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()

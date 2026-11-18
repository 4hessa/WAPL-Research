"""Audit complete policy records without rerunning timing-sensitive training."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def main():
    plan = json.loads(Path('research/pud-policy-protocol.json').read_text())
    result = json.loads(Path('results/pud-policy.json').read_text())
    source = Path('results/pud-policy-records.jsonl')
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == result['records_sha256']
    for name, digest in result['metadata']['sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    readiness = json.loads(Path('results/pud-policy-readiness.json').read_text())
    hashes = result['metadata']['sha256']
    assert hashes['research/pud-policy-protocol.json'] == readiness['protocol_sha256']
    assert all(hashes[name] == digest for name, digest in plan['input_sha256'].items())
    environment = json.loads(Path('results/pud-policy-environment.json').read_text())
    assert result['metadata'] == environment and environment['status'] == 'complete'
    manifest = json.loads(Path('data/pud_sources.json').read_text())
    folds = manifest['folds_source_indices']
    for language in plan['languages']:
        path = Path(manifest['processed'][language]['path'])
        sentences = [json.loads(line) for line in path.read_text().splitlines()]
        lengths = {s['source_index']: len(s['text'].encode('utf-8')) + 1 for s in sentences}
        for i, fold in enumerate(folds):
            probe = set(folds[(i + 1) % len(folds)])
            assert not probe.intersection(fold)
            candidates = lengths.keys() - set(fold) - probe
            # The preparation joins sentences with a newline, including the final one.
            expected = {
                'fold': i,
                'candidate_chunks': sum(lengths[k] for k in candidates) // 65,
                'probe_chunks': min(64, sum(lengths[k] for k in probe) // 65),
                'target_chunks': sum(lengths[k] for k in fold) // 65,
            }
            assert result['coverage'][language][i] == expected
    lookup = {}
    for r in rows:
        key = tuple(r[k] for k in ['language', 'fold', 'seed', 'mode', 'repeat', 'arm'])
        assert key not in lookup
        lookup[key] = r
        assert all(np.isfinite(r[k]) for k in ['initial_loss', 'final_loss', 'actual_seconds', 'overrun_seconds', 'last_accepted_seconds'])
        assert r['updates'] == len(r['choices'])
        assert all(0 <= c < 4 for c in r['choices'])
        assert r['attempted'] in [r['updates'], r['updates']+1]
        assert 0 <= r['last_accepted_seconds'] <= r['actual_seconds']
        if r['mode'] == 'time':
            assert r['last_accepted_seconds'] <= 3.
            assert r['overrun_seconds'] == max(0., r['actual_seconds']-3.)
        else:
            assert r['updates'] == 128
            assert r['overrun_seconds'] == 0.
    expected_order = []
    for lang in plan['languages']:
        for fold in plan['folds']:
            n = result['coverage'][lang][fold]['candidate_chunks']
            for seed in plan['seeds']:
                rng = np.random.default_rng(seed)
                for _ in range(32):
                    rng.integers(n, size=4)
                ids = rng.integers(n, size=(20000, 4))
                choices = rng.integers(4, size=20000)
                h = hashlib.sha256(ids.tobytes()+choices.tobytes()).hexdigest()
                group = []
                modes = ['updates', 'time'] if (fold+seed)%2 == 0 else ['time', 'updates']
                for mode in modes:
                    for repeat in range(2 if mode == 'time' else 1):
                        arms = plan['arms'] if (fold+seed+repeat)%2 == 0 else plan['arms'][::-1]
                        for arm in arms:
                            key = (lang, fold, seed, mode, repeat, arm)
                            expected_order.append(key)
                            r = lookup[key]
                            group.append(r)
                            assert r['schedule_sha256'] == h
                            if arm == 'random':
                                assert r['choices'] == choices[:r['updates']].tolist()
                assert len({r['initial_loss'] for r in group}) == 1
    assert list(lookup) == expected_order
    assert len(rows) == result['runs'] == 960
    for lang in plan['languages']:
        for mode in ['time', 'updates']:
            per = {}
            for arm in plan['arms']:
                rr = [r for r in rows if (r['language'], r['mode'], r['arm']) == (lang, mode, arm)]
                per[arm] = np.array([np.mean([r['final_loss'] for r in rr if r['seed'] == seed]) for seed in plan['seeds']])
                np.testing.assert_allclose(per[arm], result['outputs'][lang][mode][arm]['per_seed_mean_loss'], rtol=0, atol=1e-14)
                for field in ['final_loss', 'updates', 'actual_seconds', 'overrun_seconds']:
                    assert abs(np.mean([r[field] for r in rr])-result['outputs'][lang][mode][arm][field]) < 1e-14
            # Reconstruct nested averaging exactly for bootstrap rounding agreement.
            for arm in plan['arms']:
                seed_means = []
                for seed in plan['seeds']:
                    fold_means = []
                    for fold in plan['folds']:
                        count = plan['budgets']['time_repeats'] if mode == 'time' else 1
                        losses = [
                            lookup[(lang, fold, seed, mode, repeat, arm)]['final_loss']
                            for repeat in range(count)
                        ]
                        fold_means.append(np.mean(losses))
                    seed_means.append(np.mean(fold_means))
                per[arm] = np.array(seed_means)
            d = per['alignment'] - per['random']
            comparison = result['comparisons'][lang][mode]
            np.testing.assert_array_equal(d, comparison['per_seed_difference'])
            assert float(d.mean()) == comparison['difference']
            bs = np.random.default_rng(5701).choice(d, (10000, 8), replace=True).mean(axis=1)
            np.testing.assert_array_equal(np.quantile(bs, [.025, .975]), comparison['ci95'])
    tests = Path('results/pud-policy-tests.log').read_text()
    match = re.search(r'Ran (\d+) tests? in', tests)
    assert match and tests.rstrip().endswith('OK')
    audit = {
        'status': 'passed', 'records': len(rows), 'tests': int(match.group(1)),
        'source_and_record_hashes_verified': True, 'frozen_protocol_verified': True,
        'coverage_from_sentences_verified': True, 'grid_order_and_schedules_verified': True,
        'random_choice_sequences_verified': True, 'paired_metrics_and_bootstrap_verified': True,
        'timing_training_rerun': False,
        'result_sha256': hashlib.sha256(Path('results/pud-policy.json').read_bytes()).hexdigest(),
    }
    Path('results/pud-policy-audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()

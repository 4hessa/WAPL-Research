"""Compare the stage-60 PUD policies after equal numbers of accepted updates."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import time

import numpy as np
import torch

from experiments import pud_policy as original
from experiments.pud_policy_profile import provenance
from experiments.pud_reuse import CachedDecision


PROTOCOL = Path('research/pud-reuse-matched-protocol.json')
PREVIOUS = Path('results/pud-reuse.json')
PREVIOUS_RUNS = Path('results/pud-reuse-records.jsonl')
PAIRS = (('reuse4', 'fresh'), ('reuse4', 'random'), ('fresh', 'random'))
STATE_KEYS = ('language', 'fold', 'seed')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_fingerprint(value):
    """Hash tensor values and typed metadata, independent of storage addresses."""
    def encode(item):
        if isinstance(item, torch.Tensor):
            data = item.detach().cpu().contiguous().numpy().tobytes()
            return ['tensor', str(item.dtype), list(item.shape), hashlib.sha256(data).hexdigest()]
        if isinstance(item, dict):
            pairs = [[encode(k), encode(v)] for k, v in item.items()]
            return ['dict', sorted(pairs, key=lambda pair: json.dumps(pair[0]))]
        if isinstance(item, (list, tuple)):
            return [type(item).__name__, [encode(v) for v in item]]
        return [type(item).__name__, item]

    serialized = json.dumps(encode(value), separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


def run_arm(base, opt, pool, probe, ids, random_choices, target, arm, plan):
    """Commit one chosen AdamW update at every step; evaluate only between updates."""
    if arm not in ('random', 'fresh', 'reuse4'):
        raise ValueError(f'Unknown arm: {arm}')
    model, current_opt = original.clone(base, opt)
    torch.testing.assert_close(model.state_dict(), base.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(current_opt.state_dict(), opt.state_dict(), rtol=0, atol=0)
    start_hashes = dict(initial_model_sha256=state_fingerprint(model.state_dict()),
                        initial_optimizer_sha256=state_fingerprint(current_opt.state_dict()))
    cache = CachedDecision(plan['refresh_interval']) if arm == 'reuse4' else None
    selector = cache or original.decision
    selector_arm = 'random' if arm == 'random' else 'alignment'
    choices, losses = [], {}
    for step in range(plan['updates'] + 1):
        if step in plan['checkpoints']:
            loss = original.evaluate(model, target)
            if not np.isfinite(loss):
                raise ValueError(f'Nonfinite loss at checkpoint {step}')
            losses[str(step)] = loss
        if step < plan['updates']:
            model, current_opt, winner = selector(model, current_opt, pool[ids[step]], probe,
                                                   selector_arm, int(random_choices[step]))
            choices.append(winner)
    if cache is not None:
        refreshes = cache.refreshes
    else:
        refreshes = len(choices) if arm == 'fresh' else 0
    return dict(arm=arm, choices=choices, refreshes=refreshes, losses=losses, **start_hashes)


def paired_differences(arms, checkpoints):
    return {f'{a}_minus_{b}': {key: arms[a]['losses'][key] - arms[b]['losses'][key]
                               for key in checkpoints}
            for a, b in PAIRS}


def summarize(rows, plan):
    checkpoint_keys = [str(c) for c in plan['checkpoints']]
    final_key = str(plan['updates'])
    states = []
    by_language = {}
    for language in plan['languages']:
        cases = []
        for fold in plan['folds']:
            for seed in plan['seeds']:
                runs = [r for r in rows if (r['language'], r['fold'], r['seed']) == (language, fold, seed)]
                lookup = {r['arm']: r for r in runs}
                if len(runs) != len(plan['arms']) or set(lookup) != set(plan['arms']):
                    raise ValueError('Incomplete or duplicated policy arms')
                for field in ('initial_model_sha256', 'initial_optimizer_sha256', 'schedule_sha256'):
                    if len({r[field] for r in runs}) != 1:
                        raise ValueError(f'Unpaired {field}')
                if len({r['losses']['0'] for r in runs}) != 1:
                    raise ValueError('Initial checkpoint losses differ')
                contrasts = paired_differences(lookup, checkpoint_keys)
                states.append(dict(language=language, fold=fold, seed=seed,
                                   contrasts={k: v[final_key] for k, v in contrasts.items()}))
                cases.append((lookup, contrasts))
        arms = {arm: {'losses': {key: mean(lookup[arm]['losses'][key] for lookup, _ in cases)
                                 for key in checkpoint_keys}} for arm in plan['arms']}
        differences = {f'{a}_minus_{b}': {key: mean(contrasts[f'{a}_minus_{b}'][key]
                                                      for _, contrasts in cases)
                                           for key in checkpoint_keys} for a, b in PAIRS}
        by_language[language] = dict(arms=arms, contrasts={k: v[final_key] for k, v in differences.items()},
                                     contrasts_by_checkpoint=differences)
    if len(rows) != len(states) * len(plan['arms']):
        raise ValueError('Unexpected trajectory outside protocol grid')
    return dict(states=states, languages=by_language)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/pud-reuse-matched.json'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new --output path to preserve the original measurement')
    plan = json.loads(PROTOCOL.read_text())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    previous = json.loads(PREVIOUS.read_text())
    meta = provenance()
    previous_audit_path = Path('results/pud-reuse-audit.json')
    audit = json.loads(previous_audit_path.read_text())
    if audit['status'] != 'passed' or digest(PREVIOUS) != audit['result_sha256']:
        raise ValueError('Prior experiment differs from its audited version')
    if digest(PREVIOUS_RUNS) != previous['output_sha256'][PREVIOUS_RUNS.name]:
        raise ValueError('Archived decision records differ')
    for path, expected in previous['provenance']['sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError(f'Stage 60 source mismatch: {path}')
    for path in (previous_audit_path, PREVIOUS, PREVIOUS_RUNS, PROTOCOL,
                 Path('experiments/pud_reuse.py'), Path('experiments/pud_reuse_matched.py'),
                 Path('tests/test_pud_reuse_matched.py')):
        meta['sha256'][str(path)] = digest(path)
    history = [json.loads(line) for line in PREVIOUS_RUNS.read_text().splitlines()]
    coverage_ref = {tuple(r[k] for k in STATE_KEYS): r for r in previous['coverage']}
    manifest = json.loads(Path('data/pud_sources.json').read_text())
    rows, coverage = [], []
    start = time.perf_counter()
    for language in plan['languages']:
        sentences = [json.loads(s) for s in Path(manifest['processed'][language]['path']).read_text().splitlines()]
        for fold in plan['folds']:
            pool, probe, target = original.partition(sentences, manifest['folds_source_indices'], fold)
            for seed in plan['seeds']:
                state = dict(language=language, fold=fold, seed=seed)
                torch.manual_seed(seed)
                rng = np.random.default_rng(seed)
                base = original.ByteTransformer()
                opt = original.optimizer(base)
                for _ in range(32):
                    original.update(base, opt, pool[rng.integers(len(pool), size=4)])
                ids = rng.integers(len(pool), size=(20000, 4))
                random_choices = rng.integers(4, size=20000)
                schedule_hash = hashlib.sha256(ids.tobytes()+random_choices.tobytes()).hexdigest()
                info = dict(**state, pool_chunks=len(pool), probe_chunks=len(probe),
                            target_chunks=len(target), schedule_sha256=schedule_hash)
                if info != coverage_ref[(language, fold, seed)]:
                    raise ValueError('Partition or schedule differs from stage 60')
                coverage.append(info)
                for arm in plan['arms']:
                    row = run_arm(base, opt, pool, probe, ids, random_choices, target, arm, plan)
                    records = [r for r in history if r['arm'] == arm and all(r[k] == v for k, v in state.items())]
                    if len(records) != 3:
                        raise ValueError('Incomplete historical state')
                    prefixes = []
                    for r in records:
                        count = min(len(row['choices']), len(r['choices']))
                        if row['choices'][:count] != r['choices'][:count]:
                            raise ValueError(f'Published choice prefix mismatch at {state}, {arm}, repeat {r["repeat"]}')
                        prefixes.append(count)
                    historical_difference = row['losses']['0'] - records[0]['initial_loss']
                    row.update(**state, schedule_sha256=schedule_hash,
                               stage60_matched_prefix_lengths=prefixes,
                               stage60_initial_loss_difference=historical_difference)
                    rows.append(row)
                print(f'{language} fold {fold} seed {seed}: {len(rows)} complete trajectories', flush=True)
    summary = summarize(rows, plan)
    result = dict(stage=61, protocol=plan, provenance=meta, coverage=coverage, rows=rows, summary=summary,
                  wall_seconds=time.perf_counter() - start)
    with args.output.open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write('\n')


if __name__ == '__main__':
    main()

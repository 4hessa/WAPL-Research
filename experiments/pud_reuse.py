"""Stage 60: cache the probe gradient for four policy decisions."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import time
from types import FunctionType

import numpy as np
import torch

from experiments import pud_policy as original
from experiments.pud_policy_profile import check_winners, provenance


PROTOCOL = Path('research/pud-reuse-protocol.json')
PROFILE = Path('results/pud-policy-profile.json')
ARMS = ('reuse4', 'fresh', 'random')
METRICS = ('final_loss', 'updates', 'actual_seconds', 'overrun_seconds',
           'accepted_refreshes', 'attempted_refreshes')
CONTRASTS = (('reuse4', 'fresh'), ('reuse4', 'random'))


class CachedDecision:
    """Run-private probe gradient; each attempted decision advances the step counter."""

    def __init__(self, interval=4):
        if interval < 1:
            raise ValueError('Refresh interval must be positive')
        self.interval = interval
        self.attempts = 0
        self.refreshes = 0
        self.gradient = None

    def __call__(self, model, opt, candidates, probe, arm, random_choice):
        if self.attempts % self.interval == 0:
            probe_model, probe_opt = original.clone(model, opt)
            probe_opt.zero_grad()
            original.byte_loss(probe_model, probe).backward()
            self.gradient = original.flatten({k: p.grad for k, p in probe_model.named_parameters()})
            self.refreshes += 1
        self.attempts += 1
        before = {k: p.detach() for k, p in model.named_parameters()}
        best_score = -float('inf')
        best = None
        for idx in range(len(candidates)):
            trial, trial_opt = original.clone(model, opt)
            original.update(trial, trial_opt, candidates[idx:idx + 1])
            delta = {k: p.detach() - before[k] for k, p in trial.named_parameters()}
            score = original.alignment_score(delta, self.gradient)
            if score > best_score:
                best_score = score
                best = trial, trial_opt, idx
        return best


def timed_run(base, opt, pool, probe, ids, choices, arm, *, budget=3., clock=time.perf_counter):
    runner = original.run_policy
    cache = None
    if arm == 'reuse4':
        # Invoke stage 58's published deadline loop, with a single local
        # replacement. All cloning and all scoring run inside that same clock.
        cache = CachedDecision()
        namespace = dict(runner.__globals__, decision=cache)
        runner = FunctionType(runner.__code__, namespace, runner.__name__, runner.__defaults__)
    elif arm not in ('random', 'fresh'):
        raise ValueError(f'Unknown arm: {arm}')
    model, row = runner(base, opt, pool, probe, ids, choices,
                        'random' if arm == 'random' else 'alignment', 'time', budget=budget, clock=clock)
    if cache is None:
        row['attempted_refreshes'] = row['attempted'] if arm == 'fresh' else 0
        row['accepted_refreshes'] = row['updates'] if arm == 'fresh' else 0
    else:
        if row['attempted'] != cache.attempts:
            raise ValueError('Cached attempt counter differs from the deadline loop')
        row['attempted_refreshes'] = cache.refreshes
        row['accepted_refreshes'] = (row['updates'] + cache.interval - 1) // cache.interval
    return model, row


def drift_checks(base, opt, pool, probe, ids, choices, count):
    model, current_opt = original.clone(base, opt)
    cache = CachedDecision()
    for step in range(count):
        inputs = (model, current_opt, pool[ids[step]], probe, 'alignment', int(choices[step]))
        fresh = original.decision(*inputs)
        reused = cache(*inputs)
        age = step % cache.interval
        if age == 0:
            check_winners(fresh, reused)
        yield dict(step=step, age=age, fresh_choice=fresh[2], reuse_choice=reused[2],
                   refresh_parity=True if age == 0 else None)
        model, current_opt = reused[:2]


def summarize(rows, diagnostics, states):
    state_metrics = []
    for state in states:
        runs = [r for r in rows if all(r[k] == v for k, v in state.items())]
        if len(runs) != 9:
            raise ValueError('Incomplete state runs')
        arms = {}
        for arm in ARMS:
            matched = [r for r in runs if r['arm'] == arm]
            if sorted(r['repeat'] for r in matched) != [0, 1, 2]:
                raise ValueError('Unpaired repeats')
            arms[arm] = {k: mean(r[k] for r in matched) for k in METRICS}
        contrasts = {f'{a}_minus_{b}': {k: arms[a][k] - arms[b][k] for k in ('final_loss', 'updates')}
                     for a, b in CONTRASTS}
        state_metrics.append(dict(**state, arms=arms, contrasts=contrasts))
    languages = {}
    for lang in ('ar', 'en'):
        group = [r for r in state_metrics if r['language'] == lang]
        arms = {a: {k: mean(r['arms'][a][k] for r in group) for k in METRICS} for a in ARMS}
        contrasts = {}
        for a, b in CONTRASTS:
            contrast = f'{a}_minus_{b}'
            contrasts[contrast] = {k: mean(r['contrasts'][contrast][k] for r in group)
                                   for k in ('final_loss', 'updates')}
        languages[lang] = dict(arms=arms, contrasts=contrasts)
    drift = []
    for age in range(4):
        records = [r for r in diagnostics if r['age'] == age]
        drift.append(dict(age=age, decisions=len(records),
                          agreement=sum(r['fresh_choice'] == r['reuse_choice'] for r in records)))
    return dict(states=state_metrics, languages=languages, drift_by_age=drift)


def write_line(file, row):
    file.write(json.dumps(row, allow_nan=False) + '\n')
    file.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: args.output_dir / f'pud-reuse{suffix}' for name, suffix in
             [('result', '.json'), ('records', '-records.jsonl'), ('decisions', '-decisions.jsonl')]}
    if any(p.exists() for p in paths.values()):
        parser.error('Choose an unused output directory to preserve recorded runs')
    plan = json.loads(PROTOCOL.read_text())
    if tuple(plan['arms']) != ARMS or plan['refresh_interval'] != 4:
        raise ValueError('Protocol differs from the implemented policies')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    meta = provenance()
    for path in (PROFILE, PROTOCOL, Path(__file__).relative_to(Path.cwd()), Path('tests/test_pud_reuse.py')):
        meta['sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    prior = json.loads(PROFILE.read_text())
    references = {(r['language'], r['fold'], r['seed']): r for r in prior['coverage']}
    manifest = json.loads(Path('data/pud_sources.json').read_text())
    rows, diagnostic_rows, coverage, states = [], [], [], []
    start = time.perf_counter()
    with paths['records'].open('x') as records, paths['decisions'].open('x') as decisions:
        for language in plan['languages']:
            sentences = [json.loads(s) for s in Path(manifest['processed'][language]['path']).read_text().splitlines()]
            for fold in plan['folds']:
                pool, probe, target = original.partition(sentences, manifest['folds_source_indices'], fold)
                for seed in plan['seeds']:
                    index = len(states)
                    state = dict(language=language, fold=fold, seed=seed)
                    states.append(state)
                    rng = np.random.default_rng(seed)
                    torch.manual_seed(seed)
                    base = original.ByteTransformer()
                    opt = original.optimizer(base)
                    for _ in range(32):
                        original.update(base, opt, pool[rng.integers(len(pool), size=4)])
                    ids = rng.integers(len(pool), size=(20000, 4))
                    choices = rng.integers(4, size=20000)
                    digest = hashlib.sha256(ids.tobytes() + choices.tobytes()).hexdigest()
                    reference = references[language, fold, seed]
                    if digest != reference['schedule_sha256'] or len(probe) != 64:
                        raise ValueError('State inputs differ from stage 59')
                    initial = original.evaluate(base, target)
                    coverage.append(dict(**state, pool_chunks=len(pool), probe_chunks=len(probe),
                                         target_chunks=len(target), schedule_sha256=digest))
                    for repeat in range(plan['timing_repeats']):
                        shift = (index + repeat) % len(ARMS)
                        for position, arm in enumerate(ARMS[shift:] + ARMS[:shift]):
                            model, row = timed_run(base, opt, pool, probe, ids, choices, arm,
                                                   budget=plan['budget_seconds'])
                            final = original.evaluate(model, target)
                            if not np.isfinite(initial) or not np.isfinite(final):
                                raise ValueError('Nonfinite evaluation')
                            row.update(**state, arm=arm, repeat=repeat, position=position,
                                       initial_loss=initial, final_loss=final, schedule_sha256=digest)
                            rows.append(row)
                            write_line(records, row)
                    for row in drift_checks(base, opt, pool, probe, ids, choices, count=plan['drift_steps']):
                        if row['step'] == 0 and row['fresh_choice'] != reference['historical_winner']:
                            raise ValueError('Reference first choice mismatch')
                        row.update(**state, agreement=row['fresh_choice'] == row['reuse_choice'])
                        diagnostic_rows.append(row)
                        write_line(decisions, row)
                    print(f'{language} fold {fold} seed {seed}: {len(rows)} timed runs', flush=True)
    summary = summarize(rows, diagnostic_rows, states)
    digests = {paths[k].name: hashlib.sha256(paths[k].read_bytes()).hexdigest() for k in ('records', 'decisions')}
    result = dict(stage=60, protocol=plan, provenance=meta, output_sha256=digests, coverage=coverage,
                  summary=summary, runs=len(rows), diagnostic_decisions=len(diagnostic_rows),
                  wall_seconds=time.perf_counter() - start)
    with paths['result'].open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write('\n')


if __name__ == '__main__':
    main()

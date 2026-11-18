"""Paired wall-clock profile of the stage-58 alignment decision on CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time
from types import FunctionType

import numpy as np
import torch

from experiments import pud_policy as policy


LANGUAGES = ('ar', 'en')
FOLDS = (0, 5)
SEEDS = (28000, 28001)
REPEATS = 6
COMPONENTS = ('probe_clone', 'probe_gradient', 'candidate_clones',
              'candidate_updates', 'candidate_delta_scores', 'residual')


class DecisionTimer:
    """Record disjoint intervals inside the original single-threaded decision."""

    def __init__(self, clock=time.perf_counter_ns):
        self.clock = clock
        self.elapsed = dict.fromkeys(COMPONENTS, 0)
        self.calls = dict.fromkeys(('clone', 'flatten', 'update', 'alignment_score'), 0)
        self.probe_start = None
        self.delta_start = None

    def clone(self, model, opt):
        start = self.clock()
        result = policy.clone(model, opt)
        end = self.clock()
        is_probe = self.calls['clone'] == 0
        key = 'probe_clone' if is_probe else 'candidate_clones'
        self.elapsed[key] += end - start
        self.calls['clone'] += 1
        if is_probe:
            self.probe_start = end
        return result

    def flatten(self, grads):
        # decision calls flatten directly only to form the probe gradient.
        # alignment_score runs with its own original globals.
        result = policy.flatten(grads)
        self.elapsed['probe_gradient'] += self.clock() - self.probe_start
        self.calls['flatten'] += 1
        return result

    def update(self, model, opt, batch):
        start = self.clock()
        result = policy.update(model, opt, batch)
        self.delta_start = self.clock()
        self.elapsed['candidate_updates'] += self.delta_start - start
        self.calls['update'] += 1
        return result

    def alignment_score(self, delta, probe_gradient):
        score = policy.alignment_score(delta, probe_gradient)
        self.elapsed['candidate_delta_scores'] += self.clock() - self.delta_start
        self.calls['alignment_score'] += 1
        return score

    def bind(self):
        # Copy the namespace to keep all patching local to this call. The
        # bytecode and numerical routines are the ones used in stage 58.
        globals_copy = dict(policy.decision.__globals__)
        for name in self.calls:
            globals_copy[name] = getattr(self, name)
        return FunctionType(policy.decision.__code__, globals_copy, policy.decision.__name__)

    def finish(self, total_ns, candidates):
        expected = {'clone': candidates + 1, 'flatten': 1, 'update': candidates,
                    'alignment_score': candidates}
        if self.calls != expected:
            raise RuntimeError(f'Unexpected decision calls: {self.calls}')
        accounted = sum(self.elapsed.values())
        if accounted > total_ns:
            raise RuntimeError('Timed component intervals overlap or exceed the decision clock')
        self.elapsed['residual'] = total_ns - accounted
        return dict(self.elapsed)


def timed_decision(inputs, profile):
    timer = DecisionTimer() if profile else None
    decision = timer.bind() if profile else policy.decision
    start = time.perf_counter_ns()
    winner = decision(*inputs)
    duration = time.perf_counter_ns() - start
    components = timer.finish(duration, len(inputs[2])) if profile else None
    return winner, duration, components


def check_winners(left, right):
    if left[2] != right[2]:
        raise ValueError('Different winning index with timing hooks')
    torch.testing.assert_close(left[0].state_dict(), right[0].state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(left[1].state_dict(), right[1].state_dict(), rtol=0, atol=0)


def measure_state(inputs, language, fold, seed, state_index):
    # Prime both paths before collecting observations at this state.
    first, _, _ = timed_decision(inputs, profile=False)
    second, _, _ = timed_decision(inputs, profile=True)
    check_winners(first, second)
    del first, second
    rows = []
    for repeat in range(REPEATS):
        profiled_first = (state_index + repeat) % 2 == 1
        measurements = {}
        for profile in (profiled_first, not profiled_first):
            measurements[profile] = timed_decision(inputs, profile)
        plain, plain_ns, _ = measurements[False]
        profiled, profiled_ns, components = measurements[True]
        check_winners(plain, profiled)
        rows.append(dict(language=language, fold=fold, seed=seed, repeat=repeat,
                         profiled_first=profiled_first, winner=plain[2], parity=True,
                         plain_ns=plain_ns, profiled_ns=profiled_ns, component_ns=components))
        del measurements, plain, profiled
    return rows


def summarize(rows):
    plain = statistics.mean(r['plain_ns'] for r in rows)
    profiled = statistics.mean(r['profiled_ns'] for r in rows)
    differences = [r['profiled_ns'] - r['plain_ns'] for r in rows]
    components = {k: statistics.mean(r['component_ns'][k] for r in rows) for k in COMPONENTS}
    return dict(pairs=len(rows), plain_mean_ms=plain / 1e6, profiled_mean_ms=profiled / 1e6,
                paired_difference_mean_ms=statistics.mean(differences) / 1e6,
                paired_difference_median_ms=statistics.median(differences) / 1e6,
                overhead_pct=100 * (profiled - plain) / plain,
                component_mean_ms={k: v / 1e6 for k, v in components.items()},
                component_share_pct={k: 100 * v / profiled for k, v in components.items()})


def provenance():
    previous_path = Path('results/pud-policy-environment.json')
    historical = json.loads(previous_path.read_text())['sha256']
    for path, expected in historical.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Historical input mismatch: {path}')
    hashes = dict(historical)
    for path in [previous_path, Path('results/pud-policy-records.jsonl'),
                 Path('experiments/pud_policy_profile.py'), Path('tests/test_pud_policy_profile.py')]:
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    cpu_line = next(line for line in Path('/proc/cpuinfo').read_text().splitlines()
                    if line.startswith('model name'))
    return dict(sha256=hashes, python=platform.python_version(), numpy=np.__version__,
                torch=torch.__version__, platform=platform.platform(), cpu=cpu_line.split(':')[1].strip(),
                torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                openblas_num_threads=os.environ.get('OPENBLAS_NUM_THREADS'),
                deterministic=torch.are_deterministic_algorithms_enabled())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/pud-policy-profile.json'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists. Use --output to preserve both timing runs.')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    metadata = provenance()
    manifest = json.loads(Path('data/pud_sources.json').read_text())
    archived = [json.loads(line) for line in Path('results/pud-policy-records.jsonl').read_text().splitlines()]
    reference = {(r['language'], r['fold'], r['seed']): r for r in archived
                 if r['arm'] == 'alignment' and r['mode'] == 'updates'}
    rows, coverage = [], []
    started = time.perf_counter()
    for language in LANGUAGES:
        sentences = [json.loads(s) for s in Path(manifest['processed'][language]['path']).read_text().splitlines()]
        for fold in FOLDS:
            pool, probe, _ = policy.partition(sentences, manifest['folds_source_indices'], fold)
            if len(probe) != 64:
                raise ValueError('Incomplete probe')
            for seed in SEEDS:
                torch.manual_seed(seed)
                rng = np.random.default_rng(seed)
                model = policy.ByteTransformer()
                opt = policy.optimizer(model)
                for _ in range(32):
                    policy.update(model, opt, pool[rng.integers(len(pool), size=4)])
                schedule = rng.integers(len(pool), size=(20000, 4))
                choices = rng.integers(4, size=20000)
                schedule_hash = hashlib.sha256(schedule.tobytes() + choices.tobytes()).hexdigest()
                original = reference[language, fold, seed]
                if schedule_hash != original['schedule_sha256']:
                    raise ValueError('Different candidate schedule from stage 58')
                inputs = (model, opt, pool[schedule[0]], probe, 'alignment', int(choices[0]))
                measured = measure_state(inputs, language, fold, seed, len(coverage))
                if any(r['winner'] != original['choices'][0] for r in measured):
                    raise ValueError('Different first winner from stage 58')
                coverage.append(dict(language=language, fold=fold, seed=seed, pool_chunks=len(pool),
                                     probe_chunks=len(probe), candidate_ids=schedule[0].tolist(),
                                     schedule_sha256=schedule_hash, historical_winner=original['choices'][0]))
                rows.extend(measured)
    design = dict(languages=LANGUAGES, folds=FOLDS, seeds=SEEDS, repeats_per_state=REPEATS,
                  warmup_updates=32, warmup_batch=4, elapsed_seconds=time.perf_counter()-started,
                  state='first decision after warmup; each call starts from the same state',
                  excluded_from_clock=['data preparation', 'candidate indexing', 'wrapper installation',
                                       'priming decisions', 'winner/state parity checks'],
                  target_evaluations=0, runs_per_state='one discarded priming pair then six measured pairs',
                  clock='time.perf_counter_ns; CPU wall time; setup of decision arguments excluded')
    report = dict(stage=59, design=design, provenance=metadata, coverage=coverage, rows=rows,
                  summary=summarize(rows))
    with args.output.open('x') as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write('\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()

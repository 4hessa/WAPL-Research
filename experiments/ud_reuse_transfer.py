"""Test cached and freshly recomputed probe gradients on PADT and EWT."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import mean
import time

import numpy as np
import torch

from experiments import pud_policy as original
from experiments.multilingual_readiness import byte_chunks
from experiments.pud_reuse_matched import digest
from experiments.pud_reuse_one_step import fingerprints, grouped, paired_steps


PROTOCOL = Path('research/ud-reuse-transfer-protocol.json')
MANIFEST = Path('data/multilingual_sources.json')
PREVIOUS = Path('results/pud-reuse-one-step.json')
PREVIOUS_AUDIT = Path('results/pud-reuse-one-step-audit.json')


def prepare_language(splits, plan, hashes):
    chunks, tails = {}, {}
    for split in ('train', 'dev', 'test'):
        entry = splits[split]
        path = Path(entry['path'])
        blob = path.read_bytes()
        actual = hashlib.sha256(blob).hexdigest()
        if actual != entry['sha256']:
            raise ValueError(f'Prepared {split} input checksum mismatch: {path}')
        hashes[str(path)] = actual
        chunks[split] = byte_chunks(blob)
        tails[split] = len(blob) % 65
    probe_ids = np.sort(np.random.default_rng(plan['probe_seed']).choice(
        len(chunks['dev']), plan['probe_chunks'], replace=False))
    target_ids = np.sort(np.random.default_rng(plan['target_seed']).choice(
        len(chunks['test']), plan['target_chunks'], replace=False))
    population = dict(train_chunks=len(chunks['train']), dev_chunks=len(chunks['dev']),
                      test_chunks=len(chunks['test']), probe_ids=probe_ids.tolist(),
                      target_ids=target_ids.tolist(), discarded_tail_bytes=tails)
    return chunks['train'], chunks['dev'][probe_ids], chunks['test'][target_ids], population


def validate_replay_source():
    previous = json.loads(PREVIOUS.read_text())
    audit = json.loads(PREVIOUS_AUDIT.read_text())
    if audit['status'] != 'passed' or audit['result_sha256'] != digest(PREVIOUS):
        raise ValueError('Stage 62 reference differs from its audit')
    # Only reused implementation and checked-in provenance are needed from PUD.
    hashes = {}
    for name, expected in previous['provenance']['sha256'].items():
        if name.startswith('experiments/'):
            hashes[name] = digest(Path(name))
            if hashes[name] != expected:
                raise ValueError(f'Reused stage-62 source changed: {name}')
    for path in (PROTOCOL, MANIFEST, PREVIOUS, PREVIOUS_AUDIT,
                 Path('experiments/ud_reuse_transfer.py'), Path('tests/test_ud_reuse_transfer.py')):
        hashes[str(path)] = digest(path)
    return hashes


def summarize(rows, plan):
    step_count = plan['steps']
    interval = plan['refresh_interval']
    states, languages = [], {}
    for lang in plan['languages']:
        language_rows = [r for r in rows if r['language'] == lang]
        seed_means = []
        for seed in plan['seeds']:
            rr = [r for r in language_rows if r['seed'] == seed]
            if sorted(r['step'] for r in rr) != list(range(step_count)):
                raise ValueError(f'Incomplete decisions for {lang} seed {seed}')
            if any(r['age'] != r['step'] % interval for r in rr):
                raise ValueError('Recorded cache age differs from the specified interval')
            state = dict(language=lang, seed=seed, **grouped(rr, interval))
            states.append(state)
            seed_means.append(state['stale']['mean_delta'])
        languages[lang] = grouped(language_rows, interval)
        languages[lang]['stale']['mean_delta'] = mean(seed_means)
    if len(rows) != step_count * len(states):
        raise ValueError('Extraneous observations')
    return dict(states=states, languages=languages)


def run_state(train, probe, target, lang, seed, plan):
    torch.manual_seed(seed)
    model = original.ByteTransformer()
    opt = original.optimizer(model)
    rng = np.random.default_rng(seed)
    for _ in range(plan['warmup_updates']):
        original.update(model, opt, train[rng.integers(len(train), size=plan['warmup_batch_size'])])
    initial = fingerprints(model, opt)
    ids = rng.integers(len(train), size=(plan['steps'], plan['candidates_per_step']))
    choices = rng.integers(plan['candidates_per_step'], size=plan['steps'])
    schedule_hash = hashlib.sha256(ids.tobytes() + choices.tobytes()).hexdigest()
    coverage = dict(language=lang, seed=seed, initial_model_sha256=initial[0],
                    initial_optimizer_sha256=initial[1], schedule_sha256=schedule_hash)
    rows = []
    parent = initial
    for r in paired_steps(model, opt, train, probe, ids, choices, target,
                          plan['steps'], plan['refresh_interval']):
        if (r['parent_model_sha256'], r['parent_optimizer_sha256']) != parent:
            raise ValueError('Trajectory does not continue from its preceding cached update')
        parent = r['reuse_model_sha256'], r['reuse_optimizer_sha256']
        r.update(language=lang, seed=seed)
        rows.append(r)
    return rows, coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/ud-reuse-transfer.json'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new --output path to preserve existing observations')
    plan = json.loads(PROTOCOL.read_text())
    manifest = json.loads(MANIFEST.read_text())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    hashes = validate_replay_source()
    prepared = {lang: prepare_language(manifest['languages'][lang]['splits'], plan, hashes)
                for lang in plan['languages']}
    cpu_line = next(line for line in Path('/proc/cpuinfo').read_text().splitlines()
                    if line.startswith('model name'))
    meta = dict(sha256=hashes, cpu=cpu_line.split(':')[1].strip(), platform=platform.platform(),
                python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                deterministic=torch.are_deterministic_algorithms_enabled(),
                openblas_num_threads=os.environ.get('OPENBLAS_NUM_THREADS'))
    start = time.perf_counter()
    rows, coverage, populations = [], [], {}
    for lang in plan['languages']:
        train, probe, target, populations[lang] = prepared[lang]
        for seed in plan['seeds']:
            observations, info = run_state(train, probe, target, lang, seed, plan)
            rows.extend(observations)
            coverage.append(info)
            print(f'{lang} seed {seed}: {len(rows)} paired steps', flush=True)
    result = dict(stage=63, protocol=plan, provenance=meta, populations=populations, coverage=coverage,
                  rows=rows, summary=summarize(rows, plan), measurement_seconds=time.perf_counter() - start)
    with args.output.open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write('\n')


if __name__ == '__main__':
    main()

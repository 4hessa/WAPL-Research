"""Stage 64: timed reuse4, fresh and random policies on PADT and EWT.

Implements research/ud-reuse-timed-protocol.json. The policy loop and timer are
the published stage-58/60 code (experiments.pud_reuse.timed_run); this module
only supplies the PADT/EWT states, the run order and the bookkeeping.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import mean, pstdev
import time

import numpy as np
import torch

from experiments import pud_policy as original
from experiments.pud_reuse import timed_run
from experiments.pud_reuse_matched import digest
from experiments.pud_reuse_one_step import fingerprints
from experiments.ud_reuse_transfer import prepare_language


PROTOCOL = Path('research/ud-reuse-timed-protocol.json')
MANIFEST = Path('data/multilingual_sources.json')
STAGE63 = Path('results/ud-reuse-transfer.json')
STAGE63_AUDIT = Path('results/ud-reuse-transfer-audit.json')
ARMS = ('reuse4', 'fresh', 'random')
METRICS = ('final_loss', 'updates', 'attempted', 'actual_seconds', 'overrun_seconds',
           'accepted_refreshes', 'attempted_refreshes')
CONTRASTS = (('reuse4', 'fresh'), ('reuse4', 'random'), ('fresh', 'random'))


def arm_order(state_index, repeat):
    shift = (state_index + repeat) % len(ARMS)
    return ARMS[shift:] + ARMS[:shift]


def build_state(train, lang, seed, plan):
    """Warm up once per state and draw the common candidate schedule."""
    torch.manual_seed(seed)
    model = original.ByteTransformer()
    opt = original.optimizer(model)
    rng = np.random.default_rng(seed)
    for _ in range(plan['warmup_updates']):
        original.update(model, opt, train[rng.integers(len(train), size=plan['warmup_batch_size'])])
    initial = fingerprints(model, opt)
    ids = rng.integers(len(train), size=(plan['schedule_length'], plan['candidates_per_step']))
    choices = rng.integers(plan['candidates_per_step'], size=plan['schedule_length'])
    schedule = hashlib.sha256(ids.tobytes() + choices.tobytes()).hexdigest()
    return model, opt, ids, choices, dict(language=lang, seed=seed, initial_model_sha256=initial[0],
                                          initial_optimizer_sha256=initial[1], schedule_sha256=schedule)


def check_against_stage63(info, ids, reference_rows, reference_coverage):
    """Same warmed state and the same first 16 candidate sets as stage 63."""
    ref = reference_coverage[info['language'], info['seed']]
    if (info['initial_model_sha256'], info['initial_optimizer_sha256']) != (
            ref['initial_model_sha256'], ref['initial_optimizer_sha256']):
        raise ValueError('Warmed state differs from stage 63')
    rows = sorted(reference_rows[info['language'], info['seed']], key=lambda r: r['step'])
    if len(rows) != 16 or [r['candidate_ids'] for r in rows] != ids[:16].tolist():
        raise ValueError('Candidate schedule prefix differs from stage 63')


def summarize(rows, states, plan):
    seen = set()
    for row in rows:
        key = (row['language'], row['seed'], row['repeat'], row['arm'])
        if key in seen:
            raise ValueError('Duplicate run')
        seen.add(key)
        if not all(np.isfinite(row[k]) for k in ('initial_loss', 'final_loss', 'actual_seconds')):
            raise ValueError('Nonfinite run metric')
    state_metrics = []
    for state in states:
        runs = [r for r in rows if r['language'] == state['language'] and r['seed'] == state['seed']]
        if len(runs) != len(ARMS) * plan['timing_repeats']:
            raise ValueError('Incomplete state runs')
        arms = {}
        for arm in ARMS:
            matched = [r for r in runs if r['arm'] == arm]
            if sorted(r['repeat'] for r in matched) != list(range(plan['timing_repeats'])):
                raise ValueError('Unpaired repeats')
            if sorted(r['position'] for r in matched) != list(range(len(ARMS))):
                raise ValueError('Arm does not occupy each order position once')
            arms[arm] = {k: mean(r[k] for r in matched) for k in METRICS}
        contrasts = {f'{a}_minus_{b}': {k: arms[a][k] - arms[b][k] for k in ('final_loss', 'updates')}
                     for a, b in CONTRASTS}
        state_metrics.append(dict(**state, initial_loss=runs[0]['initial_loss'],
                                  arms=arms, contrasts=contrasts))
    languages = {}
    for lang in plan['languages']:
        group = [s for s in state_metrics if s['language'] == lang]
        if [s['seed'] for s in group] != plan['seeds']:
            raise ValueError('Missing language states')
        arms = {a: {k: mean(s['arms'][a][k] for s in group) for k in METRICS} for a in ARMS}
        contrasts = {}
        for a, b in CONTRASTS:
            name = f'{a}_minus_{b}'
            values = [s['contrasts'][name]['final_loss'] for s in group]
            contrasts[name] = dict(
                final_loss=mean(values), updates=mean(s['contrasts'][name]['updates'] for s in group),
                per_seed_final_loss=values, seeds_negative=sum(v < 0 for v in values),
                seeds_positive=sum(v > 0 for v in values), min=min(values), max=max(values),
                population_sd=pstdev(values))
        languages[lang] = dict(initial_loss=mean(s['initial_loss'] for s in group),
                               arms=arms, contrasts=contrasts)
    return dict(primary=dict(language='ar', contrast='reuse4_minus_fresh',
                             final_loss=languages['ar']['contrasts']['reuse4_minus_fresh']['final_loss']),
                states=state_metrics, languages=languages)


def write_line(file, row):
    file.write(json.dumps(row, allow_nan=False) + '\n')
    file.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = dict(result=args.output_dir / 'ud-reuse-timed.json',
                 records=args.output_dir / 'ud-reuse-timed-records.jsonl')
    if any(p.exists() for p in paths.values()):
        parser.error('Choose an unused output directory to preserve recorded runs')
    plan = json.loads(PROTOCOL.read_text())
    if tuple(plan['arms']) != ARMS or plan['refresh_interval'] != 4 or plan['stage'] != 64:
        raise ValueError('Protocol differs from the implemented policies')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    stage63 = json.loads(STAGE63.read_text())
    audit63 = json.loads(STAGE63_AUDIT.read_text())
    if audit63['status'] != 'passed' or audit63['result_sha256'] != digest(STAGE63):
        raise ValueError('Stage 63 reference differs from its audit')
    if stage63['protocol']['seeds'] != plan['seeds'] or any(
            stage63['protocol'][k] != plan[k] for k in ('probe_seed', 'target_seed', 'probe_chunks',
                                                         'target_chunks', 'warmup_updates')):
        raise ValueError('Stage 63 sampling settings differ from the protocol')
    hashes = {}
    for path in (PROTOCOL, MANIFEST, STAGE63, STAGE63_AUDIT, Path('experiments/ud_reuse_timed.py'),
                 Path('tests/test_ud_reuse_timed.py'), Path('experiments/pud_reuse.py'),
                 Path('experiments/pud_policy.py'), Path('experiments/ud_reuse_transfer.py'),
                 Path('experiments/multilingual_readiness.py'), Path('experiments/multilingual_policy.py'),
                 Path('experiments/head_invariance.py')):
        hashes[str(path)] = digest(path)
    manifest = json.loads(MANIFEST.read_text())
    prepared = {lang: prepare_language(manifest['languages'][lang]['splits'], plan, hashes)
                for lang in plan['languages']}
    for lang in plan['languages']:
        if prepared[lang][3] != stage63['populations'][lang]:
            raise ValueError('Probe or target sample differs from stage 63')
    ref_rows = {}
    for r in stage63['rows']:
        ref_rows.setdefault((r['language'], r['seed']), []).append(r)
    ref_coverage = {(c['language'], c['seed']): c for c in stage63['coverage']}
    cpu_line = next(line for line in Path('/proc/cpuinfo').read_text().splitlines()
                    if line.startswith('model name'))
    meta = dict(sha256=hashes, cpu=cpu_line.split(':')[1].strip(), platform=platform.platform(),
                python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                deterministic=torch.are_deterministic_algorithms_enabled(),
                openblas_num_threads=os.environ.get('OPENBLAS_NUM_THREADS'),
                logical_cpus=os.cpu_count())
    rows, coverage, states = [], [], []
    start = time.perf_counter()
    with paths['records'].open('x') as records:
        for lang in plan['languages']:
            train, probe, target, _ = prepared[lang]
            for seed in plan['seeds']:
                index = len(states)
                base, opt, ids, choices, info = build_state(train, lang, seed, plan)
                check_against_stage63(info, ids, ref_rows, ref_coverage)
                initial = original.evaluate(base, target)
                state = dict(language=lang, seed=seed)
                states.append(state)
                coverage.append(dict(**info, initial_loss=initial))
                for repeat in range(plan['timing_repeats']):
                    for position, arm in enumerate(arm_order(index, repeat)):
                        model, row = timed_run(base, opt, train, probe, ids, choices, arm,
                                               budget=plan['budget_seconds'])
                        final = original.evaluate(model, target)
                        if not np.isfinite(initial) or not np.isfinite(final):
                            raise ValueError('Nonfinite evaluation')
                        row.update(**state, arm=arm, repeat=repeat, position=position,
                                   initial_loss=initial, final_loss=final,
                                   late_attempt=row['attempted'] - row['updates'],
                                   schedule_sha256=info['schedule_sha256'])
                        rows.append(row)
                        write_line(records, row)
                print(f'{lang} seed {seed}: {len(rows)} timed runs', flush=True)
    summary = summarize(rows, states, plan)
    result = dict(stage=64, protocol=plan, provenance=meta,
                  output_sha256={paths['records'].name: digest(paths['records'])},
                  populations={lang: prepared[lang][3] for lang in plan['languages']},
                  coverage=coverage, summary=summary, runs=len(rows),
                  wall_seconds=time.perf_counter() - start)
    with paths['result'].open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write('\n')


if __name__ == '__main__':
    main()

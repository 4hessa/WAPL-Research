"""Stage 65: decision cost against selection gain on unscored PADT/EWT dev chunks.

Implements research/ud-cost-frontier-protocol.json. random and reuse4 use the
stage-58/60 code unchanged; lossK replaces only the decision function inside
the stage-58 deadline loop, so every arm is timed by the same loop.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import mean
import time
from types import FunctionType

import numpy as np
import torch

from experiments import pud_policy as original
from experiments.multilingual_policy import scores
from experiments.multilingual_readiness import byte_chunks
from experiments.pud_reuse import timed_run
from experiments.pud_reuse_matched import digest
from experiments.pud_reuse_one_step import fingerprints


PROTOCOL = Path('research/ud-cost-frontier-protocol.json')
MANIFEST = Path('data/multilingual_sources.json')
ARMS = ('random', 'loss4', 'loss16', 'loss64', 'reuse4')
GUIDED = ARMS[1:]
METRICS = ('final_loss', 'updates', 'attempted', 'actual_seconds', 'overrun_seconds')
EXPOSED_DEV = ((4800, 128), (6301, 64))  # stage-49 monitor, stage-63 probe


def arm_order(state_index, repeat):
    shift = (state_index + repeat) % len(ARMS)
    return ARMS[shift:] + ARMS[:shift]


def unscored_target(n_dev, plan):
    exposed = set()
    for seed, count in EXPOSED_DEV:
        exposed |= set(np.random.default_rng(seed).choice(n_dev, count, replace=False).tolist())
    eligible = np.array(sorted(set(range(n_dev)) - exposed))
    picked = np.random.default_rng(plan['target_seed']).choice(len(eligible), plan['target_chunks'], replace=False)
    return np.sort(eligible[picked]), sorted(exposed)


def prepare_language(splits, plan, hashes):
    chunks = {}
    for split in ('train', 'dev'):
        entry = splits[split]
        path = Path(entry['path'])
        blob = path.read_bytes()
        actual = hashlib.sha256(blob).hexdigest()
        if actual != entry['sha256']:
            raise ValueError(f'Prepared {split} input checksum mismatch: {path}')
        hashes[str(path)] = actual
        chunks[split] = byte_chunks(blob)
    n_dev = len(chunks['dev'])
    probe_ids = np.sort(np.random.default_rng(plan['probe_seed']).choice(n_dev, plan['probe_chunks'], replace=False))
    target_ids, exposed = unscored_target(n_dev, plan)
    if set(target_ids.tolist()) & set(exposed):
        raise ValueError('Target overlaps previously scored dev chunks')
    population = dict(train_chunks=len(chunks['train']), dev_chunks=n_dev, probe_ids=probe_ids.tolist(),
                      target_ids=target_ids.tolist(), excluded_dev_chunks=len(exposed))
    return chunks['train'], chunks['dev'][probe_ids], chunks['dev'][target_ids], population


class LossDecision:
    """Clone, score all candidates in one no-grad forward pass, update the highest-loss one."""

    def __call__(self, model, opt, candidates, probe, arm, random_choice):
        winner, wo = original.clone(model, opt)
        choice = int(torch.argmax(scores(winner, candidates)))
        original.update(winner, wo, candidates[choice:choice + 1])
        return winner, wo, choice


def run_arm(base, opt, pool, probe, ids, choices, arm, plan, clock=time.perf_counter):
    width = plan['candidates'][arm]
    view = ids[:, :width]
    if arm in ('random', 'reuse4'):
        model, row = timed_run(base, opt, pool, probe, view, choices, arm,
                               budget=plan['budget_seconds'], clock=clock)
    elif arm.startswith('loss'):
        runner = original.run_policy
        namespace = dict(runner.__globals__, decision=LossDecision())
        runner = FunctionType(runner.__code__, namespace, runner.__name__, runner.__defaults__)
        model, row = runner(base, opt, pool, probe, view, choices, 'alignment', 'time',
                            budget=plan['budget_seconds'], clock=clock)
        row['attempted_refreshes'] = row['accepted_refreshes'] = 0
    else:
        raise ValueError(f'Unknown arm: {arm}')
    row['candidates'] = width
    return model, row


def build_state(train, lang, seed, plan):
    torch.manual_seed(seed)
    model = original.ByteTransformer()
    opt = original.optimizer(model)
    rng = np.random.default_rng(seed)
    for _ in range(plan['warmup_updates']):
        original.update(model, opt, train[rng.integers(len(train), size=plan['warmup_batch_size'])])
    initial = fingerprints(model, opt)
    ids = rng.integers(len(train), size=(plan['schedule_length'], plan['schedule_width']))
    choices = rng.integers(4, size=plan['schedule_length'])
    schedule = hashlib.sha256(ids.tobytes() + choices.tobytes()).hexdigest()
    return model, opt, ids, choices, dict(language=lang, seed=seed, initial_model_sha256=initial[0],
                                          initial_optimizer_sha256=initial[1], schedule_sha256=schedule)


def random_curve(base, opt, pool, ids, choices, target, counts):
    """Target loss after the first n random-schedule updates, for each n in counts."""
    model, o = original.clone(base, opt)
    done, curve = 0, {}
    for n in sorted(set(counts)):
        while done < n:
            c = int(choices[done])
            original.update(model, o, pool[ids[done, :4]][c:c + 1])
            done += 1
        curve[n] = original.evaluate(model, target)
    return curve


def bootstrap(values, plan):
    values = np.asarray(values)
    rng = np.random.default_rng(plan['bootstrap_seed'])
    draws = rng.choice(values, (plan['bootstrap_resamples'], len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [.025, .975])]


def contrast(states, a, b, plan):
    values = [s['arms'][a]['final_loss'] - s['arms'][b]['final_loss'] for s in states]
    return dict(mean=mean(values), ci95=bootstrap(values, plan), per_seed=values,
                seeds_negative=sum(v < 0 for v in values), seeds_positive=sum(v > 0 for v in values))


def summarize(rows, curves, states, plan):
    seen = set()
    for row in rows:
        key = (row['language'], row['seed'], row['repeat'], row['arm'])
        if key in seen:
            raise ValueError('Duplicate run')
        seen.add(key)
        if not all(np.isfinite(row[k]) for k in ('initial_loss', 'final_loss', 'actual_seconds')):
            raise ValueError('Nonfinite run metric')
    reps = plan['timing_repeats']
    state_metrics = []
    for state in states:
        runs = [r for r in rows if r['language'] == state['language'] and r['seed'] == state['seed']]
        if len(runs) != len(ARMS) * reps:
            raise ValueError('Incomplete state runs')
        curve = curves[state['language'], state['seed']]
        initial = runs[0]['initial_loss']
        arms, terms = {}, {}
        for arm in ARMS:
            matched = [r for r in runs if r['arm'] == arm]
            if sorted(r['repeat'] for r in matched) != list(range(reps)):
                raise ValueError('Unpaired repeats')
            if sorted(r['position'] for r in matched) != list(range(len(ARMS))):
                raise ValueError('Arm does not occupy each order position once')
            arms[arm] = {k: mean(r[k] for r in matched) for k in METRICS}
            if arm != 'random':
                at_n = mean(curve[r['updates']] for r in matched)
                terms[arm] = dict(random_at_matched_updates=at_n)
        random_final = arms['random']['final_loss']
        for arm in GUIDED:
            t = terms[arm]
            a = initial - t['random_at_matched_updates']
            s = t['random_at_matched_updates'] - arms[arm]['final_loss']
            r = arms['random']['updates'] / arms[arm]['updates']
            predicted = s / a > r - 1
            observed = arms[arm]['final_loss'] < random_final
            t.update(r=r, a=a, s=s, s_over_a=s / a, selection_term=-s,
                     throughput_term=t['random_at_matched_updates'] - random_final,
                     predicted_win=bool(predicted), observed_win=bool(observed),
                     agreement=bool(predicted == observed))
        state_metrics.append(dict(**state, initial_loss=initial, arms=arms, decomposition=terms))
    languages = {}
    for lang in plan['languages']:
        group = [s for s in state_metrics if s['language'] == lang]
        if [s['seed'] for s in group] != plan['seeds']:
            raise ValueError('Missing language states')
        contrasts = {f'{a}_minus_random': contrast(group, a, 'random', plan) for a in GUIDED}
        contrasts['loss16_minus_loss4'] = contrast(group, 'loss16', 'loss4', plan)
        decomposition = {}
        for arm in GUIDED:
            d = [s['decomposition'][arm] for s in group]
            decomposition[arm] = {k: mean(x[k] for x in d) for k in
                                  ('r', 's_over_a', 'selection_term', 'throughput_term')}
            decomposition[arm].update(agreements=sum(x['agreement'] for x in d), cases=len(d),
                                      predicted_wins=sum(x['predicted_win'] for x in d),
                                      observed_wins=sum(x['observed_win'] for x in d))
        languages[lang] = dict(initial_loss=mean(s['initial_loss'] for s in group),
                               arms={a: {k: mean(s['arms'][a][k] for s in group) for k in METRICS} for a in ARMS},
                               contrasts=contrasts, decomposition=decomposition)
    cases = [s['decomposition'][a] for s in state_metrics for a in GUIDED]
    return dict(primary=dict(language='ar', contrast='loss16_minus_random',
                             **{k: languages['ar']['contrasts']['loss16_minus_random'][k] for k in ('mean', 'ci95')}),
                prediction_agreement=dict(agreements=sum(c['agreement'] for c in cases), cases=len(cases)),
                states=state_metrics, languages=languages)


def write_line(file, row):
    file.write(json.dumps(row, allow_nan=False) + '\n')
    file.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = dict(result=args.output_dir / 'ud-cost-frontier.json',
                 records=args.output_dir / 'ud-cost-frontier-records.jsonl')
    if any(p.exists() for p in paths.values()):
        parser.error('Choose an unused output directory to preserve recorded runs')
    plan = json.loads(PROTOCOL.read_text())
    if tuple(plan['arms']) != ARMS or plan['stage'] != 65 or plan['timing_repeats'] != len(ARMS):
        raise ValueError('Protocol differs from the implementation')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    hashes = {}
    for path in (PROTOCOL, MANIFEST, Path('experiments/ud_cost_frontier.py'),
                 Path('tests/test_ud_cost_frontier.py'), Path('experiments/pud_reuse.py'),
                 Path('experiments/pud_policy.py'), Path('experiments/multilingual_readiness.py'),
                 Path('experiments/multilingual_policy.py'), Path('experiments/head_invariance.py')):
        hashes[str(path)] = digest(path)
    manifest = json.loads(MANIFEST.read_text())
    prepared = {lang: prepare_language(manifest['languages'][lang]['splits'], plan, hashes)
                for lang in plan['languages']}
    cpu_line = next(line for line in Path('/proc/cpuinfo').read_text().splitlines()
                    if line.startswith('model name'))
    meta = dict(sha256=hashes, cpu=cpu_line.split(':')[1].strip(), platform=platform.platform(),
                python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                deterministic=torch.are_deterministic_algorithms_enabled(),
                openblas_num_threads=os.environ.get('OPENBLAS_NUM_THREADS'), logical_cpus=os.cpu_count())
    rows, coverage, states, curves = [], [], [], {}
    start = time.perf_counter()
    with paths['records'].open('x') as records:
        for lang in plan['languages']:
            train, probe, target, _ = prepared[lang]
            for seed in plan['seeds']:
                index = len(states)
                base, opt, ids, choices, info = build_state(train, lang, seed, plan)
                initial = original.evaluate(base, target)
                state = dict(language=lang, seed=seed)
                states.append(state)
                coverage.append(dict(**info, initial_loss=initial))
                state_rows = []
                for repeat in range(plan['timing_repeats']):
                    for position, arm in enumerate(arm_order(index, repeat)):
                        model, row = run_arm(base, opt, train, probe, ids, choices, arm, plan)
                        final = original.evaluate(model, target)
                        if not np.isfinite(initial) or not np.isfinite(final):
                            raise ValueError('Nonfinite evaluation')
                        row.update(**state, arm=arm, repeat=repeat, position=position,
                                   initial_loss=initial, final_loss=final,
                                   late_attempt=row['attempted'] - row['updates'],
                                   schedule_sha256=info['schedule_sha256'])
                        rows.append(row)
                        state_rows.append(row)
                        write_line(records, row)
                counts = [r['updates'] for r in state_rows if r['arm'] != 'random']
                curve = random_curve(base, opt, train, ids, choices, target, counts)
                curves[lang, seed] = curve
                coverage[-1]['random_curve'] = {str(k): v for k, v in sorted(curve.items())}
                print(f'{lang} seed {seed}: {len(rows)} timed runs', flush=True)
    summary = summarize(rows, curves, states, plan)
    result = dict(stage=65, protocol=plan, provenance=meta,
                  output_sha256={paths['records'].name: digest(paths['records'])},
                  populations={lang: prepared[lang][3] for lang in plan['languages']},
                  coverage=coverage, summary=summary, runs=len(rows),
                  wall_seconds=time.perf_counter() - start)
    with paths['result'].open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write('\n')


if __name__ == '__main__':
    main()

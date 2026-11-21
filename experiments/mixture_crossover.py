"""Stage 67: predict the timed crossover of guided selection under contamination.

Phase A (--phase A): equal-update runs give s/a and step-cost ratio r, and the
first-order rule writes predictions. Phase B (--phase B): timed runs on new
seeds test those predictions. Implements research/mixture-crossover-protocol.json.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import mean, median
import time

import numpy as np
import torch

from experiments import pud_policy as original
from experiments import ud_cost_frontier as frontier
from experiments.llm_selection import exposed_dev_chunks
from experiments.multilingual_readiness import byte_chunks
from experiments.pud_reuse import CachedDecision
from experiments.pud_reuse_matched import digest

PROTOCOL = Path('research/mixture-crossover-protocol.json')
MANIFEST = Path('data/multilingual_sources.json')
ARMS = ('random', 'loss4', 'reuse4')
GUIDED = ARMS[1:]
OUT = dict(A=('mixture-crossover-phaseA.json', 'mixture-crossover-phaseA-records.jsonl'),
           B=('mixture-crossover-phaseB.json', 'mixture-crossover-phaseB-records.jsonl'))
PREDICTIONS = Path('results/mixture-crossover-predictions.json')


def stage66_line_chunks(blob):
    """Byte chunks meeting any dev line that stage 66 treated as eligible."""
    exposed = exposed_dev_chunks(len(blob) // 65)
    touched, offset = set(), 0
    for line in blob.split(b'\n'):
        end = offset + len(line)
        first, last = offset // 65, max(offset, end - 1) // 65
        span = range(first, last + 1)
        if line and not any(i in exposed for i in span):
            touched.update(span)
        offset = end + 1
    return touched


def prepare(manifest, plan, hashes):
    splits = {lang: manifest['languages'][lang]['splits'] for lang in ('ar', 'en')}
    for lang in splits:
        for split in ('train', 'dev'):
            entry = splits[lang][split]
            if digest(Path(entry['path'])) != entry['sha256']:
                raise ValueError(f'Prepared {lang}/{split} checksum mismatch')
            hashes[entry['path']] = entry['sha256']
    ar_train = byte_chunks(Path(splits['ar']['train']['path']).read_bytes())
    en_train = byte_chunks(Path(splits['en']['train']['path']).read_bytes())
    dev_blob = Path(splits['ar']['dev']['path']).read_bytes()
    dev = byte_chunks(dev_blob)
    excluded = exposed_dev_chunks(len(dev)) | stage66_line_chunks(dev_blob)
    eligible = np.array(sorted(set(range(len(dev))) - excluded))
    pick = np.random.default_rng(6701).choice(len(eligible), 512, replace=False)
    target_ids = np.sort(eligible[pick])
    probe_ids = np.sort(np.random.default_rng(6301).choice(len(dev), 64, replace=False))
    pool = torch.cat([ar_train, en_train])
    population = dict(ar_train_chunks=len(ar_train), en_train_chunks=len(en_train), dev_chunks=len(dev),
                      excluded_dev_chunks=len(excluded), eligible_dev_chunks=len(eligible),
                      target_ids=target_ids.tolist(), probe_ids=probe_ids.tolist())
    return pool, len(ar_train), len(en_train), dev[probe_ids], dev[target_ids], population


def mixture(rng, rho, n_ar, n_en, shape):
    flags = rng.random(shape) < rho
    ar = rng.integers(n_ar, size=shape)
    en = rng.integers(n_en, size=shape)
    return np.where(flags, n_ar + en, ar)


def build_state(pool, n_ar, n_en, rho_index, rho, seed, length):
    torch.manual_seed(seed)
    model = original.ByteTransformer()
    opt = original.optimizer(model)
    rng = np.random.default_rng(seed + 1000 * rho_index)
    for _ in range(32):
        original.update(model, opt, pool[mixture(rng, rho, n_ar, n_en, 4)])
    ids = mixture(rng, rho, n_ar, n_en, (length, 4))
    choices = rng.integers(4, size=length)
    digest_ = hashlib.sha256(ids.tobytes() + choices.tobytes()).hexdigest()
    return model, opt, ids, choices, digest_


def fixed_run(base, opt, pool, probe, ids, choices, arm, updates, clock=time.perf_counter):
    model, o = original.clone(base, opt)
    decide = dict(random=None, loss4=frontier.LossDecision(), reuse4=CachedDecision())[arm]
    picked, seconds = [], []
    for step in range(updates):
        candidates = pool[ids[step]]
        start = clock()
        if arm == 'random':
            model, o, c = original.decision(model, o, candidates, probe, 'random', int(choices[step]))
        else:
            model, o, c = decide(model, o, candidates, probe, 'alignment', int(choices[step]))
        seconds.append(clock() - start)
        picked.append(c)
    return model, dict(choices=picked, step_seconds=seconds)


def english_share(ids, choices, n_ar):
    return mean(float(ids[i][c] >= n_ar) for i, c in enumerate(choices)) if choices else 0.


def summarize_a(rows, plan):
    phase = plan['phase_A']
    r = {arm: median(s for x in rows if x['arm'] == arm for s in x['step_seconds']) /
              median(s for x in rows if x['arm'] == 'random' for s in x['step_seconds']) for arm in GUIDED}
    per_rho = {}
    for rho in plan['rho']:
        group = {arm: sorted([x for x in rows if x['rho'] == rho and x['arm'] == arm], key=lambda x: x['seed'])
                 for arm in ARMS}
        for arm in ARMS:
            if [x['seed'] for x in group[arm]] != phase['seeds']:
                raise ValueError('Incomplete phase-A grid')
        loss0 = mean(x['loss0'] for x in group['random'])
        rand = mean(x['loss_final'] for x in group['random'])
        a = loss0 - rand
        entry = dict(loss0=loss0, random_final=rand, a=a, arms={})
        for arm in GUIDED:
            final = mean(x['loss_final'] for x in group[arm])
            s = rand - final
            entry['arms'][arm] = dict(final=final, s=s, s_over_a=s / a, r=r[arm], threshold=r[arm] - 1,
                                      predicted_guided_wins=bool(s / a > r[arm] - 1),
                                      english_share=mean(x['english_share'] for x in group[arm]))
        entry['random_english_share'] = mean(x['english_share'] for x in group['random'])
        per_rho[str(rho)] = entry
    return dict(r=r, per_rho=per_rho)


def summarize_b(rows, curves, plan, predictions):
    phase = plan['phase_B']
    per_rho, agree, cases, strict = {}, 0, 0, dict(agree=0, contradict=0, unresolved=0)
    for rho in plan['rho']:
        states = []
        for seed in phase['seeds']:
            runs = [x for x in rows if x['rho'] == rho and x['seed'] == seed]
            if len(runs) != len(ARMS) * phase['timing_repeats']:
                raise ValueError('Incomplete phase-B state')
            arms = {arm: dict(final=mean(x['final_loss'] for x in runs if x['arm'] == arm),
                              updates=mean(x['updates'] for x in runs if x['arm'] == arm)) for arm in ARMS}
            curve = curves[str(rho), seed]
            for arm in GUIDED:
                at_n = mean(curve[x['updates']] for x in runs if x['arm'] == arm)
                arms[arm].update(selection_term=arms[arm]['final'] - at_n,
                                 throughput_term=at_n - arms['random']['final'])
            states.append(arms)
        entry = dict(arms={arm: dict(final=mean(s[arm]['final'] for s in states),
                                     updates=mean(s[arm]['updates'] for s in states)) for arm in ARMS})
        for arm in GUIDED:
            diff = [s[arm]['final'] - s['random']['final'] for s in states]
            draws = np.random.default_rng(phase.get('bootstrap_seed', 6702)).choice(
                diff, (10000, len(diff)), replace=True).mean(axis=1)
            ci = [float(x) for x in np.quantile(draws, [.025, .975])]
            observed = mean(diff) < 0
            predicted = predictions['per_rho'][str(rho)]['arms'][arm]['predicted_guided_wins']
            agree += observed == predicted
            cases += 1
            if ci[1] < 0 or ci[0] > 0:
                strict['agree' if (ci[1] < 0) == predicted else 'contradict'] += 1
            else:
                strict['unresolved'] += 1
            entry['arms'][arm].update(minus_random=mean(diff), ci95=ci, per_seed=diff,
                                      observed_guided_wins=observed, predicted_guided_wins=predicted,
                                      selection_term=mean(s[arm]['selection_term'] for s in states),
                                      throughput_term=mean(s[arm]['throughput_term'] for s in states))
        per_rho[str(rho)] = entry
    return dict(primary=dict(agreements=agree, cases=cases), interval_resolved=strict, per_rho=per_rho)


def provenance(hashes):
    for p in (PROTOCOL, MANIFEST, Path('experiments/mixture_crossover.py'), Path('tests/test_mixture_crossover.py'),
              Path('experiments/ud_cost_frontier.py'), Path('experiments/pud_reuse.py'),
              Path('experiments/pud_policy.py'), Path('experiments/llm_selection.py')):
        hashes[str(p)] = digest(p)
    cpu = next(l for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name'))
    return dict(sha256=hashes, cpu=cpu.split(':')[1].strip(), platform=platform.platform(),
                python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                torch_threads=torch.get_num_threads(), logical_cpus=os.cpu_count(),
                deterministic=torch.are_deterministic_algorithms_enabled())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['A', 'B'], required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    args = parser.parse_args()
    result_path, record_path = (args.output_dir / name for name in OUT[args.phase])
    if result_path.exists() or record_path.exists():
        parser.error('Choose an unused output directory to preserve recorded runs')
    plan = json.loads(PROTOCOL.read_text())
    if tuple(plan['arms']) != ARMS or plan['stage'] != 67:
        raise ValueError('Protocol differs from the implementation')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    hashes = {}
    pool, n_ar, n_en, probe, target, population = prepare(json.loads(MANIFEST.read_text()), plan, hashes)
    if args.phase == 'B':
        predictions = json.loads(PREDICTIONS.read_text())
        hashes[str(PREDICTIONS)] = digest(PREDICTIONS)
    meta = provenance(hashes)
    rows, curves, coverage = [], {}, []
    start = time.perf_counter()
    with record_path.open('x') as records:
        if args.phase == 'A':
            for k, rho in enumerate(plan['rho']):
                for seed in plan['phase_A']['seeds']:
                    base, opt, ids, choices, sched = build_state(pool, n_ar, n_en, k, rho, seed, 20000)
                    loss0 = original.evaluate(base, target)
                    for arm in ARMS:
                        model, row = fixed_run(base, opt, pool, probe, ids, choices, arm, plan['phase_A']['updates'])
                        row.update(rho=rho, seed=seed, arm=arm, loss0=loss0, loss_final=original.evaluate(model, target),
                                   english_share=english_share(ids, row['choices'], n_ar), schedule_sha256=sched)
                        rows.append(row)
                        records.write(json.dumps(row, allow_nan=False) + '\n')
                        records.flush()
                    print(f'A rho {rho} seed {seed} done', flush=True)
            summary = summarize_a(rows, plan)
        else:
            timed_plan = dict(candidates={arm: 4 for arm in ARMS}, budget_seconds=plan['phase_B']['budget_seconds'])
            index = 0
            for k, rho in enumerate(plan['rho']):
                for seed in plan['phase_B']['seeds']:
                    base, opt, ids, choices, sched = build_state(pool, n_ar, n_en, k, rho, seed,
                                                                 plan['phase_B']['schedule_length'])
                    loss0 = original.evaluate(base, target)
                    state_rows = []
                    for repeat in range(plan['phase_B']['timing_repeats']):
                        shift = (index + repeat) % 3
                        for position, arm in enumerate(ARMS[shift:] + ARMS[:shift]):
                            model, row = frontier.run_arm(base, opt, pool, probe, ids, choices, arm, timed_plan)
                            row.update(rho=rho, seed=seed, arm=arm, repeat=repeat, position=position,
                                       initial_loss=loss0, final_loss=original.evaluate(model, target),
                                       english_share=english_share(ids, row['choices'], n_ar),
                                       late_attempt=row['attempted'] - row['updates'], schedule_sha256=sched)
                            rows.append(row)
                            state_rows.append(row)
                            records.write(json.dumps(row, allow_nan=False) + '\n')
                            records.flush()
                    counts = [x['updates'] for x in state_rows if x['arm'] != 'random']
                    curve = frontier.random_curve(base, opt, pool, ids, choices, target, counts)
                    curves[str(rho), seed] = curve
                    coverage.append(dict(rho=rho, seed=seed, schedule_sha256=sched,
                                         random_curve={str(n): v for n, v in sorted(curve.items())}))
                    index += 1
                    print(f'B rho {rho} seed {seed} done', flush=True)
            summary = summarize_b(rows, curves, plan, predictions)
    result = dict(stage=67, phase=args.phase, protocol=plan, provenance=meta, population=population,
                  coverage=coverage, output_sha256={record_path.name: digest(record_path)},
                  summary=summary, runs=len(rows), wall_seconds=time.perf_counter() - start)
    with result_path.open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write('\n')
    if args.phase == 'A':
        with PREDICTIONS.open('x') as out:
            json.dump(dict(stage=67, written_before_phase_B=True, source=result_path.name,
                           source_sha256=digest(result_path), r=summary['r'], per_rho=summary['per_rho']),
                      out, indent=2)
            out.write('\n')


if __name__ == '__main__':
    main()

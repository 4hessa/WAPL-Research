"""Paired one-update counterfactuals on the archived PUD reuse trajectories."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import time

import numpy as np
import torch

from experiments import pud_policy as original
from experiments.pud_policy_profile import check_winners, provenance
from experiments.pud_reuse import CachedDecision
from experiments.pud_reuse_matched import digest, state_fingerprint


PROTOCOL = Path('research/pud-reuse-one-step-protocol.json')
DECISIONS = Path('results/pud-reuse-decisions.jsonl')
MATCHED = Path('results/pud-reuse-matched.json')
HISTORY = Path('results/pud-reuse.json')
KEYS = ('language', 'fold', 'seed')


def fingerprints(model, opt):
    return state_fingerprint(model.state_dict()), state_fingerprint(opt.state_dict())


def read_loss(model, opt, target):
    state = fingerprints(model, opt)
    training = tuple(layer.training for layer in model.modules())
    loss = original.evaluate(model, target)
    if not np.isfinite(loss):
        raise ValueError('Nonfinite target loss')
    if state != fingerprints(model, opt) or training != tuple(layer.training for layer in model.modules()):
        raise ValueError('Target evaluation mutated the branch state or training mode')
    return loss, state


def paired_steps(model, opt, pool, probe, ids, choices, target, steps, interval):
    cache = CachedDecision(interval)
    for step in range(steps):
        parent = fingerprints(model, opt)
        inputs = (model, opt, pool[ids[step]], probe, 'alignment', int(choices[step]))
        # Both selections are finished before the target is evaluated.
        fresh = original.decision(*inputs)
        if parent != fingerprints(model, opt):
            raise ValueError('Fresh branch mutated its parent')
        reused = cache(*inputs)
        if parent != fingerprints(model, opt):
            raise ValueError('Cached branch mutated its parent')
        agreement = reused[2] == fresh[2]
        if step % interval == 0 and not agreement:
            raise ValueError('Choices diverged on a refreshed probe')
        if agreement:
            check_winners(reused, fresh)
        reuse_loss, reuse_state = read_loss(*reused[:2], target)
        fresh_loss, fresh_state = read_loss(*fresh[:2], target)
        if agreement and (reuse_loss != fresh_loss or reuse_state != fresh_state):
            raise ValueError('Identical choices yielded different states or target losses')
        yield dict(step=step, age=step % interval, candidate_ids=ids[step].tolist(),
                   fresh_choice=fresh[2], reuse_choice=reused[2], agreement=agreement,
                   fresh_loss=fresh_loss, reuse_loss=reuse_loss, delta=reuse_loss - fresh_loss,
                   parent_model_sha256=parent[0], parent_optimizer_sha256=parent[1],
                   fresh_model_sha256=fresh_state[0], fresh_optimizer_sha256=fresh_state[1],
                   reuse_model_sha256=reuse_state[0], reuse_optimizer_sha256=reuse_state[1])
        model, opt = reused[:2]


def moments(rows):
    values = [r['delta'] for r in rows]
    return dict(n=len(values), mean_delta=mean(values) if values else None,
                mean_abs_delta=mean(abs(d) for d in values) if values else None,
                worse=sum(d > 0 for d in values), better=sum(d < 0 for d in values),
                tied=sum(d == 0 for d in values))


def aggregate(rows):
    changed = [r for r in rows if not r['agreement']]
    return dict(**moments(rows), changed=len(changed), changed_only=moments(changed))


def grouped(rows, interval):
    return dict(all=aggregate(rows), stale=aggregate([r for r in rows if r['age'] != 0]),
                by_age={str(age): aggregate([r for r in rows if r['age'] == age]) for age in range(interval)})


def summarize(rows, plan, steps, interval):
    states, languages = [], {}
    for lang in plan['languages']:
        cases = [r for r in rows if r['language'] == lang]
        state_stale_means = []
        for fold in plan['folds']:
            for seed in plan['seeds']:
                rr = [r for r in cases if r['fold'] == fold and r['seed'] == seed]
                if sorted(r['step'] for r in rr) != list(range(steps)):
                    raise ValueError('Incomplete or duplicate decisions in trajectory')
                state = dict(language=lang, fold=fold, seed=seed, **grouped(rr, interval))
                states.append(state)
                state_stale_means.append(state['stale']['mean_delta'])
        # Every state contributes exactly the same number of observations at
        # each age. Averaging all stale decisions weights the four states equally.
        languages[lang] = grouped(cases, interval)
        languages[lang]['stale']['mean_delta'] = mean(state_stale_means)
    if len(rows) != steps * len(states):
        raise ValueError('Unexpected trajectory')
    return dict(states=states, languages=languages)


def verify_inputs():
    meta = provenance()
    for source in (MATCHED, HISTORY):
        audit_path = source.with_name(source.stem + '-audit.json')
        audit = json.loads(audit_path.read_text())
        if audit['status'] != 'passed' or digest(source) != audit['result_sha256']:
            raise ValueError(f'Audited input mismatch: {source}')
        meta['sha256'][str(audit_path)] = digest(audit_path)
    matched = json.loads(MATCHED.read_text())
    history = json.loads(HISTORY.read_text())
    if digest(DECISIONS) != history['output_sha256'][DECISIONS.name]:
        raise ValueError('Archived choices differ from stage 60')
    for path, known in matched['provenance']['sha256'].items():
        if digest(Path(path)) != known:
            raise ValueError(f'Stage 61 source mismatch: {path}')
    for path in (PROTOCOL, HISTORY, MATCHED, DECISIONS, Path('experiments/pud_reuse_one_step.py'),
                 Path('tests/test_pud_reuse_one_step.py'), Path('experiments/pud_reuse_matched.py'),
                 Path('experiments/pud_reuse.py')):
        meta['sha256'][str(path)] = digest(path)
    return meta, matched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/pud-reuse-one-step.json'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new --output path to preserve the original experiment')
    plan = json.loads(PROTOCOL.read_text())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    meta, matched = verify_inputs()
    steps, interval = plan['steps'], plan['refresh_interval']
    previous = [json.loads(s) for s in DECISIONS.read_text().splitlines()]
    past = {tuple(r[k] for k in (*KEYS, 'step')): r for r in previous}
    profiles = {tuple(r[k] for k in KEYS): r for r in matched['coverage']}
    starts = {tuple(r[k] for k in KEYS): r for r in matched['rows'] if r['arm'] == 'reuse4'}
    manifest = json.loads(Path('data/pud_sources.json').read_text())
    rows, coverage = [], []
    start = time.perf_counter()
    for lang in plan['languages']:
        sentences = [json.loads(s) for s in Path(manifest['processed'][lang]['path']).read_text().splitlines()]
        for fold in plan['folds']:
            pool, probe, target = original.partition(sentences, manifest['folds_source_indices'], fold)
            for seed in plan['seeds']:
                key = (lang, fold, seed)
                state = dict(zip(KEYS, key))
                torch.manual_seed(seed)
                rng = np.random.default_rng(seed)
                model = original.ByteTransformer()
                opt = original.optimizer(model)
                for _ in range(32):
                    original.update(model, opt, pool[rng.integers(len(pool), size=4)])
                ids = rng.integers(len(pool), size=(20000, 4))
                choices = rng.integers(4, size=20000)
                schedule = hashlib.sha256(ids.tobytes() + choices.tobytes()).hexdigest()
                info = dict(**state, pool_chunks=len(pool), probe_chunks=len(probe), target_chunks=len(target),
                            schedule_sha256=schedule)
                if profiles[key] != info:
                    raise ValueError('Incompatible partition or candidate schedule')
                initial_loss, initial_state = read_loss(model, opt, target)
                ref = starts[key]
                if initial_state != (ref['initial_model_sha256'], ref['initial_optimizer_sha256']):
                    raise ValueError('Reconstructed warmup differs from the archived stage-61 state')
                info.update(initial_loss=initial_loss, initial_model_sha256=initial_state[0],
                            initial_optimizer_sha256=initial_state[1])
                for row in paired_steps(model, opt, pool, probe, ids, choices, target, steps, interval):
                    archived = past[(*key, row['step'])]
                    for name in ('age', 'fresh_choice', 'reuse_choice', 'agreement'):
                        if row[name] != archived[name]:
                            raise ValueError(f'Archived stage-60 replay mismatch: {key}, {row["step"]}, {name}')
                    if row['reuse_choice'] != ref['choices'][row['step']]:
                        raise ValueError('Stage-61 trajectory prefix mismatch')
                    row.update(state)
                    rows.append(row)
                info['stage61_initial_loss_difference'] = initial_loss - ref['losses']['0']
                info['stage61_step16_difference'] = rows[-1]['reuse_loss'] - ref['losses']['16']
                for field, cp in [('stage61_initial_loss_difference', '0'), ('stage61_step16_difference', '16')]:
                    tolerance = float(np.spacing(np.float32(ref['losses'][cp])))
                    if abs(info[field]) > tolerance:
                        raise ValueError(f'Checkpoint replay exceeds one float32 ULP: {field}')
                coverage.append(info)
                print(f'{lang} fold {fold} seed {seed}: {len(rows)} paired steps', flush=True)
    report = dict(stage=62, protocol=plan, provenance=meta, coverage=coverage, rows=rows,
                  summary=summarize(rows, plan, steps, interval), wall_seconds=time.perf_counter() - start)
    with args.output.open('x') as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write('\n')


if __name__ == '__main__':
    main()

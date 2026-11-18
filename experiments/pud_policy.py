"""Stage-57 frozen policy comparison. Target tensors never enter the selector."""
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from experiments.multilingual_readiness import ByteTransformer, byte_loss
from experiments.multilingual_policy import clone, evaluate, optimizer, update
from experiments.head_invariance import canonical_head, flatten
from experiments.pud_transfer import partition


def alignment_score(delta, probe_gradient):
    score = float(-flatten(canonical_head(delta)).dot(probe_gradient))
    if not np.isfinite(score):
        raise ValueError('Nonfinite alignment score')
    return score


def decision(model, opt, candidates, probe, arm, random_choice):
    if arm == 'random':
        winner, wo = clone(model, opt)
        update(winner, wo, candidates[random_choice:random_choice + 1])
        return winner, wo, int(random_choice)
    if arm != 'alignment':
        raise ValueError('Unknown arm')
    pm, po = clone(model, opt)
    po.zero_grad()
    byte_loss(pm, probe).backward()
    pg = flatten({k: p.grad for k, p in pm.named_parameters()})
    before = {k: p.detach() for k, p in model.named_parameters()}
    best = -float('inf')
    winner = wo = None
    chosen = None
    for i in range(len(candidates)):
        trial, to = clone(model, opt)
        update(trial, to, candidates[i:i + 1])
        delta = {k: p.detach() - before[k] for k, p in trial.named_parameters()}
        score = alignment_score(delta, pg)
        if score > best:  # Strict comparison preserves first-index tie breaking.
            best, winner, wo, chosen = score, trial, to, i
    return winner, wo, chosen


def run_policy(base, opt, pool, probe, ids, choices, arm, mode,
               budget=3., fixed_updates=128, clock=time.perf_counter):
    if arm not in ['random', 'alignment'] or mode not in ['time', 'updates']:
        raise ValueError('Unknown arm or budget mode')
    model, opt = clone(base, opt)
    accepted = []
    attempted = 0
    last = 0.
    start = clock()
    limit = len(ids) if mode == 'time' else fixed_updates
    if limit > len(ids):
        raise ValueError('Insufficient schedule')
    for step in range(limit):
        if mode == 'time' and clock() - start >= budget:
            break
        attempted += 1
        trial, to, choice = decision(model, opt, pool[ids[step]], probe, arm, int(choices[step]))
        completed = clock() - start
        if mode == 'time' and completed > budget:
            break
        model, opt = trial, to
        accepted.append(choice)
        last = completed
    actual = clock() - start
    if mode == 'time' and len(accepted) == len(ids):
        raise ValueError('Schedule exhausted')
    for p in model.parameters():
        if not torch.isfinite(p).all():
            raise ValueError('Nonfinite model state')
    return model, dict(updates=len(accepted), attempted=attempted, choices=accepted,
                      last_accepted_seconds=last, actual_seconds=actual,
                      overrun_seconds=max(0., actual-budget) if mode == 'time' else 0.)


def summarize(rows, plan):
    lookup = {}
    for row in rows:
        key = tuple(row[k] for k in ['language', 'fold', 'seed', 'mode', 'repeat', 'arm'])
        if key in lookup:
            raise ValueError('Duplicate run')
        lookup[key] = row
        if not all(np.isfinite(row[k]) for k in ['initial_loss', 'final_loss', 'actual_seconds', 'last_accepted_seconds']):
            raise ValueError('Nonfinite metric')
        if row['updates'] != len(row['choices']) or row['attempted'] not in [row['updates'], row['updates']+1]:
            raise ValueError('Invalid update accounting')
        if row['mode'] == 'time' and row['last_accepted_seconds'] > plan['budgets']['seconds']:
            raise ValueError('Late update accepted')
        if row['mode'] == 'updates' and row['updates'] != plan['budgets']['fixed_updates']:
            raise ValueError('Wrong update budget')
    expected = {(l, f, s, m, r, a) for l in plan['languages'] for f in plan['folds'] for s in plan['seeds'] for m in ['time', 'updates'] for r in range(plan['budgets']['time_repeats'] if m == 'time' else 1) for a in plan['arms']}
    if set(lookup) != expected:
        raise ValueError('Incomplete run grid')
    for l in plan['languages']:
        for f in plan['folds']:
            for s in plan['seeds']:
                group = [v for k, v in lookup.items() if k[:3] == (l, f, s)]
                if len({r['schedule_sha256'] for r in group}) != 1 or len({r['initial_loss'] for r in group}) != 1:
                    raise ValueError('Unpaired schedule or initial state')
    outputs, comparisons = {}, {}
    for language in plan['languages']:
        outputs[language], comparisons[language] = {}, {}
        for mode in ['time', 'updates']:
            outputs[language][mode] = {}
            for arm in plan['arms']:
                rr = [r for r in rows if (r['language'], r['mode'], r['arm']) == (language, mode, arm)]
                per = []
                for seed in plan['seeds']:
                    folds = [np.mean([lookup[(language, f, seed, mode, repeat, arm)]['final_loss'] for repeat in range(plan['budgets']['time_repeats'] if mode == 'time' else 1)]) for f in plan['folds']]
                    per.append(float(np.mean(folds)))
                outputs[language][mode][arm] = {k: float(np.mean([r[k] for r in rr])) for k in ['final_loss', 'updates', 'actual_seconds', 'overrun_seconds']}
                outputs[language][mode][arm]['per_seed_mean_loss'] = per
            o = outputs[language][mode]
            d = np.array(o['alignment']['per_seed_mean_loss']) - np.array(o['random']['per_seed_mean_loss'])
            bs = np.random.default_rng(plan['bootstrap']['seed']).choice(d, (plan['bootstrap']['resamples'], len(d)), replace=True).mean(axis=1)
            comparisons[language][mode] = dict(difference=float(d.mean()), per_seed_difference=d.tolist(), ci95=np.quantile(bs, [.025, .975]).tolist(), role='primary' if language == 'ar' and mode == 'time' else 'secondary_diagnostic')
    return outputs, comparisons


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    plan_path = Path('research/pud-policy-protocol.json')
    plan = json.loads(plan_path.read_text())
    for name, digest in plan['input_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    manifest = json.loads(Path('data/pud_sources.json').read_text())
    journal = Path('results/pud-policy-records.jsonl')
    if journal.exists():
        raise FileExistsError('Preserve previous attempts; automatic overwrite/resume prohibited')
    files = list(plan['input_sha256']) + [str(plan_path), 'experiments/pud_policy.py', 'tests/test_pud_policy.py']
    metadata = dict(status='running', sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}, python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, platform=platform.platform(), cpu=next((x.strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')), 'unknown'), torch_threads=torch.get_num_threads(), started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), protocol_commit='bf946036783ceffaeff5b4ae1664a1e341211eea', interruptions=[])
    Path('results/pud-policy-environment.json').write_text(json.dumps(metadata, indent=2)+'\n')
    rows, coverage = [], {}
    with journal.open('x') as out:
        for language in plan['languages']:
            info = manifest['processed'][language]
            sentences = [json.loads(line) for line in Path(info['path']).read_text().splitlines()]
            coverage[language] = []
            for fold in plan['folds']:
                pool, probe, target = partition(sentences, manifest['folds_source_indices'], fold)
                assert len(probe) == 64
                coverage[language].append(dict(fold=fold, candidate_chunks=len(pool), probe_chunks=len(probe), target_chunks=len(target)))
                for seed in plan['seeds']:
                    torch.manual_seed(seed)
                    rng = np.random.default_rng(seed)
                    base = ByteTransformer()
                    opt = optimizer(base)
                    for _ in range(plan['warmup_updates']):
                        update(base, opt, pool[rng.integers(len(pool), size=plan['warmup_batch_chunks'])])
                    ids = rng.integers(len(pool), size=(plan['schedule']['length'], plan['schedule']['candidates_per_step']))
                    choices = rng.integers(plan['schedule']['candidates_per_step'], size=plan['schedule']['length'])
                    digest = hashlib.sha256(ids.tobytes()+choices.tobytes()).hexdigest()
                    initial = evaluate(base, target)
                    modes = ['updates', 'time'] if (fold+seed)%2 == 0 else ['time', 'updates']
                    for mode in modes:
                        for repeat in range(plan['budgets']['time_repeats'] if mode == 'time' else 1):
                            arms = plan['arms'] if (fold+seed+repeat)%2 == 0 else plan['arms'][::-1]
                            for arm in arms:
                                model, record = run_policy(base, opt, pool, probe, ids, choices, arm, mode, budget=plan['budgets']['seconds'], fixed_updates=plan['budgets']['fixed_updates'])
                                final = evaluate(model, target)
                                assert np.isfinite(initial) and np.isfinite(final)
                                record.update(language=language, fold=fold, seed=seed, mode=mode, repeat=repeat, arm=arm, initial_loss=initial, final_loss=final, schedule_sha256=digest)
                                rows.append(record)
                                out.write(json.dumps(record)+'\n')
                                out.flush()
                    print(f'{language} fold={fold} seed={seed}: {len(rows)}/960 runs; elapsed={time.perf_counter()-started:.1f}s', flush=True)
    outputs, comparisons = summarize(rows, plan)
    result = dict(status='complete_exploratory_policy_evaluation', outputs=outputs, comparisons=comparisons, coverage=coverage, runs=len(rows), seconds=time.perf_counter()-started, records_sha256=hashlib.sha256(journal.read_bytes()).hexdigest(), metadata=metadata)
    result['metadata']['status'] = 'complete'
    Path('results/pud-policy.json').write_text(json.dumps(result, indent=2)+'\n')
    Path('results/pud-policy-environment.json').write_text(json.dumps(result['metadata'], indent=2)+'\n')
    print(json.dumps(comparisons, indent=2), flush=True)


if __name__ == '__main__':
    main()

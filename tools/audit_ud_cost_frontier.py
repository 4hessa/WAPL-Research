"""Audit stage 65: recompute the summary and replay every recorded trajectory.

Timing is not replayable. Given recorded choices, each final model, each
random-curve point and each lossK/reuse4 decision is deterministic, so they are
recomputed exactly on the audit host.
"""
import json
from pathlib import Path

import numpy as np
import torch

from experiments import pud_policy as original
from experiments import ud_cost_frontier as frontier
from experiments.pud_reuse import CachedDecision
from tools.audit_pud_reuse import close, digest, read_json, read_rows, require


RESULT = Path('results/ud-cost-frontier.json')
RECORDS = Path('results/ud-cost-frontier-records.jsonl')
OUTPUT = Path('results/ud-cost-frontier-audit.json')


def check_timing(r, budget):
    key = (r['language'], r['seed'], r['repeat'], r['arm'])
    require(type(r['updates']) is int and r['updates'] == len(r['choices']), f'{key}: update count')
    require(r['attempted'] in (r['updates'], r['updates'] + 1), f'{key}: attempts')
    require(r['late_attempt'] == r['attempted'] - r['updates'], f'{key}: late attempt')
    require(all(type(c) is int and 0 <= c < r['candidates'] for c in r['choices']), f'{key}: choice')
    require(r['actual_seconds'] >= budget, f'{key}: early exit')
    require(r['last_accepted_seconds'] <= budget, f'{key}: deadline adherence')
    close(r['overrun_seconds'], max(0., r['actual_seconds'] - budget), f'{key}: overrun')
    expected = {'reuse4': ((r['updates'] + 3) // 4, (r['attempted'] + 3) // 4)}.get(r['arm'], (0, 0))
    require((r['accepted_refreshes'], r['attempted_refreshes']) == expected, f'{key}: refreshes')


def replay_choices(base, opt, pool, ids, choices):
    model, o = original.clone(base, opt)
    for step, c in enumerate(choices):
        original.update(model, o, pool[ids[step]][c:c + 1])
    return model


def replay_policy(base, opt, pool, probe, ids, arm, count):
    model, o = original.clone(base, opt)
    decide = CachedDecision() if arm == 'reuse4' else frontier.LossDecision()
    picked = []
    for step in range(count):
        model, o, c = decide(model, o, pool[ids[step]], probe, 'alignment', 0)
        picked.append(c)
    return picked


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    result = read_json(RESULT)
    plan = result['protocol']
    require(plan == read_json(frontier.PROTOCOL), 'protocol changed after the run')
    require(digest(RECORDS) == result['output_sha256'][RECORDS.name], 'record journal digest')
    rows = read_rows(RECORDS)
    per_state = len(frontier.ARMS) * plan['timing_repeats']
    require(len(rows) == result['runs'] == 500, 'run count')
    for path, expected in result['provenance']['sha256'].items():
        require(digest(path) == expected, f'source digest {path}')
    manifest = read_json(frontier.MANIFEST)
    states = [dict(language=l, seed=s) for l in plan['languages'] for s in plan['seeds']]
    curves = {}
    checks = dict(order=0, timing=0, random_prefix=0, policy_replay_decisions=0,
                  final_loss_replays=0, random_curve_points=0)
    for index, state in enumerate(states):
        lang, seed = state['language'], state['seed']
        if index % len(plan['seeds']) == 0:
            train, probe, target, population = frontier.prepare_language(
                manifest['languages'][lang]['splits'], plan, {})
            require(population == result['populations'][lang], 'population')
            dev_n = population['dev_chunks']
            exposed = set()
            for s, n in frontier.EXPOSED_DEV:
                exposed |= set(np.random.default_rng(s).choice(dev_n, n, replace=False).tolist())
            require(not exposed & set(population['target_ids']), 'target disjointness')
        base, opt, ids, uniform, info = frontier.build_state(train, lang, seed, plan)
        cov = result['coverage'][index]
        require(all(cov[k] == info[k] for k in info), 'state coverage')
        initial = original.evaluate(base, target)
        require(initial == cov['initial_loss'], 'initial loss')
        runs = rows[index * per_state:(index + 1) * per_state]
        for repeat in range(plan['timing_repeats']):
            for position, arm in enumerate(frontier.arm_order(index, repeat)):
                r = runs[len(frontier.ARMS) * repeat + position]
                require((r['language'], r['seed'], r['repeat'], r['position'], r['arm']) ==
                        (lang, seed, repeat, position, arm), 'run order')
                require(r['schedule_sha256'] == info['schedule_sha256'], 'schedule')
                checks['order'] += 1
        for arm in frontier.ARMS:
            width = plan['candidates'][arm]
            view = ids[:, :width]
            group = [r for r in runs if r['arm'] == arm]
            longest = max(group, key=lambda r: r['updates'])
            for r in group:
                require(r['candidates'] == width, 'candidate width')
                check_timing(r, plan['budget_seconds'])
                checks['timing'] += 1
                require(r['choices'] == longest['choices'][:r['updates']], 'prefix consistency')
            if arm == 'random':
                require(longest['choices'] == uniform[:longest['updates']].tolist(), 'random choices')
                checks['random_prefix'] += 1
            else:
                replayed = replay_policy(base, opt, train, probe, view, arm, longest['updates'])
                require(replayed == longest['choices'], f'{lang} {seed} {arm}: policy replay')
                checks['policy_replay_decisions'] += len(replayed)
            for r in group:
                final = original.evaluate(replay_choices(base, opt, train, view, r['choices']), target)
                require(final == r['final_loss'], f'{lang} {seed} {arm}: final loss replay')
                checks['final_loss_replays'] += 1
        counts = [r['updates'] for r in runs if r['arm'] != 'random']
        curve = frontier.random_curve(base, opt, train, ids, uniform, target, counts)
        require({str(k): v for k, v in sorted(curve.items())} == cov['random_curve'], 'random curve')
        checks['random_curve_points'] += len(curve)
        curves[lang, seed] = curve
        print(f'{lang} seed {seed}: audited', flush=True)
    recomputed = frontier.summarize(rows, curves, states, plan)
    require(json.dumps(recomputed, sort_keys=True) == json.dumps(result['summary'], sort_keys=True),
            'summary recomputation')
    audit = dict(status='passed', stage=65, result_sha256=digest(RESULT), records_sha256=digest(RECORDS),
                 checks=checks, notes='Timing values are not replayable; choices, decisions, random-curve '
                                      'points, final losses and the summary were recomputed on the audit host.')
    with OUTPUT.open('x') as out:
        json.dump(audit, out, indent=2)
        out.write('\n')


if __name__ == '__main__':
    main()

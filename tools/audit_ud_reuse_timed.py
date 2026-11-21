"""Audit stage 64: recompute summaries and replay every recorded trajectory.

Timing cannot be replayed. Given the recorded choices, however, each final
model is a deterministic function of the warmed state, so final losses are
recomputed exactly. Policy decisions of reuse4 and fresh are replayed without
a clock over each state's longest accepted prefix.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments import pud_policy as original
from experiments import ud_reuse_timed as timed
from experiments.pud_reuse import CachedDecision
from experiments.ud_reuse_transfer import prepare_language
from tools.audit_pud_reuse import close, digest, read_json, read_rows, require


RESULT = Path('results/ud-reuse-timed.json')
RECORDS = Path('results/ud-reuse-timed-records.jsonl')
OUTPUT = Path('results/ud-reuse-timed-audit.json')


def check_timing(r, budget):
    key = (r['language'], r['seed'], r['repeat'], r['arm'])
    require(type(r['updates']) is int and r['updates'] == len(r['choices']), f'{key}: update count')
    require(r['attempted'] in (r['updates'], r['updates'] + 1), f'{key}: attempts')
    require(r['late_attempt'] == r['attempted'] - r['updates'], f'{key}: late attempt')
    require(all(type(c) is int and 0 <= c < 4 for c in r['choices']), f'{key}: choice')
    require(r['actual_seconds'] >= budget, f'{key}: early exit')
    require(r['last_accepted_seconds'] <= budget, f'{key}: deadline adherence')
    close(r['overrun_seconds'], max(0., r['actual_seconds'] - budget), f'{key}: overrun')
    for field, n in (('accepted_refreshes', r['updates']), ('attempted_refreshes', r['attempted'])):
        expected = {'random': 0, 'fresh': n, 'reuse4': (n + 3) // 4}[r['arm']]
        require(r[field] == expected, f'{key}: {field}')


def replay_choices(base, opt, pool, ids, choices):
    model, o = original.clone(base, opt)
    for step, c in enumerate(choices):
        original.update(model, o, pool[ids[step]][c:c + 1])
    return model


def replay_policy(base, opt, pool, probe, ids, arm, count):
    model, o = original.clone(base, opt)
    cache = CachedDecision() if arm == 'reuse4' else None
    picked = []
    for step in range(count):
        inputs = (model, o, pool[ids[step]], probe, 'alignment', 0)
        model, o, c = cache(*inputs) if cache else original.decision(*inputs)
        picked.append(c)
    return picked


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    result = read_json(RESULT)
    plan = result['protocol']
    require(plan == read_json(timed.PROTOCOL), 'protocol changed after the run')
    require(digest(RECORDS) == result['output_sha256'][RECORDS.name], 'record journal digest')
    rows = read_rows(RECORDS)
    require(len(rows) == result['runs'] == 144, 'run count')
    for path, expected in result['provenance']['sha256'].items():
        require(digest(path) == expected, f'source digest {path}')
    states = [dict(language=l, seed=s) for l in plan['languages'] for s in plan['seeds']]
    recomputed = timed.summarize(rows, states, plan)
    require(json.dumps(recomputed, sort_keys=True) == json.dumps(result['summary'], sort_keys=True),
            'summary recomputation')
    stage63 = read_json(timed.STAGE63)
    ref_rows = {}
    for r in stage63['rows']:
        ref_rows.setdefault((r['language'], r['seed']), []).append(r)
    manifest = read_json(timed.MANIFEST)
    checks = dict(order=0, timing=0, random_prefix=0, policy_replay_decisions=0,
                  final_loss_replays=0, stage63_prefix=0)
    max_loss_error = 0.
    offset = 0
    for lang in plan['languages']:
        train, probe, target, population = prepare_language(manifest['languages'][lang]['splits'], plan, {})
        require(population == result['populations'][lang], 'population')
        for seed in plan['seeds']:
            index = offset // 9
            base, opt, ids, uniform, info = timed.build_state(train, lang, seed, plan)
            cov = result['coverage'][offset // 9]
            require(all(cov[k] == info[k] for k in info), 'state coverage')
            initial = original.evaluate(base, target)
            require(initial == cov['initial_loss'], 'initial loss')
            runs = rows[offset:offset + 9]
            for repeat in range(3):
                for position, arm in enumerate(timed.arm_order(index, repeat)):
                    r = runs[3 * repeat + position]
                    require((r['language'], r['seed'], r['repeat'], r['position'], r['arm']) ==
                            (lang, seed, repeat, position, arm), 'run order')
                    checks['order'] += 1
            offset += 9
            reference = sorted(ref_rows[lang, seed], key=lambda x: x['step'])
            for arm in timed.ARMS:
                group = [r for r in runs if r['arm'] == arm]
                longest = max(group, key=lambda r: r['updates'])
                for r in group:
                    check_timing(r, plan['budget_seconds'])
                    checks['timing'] += 1
                    require(r['choices'] == longest['choices'][:r['updates']], 'prefix consistency')
                if arm == 'random':
                    require(longest['choices'] == uniform[:longest['updates']].tolist(), 'random choices')
                    checks['random_prefix'] += 1
                else:
                    replayed = replay_policy(base, opt, train, probe, ids, arm, longest['updates'])
                    require(replayed == longest['choices'], f'{lang} {seed} {arm}: policy replay')
                    checks['policy_replay_decisions'] += len(replayed)
                    if arm == 'reuse4':
                        n = min(16, longest['updates'])
                        require(longest['choices'][:n] == [x['reuse_choice'] for x in reference[:n]],
                                'stage-63 reuse trajectory')
                    else:
                        require(longest['choices'][0] == reference[0]['fresh_choice'], 'stage-63 fresh step 0')
                    checks['stage63_prefix'] += 1
                for r in group:
                    final = original.evaluate(replay_choices(base, opt, train, ids, r['choices']), target)
                    max_loss_error = max(max_loss_error, abs(final - r['final_loss']))
                    require(final == r['final_loss'], f'{lang} {seed} {arm}: final loss replay')
                    checks['final_loss_replays'] += 1
            print(f'{lang} seed {seed}: audited', flush=True)
    audit = dict(status='passed', stage=64, result_sha256=digest(RESULT),
                 records_sha256=digest(RECORDS), checks=checks, max_final_loss_error=max_loss_error,
                 notes='Timing values are not replayable; choice sequences, policy decisions and '
                       'final losses were recomputed on the audit host.')
    with OUTPUT.open('x') as out:
        json.dump(audit, out, indent=2)
        out.write('\n')


if __name__ == '__main__':
    main()

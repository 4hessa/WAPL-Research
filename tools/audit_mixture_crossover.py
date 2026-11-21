"""Audit stage 67: replay phase A and phase B trajectories and recompute summaries."""
import json
from pathlib import Path

import torch

from experiments import mixture_crossover as stage
from experiments import pud_policy as original
from tools.audit_pud_reuse import digest, read_json, read_rows, require
from tools.audit_ud_cost_frontier import replay_choices, replay_policy

OUTPUT = Path('results/mixture-crossover-audit.json')


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    plan = read_json(stage.PROTOCOL)
    pool, n_ar, n_en, probe, target, population = stage.prepare(read_json(stage.MANIFEST), plan, {})
    checks = dict(phaseA_runs=0, phaseB_runs=0, policy_replay_decisions=0, random_curve_points=0)
    results = {}
    for phase in ('A', 'B'):
        result_path, record_path = (Path('results') / n for n in stage.OUT[phase])
        result = read_json(result_path)
        rows = read_rows(record_path)
        require(result['protocol'] == plan, f'{phase}: protocol changed')
        require(result['population'] == population, f'{phase}: population')
        require(digest(record_path) == result['output_sha256'][record_path.name], f'{phase}: journal digest')
        results[phase] = (result, rows)
    pred = read_json(stage.PREDICTIONS)
    require(pred['source_sha256'] == digest(Path('results') / stage.OUT['A'][0]), 'predictions source')
    require(read_json('results/mixture-crossover-predictions-addendum.json')['predictions_sha256'] ==
            digest(stage.PREDICTIONS), 'addendum source')
    resultA, rowsA = results['A']
    require(json.dumps(stage.summarize_a(rowsA, plan), sort_keys=True) == json.dumps(resultA['summary'], sort_keys=True),
            'phase-A summary')
    require(json.dumps({k: pred[k] for k in ('r', 'per_rho')}, sort_keys=True) ==
            json.dumps(resultA['summary'], sort_keys=True), 'predictions equal phase-A summary')
    for row in rowsA:
        k = plan['rho'].index(row['rho'])
        base, opt, ids, choices, sched = stage.build_state(pool, n_ar, n_en, k, row['rho'], row['seed'], 20000)
        require(sched == row['schedule_sha256'], 'phase-A schedule')
        require(original.evaluate(base, target) == row['loss0'], 'phase-A loss0')
        ticks = iter(range(10 ** 6))
        model, replay = stage.fixed_run(base, opt, pool, probe, ids, choices, row['arm'], plan['phase_A']['updates'],
                                        clock=lambda: float(next(ticks)))
        require(replay['choices'] == row['choices'], 'phase-A choices')
        require(original.evaluate(model, target) == row['loss_final'], 'phase-A final loss')
        checks['phaseA_runs'] += 1
    resultB, rowsB = results['B']
    curves = {}
    for cov in resultB['coverage']:
        rho, seed = cov['rho'], cov['seed']
        k = plan['rho'].index(rho)
        base, opt, ids, choices, sched = stage.build_state(pool, n_ar, n_en, k, rho, seed,
                                                           plan['phase_B']['schedule_length'])
        require(sched == cov['schedule_sha256'], 'phase-B schedule')
        runs = [r for r in rowsB if r['rho'] == rho and r['seed'] == seed]
        for arm in stage.ARMS:
            group = [r for r in runs if r['arm'] == arm]
            longest = max(group, key=lambda r: r['updates'])
            for r in group:
                require(r['choices'] == longest['choices'][:r['updates']], 'phase-B prefix')
                require(r['last_accepted_seconds'] <= plan['phase_B']['budget_seconds'], 'deadline')
                final = original.evaluate(replay_choices(base, opt, pool, ids, r['choices']), target)
                require(final == r['final_loss'], f'phase-B final loss {rho} {seed} {arm}')
                checks['phaseB_runs'] += 1
            if arm == 'random':
                require(longest['choices'] == choices[:longest['updates']].tolist(), 'random choices')
            else:
                replayed = replay_policy(base, opt, pool, probe, ids, arm, longest['updates'])
                require(replayed == longest['choices'], f'phase-B policy replay {rho} {seed} {arm}')
                checks['policy_replay_decisions'] += len(replayed)
        counts = [r['updates'] for r in runs if r['arm'] != 'random']
        curve = stage.frontier.random_curve(base, opt, pool, ids, choices, target, counts)
        require({str(n): v for n, v in sorted(curve.items())} == cov['random_curve'], 'random curve')
        curves[str(rho), seed] = curve
        checks['random_curve_points'] += len(curve)
        print('B', rho, seed, 'audited', flush=True)
    require(json.dumps(stage.summarize_b(rowsB, curves, plan, pred), sort_keys=True) ==
            json.dumps(resultB['summary'], sort_keys=True), 'phase-B summary')
    audit = dict(status='passed', stage=67, checks=checks,
                 sha256={n: digest(Path('results') / n) for p in stage.OUT.values() for n in p},
                 notes='Timing is not replayable; schedules, choices, decisions, final losses, random-curve '
                       'points and both summaries were recomputed on the audit host.')
    with OUTPUT.open('x') as out:
        json.dump(audit, out, indent=2)
        out.write('\n')


if __name__ == '__main__':
    main()

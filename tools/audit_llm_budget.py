"""Audit stage 68: recompute the summary, replay every run without a clock, and check
per-example batched gradients against separate backward passes on the real model."""
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments import llm_budget as stage
from experiments import llm_common as llm
from experiments import llm_selection as s66
from tools.audit_pud_reuse import digest, read_json, require

RESULT = Path('results/llm-budget.json')
RECORDS = Path('results/llm-budget-records.jsonl')
PROGRESS = Path('results/llm-budget-audit-progress.jsonl')  # resumable: one verified run per line
OUTPUT = Path('results/llm-budget-audit.json')


def verified():
    if not PROGRESS.exists():
        return set()
    return {tuple(json.loads(l)['key']) for l in PROGRESS.read_text().splitlines() if l.strip()}


def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    result = read_json(RESULT)
    plan = result['protocol']
    require(plan == read_json(stage.PROTOCOL), 'protocol changed')
    require(digest(RECORDS) == result['output_sha256'][RECORDS.name], 'journal digest')
    lines = [json.loads(x) for x in RECORDS.read_text().splitlines()]
    header, rows = lines[0], lines[1:]
    require(header['updates'] == result['updates'] == stage.budgets(header['costs'], plan), 'budgets')
    require(len(rows) == result['runs'] == 32, 'run count')
    n = result['updates']
    require(json.dumps(stage.summarize(rows, plan, n), sort_keys=True) == json.dumps(result['summary'], sort_keys=True),
            'summary')
    tokenizer = AutoTokenizer.from_pretrained(llm.MODEL_DIR)
    pool, n_ar, n_en, probe, target, population = stage.prepare(tokenizer, {})
    require(population == result['population'], 'population')
    require(not set(population['target_ids']) & set(population['probe_ids']), 'probe/target overlap')
    # Per-example gradients on the real model, at the first decision of the calibration schedule.
    model = llm.load_model(stage.CALIBRATION_SEED)
    params = llm.adapter_params(model)
    capture = stage.PerExampleCapture(model)
    ids, _ = stage.schedule(0, stage.CALIBRATION_SEED, stage.CALIBRATION_RHO, n_ar, n_en)
    batch = pool[ids[0]]
    model.zero_grad()
    capture.active = True
    llm.lm_loss(model, batch, 'none').sum().backward()
    capture.active = False
    grads = capture.per_example()
    # Pre-registered tolerance rtol 1e-4, atol 1e-6. The first audit attempt failed it
    # (results/llm-budget-audit-attempt-1.log); the outcome is recorded, not asserted.
    gradient_check = []
    for i in range(4):
        model.zero_grad()
        llm.lm_loss(model, batch[i:i + 1]).backward()
        ref = s66.flat_grad(params)
        diff = (grads[i] - ref).abs()
        allowed = 1e-6 + 1e-4 * ref.abs()
        gradient_check.append(dict(max_abs_difference=float(diff.max()),
                                   entries_outside_preregistered_tolerance=int((diff > allowed).sum()),
                                   entries=int(diff.numel()),
                                   relative_l2_difference=float(diff.norm() / ref.norm()),
                                   cosine=float(torch.nn.functional.cosine_similarity(grads[i], ref, dim=0))))
    passed = all(g['entries_outside_preregistered_tolerance'] == 0 for g in gradient_check)
    checks = dict(runs_replayed=0, choices_compared=0, per_example_gradient=gradient_check,
                  per_example_gradient_within_preregistered_tolerance=passed)
    done = verified()
    for k, rho in enumerate(plan['rho']):
        for seed in plan['seeds']:
            ids, choices = stage.schedule(k, seed, rho, n_ar, n_en)
            for row in [r for r in rows if r['rho'] == rho and r['seed'] == seed]:
                key = (rho, seed, row['arm'])
                if list(key) in [list(k) for k in done]:
                    checks['runs_replayed'] += 1
                    checks['choices_compared'] += len(row['choices'])
                    continue
                checkpoints = [0] + ([n[a] for a in stage.ARMS] if row['arm'] == 'random' else [])
                ticks = iter(range(10 ** 6))
                replay = stage.run(row['arm'], seed, pool, probe, target, ids, choices, n[row['arm']], checkpoints,
                                   clock=lambda: float(next(ticks)))
                require(replay['choices'] == row['choices'], f'{rho} {seed} {row["arm"]}: choices')
                require(replay['losses'] == row['losses'], f'{rho} {seed} {row["arm"]}: losses')
                with PROGRESS.open('a') as fh:
                    fh.write(json.dumps(dict(key=list(key))) + '\n')
                checks['runs_replayed'] += 1
                checks['choices_compared'] += len(row['choices'])
                print(rho, seed, row['arm'], 'replayed', flush=True)
    audit = dict(status='passed' if passed else 'replay passed; per-example gradient tolerance not met', stage=68, result_sha256=digest(RESULT), records_sha256=digest(RECORDS),
                 checks=checks, notes='Step times are not replayable; all choices and losses were recomputed '
                                      'exactly. See checks.per_example_gradient for the batched-versus-separate gradient comparison.')
    with OUTPUT.open('x') as out:
        json.dump(audit, out, indent=2)
        out.write('\n')


if __name__ == '__main__':
    main()

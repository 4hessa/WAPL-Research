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
OUTPUT = Path('results/llm-budget-audit.json')


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
    worst = 0.
    for i in range(4):
        model.zero_grad()
        llm.lm_loss(model, batch[i:i + 1]).backward()
        ref = s66.flat_grad(params)
        torch.testing.assert_close(grads[i], ref, rtol=1e-4, atol=1e-6)
        worst = max(worst, float((grads[i] - ref).abs().max()))
    checks = dict(runs_replayed=0, choices_compared=0, per_example_max_abs_error=worst)
    for k, rho in enumerate(plan['rho']):
        for seed in plan['seeds']:
            ids, choices = stage.schedule(k, seed, rho, n_ar, n_en)
            for row in [r for r in rows if r['rho'] == rho and r['seed'] == seed]:
                checkpoints = [0] + ([n[a] for a in stage.ARMS] if row['arm'] == 'random' else [])
                ticks = iter(range(10 ** 6))
                replay = stage.run(row['arm'], seed, pool, probe, target, ids, choices, n[row['arm']], checkpoints,
                                   clock=lambda: float(next(ticks)))
                require(replay['choices'] == row['choices'], f'{rho} {seed} {row["arm"]}: choices')
                require(replay['losses'] == row['losses'], f'{rho} {seed} {row["arm"]}: losses')
                checks['runs_replayed'] += 1
                checks['choices_compared'] += len(row['choices'])
                print(rho, seed, row['arm'], 'replayed', flush=True)
    audit = dict(status='passed', stage=68, result_sha256=digest(RESULT), records_sha256=digest(RECORDS),
                 checks=checks, notes='Step times are not replayable; all choices and losses were recomputed '
                                      'exactly; per-example gradients matched separate backward passes.')
    with OUTPUT.open('x') as out:
        json.dump(audit, out, indent=2)
        out.write('\n')


if __name__ == '__main__':
    main()

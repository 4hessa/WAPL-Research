"""Audit stage 73: journal, summary, budget, data, decontamination, pools, schedules, kept sets,
removed shares, calibration agreement; on the GPU, recompute seed-37300 alignment scores and
replay the first 50 random steps of seed 37300 on the raw pool."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from experiments import llm_gpu as s69
from experiments import llm_web_filter as stage
from tools.audit_pud_reuse import require

RESULT = Path('results/llm-web-filter.json')
JOURNAL = Path('results/llm-web-filter-journal.jsonl')
OUTPUT = Path('results/llm-web-filter-audit.json')
AUDIT_SEED, SAMPLE = 37300, 256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kept', type=Path, required=True)
    parser.add_argument('--removed', type=Path, required=True)
    parser.add_argument('--mmlu', type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--no-gpu', action='store_true', help='Skip the GPU recomputation and replay')
    args = parser.parse_args()
    plan = json.loads(stage.PROTOCOL.read_text())
    result = json.loads(RESULT.read_text())
    require(result['protocol'] == plan, 'protocol changed')
    require(s69.sha256(JOURNAL) == result['journal_sha256'], 'journal digest')
    journal = s69.read_journal(JOURNAL)
    rows = [r for r in journal if r['kind'] == 'run']
    cals = [r for r in journal if r['kind'] == 'calibration']
    require(len(cals) == 1, 'one calibration')
    cal = cals[0]
    grid = {(p, s, a) for p in stage.POOLS for s in plan['seeds'] for a in stage.ARMS}
    require(len(rows) == len(grid) and {(r['pool'], r['seed'], r['arm']) for r in rows} == grid, 'grid')
    require(all(r['device'] == cal['device'] for r in rows), 'device consistency')
    require(cal['updates'] == stage.budgets(cal['costs']), 'budget')
    require(json.dumps(stage.summarize(rows, cal, plan), sort_keys=True) ==
            json.dumps(result['summary'], sort_keys=True), 'summary recomputation')
    require(cal['fd_vs_grad'] == stage.agreement(np.array(cal['scores']['align_grad']),
                                                 np.array(cal['scores']['align_fd'])), 'calibration agreement')

    model_dir = args.models_dir / stage.MODEL
    tok = AutoTokenizer.from_pretrained(model_dir)
    probe_items, target_items, pools, population = stage.load_inputs(args, tok)  # checks all four file hashes
    require(population == cal['population'], 'samples, decontamination counts and pools')
    banned = stage.banned_ngrams(probe_items + target_items)
    for name, pool in pools.items():
        require(not any(stage.ngrams(c) & banned for c in pool.tolist()), f'{name} pool decontamination')
    checks = dict(runs=0, schedules=0, kept_sets=0)
    n = cal['updates']['align_grad']
    for k, pool in enumerate(stage.POOLS):
        for seed in plan['seeds']:
            rand, filt, digest = stage.schedule(k, seed)
            checks['schedules'] += 1
            for r in [x for x in rows if (x['pool'], x['seed']) == (pool, seed)]:
                require(r['schedule_sha256'] == digest, f'{pool} {seed}: schedule')
                require(len(r['step_seconds']) == stage.BUDGET, 'update count')
                expected = {'0', str(stage.BUDGET), str(n)}
                require(set(r['losses']) == expected == set(r['accuracy']), 'checkpoints')
                if r['arm'] == 'align_grad':
                    kept = stage.keep_top(np.array(r['scores']))
                    require(kept.tolist() == r['kept'], 'kept set from scores')
                    require(hashlib.sha256(kept.tobytes()).hexdigest() == r['kept_sha256'], 'kept digest')
                    share = float(np.mean(kept >= stage.RAW_KEPT)) if pool == 'raw' else 0.
                    require(share == r['removed_share'], 'removed share')
                    checks['kept_sets'] += 1
                checks['runs'] += 1

    gpu = None
    if not args.no_gpu:
        s69.setup_numerics()
        device = torch.device('cuda')
        require(torch.cuda.get_device_name(0) == cal['device'], 'audit device differs from the run')
        sample = np.sort(np.random.default_rng(7304).choice(stage.POOL_SIZE, SAMPLE, replace=False))
        gpu = dict(score_recomputation={})
        for pool in stage.POOLS:
            row = next(r for r in rows if (r['pool'], r['seed'], r['arm']) == (pool, AUDIT_SEED, 'align_grad'))
            scores, _ = stage.score(s69.load(model_dir, AUDIT_SEED, device), 'align_grad',
                                    pools[pool].to(device), probe_items, device)
            journaled = np.array(row['scores'])
            scale = float(np.abs(journaled).max())
            rel_sample = float(np.abs(scores[sample] - journaled[sample]).max() / scale)
            rel_all = float(np.abs(scores - journaled).max() / scale)
            require(rel_sample <= 1e-4, f'{pool} score recomputation')
            gpu['score_recomputation'][pool] = dict(max_rel_diff_sample=rel_sample, max_rel_diff_all=rel_all,
                                                     identical_kept_set=stage.keep_top(scores).tolist() == row['kept'])
        rand, _, _ = stage.schedule(stage.POOLS.index('raw'), AUDIT_SEED)
        row = next(r for r in rows if (r['pool'], r['seed'], r['arm']) == ('raw', AUDIT_SEED, 'random'))
        replay = stage.train(s69.load(model_dir, AUDIT_SEED, device), pools['raw'].to(device), rand,
                             stage.DIGEST_STEP, set(), None, device)
        require(replay['adapter_digest_50'] == row['adapter_digest_50'], 'random replay to step 50')
        gpu['random_replay_step50'] = 'identical adapter digest'
    OUTPUT.write_text(json.dumps(dict(
        status='passed', stage=73, journal_sha256=s69.sha256(JOURNAL),
        result_sha256=hashlib.sha256(RESULT.read_bytes()).hexdigest(), checks=checks, device=cal['device'], gpu=gpu,
        notes='Journal, summary, budget, data hashes, decontamination, samples and pools, schedules, kept sets, '
              'removed shares and calibration agreement on CPU; alignment scores of seed 37300 recomputed and '
              'the first 50 random steps of seed 37300 (raw pool) replayed on the GPU. Full trajectories were '
              'not replayed.'), indent=2) + '\n')
    print(json.dumps(dict(checks=checks, gpu=gpu), indent=2))


if __name__ == '__main__':
    main()

"""Audit stage 77 (research/llm-deepseek-filter-protocol.json, audit field, with amendment 1).

CPU: journal digest, summary recomputation, environments, calibration (N_uncapped, N capped by amendment 2,
scoring batch), run grid and order,
schedules, update counts, checkpoints, kept sets from journaled scores, subset digests, QA shares, random-arm QA draws,
views per chunk, model weight hashes, DeepSeek and Qwen tokenizer hashes, QA chunks (1,992) and pool rebuilt with their
digests, web eligibility and decontamination counts (Qwen 8-grams, amendment 1), target eligibility and sample
recomputed (criteria (a)-(e) and counts), target disjoint from every exposed or trained ArabicMMLU ID.
GPU (same device type): recompute the self-scores of seed 37700 at the calibration's scoring batch; replay the first
50 steps of seed 37700 for random and align_once to their step-50 adapter digests.

python -m tools.audit_llm_deepseek_filter --journal J --kept K --removed R --mmlu M [--no-gpu]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import llm_deepseek_filter as stage
from experiments import llm_gpu as s69
from experiments import llm_qa_mix as s74
from experiments import llm_web_filter as s73
from tools.audit_pud_reuse import require

RESULT = Path('results/llm-deepseek-filter.json')
OUTPUT = Path('results/llm-deepseek-filter-audit.json')
STAGE74_JOURNAL = Path('results/llm-qa-mix-journal.jsonl')
AUDIT_SEED, SCORE_TOL = 37700, 1e-4
ENV_KEYS = ('device', 'torch', 'transformers', 'cuda', 'python')


def check_calibration(cal):
    require(cal['model'] == stage.MODEL and cal['revision'] == stage.REVISION, 'calibration model')
    require(cal['updates'] == stage.uncapped_updates(cal['costs']), 'N_uncapped = floor(B - S / c_r)')
    require(cal['updates'] >= stage.N_MIN, 'N_uncapped not below the minimum')
    n = stage.updates(cal['costs'])
    require(n == min(cal['updates'], stage.N_QA), 'N = min(N_uncapped, 1992) (amendment 2)')
    batch, oom = cal['scoring_batch'], cal['scoring_oom_batches']
    require(batch in stage.SCORE_BATCHES, 'scoring batch is a protocol batch')
    require(oom == list(stage.SCORE_BATCHES[:stage.SCORE_BATCHES.index(batch)]), 'scoring batch fallback order')
    return n, batch


def check_run(row, n, batch):
    tag = f"seed {row['seed']} {row['arm']}"
    sch = stage.schedule(row['seed'], n)
    require(row['model'] == stage.MODEL, f'{tag}: model')
    require(row['schedule_sha256'] == sch['sha256'], f'{tag}: schedule')
    require(len(row['step_seconds']) == stage.BUDGET, f'{tag}: update count')
    require(set(row['losses']) == {'0', str(n), str(stage.BUDGET)} == set(row['accuracy']), f'{tag}: checkpoints')
    require(isinstance(row['adapter_digest_50'], str) and len(row['adapter_digest_50']) == 64, f'{tag}: step-50 digest')
    kept = None
    if row['arm'] == 'align_once':
        require(len(row['scores']) == stage.POOL_SIZE, f'{tag}: score count')
        kept = s73.keep_top(row['scores'], n)
        require(kept.tolist() == row['kept'], f'{tag}: kept set from scores')
        require(hashlib.sha256(kept.tobytes()).hexdigest() == row['kept_sha256'], f'{tag}: kept digest')
        require(row['scoring_seconds'] > 0, f'{tag}: scoring time')
        require(row['scoring_batch'] == batch, f'{tag}: scoring batch equals the calibration')
    else:
        require(not {'kept', 'scores'} & set(row), f'{tag}: control carries filter fields')
    order, members = stage.training_order(row['arm'], sch, kept)
    if row['arm'] == 'random_once':
        require(row['subset_sha256'] == hashlib.sha256(np.asarray(members).tobytes()).hexdigest(), f'{tag}: subset')
    if row['arm'] == 'random':
        require(row['random_qa'] == {str(t): stage.random_qa(sch['rand'], t) for t in (n, stage.BUDGET)},
                f'{tag}: random QA draws')
    else:
        require(row['qa_share'] == stage.qa_share(members), f'{tag}: QA share')
        require(row['views'][str(n)]['max'] == 1, f'{tag}: one pass by N')
    require(row['views'] == {str(t): stage.views(order, members, t) for t in (n, stage.BUDGET)}, f'{tag}: views')
    return order


def exposed_ids(test):
    """Every ArabicMMLU test ID evaluated or trained before stage 77."""
    pilot, target74 = s74.item_splits(len(test))
    ids = {int(test[i]['ID']) for i in s73.sample_target(len(test))}
    ids |= {int(test[i]['ID']) for i in np.r_[pilot, target74]}
    cal74 = next(r for r in s69.read_journal(STAGE74_JOURNAL) if r.get('kind') == 'calibration')['population']
    require(set(cal74['pilot_items']) | set(cal74['target_items']) <= ids, 'stage-74 splits match the journal')
    for path in (stage.STAGE75_JOURNAL, stage.STAGE76_JOURNAL):
        ids |= set(next(r for r in s69.read_journal(path) if r.get('kind') == 'calibration')['population']['target_items'])
    ledger = json.loads(stage.s75.LEDGER.read_text())['additional_sources']['ArabicMMLU']['stage_74']
    return ids | set(ledger['used_for_training']['ids'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('--journal', '--kept', '--removed', '--mmlu'):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--no-gpu', action='store_true')
    args = parser.parse_args()
    plan = json.loads(stage.PROTOCOL.read_text())
    result = json.loads(RESULT.read_text())
    journal = s69.read_journal(args.journal)
    require(s69.sha256(args.journal) == result['journal_sha256'], 'journal digest')
    envs = [r for r in journal if r['kind'] == 'environment']
    require(len({tuple(e[k] for k in ENV_KEYS) for e in envs}) == 1, 'environments differ')
    cals = [r for r in journal if r['kind'] == 'calibration']
    require(len(cals) == 1, 'one calibration')
    cal = cals[0]
    n, batch = check_calibration(cal)
    rows = [r for r in journal if r['kind'] == 'run']
    require([(r['seed'], r['arm']) for r in rows] == stage.run_grid(plan['design']['seeds']), 'run grid and order')
    for r in rows:
        require(r['device'] == cal['device'], 'run device equals the calibration')
        check_run(r, n, batch)
    summary = stage.summarize(rows, cal, plan)
    require(json.loads(json.dumps(summary)) == result['summary'], 'summary recomputation')

    from transformers import AutoTokenizer
    model_dir = args.models_dir / stage.MODEL
    tok_sha = stage.check_model(model_dir)  # both weight files
    decon_sha = stage.check_decon_tokenizer(args.models_dir)
    tok = AutoTokenizer.from_pretrained(model_dir)
    qwen = AutoTokenizer.from_pretrained(args.models_dir / stage.QWEN_TOKENIZER)
    probe, target, pool, population = stage.load_inputs(args, tok, qwen)  # data hashes, QA chunks, (a)-(e)
    population.update(tokenizer_sha256=tok_sha, decontamination_tokenizer_sha256=decon_sha)
    require(population == cal['population'], 'population (QA chunks, pool, web and target eligibility, target IDs)')
    require(population['qa_chunks'] == stage.N_QA and population['qa_items_packed'] <= population['trained_ids'],
            'QA chunks')
    import pandas as pd
    test = pd.read_csv(args.mmlu / 'test.csv').to_dict('records')
    exposed = exposed_ids(test)
    target_ids = set(population['target_items'])
    require(len(target_ids) == stage.TARGET_ITEMS and not target_ids & exposed, 'target disjoint from exposed IDs')

    gpu = None
    if not args.no_gpu:
        import torch
        s69.setup_numerics()
        device = torch.device('cuda')
        require(torch.cuda.get_device_name(0) == cal['device'], 'audit device type differs from the run')
        data = pool.to(device)
        row = next(r for r in rows if (r['seed'], r['arm']) == (AUDIT_SEED, 'align_once'))
        model = stage.load(model_dir, AUDIT_SEED, device)
        scores, seconds = stage.score(model, data, probe, device, batch)
        del model
        torch.cuda.empty_cache()
        journaled = np.array(row['scores'])
        rel = float(np.abs(scores - journaled).max() / np.abs(journaled).max())
        require(rel <= SCORE_TOL, 'self-score recomputation')
        gpu = dict(score_recomputation=dict(max_rel_diff=rel, audit_scoring_seconds=seconds,
                                            identical_kept_set=s73.keep_top(scores, n).tolist() == row['kept']),
                   replay_step50={})
        for arm in ('random', 'align_once'):
            ref = next(r for r in rows if (r['seed'], r['arm']) == (AUDIT_SEED, arm))
            order = check_run(ref, n, batch)
            rep = s73.train(stage.load(model_dir, AUDIT_SEED, device), data, order, s73.DIGEST_STEP, set(), None,
                            device)
            same = rep['adapter_digest_50'] == ref['adapter_digest_50']
            require(same, f'{arm} step-50 replay')
            gpu['replay_step50'][arm] = dict(identical_adapter_digest=same)
            torch.cuda.empty_cache()
    out = dict(stage=77, status='passed', journal_sha256=result['journal_sha256'], runs=len(rows), updates=n,
               scoring_batch=batch, environments=len(envs), summary_recomputed=True,
               population=dict(qa_chunks=population['qa_chunks'], web_eligible=population['web_eligible'],
                               web_decontaminated=population['web_decontaminated']),
               target=dict(items=len(target_ids), disjoint_from_exposed=len(exposed), eligibility=population['target_pool']),
               step_seconds_median={arm: float(np.median([np.median(r['step_seconds']) for r in rows if r['arm'] == arm]))
                                    for arm in stage.ARMS},
               gpu=gpu)
    OUTPUT.write_text(json.dumps(out, indent=2) + '\n')
    print('audit passed', flush=True)


if __name__ == '__main__':
    main()

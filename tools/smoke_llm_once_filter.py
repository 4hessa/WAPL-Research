"""Stage 75 smoke test before the protocol: costs and data only; no target item is evaluated, no loss is printed.

1. Rebuilds the stage-74 population and checks that the sigma 0.25 and 0.1 pools equal the stage-74 journaled digests.
2. Counts the ArabicMMLU test items still unscored and untrained, and how many survive decontamination against
   everything stage 75 would train (pool chunks of both sigmas, the 4,946 stage-74 training items in their templated
   and field forms), with the candidate tokenized as evaluated per content field
   (alone and with a leading space; v1 also tested the evaluated prompt, see eval_forms).
3. On the GPU: median random step (steps 5-24 of 24), complete align_grad scoring of the 2048-chunk pool, and the
   wall-clock of one 512-item evaluation pass on already exposed items (the stage-74 target; values discarded).
   Implied N = floor(2400 - S / c_r), kept size K = N for one pass, and runs time.

python -m tools.smoke_llm_once_filter --kept K --removed R --mmlu M [--out results/llm-once-filter-smoke.json]
"""
import argparse
import json
from pathlib import Path
import platform
import time
from statistics import median

import numpy as np

from experiments import llm_gpu as s69
from experiments import llm_qa_mix as s74
from experiments import llm_web_filter as s73

SMOKE_SEED = 75998
STAGE74_JOURNAL = Path('results/llm-qa-mix-journal.jsonl')
LEDGER = Path('data/evaluation_status.json')


def eval_forms(r, tokenizer):
    """Token sequences of each content field of a candidate item (Question, Context, every option), alone and with a
    leading space. Template tokens are left out: v1 also tested the evaluated prompt, whose 8-grams spanning
    'سؤال: ' / 'الجواب:' and a few common words matched trained chunks without any shared content."""
    seqs = []
    fields = [r['Question']] + ([r['Context']] if s73._present(r.get('Context')) else [])
    fields += [r[f'Option {j}'] for j in range(1, 6) if s73._present(r.get(f'Option {j}'))]
    for f in fields:
        for text in (str(f).strip(), ' ' + str(f).strip()):
            seqs.append(tokenizer(text, add_special_tokens=False)['input_ids'])
    return seqs


def target_candidates(test, tokenizer, pools, trained_ids, excluded_rows):
    trained = [r for r in test if int(r['ID']) in trained_ids]
    cand_rows = [i for i, r in enumerate(test) if i not in excluded_rows and int(r['ID']) not in trained_ids]
    keep, counts = s74.near_duplicates([test[i] for i in cand_rows], trained, tokenizer)  # criteria (a)-(d)
    banned = set()
    for p in pools:
        for c in p.tolist():
            banned |= s73.ngrams(c)
    for r in trained:
        banned |= s73.ngrams(tokenizer(s74.mmlu_qa_text(r), add_special_tokens=False)['input_ids'])
    e_hits, survivors = 0, []
    for k in keep:
        if any(s73.ngrams(s) & banned for s in eval_forms(test[cand_rows[k]], tokenizer)):
            e_hits += 1
        else:
            survivors.append(cand_rows[k])
    counts['trained_tokenization_8gram_after_a_to_d'] = e_hits
    return dict(candidates=len(cand_rows), decontamination=counts, eligible=len(survivors)), survivors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('--kept', '--removed', '--mmlu'):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--out', type=Path, default=Path('results/llm-once-filter-smoke.json'))
    parser.add_argument('--no-gpu', action='store_true')
    args = parser.parse_args()
    import pandas as pd
    import torch
    import transformers
    from transformers import AutoTokenizer
    s69.setup_numerics()
    model_dir = args.models_dir / s74.MODEL
    spec = json.loads(s69.PROTOCOL.read_text())['models'][s74.MODEL]
    assert s69.sha256(model_dir / 'model.safetensors') == spec['safetensors_sha256'], 'model hash'
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    t0 = time.perf_counter()
    probe, _, old_target, pools, population, _ = s74.load_inputs(args, tokenizer)  # checks the four data hashes
    cal74 = next(r for r in s69.read_journal(STAGE74_JOURNAL) if r.get('kind') == 'calibration')
    same = {s: population['pool_sha256'][s] == cal74['population']['pool_sha256'][s] for s in ('0.25', '0.1')}
    out = dict(kind='smoke', stage=75, time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               pools_equal_stage74=same, population_equal_stage74=population == cal74['population'])
    test = pd.read_csv(args.mmlu / 'test.csv').to_dict('records')
    ledger = json.loads(LEDGER.read_text())['additional_sources']['ArabicMMLU']['stage_74']
    trained_ids = set(ledger['used_for_training']['ids'])
    pilot_rows, target_rows = s74.item_splits(len(test))
    excluded = set(s73.sample_target(len(test)).tolist()) | set(pilot_rows.tolist()) | set(target_rows.tolist())
    data, _ = target_candidates(test, tokenizer, [pools[0.25], pools[0.1]], trained_ids, excluded)
    out.update(trained_ids=len(trained_ids), excluded_evaluated_rows=len(excluded), target_pool=data,
               data_seconds=time.perf_counter() - t0)
    print(json.dumps(out, indent=1), flush=True)
    if not args.no_gpu:
        device = torch.device('cuda')
        rand, _, _, _ = s74.schedule(0, SMOKE_SEED)
        row = s73.train(s69.load(model_dir, SMOKE_SEED, device), pools[0.25].to(device), rand, 24, set(), None, device)
        c_r = median(row['step_seconds'][4:])
        torch.cuda.empty_cache()
        model = s69.load(model_dir, SMOKE_SEED, device)
        _, S = s73.score(model, 'align_grad', pools[0.25].to(device), probe, device)
        s69.sync(device)
        start = time.perf_counter()
        s73.evaluate(model, old_target, device)  # exposed stage-74 items; the values are discarded unread
        s69.sync(device)
        ev = time.perf_counter() - start
        n = int(np.floor(s74.BUDGET - S / c_r + 1e-9))
        run = s74.BUDGET * c_r + 3 * ev
        out.update(device=torch.cuda.get_device_name(0), torch=torch.__version__, transformers=transformers.__version__,
                   python=platform.python_version(), random_step_seconds=c_r, scoring_seconds=S,
                   scoring_in_random_steps=S / c_r, implied_N=n, kept_once=n, kept_fraction_once=n / s74.POOL_SIZE,
                   eval_seconds_512=ev, est_run_seconds=run, est_hours_60_runs=(60 * run + 12 * S) / 3600)
        print(json.dumps({k: v for k, v in out.items() if k not in ('target_pool',)}, indent=1), flush=True)
    args.out.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()

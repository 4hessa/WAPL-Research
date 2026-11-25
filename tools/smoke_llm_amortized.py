"""Stage 76 smoke test before the protocol: costs and data only; no target item is evaluated, no loss is printed.

Question to be designed: the stage-75 arms (random, align_once, random_once) on the stage-74 sigma 0.25 pool with a
longer budget B = 4 x 2400, so that one-time scoring is a small share of B, and a kept size giving about one view
per kept chunk.

1. Rebuilds the stage-74 population and checks the sigma 0.25 and 0.1 pools against the stage-74 journaled digests.
2. Counts the ArabicMMLU test items that are unscored and untrained after stage 75 (stage-75 target excluded too) and
   clean under criteria (a)-(e) of stage 75 (the trained tokenization included), exactly as
   experiments.llm_once_filter.target_candidates.
3. On the GPU: median random step (steps 5-24 of 24), complete align_grad scoring of the 2048-chunk pool, and the
   wall-clock of one 512-item evaluation pass on already exposed items (the stage-75 target; values discarded).
4. Feasibility of the requested design: N = floor(B - S / c_r) for B in {2400, 9600}; a one-pass kept set needs
   K = N <= 2048. Run-time estimates for the candidate grids.

python -m tools.smoke_llm_amortized --kept K --removed R --mmlu M [--out results/llm-amortized-smoke.json]
"""
import argparse
import json
from pathlib import Path
import platform
import time
from statistics import median

import numpy as np

from experiments import llm_gpu as s69
from experiments import llm_once_filter as s75
from experiments import llm_qa_mix as s74
from experiments import llm_web_filter as s73

SMOKE_SEED = 76998
STAGE75_JOURNAL = Path('results/llm-once-filter-journal.jsonl')
BUDGETS = (2400, 9600)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('--kept', '--removed', '--mmlu'):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--out', type=Path, default=Path('results/llm-amortized-smoke.json'))
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
    probe, _, _, pools, population, _ = s74.load_inputs(args, tokenizer)  # checks the four data-file hashes
    same = {s: population['pool_sha256'][s] == s75.STAGE74_POOL_SHA256[s] for s in ('0.25', '0.1')}
    out = dict(kind='smoke', stage=76, time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               pools_equal_stage74=same)
    test = pd.read_csv(args.mmlu / 'test.csv').to_dict('records')
    ledger = json.loads(s75.LEDGER.read_text())['additional_sources']['ArabicMMLU']['stage_74']
    trained_ids = set(ledger['used_for_training']['ids'])
    cal75 = next(r for r in s69.read_journal(STAGE75_JOURNAL) if r.get('kind') == 'calibration')
    target75_ids = set(cal75['population']['target_items'])
    pilot_rows, target74_rows = s74.item_splits(len(test))
    excluded = set(s73.sample_target(len(test)).tolist()) | set(pilot_rows.tolist()) | set(target74_rows.tolist())
    excluded |= {i for i, r in enumerate(test) if int(r['ID']) in target75_ids}
    data, _ = s75.target_candidates(test, tokenizer, [pools[0.25], pools[0.1]], trained_ids, excluded)
    out.update(trained_ids=len(trained_ids), excluded_evaluated_rows=len(excluded), target_pool=data,
               data_seconds=time.perf_counter() - t0)
    print(json.dumps(out, indent=1), flush=True)
    device = torch.device('cuda')
    pool = pools[0.25].to(device)
    rand = np.random.default_rng(SMOKE_SEED).integers(s74.POOL_SIZE, size=24)
    row = s73.train(s69.load(model_dir, SMOKE_SEED, device), pool, rand, 24, set(), None, device)
    c_r = median(row['step_seconds'][4:])
    torch.cuda.empty_cache()
    model = s69.load(model_dir, SMOKE_SEED, device)
    _, S = s73.score(model, 'align_grad', pool, probe, device)
    target75 = s73.build_items([r for r in test if int(r['ID']) in target75_ids], tokenizer)
    s69.sync(device)
    start = time.perf_counter()
    s73.evaluate(model, target75, device)  # exposed stage-75 items; the values are discarded unread
    s69.sync(device)
    ev = time.perf_counter() - start
    feas = {}
    for B in BUDGETS:
        n = int(np.floor(B - S / c_r + 1e-9))
        feas[str(B)] = dict(N=n, scoring_share=S / c_r / B, one_pass_feasible=n <= s74.POOL_SIZE,
                            views_if_whole_pool_kept=n / s74.POOL_SIZE,
                            train_seconds=B * c_r, run_seconds=B * c_r + 3 * ev)
    per_chunk = S / c_r / s74.POOL_SIZE  # scoring cost of one chunk in random steps
    out.update(device=torch.cuda.get_device_name(0), torch=torch.__version__, transformers=transformers.__version__,
               python=platform.python_version(), random_step_seconds=c_r, scoring_seconds=S,
               scoring_in_random_steps=S / c_r, scoring_steps_per_chunk=per_chunk, eval_seconds_512=ev,
               budgets=feas,
               min_scoring_share_one_view={str(f): per_chunk / (per_chunk + f) for f in (0.25, 0.5, 0.72, 1.0)})
    print(json.dumps({k: v for k, v in out.items() if k != 'target_pool'}, indent=1), flush=True)
    args.out.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()

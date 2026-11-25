"""Stage 76 resumption check on a new machine, run before the main loop restarts.

Recomputes both model hashes, the four data-file hashes (inside load_inputs), the population (QA chunks,
web draws and decontamination counts, pool digest, target eligibility and target items) and, for every
finished run, the schedule digest, kept set (recomputed from the journaled scores), subset digests, QA shares
and views per chunk, and compares them with the journaled calibration and runs. Prints PASS/FAIL lines only;
no loss, accuracy or score is printed.

python -m tools.verify_llm_cheap_filter_resume --journal J --kept K --removed R --mmlu M --device "NVIDIA L4"
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from experiments import llm_cheap_filter as s76
from experiments import llm_gpu as s69
from experiments import llm_web_filter as s73


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('--journal', '--kept', '--removed', '--mmlu'):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--device', required=True, help='Device name of this machine (torch.cuda.get_device_name)')
    args = parser.parse_args()
    failures = []

    def check(name, ok):
        print(('PASS ' if ok else 'FAIL ') + name, flush=True)
        if not ok:
            failures.append(name)

    specs = json.loads(s69.PROTOCOL.read_text())['models']
    for name in (s76.MODEL, s76.PROXY):
        check(f'{name} model.safetensors SHA-256',
              s69.sha256(args.models_dir / name / 'model.safetensors') == specs[name]['safetensors_sha256'])
    for f in ('tokenizer.json', 'vocab.json', 'merges.txt'):
        check(f'proxy {f} equals model {f}',
              s69.sha256(args.models_dir / s76.PROXY / f) == s69.sha256(args.models_dir / s76.MODEL / f))
    journal = s69.read_journal(args.journal)
    calibration = [r for r in journal if r.get('kind') == 'calibration']
    check('exactly one calibration entry', len(calibration) == 1)
    calibration = calibration[0]
    check(f"calibration device {calibration['device']} equals this device {args.device}",
          calibration['device'] == args.device)
    n = calibration['updates']
    check(f'N = {n} equals floor(B - S / c_r)', n == int(np.floor(
        s76.BUDGET - calibration['costs']['proxy_align_grad'] / calibration['costs']['random'] + 1e-9)))
    check('N within the protocol range', s76.N_RANGE[0] <= n <= s76.N_RANGE[1])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.models_dir / s76.MODEL)
    _, target, pool, population = s76.load_inputs(args, tokenizer)  # raises on any data-file hash mismatch
    check('data file hashes (kept, removed, dev, test) and QA pool digest', True)
    for key in sorted(set(population) | set(calibration['population'])):
        check(f'population.{key} equals calibration', population.get(key) == calibration['population'].get(key))
    runs = [r for r in journal if r.get('kind') == 'run']
    keys = [(r['seed'], r['arm']) for r in runs]
    grid = s76.run_grid(json.loads(s76.PROTOCOL.read_text())['design']['seeds'])
    check(f'{len(runs)} finished runs are unique', len(set(keys)) == len(keys))
    check('finished runs are the protocol order prefix (no selective rerun)', keys == grid[:len(keys)])
    for r in runs:
        sch = s76.schedule(r['seed'], n)
        ok = (r['schedule_sha256'] == sch['sha256'] and r['device'] == calibration['device']
              and r['model'] == s76.MODEL)
        kept = None
        if r['arm'] == 'align05_once':
            kept = s73.keep_top(r['scores'], n)
            ok &= len(r['scores']) == s76.POOL_SIZE and kept.tolist() == r['kept']
            ok &= hashlib.sha256(kept.tobytes()).hexdigest() == r['kept_sha256']
        order, members = s76.training_order(r['arm'], sch, kept)
        if r['arm'] in ('random_once', 'qa_once'):
            ok &= r['subset_sha256'] == hashlib.sha256(np.asarray(members).tobytes()).hexdigest()
        if r['arm'] == 'random':
            ok &= r['random_qa'] == {str(t): s76.random_qa(sch['rand'], t) for t in (n, s76.BUDGET)}
        else:
            ok &= r['qa_share'] == s76.qa_share(members)
        ok &= r['views'] == {str(t): s76.views(order, members, t) for t in (n, s76.BUDGET)}
        ok &= sorted(r['losses']) == sorted(['0', str(n), str(s76.BUDGET)])  # keys only, values not read
        check(f"run seed={r['seed']} arm={r['arm']}: schedule, kept/subset digests, QA share, views, checkpoints",
              bool(ok))
    remaining = grid[len(keys):]
    print(f'finished {len(runs)}, remaining {len(remaining)}; next {remaining[0] if remaining else None}')
    if failures:
        sys.exit(f'{len(failures)} check(s) failed: resumption refused')
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()

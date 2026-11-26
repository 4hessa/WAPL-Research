"""Stage 77 resumption check on a new machine, run before the main loop restarts.

Recomputes both weight-file hashes and the Qwen decontamination tokenizer hash (amendment 1), the four data-file
hashes (inside load_inputs), the population (QA chunks, web draws and decontamination counts, pool digest, target
eligibility and target items, tokenizer hashes) and compares it with the journaled calibration; checks the device
type and that N_uncapped equals floor(B - S / c_r) and N = min(N_uncapped, 1992) (amendment 2); and, for every
finished run, uniqueness, protocol order, schedule digest, align_once kept set (recomputed from the journaled
scores) and its digest, random_once subset digest, QA shares, random-arm QA draws, views per chunk and checkpoint
keys. Prints PASS/FAIL lines only; no loss, accuracy or score is printed.

python -m tools.verify_llm_deepseek_filter_resume --journal J --kept K --removed R --mmlu M --device "NVIDIA L4"
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from experiments import llm_deepseek_filter as s77
from experiments import llm_gpu as s69
from experiments import llm_web_filter as s73


def check_runs(runs, n, grid, device, check):
    """Journal-only checks of the finished runs (no data or GPU needed). Returns the remaining grid."""
    keys = [(r['seed'], r['arm']) for r in runs]
    check(f'{len(runs)} finished runs are unique', len(set(keys)) == len(keys))
    check('finished runs are the protocol order prefix (no selective rerun)', keys == grid[:len(keys)])
    for r in runs:
        sch = s77.schedule(r['seed'], n)
        ok = r['schedule_sha256'] == sch['sha256'] and r['device'] == device and r['model'] == s77.MODEL
        kept = None
        if r['arm'] == 'align_once':
            kept = s73.keep_top(r['scores'], n)
            ok &= len(r['scores']) == s77.POOL_SIZE and kept.tolist() == r['kept']
            ok &= hashlib.sha256(kept.tobytes()).hexdigest() == r['kept_sha256']
        order, members = s77.training_order(r['arm'], sch, kept)
        if r['arm'] == 'random_once':
            ok &= r['subset_sha256'] == hashlib.sha256(np.asarray(members).tobytes()).hexdigest()
        if r['arm'] == 'random':
            ok &= r['random_qa'] == {str(t): s77.random_qa(sch['rand'], t) for t in (n, s77.BUDGET)}
        else:
            ok &= r['qa_share'] == s77.qa_share(members)
        ok &= r['views'] == {str(t): s77.views(order, members, t) for t in (n, s77.BUDGET)}
        expected = sorted(['0', str(n), str(s77.BUDGET)])
        ok &= sorted(r['losses']) == expected and sorted(r['accuracy']) == expected  # keys only, values not read
        check(f"run seed={r['seed']} arm={r['arm']}: schedule, kept/subset digests, QA share, views, checkpoints",
              bool(ok))
    return grid[len(keys):]


def check_calibration(journal, device, check):
    calibration = [r for r in journal if r.get('kind') == 'calibration']
    check('exactly one calibration entry', len(calibration) == 1)
    calibration = calibration[0]
    check(f"calibration device {calibration['device']} equals this device {device}", calibration['device'] == device)
    check(f"calibration model {calibration['model']} at revision {calibration['revision']}",
          calibration['model'] == s77.MODEL and calibration['revision'] == s77.REVISION)
    check(f"N_uncapped = {calibration['updates']} equals floor(B - S / c_r)",
          calibration['updates'] == s77.uncapped_updates(calibration['costs']))
    check(f'N_uncapped >= {s77.N_MIN}', calibration['updates'] >= s77.N_MIN)
    n = s77.updates(calibration['costs'])
    check(f'N = min(N_uncapped, {s77.N_QA}) = {n} (amendment 2)', n == min(calibration['updates'], s77.N_QA))
    return calibration, n


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

    model_dir = args.models_dir / s77.MODEL
    for name, digest in s77.WEIGHTS_SHA256.items():
        check(f'{name} SHA-256', s69.sha256(model_dir / name) == digest)
    check('Qwen decontamination tokenizer.json SHA-256 (amendment 1)',
          s69.sha256(args.models_dir / s77.QWEN_TOKENIZER / 'tokenizer.json') == s77.QWEN_TOKENIZER_SHA256)
    journal = s69.read_journal(args.journal)
    calibration, n = check_calibration(journal, args.device, check)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    decon = AutoTokenizer.from_pretrained(args.models_dir / s77.QWEN_TOKENIZER)
    _, _, _, population = s77.load_inputs(args, tokenizer, decon)  # raises on any data-file hash mismatch
    population.update(tokenizer_sha256=s69.sha256(model_dir / 'tokenizer.json'),
                      decontamination_tokenizer_sha256=s77.QWEN_TOKENIZER_SHA256)
    check('data file hashes (kept, removed, dev, test)', True)
    for key in sorted(set(population) | set(calibration['population'])):
        check(f'population.{key} equals calibration', population.get(key) == calibration['population'].get(key))
    runs = [r for r in journal if r.get('kind') == 'run']
    grid = s77.run_grid(json.loads(s77.PROTOCOL.read_text())['design']['seeds'])
    remaining = check_runs(runs, n, grid, calibration['device'], check)
    print(f'finished {len(runs)}, remaining {len(remaining)}; next {remaining[0] if remaining else None}')
    if failures:
        sys.exit(f'{len(failures)} check(s) failed: resumption refused')
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()

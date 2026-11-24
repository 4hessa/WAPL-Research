"""Stage 74 resumption check on a new machine, run before the main loop restarts.

Recomputes the model and data hashes, the population (item samples, decontamination counts, pool digests,
QA item IDs, web document indices) and the schedule digests, and compares them with the journaled
calibration and finished runs. Prints PASS/FAIL lines only; no loss, accuracy or score is printed.

python -m tools.verify_llm_qa_mix_resume --journal J --pilot-journal P --kept K --removed R --mmlu M
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from experiments import llm_gpu as s69
from experiments import llm_qa_mix as s74
from experiments import llm_web_filter as s73

EXPECTED_ORDER = [(sigma, seed, arm) for k, sigma in enumerate(s74.SIGMAS)
                  for i, seed in enumerate(json.loads(s74.PROTOCOL.read_text())['phase_B_main']['seeds'])
                  for arm in s74.ARMS[i % 3:] + s74.ARMS[:i % 3]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('--journal', '--pilot-journal', '--kept', '--removed', '--mmlu'):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--device', required=True, help='Device name of this machine (torch.cuda.get_device_name)')
    args = parser.parse_args()
    failures = []

    def check(name, ok):
        print(('PASS ' if ok else 'FAIL ') + name, flush=True)
        if not ok:
            failures.append(name)

    spec = json.loads(s69.PROTOCOL.read_text())['models'][s74.MODEL]
    model_dir = args.models_dir / s74.MODEL
    check('model.safetensors SHA-256', s69.sha256(model_dir / 'model.safetensors') == spec['safetensors_sha256'])
    journal = s69.read_journal(args.journal)
    pilot = s69.read_journal(args.pilot_journal)
    calibration = [r for r in journal if r.get('kind') == 'calibration']
    check('exactly one calibration entry', len(calibration) == 1)
    calibration = calibration[0]
    check(f"calibration device {calibration['device']} equals this device {args.device}", calibration['device'] == args.device)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    _, _, _, pools, population, _ = s74.load_inputs(args, tokenizer)  # also checks the four data-file hashes
    check('data file hashes (kept, removed, dev, test)', True)
    for key in sorted(set(population) | set(calibration['population'])):
        check(f'population.{key} equals calibration', population.get(key) == calibration['population'].get(key))
    pilot_runs = [r for r in pilot if r.get('kind') == 'pilot']
    check('pilot journal complete on the same population',
          len(pilot_runs) == len(s74.PILOT_SIGMAS) and all(r['population'] == population for r in pilot_runs))
    runs = [r for r in journal if r.get('kind') == 'run']
    keys = [(r['sigma'], r['seed'], r['arm']) for r in runs]
    check(f'{len(runs)} finished runs are unique', len(set(keys)) == len(keys))
    check('finished runs are the protocol order prefix (no selective rerun)', keys == EXPECTED_ORDER[:len(keys)])
    for r in runs:
        k = s74.SIGMAS.index(r['sigma'])
        _, _, subset, digest = s74.schedule(k, r['seed'])
        tag = f"run sigma={r['sigma']} seed={r['seed']} arm={r['arm']}"
        ok = r['schedule_sha256'] == digest and r['device'] == calibration['device'] and r['model'] == s74.MODEL
        if r['arm'] == 'random512':
            ok &= r['subset_sha256'] == hashlib.sha256(subset.tobytes()).hexdigest()
            ok &= r['qa_share'] == s74.qa_share(subset, r['sigma'])
        if r['arm'] == 'align_grad':
            kept = s73.keep_top(r['scores'], s74.KEEP)
            ok &= kept.tolist() == r['kept']
            ok &= hashlib.sha256(np.asarray(r['kept'], dtype=kept.dtype).tobytes()).hexdigest() == r['kept_sha256']
        check(f'{tag}: schedule digest, device, subset/kept digests', bool(ok))
    remaining = EXPECTED_ORDER[len(keys):]
    print(f'finished {len(runs)}, remaining {len(remaining)}; next {remaining[0] if remaining else None}')
    if failures:
        sys.exit(f'{len(failures)} check(s) failed: resumption refused')
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()

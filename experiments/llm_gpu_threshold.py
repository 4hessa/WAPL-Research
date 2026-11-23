"""Stage 71: contamination threshold for cost-charged alignment on Qwen2.5-1.5B (Colab GPU).

Implements research/llm-gpu-threshold-protocol.json with the stage-69/70 step loop.
"""
import argparse
import json
import os
from pathlib import Path
import platform
from statistics import mean, median
import time

import numpy as np
import torch

from experiments import llm_common as llm
from experiments import llm_gpu as s69
from experiments import llm_gpu_indomain as s70

PROTOCOL = Path('research/llm-gpu-threshold-protocol.json')
MODEL = 'qwen2.5-1.5b'
ARMS = ('random', 'graddot4_batched')
CAL_SEED = 36999


def split(eligible, chunks):
    """Exclude stage-69 and stage-70 probe/target rows, then draw the stage-71 probe and target."""
    used69 = s70.stage69_rows(len(eligible))
    rem69 = [r for r in range(len(eligible)) if r not in used69]
    p70 = np.random.default_rng(7001).choice(len(rem69), 16, replace=False)
    rest70 = np.setdiff1d(np.arange(len(rem69)), p70)
    t70 = rest70[np.random.default_rng(7002).choice(len(rest70), 256, replace=False)]
    used70 = {rem69[i] for i in np.concatenate([p70, t70])}
    remaining = [r for r in rem69 if r not in used70]
    probe = np.sort(np.random.default_rng(7101).choice(len(remaining), 16, replace=False))
    rest = np.setdiff1d(np.arange(len(remaining)), probe)
    target = np.sort(rest[np.random.default_rng(7102).choice(len(rest), 256, replace=False)])
    candidates = np.setdiff1d(rest, target)
    rows = lambda idx: [remaining[i] for i in idx]
    tensor = lambda idx: torch.tensor([chunks[r] for r in rows(idx)], dtype=torch.long)
    population = dict(eligible_articles=len(eligible), excluded_stages_69_70=len(eligible) - len(remaining),
                      probe_articles=[eligible[r] for r in rows(probe)],
                      target_articles=[eligible[r] for r in rows(target)], candidate_articles=len(candidates))
    return tensor(probe), tensor(target), tensor(candidates), population


def crossover(rhos, d):
    """First positive-to-non-positive change scanning rho upward; linear interpolation of the zero."""
    for (r0, d0), (r1, d1) in zip(zip(rhos, d), zip(rhos[1:], d[1:])):
        if d0 > 0 and d1 <= 0:
            return r0 + (r1 - r0) * d0 / (d0 - d1)
    return None


def summarize(rows, calibration, plan):
    n = calibration['updates']
    rhos, seeds = plan['rho'], plan['seeds']
    final, k = str(s69.BUDGET), str(n['graddot4_batched'])
    diffs, per_rho = [], {}
    for rho in rhos:
        group = {arm: sorted([r for r in rows if (r['rho'], r['arm']) == (rho, arm)], key=lambda r: r['seed'])
                 for arm in ARMS}
        for arm in ARMS:
            if [r['seed'] for r in group[arm]] != seeds:
                raise ValueError(f'Incomplete grid for rho {rho} {arm}')
        rand, align = group['random'], group['graddot4_batched']
        if [r['losses']['0'] for r in rand] != [r['losses']['0'] for r in align]:
            raise ValueError('Arms do not share the initial state')
        d = [a['losses'][k] - r['losses'][final] for a, r in zip(align, rand)]
        sel = [a['losses'][k] - r['losses'][k] for a, r in zip(align, rand)]
        diffs.append(d)
        per_rho[str(rho)] = dict(minus_random=mean(d), ci95=s70.bootstrap(d), per_seed=d,
                                 seeds_negative=sum(x < 0 for x in d), selection_term=mean(sel),
                                 throughput_term=mean(d) - mean(sel),
                                 random_minus_no_update=mean(r['losses'][final] - r['losses']['0'] for r in rand),
                                 align_english_share=mean(a['english_share'] for a in align),
                                 random_english_share=mean(r['english_share'] for r in rand))
    diffs = np.array(diffs)  # rho x seed
    estimate = crossover(rhos, diffs.mean(axis=1).tolist())
    rng = np.random.default_rng(7103)
    draws = []
    for _ in range(10000):
        idx = rng.integers(len(seeds), size=len(seeds))
        draws.append(crossover(rhos, diffs[:, idx].mean(axis=1).tolist()))
    found = [x for x in draws if x is not None]
    interval = [float(x) for x in np.quantile(found, [.025, .975])] if found else None
    predicted = {'0.0': 1, '0.5': -1, '0.75': -1, '0.9': -1}
    agreement = {r: (np.sign(per_rho[r]['minus_random']) == s) for r, s in predicted.items()}
    return dict(primary=dict(crossover_rho=estimate, ci95=interval, resamples_with_crossing=len(found)),
                prediction_signs_agree={r: bool(v) for r, v in agreement.items()},
                prediction_crossover_in_0_0p5=bool(estimate is not None and 0 <= estimate <= .5),
                costs=calibration['costs'], updates=n, device=calibration['device'],
                r=calibration['costs']['graddot4_batched'] / calibration['costs']['random'], per_rho=per_rho)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--wiki', type=Path, required=True)
    parser.add_argument('--summary', type=Path, default=Path('results/llm-gpu-threshold.json'))
    args = parser.parse_args()
    plan = json.loads(PROTOCOL.read_text())
    spec = json.loads(s69.PROTOCOL.read_text())['models'][MODEL]
    if plan['stage'] != 71 or tuple(plan['arms']) != ARMS:
        raise ValueError('Protocol differs from the implementation')
    s69.setup_numerics()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'
    from transformers import AutoTokenizer
    import transformers
    import pandas as pd
    if s69.sha256(args.wiki) != s70.WIKI_SHA256:
        raise ValueError('Wikipedia shard differs from the protocol')
    model_dir = args.models_dir / MODEL
    if s69.sha256(model_dir / 'model.safetensors') != spec['safetensors_sha256']:
        raise ValueError('Model weights differ from the protocol')
    en_entry = json.loads(s69.MANIFEST.read_text())['languages']['en']['splits']['train']
    if s69.sha256(en_entry['path']) != en_entry['sha256']:
        raise ValueError('English train checksum mismatch')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    texts = pd.read_parquet(args.wiki, columns=['text'])['text'].tolist()
    eligible, chunks = s70.eligible_chunks(texts, tokenizer)
    probe, target, ar, population = split(eligible, chunks)
    en = llm.token_chunks(en_entry['path'], tokenizer, 65)
    pool = torch.cat([ar, en]).to(device)
    probe, target = probe.to(device), target.to(device)
    journal = s69.read_journal(args.journal)
    done = {(r['rho'], r['seed'], r['arm']) for r in journal if r.get('kind') == 'run'}
    calibration = next((r for r in journal if r.get('kind') == 'calibration'), None)
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    with args.journal.open('a') as out:
        def log(row):
            out.write(json.dumps(row, allow_nan=False) + '\n')
            out.flush()
            os.fsync(out.fileno())
        log(dict(kind='environment', time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), torch=torch.__version__,
                 transformers=transformers.__version__, cuda=torch.version.cuda, python=platform.python_version(),
                 device=device_name, resumed_runs=len(done)))
        if calibration is None:
            ids, choices, _ = s69.schedule(0, CAL_SEED, 0.5, len(ar), len(en))
            costs = {}
            for arm in ARMS:
                row = s69.run(s69.load(model_dir, CAL_SEED, device), arm, pool, probe, None, ids, choices,
                              s69.CAL_STEPS, [], device)
                costs[arm] = median(row['step_seconds'][4:])
            updates = dict(random=s69.BUDGET,
                           graddot4_batched=int(s69.BUDGET * costs['random'] / costs['graddot4_batched'] + 1e-9))
            calibration = dict(kind='calibration', model=MODEL, device=device_name, costs=costs, updates=updates,
                               population=population)
            log(calibration)
        else:
            if calibration['device'] != device_name:
                raise ValueError(f'Resumed on {device_name}; calibrated on {calibration["device"]}')
            if calibration['population'] != population:
                raise ValueError('Evaluation sample differs from the journaled calibration')
        n = calibration['updates']
        print(MODEL, device_name, 'costs', calibration['costs'], 'updates', n, flush=True)
        for k, rho in enumerate(plan['rho']):
            for i, seed in enumerate(plan['seeds']):
                ids, choices, digest = s69.schedule(k, seed, rho, len(ar), len(en))
                for arm in ARMS[i % 2:] + ARMS[:i % 2]:
                    if (rho, seed, arm) in done:
                        continue
                    checkpoints = sorted(set(n.values())) if arm == 'random' else [n[arm]]
                    row = s69.run(s69.load(model_dir, seed, device), arm, pool, probe, target, ids, choices,
                                  n[arm], checkpoints, device)
                    row.update(kind='run', model=MODEL, rho=rho, seed=seed, arm=arm, updates=n[arm],
                               schedule_sha256=digest, device=device_name,
                               english_share=mean(float(ids[t][c] >= len(ar)) for t, c in enumerate(row['choices'])))
                    log(row)
                    print(f'rho {rho} seed {seed} {arm}: {row["losses"][str(n[arm])]:.5f}', flush=True)
    journal = s69.read_journal(args.journal)
    rows = [r for r in journal if r.get('kind') == 'run']
    result = dict(stage=71, protocol=plan, journal_sha256=s69.sha256(args.journal),
                  environments=[r for r in journal if r.get('kind') == 'environment'],
                  calibration=calibration, summary=summarize(rows, calibration, plan), runs=len(rows))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['summary']['primary'], indent=2))


if __name__ == '__main__':
    main()

"""Stage 70: in-domain candidates and a no-update baseline on GPU (run in Colab).

Implements research/llm-gpu-indomain-protocol.json. Decision rules, numerics and
the step loop are those of stage 69 (experiments.llm_gpu.run).
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

PROTOCOL = Path('research/llm-gpu-indomain-protocol.json')
ARMS = ('random', 'loss4', 'graddot4_batched')
GUIDED = ARMS[1:]
CAL_SEED = 35999
WIKI_SHA256 = 'f035e9118745d465529d48b0b1893824a4f266af2474694dcd35823e767aad17'


def eligible_chunks(texts, tokenizer):
    eligible, chunks = [], []
    for i, text in enumerate(texts):
        ids = tokenizer(text, add_special_tokens=False)['input_ids'] if text else []
        if len(ids) >= 65:
            eligible.append(i)
            chunks.append(ids[:65])
    return eligible, chunks


def stage69_rows(n):
    """Eligible-list rows used as stage-69 probe and target (same draws as llm_gpu.select_articles)."""
    probe = np.random.default_rng(6901).choice(n, 16, replace=False)
    rest = np.setdiff1d(np.arange(n), probe)
    target = rest[np.random.default_rng(6902).choice(len(rest), 256, replace=False)]
    return set(probe.tolist()) | set(target.tolist())


def split(eligible, chunks):
    remaining = [r for r in range(len(eligible)) if r not in stage69_rows(len(eligible))]
    probe = np.sort(np.random.default_rng(7001).choice(len(remaining), 16, replace=False))
    rest = np.setdiff1d(np.arange(len(remaining)), probe)
    target = np.sort(rest[np.random.default_rng(7002).choice(len(rest), 256, replace=False)])
    candidates = np.setdiff1d(rest, target)
    rows = lambda idx: [remaining[i] for i in idx]
    tensor = lambda idx: torch.tensor([chunks[r] for r in rows(idx)], dtype=torch.long)
    population = dict(eligible_articles=len(eligible), excluded_stage69=len(eligible) - len(remaining),
                      probe_articles=[eligible[r] for r in rows(probe)],
                      target_articles=[eligible[r] for r in rows(target)],
                      candidate_articles=len(candidates))
    return tensor(probe), tensor(target), tensor(candidates), population


def bootstrap(values):
    draws = np.random.default_rng(7003).choice(values, (10000, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [.025, .975])]


def summarize(rows, calibrations, plan):
    out = {}
    for name in ('qwen2.5-0.5b', 'qwen2.5-1.5b'):
        n = calibrations[name]['updates']
        per_rho = {}
        for rho in plan['rho']:
            group = {arm: sorted([r for r in rows if (r['model'], r['rho'], r['arm']) == (name, rho, arm)],
                                 key=lambda r: r['seed']) for arm in ARMS}
            for arm in ARMS:
                if [r['seed'] for r in group[arm]] != plan['seeds']:
                    raise ValueError(f'Incomplete grid for {name} rho {rho} {arm}')
            rand = group['random']
            base = [r['losses']['0'] for r in rand]
            for arm in GUIDED:
                if [r['losses']['0'] for r in group[arm]] != base:
                    raise ValueError('Arms do not share the initial state')
            final = str(s69.BUDGET)
            vs_none = [r['losses'][final] - b for r, b in zip(rand, base)]
            entry = dict(no_update=mean(base), random_final=mean(r['losses'][final] for r in rand),
                         random_minus_no_update=dict(mean=mean(vs_none), ci95=bootstrap(vs_none)),
                         random_english_share=mean(r['english_share'] for r in rand), arms={})
            for arm in GUIDED:
                k = str(n[arm])
                diff = [g['losses'][k] - r['losses'][final] for g, r in zip(group[arm], rand)]
                sel = [g['losses'][k] - r['losses'][k] for g, r in zip(group[arm], rand)]
                none = [g['losses'][k] - b for g, b in zip(group[arm], base)]
                entry['arms'][arm] = dict(updates=n[arm], final=mean(g['losses'][k] for g in group[arm]),
                                          minus_random=mean(diff), ci95=bootstrap(diff), per_seed=diff,
                                          seeds_negative=sum(d < 0 for d in diff), selection_term=mean(sel),
                                          throughput_term=mean(diff) - mean(sel),
                                          minus_no_update=dict(mean=mean(none), ci95=bootstrap(none)),
                                          english_share=mean(g['english_share'] for g in group[arm]))
            per_rho[str(rho)] = entry
        out[name] = dict(costs=calibrations[name]['costs'], updates=n, device=calibrations[name]['device'],
                         r={a: calibrations[name]['costs'][a] / calibrations[name]['costs']['random'] for a in ARMS},
                         per_rho=per_rho)
    p = out['qwen2.5-1.5b']['per_rho']['0.0']['arms']['graddot4_batched']
    return dict(primary=dict(model='qwen2.5-1.5b', rho=0.0, contrast='graddot4_batched_minus_random',
                             mean=p['minus_random'], ci95=p['ci95']), models=out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--wiki', type=Path, required=True)
    parser.add_argument('--summary', type=Path, default=Path('results/llm-gpu-indomain.json'))
    args = parser.parse_args()
    plan = json.loads(PROTOCOL.read_text())
    specs = json.loads(s69.PROTOCOL.read_text())['models']
    if plan['stage'] != 70 or tuple(plan['arms']) != ARMS:
        raise ValueError('Protocol differs from the implementation')
    s69.setup_numerics()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'
    from transformers import AutoTokenizer
    import transformers
    import pandas as pd
    if s69.sha256(args.wiki) != WIKI_SHA256:
        raise ValueError('Wikipedia shard differs from the protocol')
    texts = pd.read_parquet(args.wiki, columns=['text'])['text'].tolist()
    manifest = json.loads(s69.MANIFEST.read_text())
    en_entry = manifest['languages']['en']['splits']['train']
    if s69.sha256(en_entry['path']) != en_entry['sha256']:
        raise ValueError('English train checksum mismatch')
    journal = s69.read_journal(args.journal)
    done = {(r['model'], r['rho'], r['seed'], r['arm']) for r in journal if r.get('kind') == 'run'}
    calibrations = {r['model']: r for r in journal if r.get('kind') == 'calibration'}
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    with args.journal.open('a') as out:
        def log(row):
            out.write(json.dumps(row, allow_nan=False) + '\n')
            out.flush()
            os.fsync(out.fileno())
        log(dict(kind='environment', time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), torch=torch.__version__,
                 transformers=transformers.__version__, cuda=torch.version.cuda, python=platform.python_version(),
                 device=device_name, resumed_runs=len(done)))
        for name in ('qwen2.5-0.5b', 'qwen2.5-1.5b'):
            model_dir = args.models_dir / name
            if s69.sha256(model_dir / 'model.safetensors') != specs[name]['safetensors_sha256']:
                raise ValueError(f'{name} weights differ from the protocol')
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            eligible, chunks = eligible_chunks(texts, tokenizer)
            probe, target, ar, population = split(eligible, chunks)
            en = llm.token_chunks(en_entry['path'], tokenizer, 65)
            pool = torch.cat([ar, en]).to(device)
            probe, target = probe.to(device), target.to(device)
            if name not in calibrations:
                ids, choices, _ = s69.schedule(0, CAL_SEED, 0.5, len(ar), len(en))
                cal = {}
                for arm in ARMS:
                    row = s69.run(s69.load(model_dir, CAL_SEED, device), arm, pool, probe, None, ids, choices,
                                  s69.CAL_STEPS, [], device)
                    cal[arm] = median(row['step_seconds'][4:])
                updates = {arm: s69.BUDGET if arm == 'random' else
                           int(s69.BUDGET * cal['random'] / cal[arm] + 1e-9) for arm in ARMS}
                calibrations[name] = dict(kind='calibration', model=name, device=device_name, costs=cal,
                                          updates=updates, population=population)
                log(calibrations[name])
            else:
                if calibrations[name]['device'] != device_name:
                    raise ValueError(f'Resumed on {device_name}; {name} was calibrated on {calibrations[name]["device"]}')
                if calibrations[name]['population'] != population:
                    raise ValueError('Evaluation sample differs from the journaled calibration')
            n = calibrations[name]['updates']
            print(name, device_name, 'costs', calibrations[name]['costs'], 'updates', n, flush=True)
            for k, rho in enumerate(plan['rho']):
                for i, seed in enumerate(plan['seeds']):
                    ids, choices, digest = s69.schedule(k, seed, rho, len(ar), len(en))
                    for arm in ARMS[i % 3:] + ARMS[:i % 3]:
                        if (name, rho, seed, arm) in done:
                            continue
                        checkpoints = sorted({n[a] for a in ARMS}) if arm == 'random' else [n[arm]]
                        row = s69.run(s69.load(model_dir, seed, device), arm, pool, probe, target, ids, choices,
                                      n[arm], checkpoints, device)
                        row.update(kind='run', model=name, rho=rho, seed=seed, arm=arm, updates=n[arm],
                                   schedule_sha256=digest, device=device_name,
                                   english_share=mean(float(ids[t][c] >= len(ar)) for t, c in enumerate(row['choices'])))
                        log(row)
                        print(f'{name} rho {rho} seed {seed} {arm}: {row["losses"][str(n[arm])]:.5f}', flush=True)
    journal = s69.read_journal(args.journal)
    rows = [r for r in journal if r.get('kind') == 'run']
    result = dict(stage=70, protocol=plan, journal_sha256=s69.sha256(args.journal),
                  environments=[r for r in journal if r.get('kind') == 'environment'],
                  calibrations=calibrations, summary=summarize(rows, calibrations, plan), runs=len(rows))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['summary']['primary'], indent=2))


if __name__ == '__main__':
    main()

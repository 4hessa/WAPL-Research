"""Stage 72: same-language contamination (Arabic news, garbled Arabic) on Qwen2.5-1.5B (Colab GPU).

Implements research/llm-gpu-noise-protocol.json with the stage-69 step loop.
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

PROTOCOL = Path('research/llm-gpu-noise-protocol.json')
MODEL = 'qwen2.5-1.5b'
ARMS = ('random', 'graddot4_batched')
TYPES = ('news', 'garbled', 'english')
CAL_SEED = 37999


def _draw(n, probe_seed, target_seed):
    probe = np.random.default_rng(probe_seed).choice(n, 16, replace=False)
    rest = np.setdiff1d(np.arange(n), probe)
    target = rest[np.random.default_rng(target_seed).choice(len(rest), 256, replace=False)]
    return probe, rest, target


def split(eligible, chunks):
    """Exclude stage 69-71 probe/target rows; draw stage-72 probe and target; split the rest."""
    used69 = s70.stage69_rows(len(eligible))
    rem = [r for r in range(len(eligible)) if r not in used69]
    for probe_seed, target_seed in ((7001, 7002), (7101, 7102)):
        p, _, t = _draw(len(rem), probe_seed, target_seed)
        used = {rem[i] for i in np.concatenate([p, t])}
        rem = [r for r in rem if r not in used]
    excluded = len(eligible) - len(rem)
    probe, rest, target = _draw(len(rem), 7201, 7202)
    others = np.setdiff1d(rest, target)  # sorted positions
    clean_rows = [rem[i] for i in others[0::2]]
    garble_rows = [rem[i] for i in others[1::2]]
    rng = np.random.default_rng(7204)
    garbled = [[chunks[r][j] for j in rng.permutation(65)] for r in garble_rows]
    tensor = lambda rows: torch.tensor([chunks[r] for r in rows], dtype=torch.long)
    population = dict(eligible_articles=len(eligible), excluded_stages_69_71=excluded,
                      probe_articles=[eligible[rem[i]] for i in np.sort(probe)],
                      target_articles=[eligible[rem[i]] for i in np.sort(target)],
                      clean_articles=len(clean_rows), garbled_articles=len(garble_rows))
    return (tensor([rem[i] for i in np.sort(probe)]), tensor([rem[i] for i in np.sort(target)]),
            tensor(clean_rows), torch.tensor(garbled, dtype=torch.long), population)


def summarize(rows, calibration, plan):
    n = calibration['updates']
    final, k = str(s69.BUDGET), str(n['graddot4_batched'])
    out = {}
    for t in TYPES:
        out[t] = {}
        for rho in plan['rho']:
            group = {arm: sorted([r for r in rows if (r['type'], r['rho'], r['arm']) == (t, rho, arm)],
                                 key=lambda r: r['seed']) for arm in ARMS}
            for arm in ARMS:
                if [r['seed'] for r in group[arm]] != plan['seeds']:
                    raise ValueError(f'Incomplete grid for {t} rho {rho} {arm}')
            rand, align = group['random'], group['graddot4_batched']
            if [r['losses']['0'] for r in rand] != [r['losses']['0'] for r in align]:
                raise ValueError('Arms do not share the initial state')
            d = [a['losses'][k] - r['losses'][final] for a, r in zip(align, rand)]
            sel = [a['losses'][k] - r['losses'][k] for a, r in zip(align, rand)]
            draws = np.random.default_rng(7203).choice(d, (10000, len(d)), replace=True).mean(axis=1)
            out[t][str(rho)] = dict(minus_random=mean(d), ci95=[float(x) for x in np.quantile(draws, [.025, .975])],
                                    per_seed=d, seeds_negative=sum(x < 0 for x in d), selection_term=mean(sel),
                                    throughput_term=mean(d) - mean(sel),
                                    random_minus_no_update=mean(r['losses'][final] - r['losses']['0'] for r in rand),
                                    align_contaminant_share=mean(a['contaminant_share'] for a in align),
                                    random_contaminant_share=mean(r['contaminant_share'] for r in rand))
    p = out['garbled']['0.5']
    predicted = {('garbled', '0.5'): True, ('garbled', '0.9'): True, ('english', '0.9'): True}
    return dict(primary=dict(type='garbled', rho=0.5, mean=p['minus_random'], ci95=p['ci95']),
                prediction_negative_holds={f'{t}@{r}': out[t][r]['minus_random'] < 0 for (t, r) in predicted},
                costs=calibration['costs'], updates=n, device=calibration['device'],
                r=calibration['costs']['graddot4_batched'] / calibration['costs']['random'], types=out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--wiki', type=Path, required=True)
    parser.add_argument('--summary', type=Path, default=Path('results/llm-gpu-noise.json'))
    args = parser.parse_args()
    plan = json.loads(PROTOCOL.read_text())
    spec = json.loads(s69.PROTOCOL.read_text())['models'][MODEL]
    if plan['stage'] != 72 or tuple(plan['arms']) != ARMS or tuple(plan['contamination_types']) != TYPES:
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
    splits = {lang: json.loads(s69.MANIFEST.read_text())['languages'][lang]['splits']['train'] for lang in ('ar', 'en')}
    for entry in splits.values():
        if s69.sha256(entry['path']) != entry['sha256']:
            raise ValueError('Train checksum mismatch')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    texts = pd.read_parquet(args.wiki, columns=['text'])['text'].tolist()
    eligible, chunks = s70.eligible_chunks(texts, tokenizer)
    probe, target, clean, garbled, population = split(eligible, chunks)
    contaminants = dict(news=llm.token_chunks(splits['ar']['path'], tokenizer, 65), garbled=garbled,
                        english=llm.token_chunks(splits['en']['path'], tokenizer, 65))
    pools = {t: torch.cat([clean, contaminants[t]]).to(device) for t in TYPES}
    probe, target = probe.to(device), target.to(device)
    journal = s69.read_journal(args.journal)
    done = {(r['type'], r['rho'], r['seed'], r['arm']) for r in journal if r.get('kind') == 'run'}
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
            t = TYPES.index('english')
            ids, choices, _ = s69.schedule(3 * t, CAL_SEED, 0.5, len(clean), len(contaminants['english']))
            costs = {}
            for arm in ARMS:
                row = s69.run(s69.load(model_dir, CAL_SEED, device), arm, pools['english'], probe, None, ids, choices,
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
                raise ValueError('Samples differ from the journaled calibration')
        n = calibration['updates']
        print(MODEL, device_name, 'costs', calibration['costs'], 'updates', n, flush=True)
        for t, kind in enumerate(TYPES):
            for k, rho in enumerate(plan['rho']):
                for i, seed in enumerate(plan['seeds']):
                    ids, choices, digest = s69.schedule(3 * t + k, seed, rho, len(clean), len(contaminants[kind]))
                    for arm in ARMS[i % 2:] + ARMS[:i % 2]:
                        if (kind, rho, seed, arm) in done:
                            continue
                        checkpoints = sorted(set(n.values())) if arm == 'random' else [n[arm]]
                        row = s69.run(s69.load(model_dir, seed, device), arm, pools[kind], probe, target, ids, choices,
                                      n[arm], checkpoints, device)
                        row.update(kind='run', model=MODEL, type=kind, rho=rho, seed=seed, arm=arm, updates=n[arm],
                                   schedule_sha256=digest, device=device_name,
                                   contaminant_share=mean(float(ids[s][c] >= len(clean)) for s, c in enumerate(row['choices'])))
                        log(row)
                        print(f'{kind} rho {rho} seed {seed} {arm}: {row["losses"][str(n[arm])]:.5f}', flush=True)
    journal = s69.read_journal(args.journal)
    rows = [r for r in journal if r.get('kind') == 'run']
    result = dict(stage=72, protocol=plan, journal_sha256=s69.sha256(args.journal),
                  environments=[r for r in journal if r.get('kind') == 'environment'],
                  calibration=calibration, summary=summarize(rows, calibration, plan), runs=len(rows))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['summary']['primary'], indent=2))


if __name__ == '__main__':
    main()

"""Stage 69: cost-charged LLM selection on a GPU, two Qwen2.5 sizes (run in Colab).

Implements research/llm-gpu-scale-protocol.json. Decision rules match stage 68;
this module is device-aware and resumable from a JSON-lines journal.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from statistics import mean, median
import time

import numpy as np
import torch

from experiments import llm_common as llm
from experiments.llm_budget import PerExampleCapture
from experiments.llm_selection import flat_grad, set_grad

PROTOCOL = Path('research/llm-gpu-scale-protocol.json')
MANIFEST = Path('data/multilingual_sources.json')
ARMS = ('random', 'loss4', 'graddot4', 'graddot4_batched')
GUIDED = ARMS[1:]
BUDGET = 400
CAL_SEED, CAL_RHO, CAL_STEPS = 34999, 0.5, 24
EVAL_BATCH = 32


def sha256(path):
    return llm.file_sha256(path)


def setup_numerics():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def load(model_dir, seed, device):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    return llm.add_lora(model, seed).to(device)


def select_articles(texts, tokenizer, plan):
    """First 65 tokens of eligible articles; probe and target drawn disjointly."""
    eligible, chunks = [], []
    for i, text in enumerate(texts):
        ids = tokenizer(text, add_special_tokens=False)['input_ids'] if text else []
        if len(ids) >= 65:
            eligible.append(i)
            chunks.append(ids[:65])
    order = np.random.default_rng(6901).choice(len(eligible), 16, replace=False)
    rest = np.setdiff1d(np.arange(len(eligible)), order)
    target = rest[np.random.default_rng(6902).choice(len(rest), 256, replace=False)]
    probe_rows, target_rows = np.sort(order), np.sort(target)
    tensor = lambda rows: torch.tensor([chunks[r] for r in rows], dtype=torch.long)
    population = dict(articles=len(texts), eligible_articles=len(eligible),
                      probe_articles=[eligible[r] for r in probe_rows],
                      target_articles=[eligible[r] for r in target_rows])
    return tensor(probe_rows), tensor(target_rows), population


def schedule(k, seed, rho, n_ar, n_en, length=4096):
    rng = np.random.default_rng(seed + 1000 * k)
    flags = rng.random((length, 4)) < rho
    ar = rng.integers(n_ar, size=(length, 4))
    en = rng.integers(n_en, size=(length, 4))
    ids = np.where(flags, n_ar + en, ar)
    choices = rng.integers(4, size=length)
    return ids, choices, hashlib.sha256(ids.tobytes() + choices.tobytes()).hexdigest()


@torch.no_grad()
def evaluate(model, target):
    total = 0.
    for start in range(0, len(target), EVAL_BATCH):
        batch = target[start:start + EVAL_BATCH]
        total += float(llm.lm_loss(model, batch)) * len(batch)
    return total / len(target)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def run(model, arm, pool, probe, target, ids, choices, updates, checkpoints, device, clock=time.perf_counter):
    params = llm.adapter_params(model)
    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.)
    capture = PerExampleCapture(model) if arm == 'graddot4_batched' else None
    losses = {0: evaluate(model, target)} if target is not None else {}
    picked, seconds, probe_grad = [], [], None
    for step in range(updates):
        candidates = pool[ids[step]]
        sync(device)
        start = clock()
        if arm in ('graddot4', 'graddot4_batched') and step % 4 == 0:
            opt.zero_grad()
            llm.lm_loss(model, probe).backward()
            probe_grad = flat_grad(params)
        if arm == 'random':
            c = int(choices[step])
        elif arm == 'loss4':
            with torch.no_grad():
                c = int(torch.argmax(llm.lm_loss(model, candidates, 'none')))
        if arm in ('random', 'loss4'):
            opt.zero_grad()
            llm.lm_loss(model, candidates[c:c + 1]).backward()
        elif arm == 'graddot4':
            grads = []
            for i in range(len(candidates)):
                opt.zero_grad()
                llm.lm_loss(model, candidates[i:i + 1]).backward()
                grads.append(flat_grad(params))
            grads = torch.stack(grads)
        elif arm == 'graddot4_batched':
            opt.zero_grad()
            capture.active = True
            llm.lm_loss(model, candidates, 'none').sum().backward()
            capture.active = False
            grads = capture.per_example()
        else:
            raise ValueError(f'Unknown arm: {arm}')
        if arm.startswith('graddot'):
            scores = grads @ probe_grad
            if not torch.isfinite(scores).all():
                raise ValueError('Nonfinite alignment score')
            c = int(torch.argmax(scores))
            set_grad(params, grads[c])
        opt.step()
        sync(device)
        seconds.append(clock() - start)
        picked.append(c)
        if target is not None and step + 1 in checkpoints:
            losses[step + 1] = evaluate(model, target)
    if not all(math.isfinite(v) for v in losses.values()):
        raise ValueError('Nonfinite evaluation')
    return dict(choices=picked, step_seconds=seconds, losses={str(k): v for k, v in losses.items()})


def budgets(costs):
    return {arm: BUDGET if arm == 'random' else math.floor(BUDGET * costs['random'] / costs[arm] + 1e-9)
            for arm in ARMS}


def bootstrap(values):
    draws = np.random.default_rng(6903).choice(values, (10000, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [.025, .975])]


def summarize(rows, calibrations, plan):
    out = {}
    for name in plan['models']:
        n = calibrations[name]['updates']
        per_rho = {}
        for rho in plan['rho']:
            group = {arm: sorted([r for r in rows if (r['model'], r['rho'], r['arm']) == (name, rho, arm)],
                                 key=lambda r: r['seed']) for arm in ARMS}
            for arm in ARMS:
                if [r['seed'] for r in group[arm]] != plan['seeds']:
                    raise ValueError(f'Incomplete grid for {name} rho {rho} {arm}')
            rand = group['random']
            entry = dict(loss0=mean(r['losses']['0'] for r in rand), random_final=mean(r['losses'][str(BUDGET)] for r in rand),
                         random_english_share=mean(r['english_share'] for r in rand), arms={})
            for arm in GUIDED:
                k = str(n[arm])
                diff = [g['losses'][k] - r['losses'][str(BUDGET)] for g, r in zip(group[arm], rand)]
                sel = [g['losses'][k] - r['losses'][k] for g, r in zip(group[arm], rand)]
                entry['arms'][arm] = dict(updates=n[arm], final=mean(g['losses'][k] for g in group[arm]),
                                          minus_random=mean(diff), ci95=bootstrap(diff), per_seed=diff,
                                          seeds_negative=sum(d < 0 for d in diff), selection_term=mean(sel),
                                          throughput_term=mean(diff) - mean(sel),
                                          english_share=mean(g['english_share'] for g in group[arm]))
            per_rho[str(rho)] = entry
        out[name] = dict(costs=calibrations[name]['costs'], updates=n,
                         r={a: calibrations[name]['costs'][a] / calibrations[name]['costs']['random'] for a in ARMS},
                         per_rho=per_rho)
    primary = out['qwen2.5-1.5b']['per_rho']['0.9']['arms']['graddot4_batched']
    return dict(primary=dict(model='qwen2.5-1.5b', rho=0.9, contrast='graddot4_batched_minus_random',
                             mean=primary['minus_random'], ci95=primary['ci95']), models=out)


def read_journal(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True, help='JSON-lines journal, e.g. on Google Drive')
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--wiki', type=Path, required=True, help='Pinned Arabic Wikipedia parquet shard')
    parser.add_argument('--summary', type=Path, default=Path('results/llm-gpu-scale.json'))
    args = parser.parse_args()
    plan = json.loads(PROTOCOL.read_text())
    if tuple(plan['arms']) != ARMS or plan['stage'] != 69:
        raise ValueError('Protocol differs from the implementation')
    setup_numerics()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from transformers import AutoTokenizer
    import transformers
    if sha256(args.wiki) != plan['evaluation_source'].split('SHA-256 ')[1].split(';')[0]:
        raise ValueError('Wikipedia shard differs from the protocol')
    import pandas as pd
    texts = pd.read_parquet(args.wiki, columns=['text'])['text'].tolist()
    manifest = json.loads(MANIFEST.read_text())
    splits = {lang: manifest['languages'][lang]['splits'] for lang in ('ar', 'en')}
    for lang in splits:
        if sha256(splits[lang]['train']['path']) != splits[lang]['train']['sha256']:
            raise ValueError(f'{lang} train checksum mismatch')
    journal = read_journal(args.journal)
    done = {(r['model'], r['rho'], r['seed'], r['arm']) for r in journal if r.get('kind') == 'run'}
    calibrations = {r['model']: r for r in journal if r.get('kind') == 'calibration'}
    env = dict(kind='environment', time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), torch=torch.__version__,
               transformers=transformers.__version__, cuda=torch.version.cuda, python=platform.python_version(),
               device=torch.cuda.get_device_name(0) if device.type == 'cuda' else platform.processor(),
               resumed_runs=len(done))
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    with args.journal.open('a') as out:
        def log(row):
            out.write(json.dumps(row, allow_nan=False) + '\n')
            out.flush()
            os.fsync(out.fileno())
        log(env)
        for name, spec in plan['models'].items():
            model_dir = args.models_dir / name
            if sha256(model_dir / 'model.safetensors') != spec['safetensors_sha256']:
                raise ValueError(f'{name} weights differ from the protocol')
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            ar = llm.token_chunks(splits['ar']['train']['path'], tokenizer, 65)
            en = llm.token_chunks(splits['en']['train']['path'], tokenizer, 65)
            pool = torch.cat([ar, en]).to(device)
            probe, target, population = select_articles(texts, tokenizer, plan)
            probe, target = probe.to(device), target.to(device)
            if name not in calibrations:
                ids, choices, _ = schedule(0, CAL_SEED, CAL_RHO, len(ar), len(en))
                cal = {}
                for arm in ARMS:
                    row = run(load(model_dir, CAL_SEED, device), arm, pool, probe, None, ids, choices, CAL_STEPS, [], device)
                    cal[arm] = median(row['step_seconds'][4:])
                calibrations[name] = dict(kind='calibration', model=name, costs=cal, updates=budgets(cal),
                                          population=population)
                log(calibrations[name])
            elif calibrations[name]['population'] != population:
                raise ValueError('Evaluation sample differs from the journaled calibration')
            n = calibrations[name]['updates']
            print(name, 'costs', calibrations[name]['costs'], 'updates', n, flush=True)
            for k, rho in enumerate(plan['rho']):
                for i, seed in enumerate(plan['seeds']):
                    ids, choices, digest = schedule(k, seed, rho, len(ar), len(en))
                    for arm in ARMS[i % 4:] + ARMS[:i % 4]:
                        if (name, rho, seed, arm) in done:
                            continue
                        checkpoints = sorted({n[a] for a in ARMS}) if arm == 'random' else [n[arm]]
                        row = run(load(model_dir, seed, device), arm, pool, probe, target, ids, choices,
                                  n[arm], checkpoints, device)
                        row.update(kind='run', model=name, rho=rho, seed=seed, arm=arm, updates=n[arm],
                                   schedule_sha256=digest,
                                   english_share=mean(float(ids[t][c] >= len(ar)) for t, c in enumerate(row['choices'])))
                        log(row)
                        print(f'{name} rho {rho} seed {seed} {arm}: {row["losses"][str(n[arm])]:.5f}', flush=True)
    journal = read_journal(args.journal)
    rows = [r for r in journal if r.get('kind') == 'run']
    result = dict(stage=69, protocol=plan, journal_sha256=sha256(args.journal),
                  environments=[r for r in journal if r.get('kind') == 'environment'],
                  calibrations=calibrations, summary=summarize(rows, calibrations, plan), runs=len(rows))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['summary']['primary'], indent=2))


if __name__ == '__main__':
    main()

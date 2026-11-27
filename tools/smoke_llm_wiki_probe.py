"""Stage 78 smoke 2 (before the protocol): can the 0.5B proxy separate Arabic Wikipedia from web chunks?

Implements research/llm-wiki-probe-smoke-rule.json, written and pushed before this code. Exposed data only: the
probe is the 120 ArabicMMLU All/dev items; no test item is evaluated or scored and no model is trained.

Probes: P0 = stage-76 cloze probe (experiments.llm_web_filter.probe_gradient); P1 = dev items as plain text
((Context + newline) + Question + newline + correct option, first 192 tokens, mean next-token NLL, one item per pass).
Pools: A = Wikipedia 0-511 + stage-76 kept web 0-767 + removed web 0-767 (the smoke-1 pool); B = Wikipedia 512-1023 +
kept web 768-1535 + removed web 768-1535. Seeds 78900, 78901. The decision rule is applied as written.

python -m tools.smoke_llm_wiki_probe --kept K --removed R --mmlu M --sources DIR [--out results/llm-wiki-probe-smoke.json]
"""
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from experiments import llm_cheap_filter as s76
from experiments import llm_common as llm
from experiments import llm_gpu as s69
from experiments import llm_web_filter as s73
from experiments.llm_budget import PerExampleCapture
from experiments.llm_selection import flat_grad
from tools import smoke_llm_natural_sources as sm1

RULE = Path('research/llm-wiki-probe-smoke-rule.json')
SMOKE1 = Path('results/llm-natural-source-smoke.json')
SEEDS, POOLS, PROBES = (78900, 78901), ('A', 'B'), ('P0_cloze', 'P1_plain')
PLAIN_MAX = 192
SMOKE1_AUC, SANITY_TOL = 0.644622802734375, 1e-6


def plain_items(records, tokenizer):
    out = []
    for r in records:
        ctx = str(r['Context']) + '\n' if s73._present(r.get('Context')) else ''
        answer = r[f'Option {s73.LETTERS.index(str(r["Answer Key"]).strip()) + 1}']
        out.append(tokenizer(ctx + str(r['Question']) + '\n' + str(answer), add_special_tokens=False)['input_ids'][:PLAIN_MAX])
    return out


def plain_probe_gradient(model, seqs, device):
    params = llm.adapter_params(model)
    for p in params:
        p.grad = None
    for ids in seqs:
        (llm.lm_loss(model, torch.tensor([ids], device=device)) / len(seqs)).backward()
    g = flat_grad(params)
    for p in params:
        p.grad = None
    return g


def score(model, probe, pool, probe_items, plain, device):
    """Scores and seconds; P0 is exactly experiments.llm_web_filter.score(align_grad)."""
    if probe == 'P0_cloze':
        return s73.score(model, 'align_grad', pool, probe_items, device)
    capture = PerExampleCapture(model)
    s69.sync(device)
    start = time.perf_counter()
    g = plain_probe_gradient(model, plain, device)
    scores = s73.score_grad(model, pool, g, capture)
    s69.sync(device)
    return scores.double().cpu().numpy(), time.perf_counter() - start


def decide(cells, rule_thresholds=dict(mean_auc=0.80, min_auc=0.75, share=0.50, time_ratio=1.25)):
    t0 = np.mean([c['seconds'] for c in cells['P0_cloze'].values()])
    out = {}
    for probe, cs in cells.items():
        aucs = [c['auc'] for c in cs.values()]
        share = float(np.mean([c['wiki_share_top512'] for c in cs.values()]))
        ratio = float(np.mean([c['seconds'] for c in cs.values()]) / t0)
        checks = {'mean AUC >= 0.80': float(np.mean(aucs)) >= rule_thresholds['mean_auc'],
                  'every AUC >= 0.75': min(aucs) >= rule_thresholds['min_auc'],
                  'mean top-512 share >= 0.50': share >= rule_thresholds['share'],
                  'time <= 1.25 x P0': ratio <= rule_thresholds['time_ratio']}
        out[probe] = dict(mean_auc=float(np.mean(aucs)), min_auc=float(min(aucs)), mean_share_top512=share,
                          time_ratio_vs_P0=ratio, checks=checks, passes=all(checks.values()))
    passing = [p for p in PROBES if out[p]['passes']]
    if not passing:
        return out, dict(decision='stop', probe=None)
    if len(passing) == 2 and abs(out['P0_cloze']['mean_auc'] - out['P1_plain']['mean_auc']) < 0.01:
        chosen = 'P0_cloze'
    else:
        chosen = max(passing, key=lambda p: out[p]['mean_auc'])
    return out, dict(decision='proceed', probe=chosen)


def main():
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('--kept', '--removed', '--mmlu', '--sources'):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--out', type=Path, default=Path('results/llm-wiki-probe-smoke.json'))
    args = parser.parse_args()
    import pandas as pd
    import transformers
    from transformers import AutoTokenizer
    s69.setup_numerics()
    spec = json.loads(s69.PROTOCOL.read_text())['models']['qwen2.5-0.5b']
    assert s69.sha256(args.models_dir / 'qwen2.5-0.5b' / 'model.safetensors') == spec['safetensors_sha256']
    tokenizer = AutoTokenizer.from_pretrained(args.models_dir / 'qwen2.5-1.5b')
    probe_items, _, pool76, pop = s76.load_inputs(args, tokenizer)
    cal = next(r for r in s69.read_journal(sm1.STAGE76_JOURNAL) if r.get('kind') == 'calibration')
    assert pop['pool_sha256'] == cal['population']['pool_sha256'], 'stage-76 pool differs'
    dev = pd.read_csv(args.mmlu / 'dev.csv').to_dict('records')
    test = pd.read_csv(args.mmlu / 'test.csv').to_dict('records')
    token_sets, chars = sm1.banned_sets(dev + test, tokenizer)
    rel, column, _ = sm1.SOURCES['wikipedia']
    path = args.sources / rel
    info, _, chunks, _ = sm1.text_chunks(path, column, tokenizer, sm1.MIN_CHARS['wikipedia'])
    keep, counts = sm1.decontaminate(chunks, tokenizer, token_sets, chars)
    s1 = json.loads(SMOKE1.read_text())['sources']['wikipedia']
    assert (info['eligible'], len(keep), counts) == (s1['eligible'], s1['clean_chunks'], s1['chunk_decontamination'])
    pick = np.random.default_rng(7810).choice(len(keep), sm1.N, replace=False)
    wiki = torch.tensor([chunks[keep[i]] for i in pick], dtype=torch.long)
    web_k, web_r = pool76[s76.N_QA:s76.N_QA + s76.N_WEB], pool76[s76.N_QA + s76.N_WEB:]
    pools = {'A': torch.cat([wiki[:512], web_k[:768], web_r[:768]]),
             'B': torch.cat([wiki[512:1024], web_k[768:1536], web_r[768:1536]])}
    plain = plain_items(dev, tokenizer)
    device = torch.device('cuda')
    out = dict(kind='smoke', stage='78 smoke 2', rule=RULE.as_posix(), rule_sha256=s69.sha256(RULE),
               time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), device=torch.cuda.get_device_name(0),
               torch=torch.__version__, transformers=transformers.__version__, python=platform.python_version(),
               wikipedia_file_sha256=s69.sha256(path), pool_sha256={k: s73.tensor_sha256(v) for k, v in pools.items()},
               plain_probe_tokens=dict(mean=float(np.mean([len(s) for s in plain])), max=max(len(s) for s in plain)),
               cells={p: {} for p in PROBES})
    for probe in PROBES:
        for seed in SEEDS:
            for name in POOLS:
                model = s69.load(args.models_dir / 'qwen2.5-0.5b', seed, device)
                scores, seconds = score(model, probe, pools[name].to(device), probe_items, plain, device)
                del model
                gc.collect(), torch.cuda.empty_cache()
                cell = dict(auc=sm1.auc(scores, 512), wiki_share_top512=float(np.mean(s73.keep_top(scores, 512) < 512)),
                            wiki_share_top256=float(np.mean(s73.keep_top(scores, 256) < 512)), seconds=seconds)
                out['cells'][probe][f'{seed}_{name}'] = cell
                print(probe, seed, name, json.dumps(cell), flush=True)
    s = out['cells']['P0_cloze']['78900_A']['auc']
    out['sanity'] = dict(smoke1_auc=SMOKE1_AUC, reproduced=s, passes=abs(s - SMOKE1_AUC) <= SANITY_TOL)
    if out['sanity']['passes']:
        out['per_probe'], out['decision'] = decide(out['cells'])
    else:
        out['decision'] = dict(decision='none: sanity check failed, report to the owner', probe=None)
    args.out.write_text(json.dumps(out, indent=2) + '\n')
    print('decision', json.dumps(out['decision']), flush=True)


if __name__ == '__main__':
    main()

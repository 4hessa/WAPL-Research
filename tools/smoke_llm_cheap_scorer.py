"""Stage 76 smoke test before the protocol: cost and ranking agreement of cheaper alignment scorers.

No target item is evaluated and no loss is printed. Everything scored here is already exposed: the stage-74 sigma
0.25 pool (512 in-format QA chunks first, then 1536 web chunks), the 120 All/dev probe items, and the full-scorer
scores journaled by stage 75 (align_once rows, seeds 37500 and 37501).

Scorers (gradient alignment with the probe at the seed's initial adapter, probe gradient included in the time):
  full        Qwen2.5-1.5B, LoRA on q/v of all 28 layers, dA and dB (the stage 73-75 scorer), batch 16.
  *_dB        Only the B factors are scored. At the initial adapter B = 0, so dA = 0 and the score is unchanged
              mathematically; this checks that numerically.
  1.5b_lastK  LoRA gradients of the last K layers only (earlier adapters frozen, so backpropagation stops there).
  0.5b*       Qwen2.5-0.5B (same tokenizer files, pinned revision) as proxy, with its own LoRA at the same seed.
Agreement with the journaled full scores of the same seed: Spearman, overlap of the top 512, QA share of the top
512 and top 256, and QA-vs-web AUC (QA labels are pool positions, not outcomes).

python -m tools.smoke_llm_cheap_scorer --kept K --removed R --mmlu M [--out results/llm-cheap-scorer-smoke.json]
"""
import argparse
import gc
import json
from pathlib import Path
import platform
import time
from statistics import median

import numpy as np
import torch

from experiments import llm_common as llm
from experiments import llm_gpu as s69
from experiments import llm_once_filter as s75
from experiments import llm_qa_mix as s74
from experiments import llm_web_filter as s73
from experiments.llm_selection import flat_grad

STAGE75_JOURNAL = Path('results/llm-once-filter-journal.jsonl')
SEEDS = (37500, 37501)
N_QA = 512
# name: (model, last_k or None for all layers, dB only, batch)
SCORERS = {
    'full': ('qwen2.5-1.5b', None, False, 16),
    'full_dB': ('qwen2.5-1.5b', None, True, 16),
    '1.5b_last8_b64': ('qwen2.5-1.5b', 8, True, 64),
    '1.5b_last4_b64': ('qwen2.5-1.5b', 4, True, 64),
    '1.5b_last1_b64': ('qwen2.5-1.5b', 1, True, 64),
    '0.5b_dB': ('qwen2.5-0.5b', None, True, 16),
    '0.5b_dB_b64': ('qwen2.5-0.5b', None, True, 64),
    '0.5b_last4_b64': ('qwen2.5-0.5b', 4, True, 64),
}


class Capture:
    """Per-example LoRA gradients for the trainable LoRA modules only (rows in llm.adapter_params order)."""

    def __init__(self, model):
        self.modules = [m for m in model.modules() if isinstance(m, llm.LoRALinear)
                        and (m.A.requires_grad or m.B.requires_grad)]
        self.active, self.inputs, self.grads = False, {}, {}
        for m in self.modules:
            m.register_forward_hook(self._forward)

    def _forward(self, module, args, output):
        if self.active:
            self.inputs[module] = args[0].detach()
            output.register_hook(lambda g, module=module: self.grads.__setitem__(module, g.detach()))

    def per_example(self):
        parts = []
        for m in self.modules:
            x, g = self.inputs[m], self.grads[m]
            if m.A.requires_grad:
                parts.append((m.scale * torch.einsum('btr,bti->bri', g @ m.B.detach(), x)).reshape(len(x), -1))
            if m.B.requires_grad:
                parts.append((m.scale * torch.einsum('bto,btr->bor', g, x @ m.A.detach().t())).reshape(len(x), -1))
        self.inputs, self.grads = {}, {}
        return torch.cat(parts, dim=1)


def restrict(model, last_k, db_only):
    """Freeze the LoRA factors a cheap scorer does not use."""
    layers = model.model.layers
    for i, layer in enumerate(layers):
        for name in ('q_proj', 'v_proj'):
            m = getattr(layer.self_attn, name)
            on = last_k is None or i >= len(layers) - last_k
            m.A.requires_grad_(on and not db_only)
            m.B.requires_grad_(on)


def score(model, pool, probe_items, batch, device):
    capture = Capture(model)
    params = llm.adapter_params(model)
    s69.sync(device)
    start = time.perf_counter()
    g = s73.probe_gradient(model, probe_items, device)
    out = []
    for s in range(0, len(pool), batch):
        for p in params:
            p.grad = None
        capture.active = True
        llm.lm_loss(model, pool[s:s + batch], 'none').sum().backward()
        capture.active = False
        out.append(capture.per_example() @ g)
    s69.sync(device)
    seconds = time.perf_counter() - start
    scores = torch.cat(out).double().cpu().numpy()
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite score')
    return scores, seconds


def auc(scores, n_pos):
    r = np.argsort(np.argsort(scores, kind='stable'), kind='stable').astype(float) + 1
    n_neg = len(scores) - n_pos
    return float((r[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def agreement(scores, ref):
    top = lambda x, k: set(s73.keep_top(x, k).tolist())
    return dict(spearman=float(s73.spearman(scores, ref)), top512_overlap=len(top(scores, 512) & top(ref, 512)) / 512,
                qa_share_top512=sum(i < N_QA for i in top(scores, 512)) / 512,
                qa_share_top256=sum(i < N_QA for i in top(scores, 256)) / 256, qa_auc=auc(scores, N_QA))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('--kept', '--removed', '--mmlu'):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--out', type=Path, default=Path('results/llm-cheap-scorer-smoke.json'))
    args = parser.parse_args()
    import transformers
    from transformers import AutoTokenizer
    s69.setup_numerics()
    specs = json.loads(s69.PROTOCOL.read_text())['models']
    for m in ('qwen2.5-1.5b', 'qwen2.5-0.5b'):
        assert s69.sha256(args.models_dir / m / 'model.safetensors') == specs[m]['safetensors_sha256'], m
    for f in ('tokenizer.json', 'vocab.json', 'merges.txt'):
        assert s69.sha256(args.models_dir / 'qwen2.5-0.5b' / f) == s69.sha256(args.models_dir / 'qwen2.5-1.5b' / f), f
    tokenizer = AutoTokenizer.from_pretrained(args.models_dir / 'qwen2.5-1.5b')
    probe, _, _, pools, population, _ = s74.load_inputs(args, tokenizer)
    assert population['pool_sha256']['0.25'] == s75.STAGE74_POOL_SHA256['0.25']
    journal = s69.read_journal(STAGE75_JOURNAL)
    ref = {r['seed']: np.asarray(r['scores']) for r in journal
           if r.get('kind') == 'run' and r['arm'] == 'align_once' and r['seed'] in SEEDS}
    device = torch.device('cuda')
    pool = pools[0.25].to(device)
    out = dict(kind='smoke', stage=76, time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               device=torch.cuda.get_device_name(0), torch=torch.__version__, transformers=transformers.__version__,
               python=platform.python_version(), tokenizer_files_identical=True, scorers={})
    rand = np.random.default_rng(76997).integers(len(pool), size=24)
    row = s73.train(s69.load(args.models_dir / 'qwen2.5-1.5b', 76997, device), pool, rand, 24, set(), None, device)
    c_r = out['random_step_seconds'] = median(row['step_seconds'][4:])
    gc.collect(), torch.cuda.empty_cache()
    for name, (model_name, last_k, db_only, batch) in SCORERS.items():
        res = dict(model=model_name, last_k=last_k, db_only=db_only, batch=batch, seeds={})
        for seed in SEEDS:
            model = s69.load(args.models_dir / model_name, seed, device)
            restrict(model, last_k, db_only)
            try:
                scores, seconds = score(model, pool, probe, batch, device)
            except torch.OutOfMemoryError:
                scores = None
            del model
            gc.collect(), torch.cuda.empty_cache()
            if scores is None:
                res['seeds'][str(seed)] = 'out of memory'
                print(name, seed, 'out of memory', flush=True)
                continue
            res['seeds'][str(seed)] = dict(seconds=seconds, steps_per_chunk=seconds / c_r / len(pool),
                                           agreement=agreement(scores, ref[seed]),
                                           max_rel_diff_vs_journal=float(np.max(np.abs(scores - ref[seed]))
                                                                         / np.max(np.abs(ref[seed]))))
            print(name, seed, json.dumps(res['seeds'][str(seed)]), flush=True)
        if any(isinstance(v, str) for v in res['seeds'].values()):
            out['scorers'][name] = res
            continue
        res['seconds'] = max(v['seconds'] for v in res['seeds'].values())
        res['steps_per_chunk'] = res['seconds'] / c_r / len(pool)
        res['speedup_vs_full'] = None
        out['scorers'][name] = res
    full = out['scorers']['full']['seconds']
    for res in out['scorers'].values():
        if 'seconds' in res:
            res['speedup_vs_full'] = full / res['seconds']
    args.out.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: dict(steps_per_chunk=v['steps_per_chunk'], speedup=v['speedup_vs_full'],
                              **v['seeds'][str(SEEDS[0])]['agreement']) for k, v in out['scorers'].items()
                      if 'seconds' in v}, indent=1))


if __name__ == '__main__':
    main()

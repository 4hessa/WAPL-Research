"""Stage 68: cost-charged selection on Qwen2.5-0.5B with batched per-example LoRA gradients.

Implements research/llm-budget-protocol.json. random, loss4 and graddot4 call
the unchanged stage-66 run_arm; graddot4_batched recovers each candidate's
LoRA gradient from one batched backward pass.
"""
import argparse
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
from experiments import llm_selection as s66
from experiments.pud_reuse_matched import digest

PROTOCOL = Path('research/llm-budget-protocol.json')
ARMS = ('random', 'loss4', 'graddot4', 'graddot4_batched')
GUIDED = ARMS[1:]
CALIBRATION_SEED = 33999
CALIBRATION_RHO = 0.5
CALIBRATION_STEPS = 12


class PerExampleCapture:
    """Record LoRA layer inputs and output gradients to form per-example adapter gradients."""

    def __init__(self, model):
        self.modules = [m for m in model.modules() if isinstance(m, llm.LoRALinear)]
        self.active = False
        self.inputs, self.grads = {}, {}
        for m in self.modules:
            m.register_forward_hook(self._forward)

    def _forward(self, module, args, output):
        if not self.active:
            return
        self.inputs[module] = args[0].detach()

        def keep(g, module=module):
            self.grads[module] = g.detach()
        output.register_hook(keep)

    def per_example(self):
        """Rows follow llm.adapter_params order: for each LoRA module, A then B."""
        parts = []
        for m in self.modules:
            x, g = self.inputs[m], self.grads[m]
            u = x @ m.A.detach().t()
            dA = m.scale * torch.einsum('btr,bti->bri', g @ m.B.detach(), x)
            dB = m.scale * torch.einsum('bto,btr->bor', g, u)
            parts += [dA.reshape(len(x), -1), dB.reshape(len(x), -1)]
        self.inputs, self.grads = {}, {}
        return torch.cat(parts, dim=1)


def run_batched(seed, train, probe, target, ids, checkpoints, updates, clock=time.perf_counter):
    model = llm.load_model(seed)
    params = llm.adapter_params(model)
    capture = PerExampleCapture(model)
    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.)
    losses = {0: s66.evaluate(model, target)}
    picked, seconds, probe_grad = [], [], None
    for step in range(updates):
        candidates = train[ids[step]]
        start = clock()
        if step % 4 == 0:
            opt.zero_grad()
            llm.lm_loss(model, probe).backward()
            probe_grad = s66.flat_grad(params)
        opt.zero_grad()
        capture.active = True
        llm.lm_loss(model, candidates, 'none').sum().backward()
        capture.active = False
        grads = capture.per_example()
        scores = grads @ probe_grad
        if not torch.isfinite(scores).all():
            raise ValueError('Nonfinite alignment score')
        c = int(torch.argmax(scores))
        s66.set_grad(params, grads[c])
        opt.step()
        seconds.append(clock() - start)
        picked.append(c)
        if step + 1 in checkpoints:
            losses[step + 1] = s66.evaluate(model, target)
    return dict(choices=picked, step_seconds=seconds, losses={str(k): v for k, v in losses.items()})


def run(arm, seed, train, probe, target, ids, choices, updates, checkpoints, clock=time.perf_counter):
    checkpoints = sorted(set(checkpoints) | {updates})
    if arm == 'graddot4_batched':
        return run_batched(seed, train, probe, target, ids, checkpoints, updates, clock)
    plan = dict(updates=updates, checkpoints=checkpoints)
    return s66.run_arm(seed, train, probe, target, ids, choices, arm, plan, clock=clock)


def prepare(tokenizer, hashes):
    manifest = json.loads(s66.MANIFEST.read_text())
    plan66 = json.loads(s66.PROTOCOL.read_text())
    splits = {lang: manifest['languages'][lang]['splits'] for lang in ('ar', 'en')}
    for lang in splits:
        for split in ('train', 'dev'):
            entry = splits[lang][split]
            if digest(Path(entry['path'])) != entry['sha256']:
                raise ValueError('Prepared input checksum mismatch')
            hashes[entry['path']] = entry['sha256']
    ar_train, probe, _, pop66 = s66.prepare_language(splits['ar'], tokenizer, plan66, {})
    en_train = llm.token_chunks(splits['en']['train']['path'], tokenizer, 65)
    text, _, _ = s66.eligible_dev_text(Path(splits['ar']['dev']['path']).read_bytes())
    tokens = tokenizer(text, add_special_tokens=False)['input_ids']
    full = len(tokens) // 65
    dev = torch.tensor(tokens[:full * 65], dtype=torch.long).reshape(full, 65)
    used = set(pop66['probe_ids']) | set(pop66['target_ids'])
    target_ids = [i for i in range(full) if i not in used]
    pool = torch.cat([ar_train, en_train])
    population = dict(ar_train_chunks=len(ar_train), en_train_chunks=len(en_train), eligible_dev_chunks=full,
                      target_ids=target_ids, probe_ids=pop66['probe_ids'])
    return pool, len(ar_train), len(en_train), probe, dev[target_ids], population


def schedule(k, seed, rho, n_ar, n_en, length=512):
    rng = np.random.default_rng(seed + 1000 * k)
    flags = rng.random((length, 4)) < rho
    ar = rng.integers(n_ar, size=(length, 4))
    en = rng.integers(n_en, size=(length, 4))
    ids = np.where(flags, n_ar + en, ar)
    choices = rng.integers(4, size=length)
    return ids, choices


def budgets(costs, plan):
    total = 96
    return {arm: total if arm == 'random' else math.floor(total * costs['random'] / costs[arm] + 1e-9) for arm in ARMS}


def summarize(rows, plan, n):
    out = {}
    for rho in plan['rho']:
        group = {arm: sorted([r for r in rows if r['rho'] == rho and r['arm'] == arm], key=lambda r: r['seed'])
                 for arm in ARMS}
        for arm in ARMS:
            if [r['seed'] for r in group[arm]] != plan['seeds']:
                raise ValueError('Incomplete grid')
        rand = group['random']
        loss0 = [r['losses']['0'] for r in rand]
        entry = dict(loss0=mean(loss0), random_final=mean(r['losses'][str(n['random'])] for r in rand),
                     random_english_share=mean(r['english_share'] for r in rand), arms={})
        for arm in GUIDED:
            if [r['losses']['0'] for r in group[arm]] != loss0:
                raise ValueError('Arms do not share the initial state')
            k = str(n[arm])
            diff = [g['losses'][k] - r['losses'][str(n['random'])] for g, r in zip(group[arm], rand)]
            selection = [g['losses'][k] - r['losses'][k] for g, r in zip(group[arm], rand)]
            entry['arms'][arm] = dict(updates=n[arm], final=mean(g['losses'][k] for g in group[arm]),
                                      minus_random=mean(diff), per_seed=diff,
                                      seeds_negative=sum(d < 0 for d in diff),
                                      selection_term=mean(selection),
                                      throughput_term=mean(r['losses'][k] - r['losses'][str(n['random'])] for r in rand),
                                      english_share=mean(g['english_share'] for g in group[arm]))
        pairs = [(a['choices'], b['choices']) for a, b in zip(group['graddot4'], group['graddot4_batched'])]
        entry['graddot_agreement_prefix'] = [next((i for i, (x, y) in enumerate(zip(p, q)) if x != y), min(len(p), len(q)))
                                             for p, q in pairs]
        out[str(rho)] = entry
    return dict(primary=dict(rho=0.9, contrast='graddot4_batched_minus_random',
                             mean=out['0.9']['arms']['graddot4_batched']['minus_random']), per_rho=out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    args = parser.parse_args()
    paths = dict(result=args.output_dir / 'llm-budget.json', records=args.output_dir / 'llm-budget-records.jsonl')
    if any(p.exists() for p in paths.values()):
        parser.error('Choose an unused output directory to preserve recorded runs')
    plan = json.loads(PROTOCOL.read_text())
    if tuple(plan['arms']) != ARMS or plan['stage'] != 68:
        raise ValueError('Protocol differs from the implementation')
    if llm.file_sha256(llm.MODEL_DIR / 'model.safetensors') != llm.MODEL_SHA256:
        raise ValueError('Model weights differ')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    from transformers import AutoTokenizer
    import transformers
    tokenizer = AutoTokenizer.from_pretrained(llm.MODEL_DIR)
    hashes = {p: digest(Path(p)) for p in (str(PROTOCOL), 'experiments/llm_budget.py', 'experiments/llm_common.py',
                                          'experiments/llm_selection.py', 'tests/test_llm_budget.py',
                                          str(s66.PROTOCOL), str(llm.MODEL_DIR / 'tokenizer.json'))}
    hashes[str(llm.MODEL_DIR / 'model.safetensors')] = llm.MODEL_SHA256
    pool, n_ar, n_en, probe, target, population = prepare(tokenizer, hashes)
    cpu = next(l for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name'))
    meta = dict(sha256=hashes, cpu=cpu.split(':')[1].strip(), platform=platform.platform(),
                python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
                torch_threads=torch.get_num_threads(), logical_cpus=os.cpu_count())
    start = time.perf_counter()
    # Calibration on train candidates only; the target is never evaluated here.
    ids, choices = schedule(0, CALIBRATION_SEED, CALIBRATION_RHO, n_ar, n_en)
    calibration = {}
    stand_in = pool[:1]  # a train chunk; run_arm evaluates at step 0, outside the step timer
    for arm in ARMS:
        row = run(arm, CALIBRATION_SEED, pool, probe, stand_in, ids, choices, CALIBRATION_STEPS, [])
        calibration[arm] = dict(step_seconds=row['step_seconds'], cost=median(row['step_seconds'][2:]))
    costs = {arm: calibration[arm]['cost'] for arm in ARMS}
    n = budgets(costs, plan)
    print('costs', costs, 'updates', n, flush=True)
    rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with paths['records'].open('x') as records:
        records.write(json.dumps(dict(calibration=calibration, costs=costs, updates=n)) + '\n')
        records.flush()
        for k, rho in enumerate(plan['rho']):
            for i, seed in enumerate(plan['seeds']):
                ids, choices = schedule(k, seed, rho, n_ar, n_en)
                order = ARMS[i % 4:] + ARMS[:i % 4]
                for arm in order:
                    checkpoints = [0] + ([n[a] for a in ARMS] if arm == 'random' else [])
                    row = run(arm, seed, pool, probe, target, ids, choices, n[arm], checkpoints)
                    row.update(rho=rho, seed=seed, arm=arm, updates=n[arm],
                               english_share=mean(float(ids[t][c] >= n_ar) for t, c in enumerate(row['choices'])))
                    rows.append(row)
                    records.write(json.dumps(row, allow_nan=False) + '\n')
                    records.flush()
                    print(f'rho {rho} seed {seed} {arm}: n {n[arm]} loss {row["losses"][str(n[arm])]:.5f}', flush=True)
    result = dict(stage=68, protocol=plan, provenance=meta, population=population, costs=costs, updates=n,
                  r={arm: costs[arm] / costs['random'] for arm in ARMS},
                  output_sha256={paths['records'].name: digest(paths['records'])},
                  summary=summarize(rows, plan, n), runs=len(rows), wall_seconds=time.perf_counter() - start)
    with paths['result'].open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write('\n')


if __name__ == '__main__':
    main()

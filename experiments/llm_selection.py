"""Stage 66: selection gain and decision cost in LoRA adaptation of Qwen2.5-0.5B.

Implements research/llm-selection-protocol.json. Every arm runs exactly the
same number of updates; per-step wall time is recorded to estimate r.
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
from experiments.pud_reuse_matched import digest
from experiments.ud_cost_frontier import unscored_target
from experiments.ud_cost_frontier import PROTOCOL as STAGE65_PROTOCOL

PROTOCOL = Path('research/llm-selection-protocol.json')
MANIFEST = Path('data/multilingual_sources.json')
ARMS = ('random', 'loss4', 'graddot4')
EVAL_BATCH = 16


def exposed_dev_chunks(n_dev):
    exposed = set()
    for seed, count in ((4800, 128), (6301, 64)):
        exposed |= set(np.random.default_rng(seed).choice(n_dev, count, replace=False).tolist())
    stage65 = json.loads(STAGE65_PROTOCOL.read_text())
    target65, _ = unscored_target(n_dev, stage65)
    return exposed | set(target65.tolist())


def eligible_dev_text(blob):
    """Dev lines whose byte span meets no previously scored 65-byte chunk."""
    exposed = exposed_dev_chunks(len(blob) // 65)
    kept, dropped, offset = [], 0, 0
    for line in blob.split(b'\n'):
        end = offset + len(line)
        first, last = offset // 65, max(offset, end - 1) // 65
        if line and not any(i in exposed for i in range(first, last + 1)):
            kept.append(line)
        elif line:
            dropped += 1
        offset = end + 1
    return b'\n'.join(kept).decode('utf-8'), len(kept), dropped


def prepare_language(splits, tokenizer, plan, hashes):
    for split in ('train', 'dev'):
        entry = splits[split]
        if digest(Path(entry['path'])) != entry['sha256']:
            raise ValueError(f'Prepared {split} input checksum mismatch')
        hashes[entry['path']] = entry['sha256']
    train = llm.token_chunks(splits['train']['path'], tokenizer, 65)
    text, kept, dropped = eligible_dev_text(Path(splits['dev']['path']).read_bytes())
    ids = tokenizer(text, add_special_tokens=False)['input_ids']
    full = len(ids) // 65
    dev = torch.tensor(ids[:full * 65], dtype=torch.long).reshape(full, 65)
    probe_ids = np.sort(np.random.default_rng(plan['probe_seed']).choice(full, plan['probe_chunks'], replace=False))
    rest = np.setdiff1d(np.arange(full), probe_ids)
    target_ids = np.sort(rest[np.random.default_rng(plan['target_seed']).choice(len(rest), plan['target_chunks'], replace=False)])
    population = dict(train_chunks=len(train), eligible_dev_lines=kept, excluded_dev_lines=dropped,
                      eligible_dev_chunks=full, probe_ids=probe_ids.tolist(), target_ids=target_ids.tolist())
    return train, dev[probe_ids], dev[target_ids], population


@torch.no_grad()
def evaluate(model, target):
    total = 0.
    for start in range(0, len(target), EVAL_BATCH):
        batch = target[start:start + EVAL_BATCH]
        total += float(llm.lm_loss(model, batch)) * len(batch)
    return total / len(target)


def flat_grad(params):
    return torch.cat([p.grad.reshape(-1) for p in params]).clone()


def set_grad(params, flat):
    offset = 0
    for p in params:
        n = p.numel()
        p.grad = flat[offset:offset + n].view_as(p).clone()
        offset += n


def run_arm(seed, train, probe, target, ids, choices, arm, plan, clock=time.perf_counter):
    model = llm.load_model(seed)
    params = llm.adapter_params(model)
    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.)
    losses = {0: evaluate(model, target)}
    picked, seconds, probe_grad = [], [], None
    for step in range(plan['updates']):
        candidates = train[ids[step]]
        start = clock()
        if arm == 'random':
            c = int(choices[step])
            opt.zero_grad()
            llm.lm_loss(model, candidates[c:c + 1]).backward()
        elif arm == 'loss4':
            with torch.no_grad():
                c = int(torch.argmax(llm.lm_loss(model, candidates, 'none')))
            opt.zero_grad()
            llm.lm_loss(model, candidates[c:c + 1]).backward()
        elif arm == 'graddot4':
            if step % 4 == 0:
                opt.zero_grad()
                llm.lm_loss(model, probe).backward()
                probe_grad = flat_grad(params)
            best, c, winner = -float('inf'), None, None
            for i in range(len(candidates)):
                opt.zero_grad()
                llm.lm_loss(model, candidates[i:i + 1]).backward()
                g = flat_grad(params)
                score = float(g.dot(probe_grad))
                if not np.isfinite(score):
                    raise ValueError('Nonfinite alignment score')
                if score > best:
                    best, c, winner = score, i, g
            set_grad(params, winner)
        else:
            raise ValueError(f'Unknown arm: {arm}')
        opt.step()
        seconds.append(clock() - start)
        picked.append(c)
        if step + 1 in plan['checkpoints']:
            losses[step + 1] = evaluate(model, target)
    if not all(np.isfinite(v) for v in losses.values()):
        raise ValueError('Nonfinite evaluation')
    return dict(choices=picked, step_seconds=seconds, losses={str(k): v for k, v in losses.items()})


def schedule(seed, n_train, plan):
    rng = np.random.default_rng(seed)
    ids = rng.integers(n_train, size=(plan['updates'], plan['candidates_per_step']))
    choices = rng.integers(plan['candidates_per_step'], size=plan['updates'])
    return ids, choices


def summarize(rows, plan):
    seen = {(r['language'], r['seed'], r['arm']) for r in rows}
    if len(seen) != len(rows) or len(rows) != len(plan['languages']) * len(plan['seeds']) * len(ARMS):
        raise ValueError('Incomplete or duplicate runs')
    final = str(plan['updates'])
    languages = {}
    for lang in plan['languages']:
        group = {arm: sorted([r for r in rows if r['language'] == lang and r['arm'] == arm],
                             key=lambda r: r['seed']) for arm in ARMS}
        for arm in ARMS:
            if [r['seed'] for r in group[arm]] != plan['seeds']:
                raise ValueError('Missing seeds')
        initial = [r['losses']['0'] for r in group['random']]
        for arm in ARMS:
            if [r['losses']['0'] for r in group[arm]] != initial:
                raise ValueError('Arms do not share the initial state')
        arms = {}
        for arm in ARMS:
            arms[arm] = dict(loss={k: mean(r['losses'][k] for r in group[arm]) for k in map(str, plan['checkpoints'])},
                             median_step_seconds=median(s for r in group[arm] for s in r['step_seconds']),
                             mean_step_seconds=mean(s for r in group[arm] for s in r['step_seconds']))
        random_time = arms['random']['median_step_seconds']
        a = arms['random']['loss']['0'] - arms['random']['loss'][final]
        contrasts, decomposition = {}, {}
        for arm in ARMS[1:]:
            per_seed = {k: [x['losses'][k] - y['losses'][k] for x, y in zip(group[arm], group['random'])]
                        for k in map(str, plan['checkpoints'][1:])}
            contrasts[f'{arm}_minus_random'] = {k: dict(mean=mean(v), per_seed=v,
                                                        seeds_negative=sum(x < 0 for x in v)) for k, v in per_seed.items()}
            s = arms['random']['loss'][final] - arms[arm]['loss'][final]
            r = arms[arm]['median_step_seconds'] / random_time
            decomposition[arm] = dict(a=a, s=s, s_over_a=s / a, r=r, break_even_s_over_a=r - 1,
                                      pays_off_first_order=bool(s / a > r - 1))
        languages[lang] = dict(arms=arms, contrasts=contrasts, decomposition=decomposition)
    return dict(primary=dict(language='ar', contrast='graddot4_minus_random', updates=plan['updates'],
                             mean=languages['ar']['contrasts']['graddot4_minus_random'][final]['mean']),
                languages=languages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    args = parser.parse_args()
    paths = dict(result=args.output_dir / 'llm-selection.json', records=args.output_dir / 'llm-selection-records.jsonl')
    if any(p.exists() for p in paths.values()):
        parser.error('Choose an unused output directory to preserve recorded runs')
    plan = json.loads(PROTOCOL.read_text())
    if tuple(plan['arms']) != ARMS or plan['stage'] != 66:
        raise ValueError('Protocol differs from the implementation')
    if llm.file_sha256(llm.MODEL_DIR / 'model.safetensors') != plan['model']['safetensors_sha256']:
        raise ValueError('Model weights differ from the protocol')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    from transformers import AutoTokenizer
    import transformers
    tokenizer = AutoTokenizer.from_pretrained(llm.MODEL_DIR)
    hashes = {p: digest(Path(p)) for p in (str(PROTOCOL), str(MANIFEST), 'experiments/llm_selection.py',
                                          'experiments/llm_common.py', 'tests/test_llm_selection.py',
                                          'experiments/ud_cost_frontier.py', str(STAGE65_PROTOCOL),
                                          str(llm.MODEL_DIR / 'tokenizer.json'), str(llm.MODEL_DIR / 'config.json'))}
    hashes[str(llm.MODEL_DIR / 'model.safetensors')] = plan['model']['safetensors_sha256']
    manifest = json.loads(MANIFEST.read_text())
    prepared = {lang: prepare_language(manifest['languages'][lang]['splits'], tokenizer, plan, hashes)
                for lang in plan['languages']}
    cpu_line = next(l for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name'))
    meta = dict(sha256=hashes, cpu=cpu_line.split(':')[1].strip(), platform=platform.platform(),
                python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                transformers=transformers.__version__, torch_threads=torch.get_num_threads(),
                deterministic=torch.are_deterministic_algorithms_enabled(), logical_cpus=os.cpu_count())
    rows = []
    start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with paths['records'].open('x') as records:
        for lang in plan['languages']:
            train, probe, target, _ = prepared[lang]
            for i, seed in enumerate(plan['seeds']):
                ids, choices = schedule(seed, len(train), plan)
                order = ARMS[i % 3:] + ARMS[:i % 3]
                for arm in order:
                    row = run_arm(seed, train, probe, target, ids, choices, arm, plan)
                    row.update(language=lang, seed=seed, arm=arm, candidate_ids=ids.tolist())
                    rows.append(row)
                    records.write(json.dumps(row, allow_nan=False) + '\n')
                    records.flush()
                    print(f'{lang} seed {seed} {arm}: loss64 {row["losses"][str(plan["updates"])]:.5f}', flush=True)
    result = dict(stage=66, protocol=plan, provenance=meta,
                  output_sha256={paths['records'].name: digest(paths['records'])},
                  populations={lang: prepared[lang][3] for lang in plan['languages']},
                  summary=summarize(rows, plan), runs=len(rows), wall_seconds=time.perf_counter() - start)
    with paths['result'].open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write('\n')


if __name__ == '__main__':
    main()

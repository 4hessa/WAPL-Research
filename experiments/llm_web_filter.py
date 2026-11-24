"""Stage 73: offline alignment filtering of real Arabic web text toward ArabicMMLU (Colab GPU).

Implements research/llm-web-filter-protocol.json. Numerics, LoRA and optimizer are
those of stages 69-72 (experiments.llm_gpu); the filter scores the pool once at the
seed's initial adapter and charges the full scoring time to the run.
"""
import argparse
import gc
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
from torch import nn

from experiments import llm_common as llm
from experiments import llm_gpu as s69
from experiments.llm_budget import PerExampleCapture
from experiments.llm_selection import flat_grad

PROTOCOL = Path('research/llm-web-filter-protocol.json')
MODEL = 'qwen2.5-1.5b'
ARMS = ('random', 'align_grad')
FILTERS = ('align_grad',)
POOLS = ('kept', 'raw')
BUDGET = 2400
POOL_SIZE, KEEP, RAW_KEPT = 2048, 512, 1024
TARGET_ITEMS = 512
CHUNK, NGRAM = 65, 8
PROMPT_MAX, CONT_MAX = 192, 64
REMOVED_ROW_GROUPS = 20
CAL_SEED, CAL_STEPS = 73999, 24
EPS = 0.01
GRAD_BATCH, FD_BATCH, EVAL_BATCH, PROBE_BATCH = 16, 16, 32, 8
DIGEST_STEP = 50
KEPT_SHA256 = 'a84851f790551b08c141eab938448961812582629dbbabc737d997d519b3ed39'
REMOVED_SHA256 = '2c3b8cf5ad625369fffbab09ba7b6dc81578c436216fd5a29ce547f8eeb29295'
DEV_SHA256 = '3c72bd01f3acf6c1ed8acf28ba344ea739fcf9ee7f999802ced958e3b4d8b5dc'
TEST_SHA256 = 'c4ba4d6c6903d2b7b4415f18c9bb3173ba55c1344a91339769deba931d0c2782'
LETTERS = 'ABCDE'


# ---------------------------------------------------------------- data

def _present(value):
    return value is not None and not (isinstance(value, float) and math.isnan(value)) and str(value).strip() != ''


def build_items(records, tokenizer):
    """ArabicMMLU rows (dicts) -> items with prompt ids, option continuation ids and the correct index."""
    items = []
    for row in records:
        prompt = (str(row['Context']) + '\n' if _present(row.get('Context')) else '') + \
            'سؤال: ' + str(row['Question']) + '\nالجواب:'
        options = [str(row[f'Option {j}']) for j in range(1, 6) if _present(row.get(f'Option {j}'))]
        correct = LETTERS.index(str(row['Answer Key']).strip())
        if correct >= len(options):
            raise ValueError(f'Answer key outside options for item {row.get("ID")}')
        p = tokenizer(prompt, add_special_tokens=False)['input_ids'][-PROMPT_MAX:]
        conts = [tokenizer(' ' + o, add_special_tokens=False)['input_ids'][:CONT_MAX] for o in options]
        if not p or any(len(c) == 0 for c in conts):
            raise ValueError(f'Empty prompt or option for item {row.get("ID")}')
        items.append(dict(id=int(row['ID']), prompt=p, options=conts, correct=correct))
    return items


def sample_target(n_test, size=TARGET_ITEMS, seed=7302):
    return np.sort(np.random.default_rng(seed).choice(n_test, size, replace=False))


def ngrams(ids, n=NGRAM):
    return {tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)}


def banned_ngrams(items):
    out = set()
    for it in items:
        out |= ngrams(it['prompt'] + it['options'][it['correct']])
    return out


def first_tokens(texts, tokenizer, length=CHUNK, chars=4000):
    """First `length` tokens of every text that has at least `length` tokens, with document positions."""
    rows, chunks = [], []
    for i, text in enumerate(texts):
        if not text:
            continue
        ids = tokenizer(text[:chars], add_special_tokens=False)['input_ids']
        if len(text) > chars and len(ids) < length + 16:  # truncation could matter only for extreme texts
            ids = tokenizer(text, add_special_tokens=False)['input_ids']
        if len(ids) >= length:
            rows.append(i)
            chunks.append(ids[:length])
    return rows, chunks


def decontaminate(rows, chunks, banned):
    keep = [k for k, c in enumerate(chunks) if not (ngrams(c) & banned)]
    return [rows[k] for k in keep], [chunks[k] for k in keep], len(chunks) - len(keep)


def build_pools(kept_rows, kept_chunks, removed_rows, removed_chunks):
    if len(kept_chunks) < POOL_SIZE or len(removed_chunks) < POOL_SIZE - RAW_KEPT:
        raise ValueError('Not enough eligible web chunks')
    k = np.random.default_rng(7311).choice(len(kept_chunks), POOL_SIZE, replace=False)
    r = np.random.default_rng(7312).choice(len(removed_chunks), POOL_SIZE - RAW_KEPT, replace=False)
    kept = torch.tensor([kept_chunks[i] for i in k], dtype=torch.long)
    removed = torch.tensor([removed_chunks[i] for i in r], dtype=torch.long)
    pools = dict(kept=kept, raw=torch.cat([kept[:RAW_KEPT], removed]))
    population = dict(kept_documents=[int(kept_rows[i]) for i in k], removed_documents=[int(removed_rows[i]) for i in r],
                      pool_sha256={name: tensor_sha256(t) for name, t in pools.items()})
    return pools, population


def tensor_sha256(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def schedule(k, seed, length=BUDGET):
    rng = np.random.default_rng(seed + 1000 * k)
    rand = rng.integers(POOL_SIZE, size=length)
    filt = rng.integers(KEEP, size=length)
    return rand, filt, hashlib.sha256(rand.tobytes() + filt.tobytes()).hexdigest()


def keep_top(scores, keep=KEEP):
    """Indices of the `keep` largest scores; ties go to the lower index."""
    return np.sort(np.argsort(-np.asarray(scores, dtype=np.float64), kind='stable')[:keep])


# ---------------------------------------------------------------- task loss

def _sequences(items, correct_only):
    seqs = []
    for i, it in enumerate(items):
        for j, c in enumerate(it['options']):
            if correct_only and j != it['correct']:
                continue
            seqs.append((i, j, it['prompt'] + c, len(it['prompt']), len(c)))
    return seqs


def _batch(seqs, device):
    width = max(len(s[2]) for s in seqs)
    ids = torch.zeros((len(seqs), width), dtype=torch.long)
    mask = torch.zeros((len(seqs), width - 1))
    for b, (_, _, toks, p, c) in enumerate(seqs):
        ids[b, :len(toks)] = torch.tensor(toks)
        mask[b, p - 1:p - 1 + c] = 1.  # logits at t predict token t+1
    return ids.to(device), mask.to(device)


def continuation_nll(model, seqs, device):
    """Mean token NLL of each continuation (right padding is invisible to earlier causal positions)."""
    ids, mask = _batch(seqs, device)
    logits = model(input_ids=ids[:, :-1]).logits
    tok = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1),
                                      reduction='none').reshape(mask.shape)
    return (tok * mask).sum(dim=1) / mask.sum(dim=1)


@torch.no_grad()
def evaluate(model, items, device):
    """(mean NLL of the correct continuation, accuracy by lowest mean continuation NLL)."""
    seqs = sorted(_sequences(items, False), key=lambda s: len(s[2]))
    nll = {}
    for start in range(0, len(seqs), EVAL_BATCH):
        part = seqs[start:start + EVAL_BATCH]
        for s, v in zip(part, continuation_nll(model, part, device).tolist()):
            nll[(s[0], s[1])] = v
    correct, hits = [], 0
    for i, it in enumerate(items):
        vals = [nll[(i, j)] for j in range(len(it['options']))]
        correct.append(vals[it['correct']])
        hits += int(int(np.argmin(vals)) == it['correct'])
    loss, acc = mean(correct), hits / len(items)
    if not math.isfinite(loss):
        raise ValueError('Nonfinite evaluation')
    return loss, acc


def probe_gradient(model, items, device):
    params = llm.adapter_params(model)
    for p in params:
        p.grad = None
    seqs = _sequences(items, True)
    for start in range(0, len(seqs), PROBE_BATCH):
        part = seqs[start:start + PROBE_BATCH]
        (continuation_nll(model, part, device).sum() / len(seqs)).backward()
    g = flat_grad(params)
    for p in params:
        p.grad = None
    return g


# ---------------------------------------------------------------- scoring

def score_grad(model, pool, probe_grad, capture, batch=GRAD_BATCH):
    params = llm.adapter_params(model)
    out = []
    for start in range(0, len(pool), batch):
        for p in params:
            p.grad = None
        capture.active = True
        llm.lm_loss(model, pool[start:start + batch], 'none').sum().backward()
        capture.active = False
        out.append(capture.per_example() @ probe_grad)
    for p in params:
        p.grad = None
    scores = torch.cat(out)
    if not torch.isfinite(scores).all():
        raise ValueError('Nonfinite alignment score')
    return scores


@torch.no_grad()
def score_fd(model, pool, probe_grad, eps=EPS, batch=FD_BATCH):
    params = llm.adapter_params(model)
    u = probe_grad / probe_grad.norm()
    base = [p.detach().clone() for p in params]

    def shifted(sign):
        offset = 0
        for p, b in zip(params, base):
            n = p.numel()
            p.copy_(b + sign * eps * u[offset:offset + n].view_as(p))
            offset += n
        return torch.cat([llm.lm_loss(model, pool[s:s + batch], 'none') for s in range(0, len(pool), batch)])

    plus, minus = shifted(1.), shifted(-1.)
    for p, b in zip(params, base):
        p.copy_(b)
    scores = (plus - minus) / (2 * eps)
    if not torch.isfinite(scores).all():
        raise ValueError('Nonfinite alignment score')
    return scores


def score(model, arm, pool, probe_items, device, clock=time.perf_counter):
    """Full scoring of the pool (probe gradient included); returns scores and charged seconds."""
    capture = PerExampleCapture(model) if arm == 'align_grad' else None
    s69.sync(device)
    start = clock()
    g = probe_gradient(model, probe_items, device)
    scores = score_grad(model, pool, g, capture) if arm == 'align_grad' else score_fd(model, pool, g)
    s69.sync(device)
    return scores.double().cpu().numpy(), clock() - start


# ---------------------------------------------------------------- training

def adapter_digest(model):
    return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in llm.adapter_params(model))).hexdigest()


def train(model, data, order, updates, checkpoints, evaluator, device, clock=time.perf_counter):
    params = llm.adapter_params(model)
    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.)
    losses, accs, seconds, digest = {}, {}, [], None
    if evaluator is not None and 0 in checkpoints:
        losses['0'], accs['0'] = evaluator(model)
    for step in range(updates):
        s69.sync(device)
        start = clock()
        opt.zero_grad()
        llm.lm_loss(model, data[order[step]:order[step] + 1]).backward()
        opt.step()
        s69.sync(device)
        seconds.append(clock() - start)
        if step + 1 == DIGEST_STEP:
            digest = adapter_digest(model)
        if evaluator is not None and step + 1 in checkpoints:
            losses[str(step + 1)], accs[str(step + 1)] = evaluator(model)
    return dict(losses=losses, accuracy=accs, step_seconds=seconds, adapter_digest_50=digest)


def budgets(costs):
    return {arm: math.floor(BUDGET - costs[arm] / costs['random'] + 1e-9) for arm in FILTERS}


def agreement(grad_scores, fd_scores):
    a, b = keep_top(grad_scores), keep_top(fd_scores)
    return dict(spearman=spearman(grad_scores, fd_scores), kept_overlap=len(set(a.tolist()) & set(b.tolist())) / KEEP)


# ---------------------------------------------------------------- summary

def bootstrap(values):
    draws = np.random.default_rng(7303).choice(values, (10000, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [.025, .975])]


def spearman(a, b):
    ra, rb = (np.argsort(np.argsort(x, kind='stable'), kind='stable').astype(float) for x in (a, b))
    return float(np.corrcoef(ra, rb)[0, 1])


def summarize(rows, calibration, plan):
    n = calibration['updates']
    final = str(BUDGET)
    out = {}
    for pool in POOLS:
        group = {arm: sorted([r for r in rows if (r['pool'], r['arm']) == (pool, arm)], key=lambda r: r['seed'])
                 for arm in ARMS}
        for arm in ARMS:
            if [r['seed'] for r in group[arm]] != plan['seeds']:
                raise ValueError(f'Incomplete grid for {pool} {arm}')
        rand = group['random']
        for arm in FILTERS:
            if [r['losses']['0'] for r in group[arm]] != [r['losses']['0'] for r in rand]:
                raise ValueError('Arms do not share the initial state')
        entry = dict(loss0=mean(r['losses']['0'] for r in rand), accuracy0=mean(r['accuracy']['0'] for r in rand),
                     random_final=mean(r['losses'][final] for r in rand),
                     random_minus_no_update=mean(r['losses'][final] - r['losses']['0'] for r in rand),
                     random_accuracy_final=mean(r['accuracy'][final] for r in rand), arms={})
        for arm in FILTERS:
            k = str(n[arm])
            g = group[arm]
            if n[arm] < 1:
                entry['arms'][arm] = dict(updates=n[arm], infeasible=True)
                continue
            charged = [a['losses'][k] - r['losses'][final] for a, r in zip(g, rand)]
            amortized = [a['losses'][final] - r['losses'][final] for a, r in zip(g, rand)]
            sel = [a['losses'][k] - r['losses'][k] for a, r in zip(g, rand)]
            acc = [a['accuracy'][k] - r['accuracy'][final] for a, r in zip(g, rand)]
            acc_am = [a['accuracy'][final] - r['accuracy'][final] for a, r in zip(g, rand)]
            entry['arms'][arm] = dict(
                updates=n[arm], charged=mean(charged), charged_ci95=bootstrap(charged), charged_per_seed=charged,
                seeds_negative=sum(d < 0 for d in charged), amortized=mean(amortized),
                amortized_ci95=bootstrap(amortized), selection_term=mean(sel), throughput_term=mean(charged) - mean(sel),
                accuracy_charged=mean(acc), accuracy_charged_ci95=bootstrap(acc), accuracy_amortized=mean(acc_am),
                removed_share=mean(a['removed_share'] for a in g), scoring_seconds=mean(a['scoring_seconds'] for a in g))
        out[pool] = entry
    p = out['raw']['arms']['align_grad']
    pred = {'raw align_grad negative': p.get('charged', 0) < 0,
            'kept align_grad positive': out['kept']['arms']['align_grad'].get('charged', 0) > 0}
    return dict(primary=dict(pool='raw', arm='align_grad', contrast='charged', mean=p.get('charged'),
                             ci95=p.get('charged_ci95')),
                prediction_holds=pred, costs=calibration['costs'], updates=n, device=calibration['device'],
                scoring_in_random_steps={a: calibration['costs'][a] / calibration['costs']['random'] for a in ('align_grad', 'align_fd')},
                fd_vs_grad=calibration['fd_vs_grad'],
                pools=out)


# ---------------------------------------------------------------- main

def load_inputs(args, tokenizer):
    import pandas as pd
    import pyarrow.parquet as pq
    for path, digest in ((args.kept, KEPT_SHA256), (args.removed, REMOVED_SHA256),
                         (args.mmlu / 'dev.csv', DEV_SHA256), (args.mmlu / 'test.csv', TEST_SHA256)):
        if s69.sha256(path) != digest:
            raise ValueError(f'{path} differs from the protocol')
    dev = pd.read_csv(args.mmlu / 'dev.csv', keep_default_na=True).to_dict('records')
    test = pd.read_csv(args.mmlu / 'test.csv', keep_default_na=True).to_dict('records')
    target_rows = sample_target(len(test))
    probe_items = build_items(dev, tokenizer)
    target_items = build_items([test[i] for i in target_rows], tokenizer)
    banned = banned_ngrams(probe_items + target_items)
    kept_texts = pq.read_table(args.kept, columns=['text'])['text'].to_pylist()
    removed_texts = pq.ParquetFile(args.removed).read_row_groups(list(range(REMOVED_ROW_GROUPS)),
                                                                  columns=['text'])['text'].to_pylist()
    kr, kc = first_tokens(kept_texts, tokenizer)
    rr, rc = first_tokens(removed_texts, tokenizer)
    kr, kc, k_drop = decontaminate(kr, kc, banned)
    rr, rc, r_drop = decontaminate(rr, rc, banned)
    pools, population = build_pools(kr, kc, rr, rc)
    population.update(kept_documents_read=len(kept_texts), removed_documents_read=len(removed_texts),
                      kept_eligible=len(kc) + k_drop, removed_eligible=len(rc) + r_drop,
                      kept_decontaminated=k_drop, removed_decontaminated=r_drop,
                      probe_items=[it['id'] for it in probe_items], target_items=[it['id'] for it in target_items],
                      banned_ngrams=len(banned))
    return probe_items, target_items, pools, population


def main():
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')  # before CUDA initialises
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--kept', type=Path, required=True)
    parser.add_argument('--removed', type=Path, required=True)
    parser.add_argument('--mmlu', type=Path, required=True, help='Directory with ArabicMMLU All/dev.csv and test.csv')
    parser.add_argument('--summary', type=Path, default=Path('results/llm-web-filter.json'))
    args = parser.parse_args()
    plan = json.loads(PROTOCOL.read_text())
    if plan['stage'] != 73 or tuple(plan['arms']) != ARMS:
        raise ValueError('Protocol differs from the implementation')
    s69.setup_numerics()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'
    from transformers import AutoTokenizer
    import transformers
    spec = json.loads(s69.PROTOCOL.read_text())['models'][MODEL]
    model_dir = args.models_dir / MODEL
    if s69.sha256(model_dir / 'model.safetensors') != spec['safetensors_sha256']:
        raise ValueError('Model weights differ from the protocol')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    probe_items, target_items, pools, population = load_inputs(args, tokenizer)
    pools = {k: v.to(device) for k, v in pools.items()}
    evaluator = lambda m: evaluate(m, target_items, device)
    journal = s69.read_journal(args.journal)
    done = {(r['pool'], r['seed'], r['arm']) for r in journal if r.get('kind') == 'run'}
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
            rand, _, _ = schedule(POOLS.index('raw'), CAL_SEED, CAL_STEPS)
            row = train(s69.load(model_dir, CAL_SEED, device), pools['raw'], rand, CAL_STEPS, [], None, device)
            costs = dict(random=median(row['step_seconds'][4:]))
            cal_scores = {}
            for arm in ('align_grad', 'align_fd'):
                cal_scores[arm], costs[arm] = score(s69.load(model_dir, CAL_SEED, device), arm, pools['raw'],
                                                    probe_items, device)
            calibration = dict(kind='calibration', model=MODEL, device=device_name, costs=costs,
                               updates=budgets(costs), population=population,
                               fd_vs_grad=agreement(cal_scores['align_grad'], cal_scores['align_fd']),
                               scores={a: v.tolist() for a, v in cal_scores.items()})
            log(calibration)
        else:
            if calibration['device'] != device_name:
                raise ValueError(f'Resumed on {device_name}; calibrated on {calibration["device"]}')
            if calibration['population'] != population:
                raise ValueError('Samples differ from the journaled calibration')
        n = calibration['updates']
        print(MODEL, device_name, 'costs', calibration['costs'], 'updates', n, flush=True)
        total = len(POOLS) * len(plan['seeds']) * len(ARMS)
        for k, pool in enumerate(POOLS):
            data = pools[pool]
            for i, seed in enumerate(plan['seeds']):
                rand, filt, digest = schedule(k, seed)
                for arm in ARMS[i % 2:] + ARMS[:i % 2]:
                    if (pool, seed, arm) in done:
                        continue
                    model = s69.load(model_dir, seed, device)
                    extra = {}
                    if arm == 'random':
                        order = rand
                        checkpoints = {0, BUDGET} | {n[a] for a in FILTERS if n[a] >= 1}
                    else:
                        scores, seconds = score(model, arm, data, probe_items, device)
                        kept = keep_top(scores)
                        order = kept[filt]
                        checkpoints = {0, BUDGET} | ({n[arm]} if n[arm] >= 1 else set())
                        extra = dict(scores=scores.tolist(), kept=kept.tolist(), scoring_seconds=seconds,
                                     kept_sha256=hashlib.sha256(kept.tobytes()).hexdigest(),
                                     removed_share=float(np.mean(kept >= RAW_KEPT)) if pool == 'raw' else 0.)
                    row = train(model, data, order, BUDGET, checkpoints, evaluator, device)
                    row.update(extra, kind='run', model=MODEL, pool=pool, seed=seed, arm=arm, device=device_name,
                               schedule_sha256=digest)
                    log(row)
                    done.add((pool, seed, arm))
                    del model  # capture hooks form a reference cycle; free GPU memory before the next run
                    gc.collect()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    print(f'{len(done)}/{total} runs complete', flush=True)  # outcomes are not printed
    journal = s69.read_journal(args.journal)
    rows = [r for r in journal if r.get('kind') == 'run']
    result = dict(stage=73, protocol=plan, journal_sha256=s69.sha256(args.journal),
                  environments=[r for r in journal if r.get('kind') == 'environment'],
                  calibration=calibration, summary=summarize(rows, calibration, plan), runs=len(rows))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('summary written to', args.summary)


if __name__ == '__main__':
    main()

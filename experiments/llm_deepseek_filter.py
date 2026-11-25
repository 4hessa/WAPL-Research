"""Stage 77: the stage-76 one-pass offline filter on a different model family (deepseek-llm-7b-base, self-scoring).

Implements research/llm-deepseek-filter-protocol.json (with amendment 1) with the stage 73-76 building blocks.
Differences from stage 76: bfloat16 base weights with float32 LoRA factors, logits and per-example capture; 147-token
chunks in DeepSeek tokenization; the trained model scores the pool itself; M = 7968 (1,992 QA chunks), B = 7400, four
seeds, three arms. Amendment 1: decontamination (web chunks, target criteria (d) and (e)) uses the Qwen2.5-1.5B
tokenizer with token 8-grams on the decoded text, the stage 74-76 rule; no Qwen model is run. Amendment 2:
N = min(N_uncapped, 1992); the calibration journals N_uncapped and N is recomputed from the journaled costs.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
from statistics import mean, median
import time

import numpy as np
import torch

from experiments import llm_common as llm
from experiments import llm_gpu as s69
from experiments import llm_once_filter as s75
from experiments import llm_qa_mix as s74
from experiments import llm_web_filter as s73
from experiments.llm_budget import PerExampleCapture
from experiments.llm_selection import flat_grad

PROTOCOL = Path('research/llm-deepseek-filter-protocol.json')
MODEL = 'deepseek-llm-7b-base'
REVISION = '7683fea62db869066ddaff6a41d032262c490d4f'
WEIGHTS_SHA256 = {
    'pytorch_model-00001-of-00002.bin': '950392f338e46bc0807acfedd86dc7da12745b8835b78e704e98d7e26d2e618d',
    'pytorch_model-00002-of-00002.bin': '9883470e0517cadf59a6f163e0c58e0854de662de0f7a68a1b28d95e209de34c',
}
QWEN_TOKENIZER = 'qwen2.5-1.5b'  # amendment 1: decontamination only
QWEN_TOKENIZER_SHA256 = 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'
ARMS = ('random', 'align_once', 'random_once')
CHUNK, BUDGET, N_QA, N_WEB = 147, 7400, 1992, 2988
POOL_SIZE = N_QA + 2 * N_WEB  # 7968
CAL_SEED, TARGET_SEED, TARGET_ITEMS, BOOT_SEED = 77999, 7722, 512, 7703
QA_ORDER_SEED, WEB_SEEDS = 7411, (7711, 7712)
N_MIN = 1000  # amendment 2: abort only if N_uncapped < 1000; N is capped at N_QA
PROBE_BATCH, SCORE_BATCHES, EVAL_BATCH = 2, (8, 4), 8
STAGE75_JOURNAL = Path('results/llm-once-filter-journal.jsonl')
STAGE76_JOURNAL = Path('results/llm-cheap-filter-journal.jsonl')
INTERIM = 'results/stage77-interim'


# ---------------------------------------------------------------- numerics: bf16 base, float32 adapters and logits

class MixedLoRALinear(llm.LoRALinear):
    """LoRA on a bfloat16 base: the adapter path runs in float32 and is cast back before the residual add.
    For a float32 base this equals llm.LoRALinear.forward exactly."""

    def forward(self, x):
        return self.base(x) + ((x.to(self.A.dtype) @ self.A.t() @ self.B.t()) * self.scale).to(x.dtype)


def add_lora(model, seed, rank=8, alpha=16, targets=('q_proj', 'v_proj')):
    """llm.add_lora with MixedLoRALinear (same initialisation order and generator), logits cast to float32."""
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    g = torch.Generator().manual_seed(seed)
    for layer in model.model.layers:
        for name in targets:
            setattr(layer.self_attn, name, MixedLoRALinear(getattr(layer.self_attn, name), rank, alpha, g))
    model.lm_head.register_forward_hook(lambda m, i, o: o.float())
    model.eval()
    return model


class FloatCapture(PerExampleCapture):
    def per_example(self):
        self.inputs = {m: x.float() for m, x in self.inputs.items()}
        self.grads = {m: g.float() for m, g in self.grads.items()}
        return super().per_example()


def load(model_dir, seed, device):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    return add_lora(model, seed).to(device)


def check_model(model_dir):
    for name, digest in WEIGHTS_SHA256.items():
        if s69.sha256(model_dir / name) != digest:
            raise ValueError(f'{name} differs from the protocol')
    return s69.sha256(model_dir / 'tokenizer.json')


# ---------------------------------------------------------------- evaluation and scoring with explicit batches

@torch.no_grad()
def evaluate(model, items, device, batch=EVAL_BATCH):
    """experiments.llm_web_filter.evaluate with an explicit batch size."""
    seqs = sorted(s73._sequences(items, False), key=lambda s: len(s[2]))
    nll = {}
    for start in range(0, len(seqs), batch):
        part = seqs[start:start + batch]
        for s, v in zip(part, s73.continuation_nll(model, part, device).tolist()):
            nll[(s[0], s[1])] = v
    correct, hits = [], 0
    for i, it in enumerate(items):
        vals = [nll[(i, j)] for j in range(len(it['options']))]
        correct.append(vals[it['correct']])
        hits += int(int(np.argmin(vals)) == it['correct'])
    loss = mean(correct)
    if not math.isfinite(loss):
        raise ValueError('Nonfinite evaluation')
    return loss, hits / len(items)


def probe_gradient(model, items, device, batch=PROBE_BATCH):
    """experiments.llm_web_filter.probe_gradient with an explicit batch size."""
    params = llm.adapter_params(model)
    for p in params:
        p.grad = None
    seqs = s73._sequences(items, True)
    for start in range(0, len(seqs), batch):
        part = seqs[start:start + batch]
        (s73.continuation_nll(model, part, device).sum() / len(seqs)).backward()
    g = flat_grad(params)
    for p in params:
        p.grad = None
    return g


def score(model, pool, probe_items, device, batch, clock=time.perf_counter):
    """Full self-scoring (probe gradient included) by gradient alignment; returns scores and charged seconds."""
    capture = FloatCapture(model)
    s69.sync(device)
    start = clock()
    g = probe_gradient(model, probe_items, device)
    scores = s73.score_grad(model, pool, g, capture, batch=batch)
    s69.sync(device)
    return scores.double().cpu().numpy(), clock() - start


# ---------------------------------------------------------------- data

def qa_chunks(trained_rows, tokenizer):
    items = [tokenizer(s74.mmlu_qa_text(r), add_special_tokens=False)['input_ids'] for r in trained_rows]
    members = []
    chunks = s74.pack(items, np.random.default_rng(QA_ORDER_SEED).permutation(len(items)), length=CHUNK,
                      members=members)
    if len(chunks) != N_QA:
        raise ValueError(f'{len(chunks)} QA chunks at {CHUNK} tokens, protocol {N_QA}')
    return torch.tensor(chunks, dtype=torch.long), members


class Chunks(list):
    """A list of token lists that experiments.llm_once_filter.target_candidates can read like a pool tensor."""

    def tolist(self):
        return self


def qwen_view(chunks, tokenizer, decon_tokenizer):
    """Amendment 1: each DeepSeek chunk decoded to text and tokenized by the decontamination (Qwen) tokenizer."""
    return Chunks(decon_tokenizer(tokenizer.decode(c), add_special_tokens=False)['input_ids'] for c in chunks)


def decontaminate(rows, chunks, view, banned):
    """experiments.llm_web_filter.decontaminate with the n-gram test applied to `view` (same order as `chunks`)."""
    keep = [k for k, v in enumerate(view) if not (s73.ngrams(v) & banned)]
    return [rows[k] for k in keep], [chunks[k] for k in keep], len(chunks) - len(keep)


def check_decon_tokenizer(models_dir):
    if s69.sha256(models_dir / QWEN_TOKENIZER / 'tokenizer.json') != QWEN_TOKENIZER_SHA256:
        raise ValueError('Qwen tokenizer differs from amendment 1')
    return QWEN_TOKENIZER_SHA256


def web_draws(n_kept, n_removed):
    if min(n_kept, n_removed) < N_WEB:
        raise SystemExit(f'Eligible web chunks {n_kept}, {n_removed}: fewer than {N_WEB}')
    return tuple(np.random.default_rng(s).choice(n, N_WEB, replace=False) for s, n in zip(WEB_SEEDS, (n_kept, n_removed)))


def qa_share(indices):
    return float(np.mean(np.asarray(indices) < N_QA))


def excluded_rows(test):
    """Rows of every earlier evaluation set: stage-73 target, stage-74 pilot and target, stage-75 and -76 targets."""
    pilot, target74 = s74.item_splits(len(test))
    ids = set()
    for path in (STAGE75_JOURNAL, STAGE76_JOURNAL):
        ids |= set(next(r for r in s69.read_journal(path) if r.get('kind') == 'calibration')['population']['target_items'])
    rows = set(s73.sample_target(len(test)).tolist()) | set(pilot.tolist()) | set(target74.tolist())
    return rows | {i for i, r in enumerate(test) if int(r['ID']) in ids}


def load_inputs(args, tokenizer, decon_tokenizer):
    """Probe, target items, pool (QA 0-1991, kept web 1992-4979, removed web 4980-7967) and the population record.
    `tokenizer` is the DeepSeek tokenizer (training, probe, target); `decon_tokenizer` the Qwen one (amendment 1)."""
    import pandas as pd
    import pyarrow.parquet as pq
    for path, digest in ((args.kept, s73.KEPT_SHA256), (args.removed, s73.REMOVED_SHA256),
                         (args.mmlu / 'dev.csv', s73.DEV_SHA256), (args.mmlu / 'test.csv', s73.TEST_SHA256)):
        if s69.sha256(path) != digest:
            raise ValueError(f'{path} differs from the protocol')
    dev = pd.read_csv(args.mmlu / 'dev.csv').to_dict('records')
    test = pd.read_csv(args.mmlu / 'test.csv').to_dict('records')
    probe = s73.build_items(dev, tokenizer)
    trained_ids = set(json.loads(s75.LEDGER.read_text())['additional_sources']['ArabicMMLU']['stage_74']
                      ['used_for_training']['ids'])
    by_id = {int(r['ID']): r for r in test}
    qa, members = qa_chunks([by_id[i] for i in sorted(trained_ids)], tokenizer)
    banned = s74.content_banned(dev + test, decon_tokenizer)
    kept_texts = pq.read_table(args.kept, columns=['text'])['text'].to_pylist()
    removed_texts = pq.ParquetFile(args.removed).read_row_groups(list(range(s73.REMOVED_ROW_GROUPS)),
                                                                  columns=['text'])['text'].to_pylist()
    web = {}
    for name, texts in (('kept', kept_texts), ('removed', removed_texts)):
        rows, chunks = s73.first_tokens(texts, tokenizer, length=CHUNK)
        web[name] = decontaminate(rows, chunks, qwen_view(chunks, tokenizer, decon_tokenizer), banned)
    (kr, kc, k_drop), (rr, rc, r_drop) = web['kept'], web['removed']
    k_idx, r_idx = web_draws(len(kc), len(rc))
    pool = torch.cat([qa, torch.tensor([kc[i] for i in k_idx], dtype=torch.long),
                      torch.tensor([rc[i] for i in r_idx], dtype=torch.long)])
    excluded = excluded_rows(test)
    counts, eligible = s75.target_candidates(test, decon_tokenizer, [qwen_view(pool.tolist(), tokenizer, decon_tokenizer)],
                                             trained_ids, excluded)
    if len(eligible) < TARGET_ITEMS:
        raise SystemExit('Fewer than 512 eligible target rows')
    rows = np.asarray(eligible)
    rows = np.sort(rows[np.random.default_rng(TARGET_SEED).choice(len(rows), TARGET_ITEMS, replace=False)])
    target = s73.build_items([test[i] for i in rows], tokenizer)
    population = dict(qa_chunks=len(qa), qa_pool_sha256=s73.tensor_sha256(qa),
                      qa_items_packed=sum(len(m) for m in members), web_eligible=[len(kc), len(rc)],
                      web_decontaminated=[k_drop, r_drop], kept_documents=[int(kr[i]) for i in k_idx],
                      removed_documents=[int(rr[i]) for i in r_idx], pool_sha256=s73.tensor_sha256(pool),
                      trained_ids=len(trained_ids), excluded_evaluated_rows=len(excluded), target_pool=counts,
                      target_items=[it['id'] for it in target])
    return probe, target, pool, population


# ---------------------------------------------------------------- schedules

def schedule(seed, n):
    rng = np.random.default_rng(seed)
    rand = rng.integers(POOL_SIZE, size=BUDGET)
    subset_n = np.sort(rng.choice(POOL_SIZE, n, replace=False))
    once = np.concatenate([rng.permutation(n) for _ in range(math.ceil(BUDGET / n))])[:BUDGET]
    digest = hashlib.sha256(b''.join(a.tobytes() for a in (rand, subset_n, once))).hexdigest()
    return dict(rand=rand, subset_n=subset_n, once=once, sha256=digest)


def arm_order(i):
    return ARMS[i % len(ARMS):] + ARMS[:i % len(ARMS)]


def run_grid(seeds):
    return [(seed, arm) for i, seed in enumerate(seeds) for arm in arm_order(i)]


def views(order, members, steps):
    counts = np.bincount(np.asarray(order[:steps]), minlength=POOL_SIZE)[np.asarray(members)]
    return dict(max=int(counts.max()), mean=steps / len(members))


def training_order(arm, sch, kept=None):
    if arm == 'random':
        return sch['rand'], np.arange(POOL_SIZE)
    members = dict(align_once=kept, random_once=sch['subset_n'])[arm]
    return members[sch['once']], members


def random_qa(rand, steps):
    first = np.asarray(rand[:steps])
    return dict(drawn=int((first < N_QA).sum()), distinct=int(len(set(first[first < N_QA].tolist()))))


def uncapped_updates(costs):
    return int(np.floor(BUDGET - costs['self_align_grad'] / costs['random'] + 1e-9))


def updates(costs):
    """Amendment 2: N = min(N_uncapped, 1992), the QA supply."""
    return min(uncapped_updates(costs), N_QA)


# ---------------------------------------------------------------- summary

def bootstrap(values):
    draws = np.random.default_rng(BOOT_SEED).choice(values, (10000, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [.025, .975])]


def contrast(values):
    return dict(mean=mean(values), ci95=bootstrap(values), per_seed=values, seeds_negative=sum(v < 0 for v in values))


def summarize(rows, calibration, plan):
    if calibration['updates'] != uncapped_updates(calibration['costs']):
        raise ValueError('Journaled N_uncapped differs from the costs')
    n, B = str(updates(calibration['costs'])), str(BUDGET)
    seeds = plan['design']['seeds']
    g = {arm: sorted([r for r in rows if r['arm'] == arm], key=lambda r: r['seed']) for arm in ARMS}
    for arm in ARMS:
        if [r['seed'] for r in g[arm]] != seeds:
            raise ValueError(f'Incomplete grid for {arm}')
        if [r['losses']['0'] for r in g[arm]] != [r['losses']['0'] for r in g['random']]:
            raise ValueError('Arms do not share the initial state')
    R, A, RO = (g[a] for a in ARMS)
    d = lambda x, kx, y, ky, key='losses': [a[key][kx] - b[key][ky] for a, b in zip(x, y)]
    primary, sel = d(A, n, R, B), d(A, n, R, n)
    out = dict(
        loss0=mean(r['losses']['0'] for r in R), accuracy0=mean(r['accuracy']['0'] for r in R),
        means={arm: {k: mean(r['losses'][k] for r in g[arm]) for k in ('0', n, B)} for arm in ARMS},
        accuracy_means={arm: {k: mean(r['accuracy'][k] for r in g[arm]) for k in ('0', n, B)} for arm in ARMS},
        charged=contrast(primary), matched=contrast(d(A, n, RO, n)), one_view=contrast(d(RO, n, R, n)),
        selection_term=mean(sel), throughput_term=mean(primary) - mean(sel),
        amortized=contrast(d(A, B, R, B)),
        accuracy_charged=contrast(d(A, n, R, B, 'accuracy')), accuracy_matched=contrast(d(A, n, RO, n, 'accuracy')),
        qa_share={arm: mean(r['qa_share'] for r in g[arm]) for arm in ARMS if arm != 'random'},
        random_qa={k: {f: mean(r['random_qa'][k][f] for r in R) for f in ('drawn', 'distinct')} for k in (n, B)},
        random_minus_no_update=mean(d(R, B, R, '0')),
        scoring_seconds=mean(r['scoring_seconds'] for r in A))
    return dict(primary=dict(contrast='align_once at N minus random at 7400', mean=out['charged']['mean'],
                             ci95=out['charged']['ci95']),
                prediction_holds={'qa share of align_once kept > 0.85': out['qa_share']['align_once'] > .85,
                                  'primary negative': out['charged']['mean'] < 0,
                                  'align_once minus random_once at N negative': out['matched']['mean'] < 0},
                costs=calibration['costs'], updates=int(n), updates_uncapped=calibration['updates'],
                unused_charged_steps=calibration['updates'] - int(n), device=calibration['device'],
                scoring_batch=calibration['scoring_batch'],
                scoring_in_random_steps=calibration['costs']['self_align_grad'] / calibration['costs']['random'],
                arms=out)


# ---------------------------------------------------------------- backup

def backup(journal_path, log_path, worktree, branch, run=subprocess.run):
    """Copy the journal (and run log) to the journal branch, commit and push; a failed push logs git's stderr and is
    retried once. Failures are reported, not fatal."""
    if worktree is None:
        return
    try:
        dest = worktree / INTERIM
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(journal_path, dest / 'journal.jsonl')
        if log_path is not None and log_path.exists():
            shutil.copy2(log_path, dest / 'run.log')
        runs = sum(r.get('kind') == 'run' for r in s69.read_journal(journal_path))
        stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        git = lambda *a: run(['git', '-C', str(worktree), *a], check=True, capture_output=True, text=True)
        git('add', INTERIM)
        if run(['git', '-C', str(worktree), 'diff', '--cached', '--quiet']).returncode == 0:
            return
        git('commit', '-q', '-m', f'Stage 77 interim journal snapshot: {runs} finished runs ({stamp})')
    except Exception as exc:
        print(f'BACKUP FAILED (commit): {type(exc).__name__}: {getattr(exc, "stderr", "") or exc}', flush=True)
        return
    for attempt in (1, 2):
        try:
            git('push', '-q', 'origin', f'HEAD:{branch}')
            print(f'backup pushed: {runs} finished runs' + (' (retry)' if attempt == 2 else ''), flush=True)
            return
        except Exception as exc:  # the next backup retries with the whole journal
            print(f'BACKUP FAILED (push attempt {attempt}): {type(exc).__name__}: '
                  f'{(getattr(exc, "stderr", "") or str(exc)).strip()}', flush=True)


# ---------------------------------------------------------------- main

def calibrate(model_dir, pool, probe, device, free):
    """c_r from 24 random steps, then full self-scoring at batch 8 (batch 4 if batch 8 runs out of memory)."""
    model = load(model_dir, CAL_SEED, device)
    rand = np.random.default_rng(CAL_SEED).integers(POOL_SIZE, size=24)
    row = s73.train(model, pool, rand, 24, set(), None, device)
    del model
    free()
    costs, oom = dict(random=median(row['step_seconds'][4:])), []
    for batch in SCORE_BATCHES:
        model = load(model_dir, CAL_SEED, device)
        try:
            _, costs['self_align_grad'] = score(model, pool, probe, device, batch)
            break
        except torch.OutOfMemoryError:
            oom.append(batch)
        finally:
            del model
            free()
    else:
        raise SystemExit('Scoring ran out of memory at every protocol batch size')
    return costs, batch, oom


def main():
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--kept', type=Path, required=True)
    parser.add_argument('--removed', type=Path, required=True)
    parser.add_argument('--mmlu', type=Path, required=True)
    parser.add_argument('--run-log', type=Path, default=None)
    parser.add_argument('--backup-worktree', type=Path, default=None)
    parser.add_argument('--backup-branch', default='colab/stage77-journal')
    parser.add_argument('--summary', type=Path, default=Path('results/llm-deepseek-filter.json'))
    args = parser.parse_args()
    plan = json.loads(PROTOCOL.read_text())
    if plan['stage'] != 77 or tuple(plan['design']['arms']) != ARMS:
        raise ValueError('Protocol differs from the implementation')
    import transformers
    from transformers import AutoTokenizer
    s69.setup_numerics()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'
    model_dir = args.models_dir / MODEL
    tokenizer_sha = check_model(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    decon_sha = check_decon_tokenizer(args.models_dir)
    probe, target, pool, population = load_inputs(args, tokenizer,
                                                  AutoTokenizer.from_pretrained(args.models_dir / QWEN_TOKENIZER))
    population.update(tokenizer_sha256=tokenizer_sha, decontamination_tokenizer_sha256=decon_sha)
    pool = pool.to(device)
    env = dict(kind='environment', time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), torch=torch.__version__,
               transformers=transformers.__version__, cuda=torch.version.cuda, python=platform.python_version(),
               device=device_name)
    journal = s69.read_journal(args.journal)
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    seeds = plan['design']['seeds']

    def free():
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    with args.journal.open('a') as out:
        def log(row):
            out.write(json.dumps(row, allow_nan=False) + '\n')
            out.flush()
            os.fsync(out.fileno())
        log(dict(env, resumed_runs=sum(r.get('kind') == 'run' for r in journal)))
        done = {(r['seed'], r['arm']) for r in journal if r.get('kind') == 'run'}
        calibration = next((r for r in journal if r.get('kind') == 'calibration'), None)
        if calibration is None:
            costs, batch, oom = calibrate(model_dir, pool, probe, device, free)
            calibration = dict(kind='calibration', model=MODEL, revision=REVISION, device=device_name, costs=costs,
                               scoring_batch=batch, scoring_oom_batches=oom, updates=uncapped_updates(costs),
                               population=population)
            log(calibration)
            backup(args.journal, args.run_log, args.backup_worktree, args.backup_branch)
        elif calibration['device'] != device_name or calibration['population'] != population:
            raise ValueError('Resumed on another device type or with different samples')
        if calibration['updates'] != uncapped_updates(calibration['costs']):
            raise ValueError('Journaled N_uncapped differs from the costs')
        if calibration['updates'] < N_MIN:
            raise SystemExit(f"N_uncapped = {calibration['updates']} below {N_MIN}: design infeasible")
        n, batch = updates(calibration['costs']), calibration['scoring_batch']
        print(MODEL, device_name, 'updates', n, 'uncapped', calibration['updates'], 'scoring batch', batch,
              flush=True)  # costs only; no outcome
        ev = lambda m: evaluate(m, target, device)
        grid = run_grid(seeds)
        for seed, arm in grid:
            if (seed, arm) in done:
                continue
            sch = schedule(seed, n)
            extra, kept = {}, None
            model = load(model_dir, seed, device)
            if arm == 'align_once':
                scores, seconds = score(model, pool, probe, device, batch)
                kept = s73.keep_top(scores, n)
                extra = dict(scores=scores.tolist(), kept=kept.tolist(), scoring_seconds=seconds, scoring_batch=batch,
                             kept_sha256=hashlib.sha256(kept.tobytes()).hexdigest())
                free()
            order, members = training_order(arm, sch, kept)
            if arm == 'random_once':
                extra['subset_sha256'] = hashlib.sha256(np.asarray(members).tobytes()).hexdigest()
            if arm == 'random':
                extra['random_qa'] = {str(t): random_qa(sch['rand'], t) for t in (n, BUDGET)}
            else:
                extra['qa_share'] = qa_share(members)
            extra['views'] = {str(t): views(order, members, t) for t in (n, BUDGET)}
            row = s73.train(model, pool, order, BUDGET, {0, n, BUDGET}, ev, device)
            row.update(extra, kind='run', model=MODEL, seed=seed, arm=arm, device=device_name,
                       schedule_sha256=sch['sha256'])
            log(row)
            done.add((seed, arm))
            del model
            free()
            print(f'{len(done)}/{len(grid)} runs complete', flush=True)  # outcomes are not printed
            backup(args.journal, args.run_log, args.backup_worktree, args.backup_branch)
    journal = s69.read_journal(args.journal)
    rows = [r for r in journal if r.get('kind') == 'run']
    result = dict(stage=77, protocol=plan, journal_sha256=s69.sha256(args.journal),
                  environments=[r for r in journal if r.get('kind') == 'environment'],
                  calibration={k: v for k, v in calibration.items() if k != 'population'},
                  summary=summarize(rows, calibration, plan), runs=len(rows))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('summary written to', args.summary)


if __name__ == '__main__':
    main()

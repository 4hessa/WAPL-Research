"""Stage 74: web pool mixed with in-format Aya QA; offline alignment filter vs random and a
repetition control (random512); phase A is a feasibility pilot on separate items.

Implements research/llm-qa-mix-protocol.json with the stage-73 building blocks.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import mean, median
import time

import numpy as np
import torch

from experiments import llm_gpu as s69
from experiments import llm_web_filter as s73

PROTOCOL = Path('research/llm-qa-mix-protocol.json')
MODEL = s73.MODEL
ARMS = ('random', 'align_grad', 'random512')
SIGMAS = (0.25, 0.1)
PILOT_SIGMAS = (1.0, 0.25)
PILOT_SEED, CAL_SEED = 74900, 74999
BUDGET, POOL_SIZE, KEEP, CHUNK = 2400, 2048, 512, 65
N_KEPT_WEB, N_REMOVED_WEB = 1843, 922
AYA_SHA256 = '51baa85043b569ba117f41516b32d1b0e4e2647fb203a08933f98a556fb1fb06'


# ---------------------------------------------------------------- data

def content_banned(records, tokenizer):
    """Token 8-grams of Question, Context and correct option of every ArabicMMLU item, each tokenized alone."""
    out = set()
    for r in records:
        fields = [r['Question']] + ([r['Context']] if s73._present(r.get('Context')) else [])
        fields.append(r[f'Option {s73.LETTERS.index(str(r["Answer Key"]).strip()) + 1}'])
        for f in fields:
            out |= s73.ngrams(tokenizer(str(f), add_special_tokens=False)['input_ids'])
    return out


def qa_text(q, a):
    return 'سؤال: ' + str(q).strip() + '\nالجواب: ' + str(a).strip() + '\n\n'


def clean_qa(pairs, tokenizer, banned):
    """Formatted token ids of the QA pairs whose inputs and targets share no banned 8-gram."""
    kept, dropped = [], 0
    for q, a in pairs:
        if any(s73.ngrams(tokenizer(str(t), add_special_tokens=False)['input_ids']) & banned for t in (q, a)):
            dropped += 1
            continue
        kept.append(tokenizer(qa_text(q, a), add_special_tokens=False)['input_ids'])
    return kept, dropped


def pack(items, order, length=CHUNK):
    """Chunks that each start at an item start; short items absorb the following ones; remainders discarded."""
    chunks, buf = [], []
    for i in order:
        buf = buf + list(items[i])
        if len(buf) >= length:
            chunks.append(buf[:length])
            buf = []
    return chunks


def qa_list(items):
    chunks = pack(items, np.random.default_rng(7411).permutation(len(items)))
    if len(chunks) < POOL_SIZE:
        raise ValueError('Not enough QA chunks')
    order = np.random.default_rng(7412).permutation(len(chunks))[:POOL_SIZE]
    return torch.tensor([chunks[i] for i in order], dtype=torch.long), len(chunks)


def web_lists(kept_chunks, removed_chunks):
    k = np.random.default_rng(7413).choice(len(kept_chunks), N_KEPT_WEB, replace=False)
    r = np.random.default_rng(7414).choice(len(removed_chunks), N_REMOVED_WEB, replace=False)
    return (torch.tensor([kept_chunks[i] for i in k], dtype=torch.long),
            torch.tensor([removed_chunks[i] for i in r], dtype=torch.long), k.tolist(), r.tolist())


def composition(sigma, m=POOL_SIZE):
    n_qa = int(round(sigma * m))
    n_web = m - n_qa
    return n_qa, n_web - n_web // 2, n_web // 2


def build_pool(qa, kept, removed, sigma):
    n_qa, n_k, n_r = composition(sigma)
    return torch.cat([qa[:n_qa], kept[:n_k], removed[:n_r]])


def item_splits(n_test):
    used73 = s73.sample_target(n_test)
    rem = np.setdiff1d(np.arange(n_test), used73)
    pilot = np.sort(rem[np.random.default_rng(7421).choice(len(rem), 256, replace=False)])
    rem2 = np.setdiff1d(rem, pilot)
    target = np.sort(rem2[np.random.default_rng(7422).choice(len(rem2), 512, replace=False)])
    return pilot, target


def schedule(k, seed):
    rng = np.random.default_rng(seed + 1000 * k)
    rand = rng.integers(POOL_SIZE, size=BUDGET)
    filt = rng.integers(KEEP, size=BUDGET)
    subset = np.sort(rng.choice(POOL_SIZE, KEEP, replace=False))
    return rand, filt, subset, hashlib.sha256(rand.tobytes() + filt.tobytes() + subset.tobytes()).hexdigest()


def qa_share(indices, sigma):
    return float(np.mean(np.asarray(indices) < composition(sigma)[0]))


# ---------------------------------------------------------------- summary

def bootstrap(values):
    draws = np.random.default_rng(7403).choice(values, (10000, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [.025, .975])]


def contrast(values):
    return dict(mean=mean(values), ci95=bootstrap(values), per_seed=values, seeds_negative=sum(v < 0 for v in values))


def gate(pilot_rows):
    row = next(r for r in pilot_rows if r['sigma'] == 0.25)
    return row['losses'][str(BUDGET)] < row['losses']['0']


def summarize(rows, calibration, plan):
    n, B = str(calibration['updates']), str(BUDGET)
    out = {}
    for sigma in plan['phase_B_main']['sigma']:
        g = {arm: sorted([r for r in rows if (r['sigma'], r['arm']) == (sigma, arm)], key=lambda r: r['seed'])
             for arm in ARMS}
        for arm in ARMS:
            if [r['seed'] for r in g[arm]] != plan['phase_B_main']['seeds']:
                raise ValueError(f'Incomplete grid for sigma {sigma} {arm}')
            if [r['losses']['0'] for r in g[arm]] != [r['losses']['0'] for r in g['random']]:
                raise ValueError('Arms do not share the initial state')
        R, A, C = g['random'], g['align_grad'], g['random512']
        d = lambda x, kx, y, ky, key='losses': [a[key][kx] - b[key][ky] for a, b in zip(x, y)]
        charged, sel = d(A, n, R, B), d(A, n, R, n)
        out[str(sigma)] = dict(
            loss0=mean(r['losses']['0'] for r in R), accuracy0=mean(r['accuracy']['0'] for r in R),
            means={arm: {k: mean(r['losses'][k] for r in g[arm]) for k in ('0', n, B)} for arm in ARMS},
            random_minus_no_update=mean(d(R, B, R, '0')), random512_minus_no_update=mean(d(C, n, C, '0')),
            charged=contrast(charged), matched_selection=contrast(d(A, n, C, n)),
            repetition=contrast(d(C, n, R, n)), selection_term=mean(sel), throughput_term=mean(charged) - mean(sel),
            amortized=contrast(d(A, B, R, B)), amortized_vs_random512=contrast(d(A, B, C, B)),
            accuracy_charged=contrast(d(A, n, R, B, 'accuracy')),
            accuracy_matched=contrast(d(A, n, C, n, 'accuracy')),
            qa_share_align=mean(r['qa_share'] for r in A), qa_share_random512=mean(r['qa_share'] for r in C),
            scoring_seconds=mean(r['scoring_seconds'] for r in A))
    p, s2 = out['0.25'], out['0.1']
    return dict(primary=dict(sigma=0.25, contrast='align_grad at N minus random at 2400',
                             mean=p['charged']['mean'], ci95=p['charged']['ci95']),
                prediction_holds={'sigma0.25 qa share > 0.5': p['qa_share_align'] > .5,
                                  'sigma0.25 primary negative': p['charged']['mean'] < 0,
                                  'sigma0.25 matched selection negative': p['matched_selection']['mean'] < 0,
                                  'sigma0.1 qa share > 0.2': s2['qa_share_align'] > .2},
                costs=calibration['costs'], updates=calibration['updates'], device=calibration['device'],
                scoring_in_random_steps=calibration['costs']['align_grad'] / calibration['costs']['random'],
                sigmas=out)


# ---------------------------------------------------------------- main

def load_inputs(args, tokenizer):
    import pandas as pd
    import pyarrow.parquet as pq
    for path, digest in ((args.kept, s73.KEPT_SHA256), (args.removed, s73.REMOVED_SHA256), (args.aya, AYA_SHA256),
                         (args.mmlu / 'dev.csv', s73.DEV_SHA256), (args.mmlu / 'test.csv', s73.TEST_SHA256)):
        if s69.sha256(path) != digest:
            raise ValueError(f'{path} differs from the protocol')
    dev = pd.read_csv(args.mmlu / 'dev.csv').to_dict('records')
    test = pd.read_csv(args.mmlu / 'test.csv').to_dict('records')
    banned = content_banned(dev + test, tokenizer)
    pilot_rows, target_rows = item_splits(len(test))
    probe = s73.build_items(dev, tokenizer)
    pilot = s73.build_items([test[i] for i in pilot_rows], tokenizer)
    target = s73.build_items([test[i] for i in target_rows], tokenizer)
    aya = pd.read_parquet(args.aya, columns=['inputs', 'targets', 'language'])
    aya = aya[aya.language == 'Standard Arabic']
    items, qa_dropped = clean_qa(list(zip(aya.inputs, aya.targets)), tokenizer, banned)
    qa, n_qa_chunks = qa_list(items)
    kept_texts = pq.read_table(args.kept, columns=['text'])['text'].to_pylist()
    removed_texts = pq.ParquetFile(args.removed).read_row_groups(list(range(s73.REMOVED_ROW_GROUPS)),
                                                                  columns=['text'])['text'].to_pylist()
    kr, kc, k_drop = s73.decontaminate(*s73.first_tokens(kept_texts, tokenizer), banned)
    rr, rc, r_drop = s73.decontaminate(*s73.first_tokens(removed_texts, tokenizer), banned)
    kept, removed, k_idx, r_idx = web_lists(kc, rc)
    pools = {s: build_pool(qa, kept, removed, s) for s in SIGMAS + PILOT_SIGMAS}
    population = dict(banned_ngrams=len(banned), aya_standard_arabic=len(aya), aya_dropped=qa_dropped,
                      aya_items=len(items), qa_chunks=n_qa_chunks, kept_decontaminated=k_drop,
                      removed_decontaminated=r_drop, kept_documents=[int(kr[i]) for i in k_idx],
                      removed_documents=[int(rr[i]) for i in r_idx],
                      pilot_items=[it['id'] for it in pilot], target_items=[it['id'] for it in target],
                      pool_sha256={str(s): s73.tensor_sha256(p) for s, p in pools.items()})
    return probe, pilot, target, pools, population, banned


def main():
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('A', 'B'), required=True)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--pilot-journal', type=Path, required=True, help='Phase A journal (read by phase B for the gate)')
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--kept', type=Path, required=True)
    parser.add_argument('--removed', type=Path, required=True)
    parser.add_argument('--aya', type=Path, required=True)
    parser.add_argument('--mmlu', type=Path, required=True)
    parser.add_argument('--summary', type=Path, default=Path('results/llm-qa-mix.json'))
    args = parser.parse_args()
    plan = json.loads(PROTOCOL.read_text())
    if plan['stage'] != 74 or tuple(plan['phase_B_main']['arms']) != ARMS or tuple(plan['phase_B_main']['sigma']) != SIGMAS:
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
    probe, pilot, target, pools, population, _ = load_inputs(args, tokenizer)
    pools = {k: v.to(device) for k, v in pools.items()}
    env = dict(kind='environment', time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), torch=torch.__version__,
               transformers=transformers.__version__, cuda=torch.version.cuda, python=platform.python_version(),
               device=device_name, phase=args.phase)
    journal_path = args.pilot_journal if args.phase == 'A' else args.journal
    journal = s69.read_journal(journal_path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)

    def free():
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if args.phase == 'B':
        pilot_journal = s69.read_journal(args.pilot_journal)
        pilot_runs = [r for r in pilot_journal if r.get('kind') == 'pilot']
        if len(pilot_runs) != len(PILOT_SIGMAS) or pilot_runs[0]['population'] != population:
            raise ValueError('Phase A incomplete or on different samples')
        if not gate(pilot_runs):
            raise SystemExit('Gate failed: phase B is not run (protocol phase_A_pilot.gate)')
    with journal_path.open('a') as out:
        def log(row):
            out.write(json.dumps(row, allow_nan=False) + '\n')
            out.flush()
            os.fsync(out.fileno())
        log(dict(env, resumed_runs=sum(r.get('kind') in ('run', 'pilot') for r in journal)))
        if args.phase == 'A':
            done = {r['sigma'] for r in journal if r.get('kind') == 'pilot'}
            ev = lambda m: s73.evaluate(m, pilot, device)
            for sigma in PILOT_SIGMAS:
                if sigma in done:
                    continue
                rand, _, _, digest = schedule(0, PILOT_SEED)
                model = s69.load(model_dir, PILOT_SEED, device)
                row = s73.train(model, pools[sigma], rand, BUDGET, {0, BUDGET}, ev, device)
                row.update(kind='pilot', sigma=sigma, seed=PILOT_SEED, schedule_sha256=digest, device=device_name,
                           population=population)
                log(row)
                del model
                free()
                print(f'pilot sigma {sigma} complete', flush=True)  # outcome not printed
            return
        done = {(r['sigma'], r['seed'], r['arm']) for r in journal if r.get('kind') == 'run'}
        calibration = next((r for r in journal if r.get('kind') == 'calibration'), None)
        if calibration is None:
            rand, _, _, _ = schedule(0, CAL_SEED)
            row = s73.train(s69.load(model_dir, CAL_SEED, device), pools[0.25], rand, 24, set(), None, device)
            costs = dict(random=median(row['step_seconds'][4:]))
            free()
            _, costs['align_grad'] = s73.score(s69.load(model_dir, CAL_SEED, device), 'align_grad', pools[0.25], probe, device)
            free()
            calibration = dict(kind='calibration', model=MODEL, device=device_name, costs=costs,
                               updates=int(np.floor(BUDGET - costs['align_grad'] / costs['random'] + 1e-9)),
                               population=population)
            log(calibration)
        elif calibration['device'] != device_name or calibration['population'] != population:
            raise ValueError('Resumed on another device or with different samples')
        n = calibration['updates']
        if n < 1:
            raise SystemExit('align_grad infeasible within the budget')
        print(MODEL, device_name, 'costs', calibration['costs'], 'updates', n, flush=True)
        ev = lambda m: s73.evaluate(m, target, device)
        total = len(SIGMAS) * len(plan['phase_B_main']['seeds']) * len(ARMS)
        for k, sigma in enumerate(SIGMAS):
            data = pools[sigma]
            for i, seed in enumerate(plan['phase_B_main']['seeds']):
                rand, filt, subset, digest = schedule(k, seed)
                for arm in ARMS[i % 3:] + ARMS[:i % 3]:
                    if (sigma, seed, arm) in done:
                        continue
                    model = s69.load(model_dir, seed, device)
                    extra = {}
                    if arm == 'random':
                        order = rand
                    elif arm == 'random512':
                        order = subset[filt]
                        extra = dict(subset_sha256=hashlib.sha256(subset.tobytes()).hexdigest(),
                                     qa_share=qa_share(subset, sigma))
                    else:
                        scores, seconds = s73.score(model, 'align_grad', data, probe, device)
                        kept = s73.keep_top(scores, KEEP)
                        order = kept[filt]
                        extra = dict(scores=scores.tolist(), kept=kept.tolist(), scoring_seconds=seconds,
                                     kept_sha256=hashlib.sha256(kept.tobytes()).hexdigest(), qa_share=qa_share(kept, sigma))
                    row = s73.train(model, data, order, BUDGET, {0, n, BUDGET}, ev, device)
                    row.update(extra, kind='run', model=MODEL, sigma=sigma, seed=seed, arm=arm, device=device_name,
                               schedule_sha256=digest)
                    log(row)
                    done.add((sigma, seed, arm))
                    del model
                    free()
                    print(f'{len(done)}/{total} runs complete', flush=True)  # outcomes are not printed
    journal = s69.read_journal(args.journal)
    rows = [r for r in journal if r.get('kind') == 'run']
    pilot_journal = s69.read_journal(args.pilot_journal)
    result = dict(stage=74, protocol=plan, journal_sha256=s69.sha256(args.journal),
                  pilot_journal_sha256=s69.sha256(args.pilot_journal),
                  environments=[r for r in journal + pilot_journal if r.get('kind') == 'environment'],
                  pilot=[{k: r[k] for k in ('sigma', 'seed', 'losses', 'accuracy')} for r in pilot_journal
                         if r.get('kind') == 'pilot'],
                  gate_passed=True, calibration={k: v for k, v in calibration.items() if k != 'population'},
                  summary=summarize(rows, calibration, plan), runs=len(rows))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('summary written to', args.summary)


if __name__ == '__main__':
    main()

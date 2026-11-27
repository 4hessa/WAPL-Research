"""Stage 78 smoke test before the protocol: natural useful-data candidates that are not ArabicMMLU.

Everything evaluated here is already exposed: the stage-76 target (512 items, now exposed) and the stage-73 target
(512 items, exposed); the proxy alignment probe is the 120 All/dev items (exposed). No unexposed ArabicMMLU item is
evaluated and no ArabicMMLU item is trained.

Per candidate source:
  1. Chunks: first 65 Qwen tokens of each document (as the FineWeb-2 chunks of stages 73-77); EXAMS Arabic items are
     written in the stage-74 QA template and packed into 65-token chunks (experiments.llm_qa_mix.pack).
  2. Decontamination against all 14,575 ArabicMMLU items (All/dev + All/test), on the trained tokenization:
       c1 content 8-gram: a token 8-gram of Question, Context or the correct option, each tokenized alone (stage 74);
       c2 template 8-gram: a token 8-gram of an item as evaluated (prompt + correct continuation) or in the training
          template (experiments.llm_qa_mix.mmlu_qa_text);
       c3 option 8-gram: a token 8-gram of any field incl. every option, alone and with a leading space (stage 75 (e));
       c4 character 25-gram: a 25-character shingle of normalised (diacritics, punctuation and spaces removed)
          Context + Question or of a correct option of at least 25 characters; tokenizer independent.
     EXAMS items are also checked with stage-74 criteria (a)-(c) (identical, Jaccard >= 0.5, containment).
  3. Random one-pass training (Qwen2.5-1.5B, LoRA, numerics of stages 73-76): 1633 distinct clean chunks (N of stage
     76), one pass in random order; target loss and accuracy at steps 816 and 1633 on both exposed targets; step 0 is
     the same for every condition (LoRA B = 0). References: FineWeb-2 web chunks of the stage-76 pool (unsuitable bulk)
     and the stage-74 ArabicMMLU QA chunks (the useful data of stages 74-77).
  4. Proxy separability: 0.5B gradient-alignment scores (stage-76 scorer) of a 2048-chunk pool with 512 candidate
     chunks and 1536 stage-76 web chunks (768 kept, 768 removed); AUC of candidate vs web and candidate share of the
     top 512.

python -m tools.smoke_llm_natural_sources --kept K --removed R --mmlu M --sources DIR [--out results/...json]
"""
import argparse
import gc
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from experiments import llm_cheap_filter as s76
from experiments import llm_gpu as s69
from experiments import llm_once_filter as s75
from experiments import llm_qa_mix as s74
from experiments import llm_web_filter as s73

N, SEEDS, CHECKPOINTS = 1633, (78900, 78901), (816, 1633)
CHAR_N = 25
MAX_DOCS = 40000
STAGE76_JOURNAL = Path('results/llm-cheap-filter-journal.jsonl')
MIN_CHARS = {'wikipedia': 2000}  # Wikipedia: skip stubs (most articles of the shard are short biographies/places)
SOURCES = {  # name: (relative path, text column or 'exams', licence)
    'wikipedia': ('wikimedia__wikipedia/20231101.ar/train-00004-of-00007.parquet', 'text',
                  'CC BY-SA 3.0 / GFDL (wikimedia/wikipedia b04c8d1)'),
    'fineweb2_hq': ('epfml__FineWeb2-HQ/arb_Arab/000_00000.parquet', 'text', 'ODC-By 1.0 (epfml/FineWeb2-HQ c0c06e9)'),
    'fineweb_edu_ar_mt': ('nizarun__FineWeb-Edu-Arabic-24M/data/model-a-06-of-07.parquet', 'ar_text',
                          'ODC-By 1.0, machine translated (nizarun/FineWeb-Edu-Arabic-24M f53906e)'),
    'exams_ar': ('mhardalov__exams/multilingual', 'exams', 'CC BY-SA 4.0 (mhardalov/exams 4ff1080)'),
}


# ---------------------------------------------------------------- decontamination

def char_shingles(text, n=CHAR_N):
    t = s74.normalise(text)
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def banned_sets(records, tokenizer):
    c1 = s74.content_banned(records, tokenizer)
    c2 = s73.banned_ngrams(s73.build_items(records, tokenizer))
    c3, c4 = set(), set()
    for r in records:
        c2 |= s73.ngrams(tokenizer(s74.mmlu_qa_text(r), add_special_tokens=False)['input_ids'])
        for seq in s75.eval_forms(r, tokenizer):
            c3 |= s73.ngrams(seq)
        c4 |= char_shingles(s74.question_text(r))
        c4 |= char_shingles(str(r[f'Option {s73.LETTERS.index(str(r["Answer Key"]).strip()) + 1}']))
    return dict(c1=c1, c2=c2, c3=c3), c4


def decontaminate(chunks, tokenizer, token_sets, chars):
    counts = {k: 0 for k in list(token_sets) + ['c4']}
    keep = []
    for i, c in enumerate(chunks):
        g = s73.ngrams(c)
        hits = [k for k, s in token_sets.items() if g & s]
        if char_shingles(tokenizer.decode(c)) & chars:
            hits.append('c4')
        for h in hits:
            counts[h] += 1
        if not hits:
            keep.append(i)
    counts['dropped_total'] = len(chunks) - len(keep)
    return keep, counts


# ---------------------------------------------------------------- sources

def text_chunks(path, column, tokenizer, min_chars=0):
    import pyarrow.parquet as pq
    f = pq.ParquetFile(path)
    texts, g, read = [], 0, 0
    while sum(1 for t in texts if t) < MAX_DOCS and g < f.num_row_groups:  # cap counts documents long enough
        part = f.read_row_group(g, columns=[column])[column].to_pylist()
        read += len(part)
        texts += [t if t and len(t) >= min_chars else '' for t in part]
        g += 1
    rows, chunks = s73.first_tokens(texts, tokenizer)
    return dict(documents_read=read, min_chars=min_chars, eligible=len(chunks)), rows, chunks, None


def exams_records(directory):
    import pandas as pd
    out = []
    for split in ('train', 'validation', 'test'):
        df = pd.read_parquet(Path(directory) / f'{split}-00000-of-00001.parquet')
        for r in df.to_dict('records'):
            info = r['info']
            if str(info.get('language')) != 'Arabic':
                continue
            q = r['question']
            labels, texts = list(q['choices']['label']), list(q['choices']['text'])
            if r['answerKey'] not in labels:
                continue
            out.append(dict(split=split, id=r['id'], subject=str(info.get('subject')), grade=info.get('grade'),
                            Question=str(q['stem']), Answer=str(texts[labels.index(r['answerKey'])]),
                            options=[str(t) for t in texts]))
    return out


def exams_chunks(directory, tokenizer, mmlu, token_sets, chars):
    recs = exams_records(directory)
    cand = [dict(Question=r['Question'], Context=None, **{'Option 1': r['Answer']}, **{'Answer Key': 'A'})
            for r in recs]
    keep, near = s74.near_duplicates(cand, mmlu, tokenizer)  # (a)-(d) vs every ArabicMMLU item
    items, kept_recs, field_hits = [], [], 0
    for k in keep:
        r = recs[k]
        if any(s73.ngrams(tokenizer(t, add_special_tokens=False)['input_ids']) & token_sets['c3']
               for f in [r['Question']] + r['options'] for t in (f.strip(), ' ' + f.strip())) \
                or char_shingles(r['Question'] + r['Answer']) & chars:
            field_hits += 1
            continue
        kept_recs.append(r)
        items.append(tokenizer(s74.qa_text(r['Question'], r['Answer']), add_special_tokens=False)['input_ids'])
    chunks = s74.pack(items, np.random.default_rng(7801).permutation(len(items)))
    subjects = {}
    for r in kept_recs:
        subjects[r['subject']] = subjects.get(r['subject'], 0) + 1
    info = dict(arabic_items=len(recs), item_near_duplicates=near, item_field_or_char_hits=field_hits,
                clean_items=len(items), subjects=subjects, eligible=len(chunks))
    return info, list(range(len(chunks))), chunks, kept_recs


# ---------------------------------------------------------------- GPU parts

def run_condition(model_dir, data, seed, targets, device):
    order = np.random.default_rng(seed).permutation(len(data))
    model = s69.load(model_dir, seed, device)
    def evaluator(m):  # s73.train unpacks (loss, accuracy); here each is a dict over the two exposed targets
        res = {k: s73.evaluate(m, items, device) for k, items in targets.items()}
        return {k: v[0] for k, v in res.items()}, {k: v[1] for k, v in res.items()}
    row = s73.train(model, data, order, len(data), set(CHECKPOINTS), evaluator, device)
    del model
    gc.collect(), torch.cuda.empty_cache()
    return {str(c): {k: dict(loss=row['losses'][str(c)][k], accuracy=row['accuracy'][str(c)][k]) for k in targets}
            for c in CHECKPOINTS}, float(np.median(row['step_seconds']))


def auc(scores, n_pos):
    r = np.argsort(np.argsort(scores, kind='stable'), kind='stable').astype(float) + 1
    return float((r[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * (len(scores) - n_pos)))


def main():
    import os
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')  # before CUDA initialises
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('--kept', '--removed', '--mmlu', '--sources'):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--out', type=Path, default=Path('results/llm-natural-source-smoke.json'))
    parser.add_argument('--skip-gpu', action='store_true')
    args = parser.parse_args()
    import pandas as pd
    import transformers
    from transformers import AutoTokenizer
    s69.setup_numerics()
    specs = json.loads(s69.PROTOCOL.read_text())['models']
    for m in ('qwen2.5-1.5b', 'qwen2.5-0.5b'):
        assert s69.sha256(args.models_dir / m / 'model.safetensors') == specs[m]['safetensors_sha256'], m
    tokenizer = AutoTokenizer.from_pretrained(args.models_dir / 'qwen2.5-1.5b')
    t0 = time.time()
    probe, target76, pool76, pop = s76.load_inputs(args, tokenizer)  # checks the data-file hashes
    cal = next(r for r in s69.read_journal(STAGE76_JOURNAL) if r.get('kind') == 'calibration')
    assert pop['pool_sha256'] == cal['population']['pool_sha256'], 'stage-76 pool differs'
    assert pop['target_items'] == cal['population']['target_items'], 'stage-76 target differs'
    dev = pd.read_csv(args.mmlu / 'dev.csv').to_dict('records')
    test = pd.read_csv(args.mmlu / 'test.csv').to_dict('records')
    target73 = s73.build_items([test[i] for i in s73.sample_target(len(test))], tokenizer)
    mmlu = dev + test
    token_sets, chars = banned_sets(mmlu, tokenizer)
    out = dict(kind='smoke', stage=78, time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               transformers=transformers.__version__, torch=torch.__version__, python=platform.python_version(),
               mmlu_items=len(mmlu), banned=dict({k: len(v) for k, v in token_sets.items()}, c4=len(chars)),
               evaluation_sets=dict(stage76_target=len(target76), stage73_target=len(target73)), sources={})
    data = {'web_fineweb2': pool76[s76.N_QA:], 'qa_arabicmmlu': pool76[:s76.N_QA]}
    for k, (name, (rel, reader, licence)) in enumerate(SOURCES.items()):
        path = args.sources / rel
        if reader == 'exams':
            info, rows, chunks, recs = exams_chunks(path, tokenizer, mmlu, token_sets, chars)
        else:
            info, rows, chunks, recs = text_chunks(path, reader, tokenizer, MIN_CHARS.get(name, 0))
        keep, counts = decontaminate(chunks, tokenizer, token_sets, chars)
        info.update(licence=licence, file_sha256=None if reader == 'exams' else s69.sha256(path),
                    chunk_decontamination=counts, clean_chunks=len(keep))
        if len(keep) >= N:
            pick = np.random.default_rng(7810 + k).choice(len(keep), N, replace=False)
            data[name] = torch.tensor([chunks[keep[i]] for i in pick], dtype=torch.long)
        else:
            info['note'] = f'fewer than {N} clean chunks: not trained'
            if keep:
                data[name] = torch.tensor([chunks[i] for i in keep], dtype=torch.long)
        out['sources'][name] = info
        print(name, json.dumps({x: info[x] for x in ('eligible', 'chunk_decontamination', 'clean_chunks')}), flush=True)
    out['data_seconds'] = time.time() - t0
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
    if args.skip_gpu:
        return
    device = torch.device('cuda')
    out['device'] = torch.cuda.get_device_name(0)
    targets = dict(stage76_target=target76, stage73_target=target73)
    base = s69.load(args.models_dir / 'qwen2.5-1.5b', SEEDS[0], device)
    out['step0'] = {k: dict(zip(('loss', 'accuracy'), s73.evaluate(base, v, device))) for k, v in targets.items()}
    del base
    gc.collect(), torch.cuda.empty_cache()
    out['training'] = {}
    for name, chunks in data.items():
        if len(chunks) < N:
            continue
        chunks = chunks.to(device)
        out['training'][name] = {}
        for seed in SEEDS:
            res, step_s = run_condition(args.models_dir / 'qwen2.5-1.5b', chunks, seed, targets, device)
            out['training'][name][str(seed)] = dict(checkpoints=res, median_step_seconds=step_s)
            print(name, seed, json.dumps(res), flush=True)
            args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
    # proxy separability (seed 78900): 512 candidate chunks + 1536 stage-76 web chunks
    web = torch.cat([pool76[s76.N_QA:s76.N_QA + 768], pool76[s76.N_QA + s76.N_WEB:s76.N_QA + s76.N_WEB + 768]])
    out['proxy'] = {}
    for name, chunks in data.items():
        if name == 'web_fineweb2' or len(chunks) < 512:
            continue
        mix = torch.cat([chunks[:512], web]).to(device)
        model = s69.load(args.models_dir / 'qwen2.5-0.5b', SEEDS[0], device)
        scores, seconds = s73.score(model, 'align_grad', mix, probe, device)
        del model
        gc.collect(), torch.cuda.empty_cache()
        top = s73.keep_top(scores, 512)
        out['proxy'][name] = dict(auc=auc(scores, 512), candidate_share_top512=float(np.mean(top < 512)),
                                  candidate_share_top256=float(np.mean(s73.keep_top(scores, 256) < 512)),
                                  seconds=seconds)
        print('proxy', name, json.dumps(out['proxy'][name]), flush=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()

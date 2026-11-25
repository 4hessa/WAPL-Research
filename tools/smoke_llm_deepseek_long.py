"""Stage 77 smoke test 2 before the protocol (owner's choice: option A, deepseek-llm-7b-base, 147-token chunks).

Exposed data only and no evaluation of any ArabicMMLU item. Measures, for deepseek-llm-7b-base in bfloat16 (LoRA
factors and logits in float32, as in tools/smoke_llm_deepseek.py):
  supply   QA chunks from packing the 4,946 stage-74 trained items at 147 tokens (seed 7411 order, as stage 74), the
           share of those items that fit in one chunk, and kept/removed FineWeb-2 documents with >= 147 tokens
           (whole files as in stage 74, before decontamination; the stage-76 web documents separately)
  cost     random-step cost at 147 tokens (median of steps 5-24, as the calibration, and of 200 steps), full
           alignment scoring cost on 2048 chunks (512 QA, 768 kept, 768 removed) with the 120 dev items as probe,
           QA-vs-web AUC and peak memory
  replay   two 50-step random replays from the same seed: identical adapter digests or not (bf16 determinism)
Then a run-time estimate for the option-A design (M = 8192, QA share 0.25, B = 5000, six seeds, four arms).

python -m tools.smoke_llm_deepseek_long --kept K --removed R --mmlu M
"""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from experiments import llm_common as llm
from experiments import llm_gpu as s69
from experiments import llm_qa_mix as s74
from experiments import llm_web_filter as s73
from tools import smoke_llm_deepseek as smoke1

OUT = Path('results/llm-deepseek-long-smoke.json')
SPEC = smoke1.CANDIDATES['llm-7b']
LENGTH, SEED = 147, 77002
B, M, N_QA, SEEDS, ARMS, ALIGN_RUNS = 5000, 8192, 2048, 6, 4, 6
EVAL_SECONDS_SMOKE1 = 'results/llm-deepseek-smoke.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('--kept', '--removed', '--mmlu'):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    args = parser.parse_args()
    import pandas as pd
    import pyarrow.parquet as pq
    import transformers
    from transformers import AutoTokenizer
    s69.setup_numerics()
    device = torch.device('cuda')
    llm.LoRALinear.forward = smoke1._mixed_forward
    s73.PerExampleCapture = smoke1.FloatCapture
    for path, digest in ((args.kept, s73.KEPT_SHA256), (args.removed, s73.REMOVED_SHA256),
                         (args.mmlu / 'dev.csv', s73.DEV_SHA256), (args.mmlu / 'test.csv', s73.TEST_SHA256)):
        if s69.sha256(path) != digest:
            raise ValueError(f'{path} differs from the protocol')
    tok = AutoTokenizer.from_pretrained(args.models_dir / SPEC['dir'])
    dev = pd.read_csv(args.mmlu / 'dev.csv').to_dict('records')
    test = pd.read_csv(args.mmlu / 'test.csv').to_dict('records')
    by_id = {int(r['ID']): r for r in test}
    ledger = json.loads(smoke1.LEDGER.read_text())['additional_sources']['ArabicMMLU']['stage_74']
    trained = [by_id[i] for i in sorted(ledger['used_for_training']['ids'])]
    cal76 = next(r for r in s69.read_journal(smoke1.STAGE76_JOURNAL) if r.get('kind') == 'calibration')['population']
    res = dict(stage='77 smoke 2 (before protocol)', model=SPEC['repo'], revision=SPEC['revision'], dtype='bfloat16',
               chunk_tokens=LENGTH, torch=torch.__version__, transformers=transformers.__version__,
               cuda=torch.version.cuda, python=platform.python_version(), device=torch.cuda.get_device_name(0),
               data='exposed only; no ArabicMMLU item evaluated')

    # supply
    items = [tok(s74.mmlu_qa_text(r), add_special_tokens=False)['input_ids'] for r in trained]
    chunks = s74.pack(items, np.random.default_rng(7411).permutation(len(items)), length=LENGTH)
    res['qa'] = dict(items=len(items), chunks_available=len(chunks),
                     items_longer_than_chunk=float(np.mean([len(x) > LENGTH for x in items])))
    if len(chunks) < N_QA:
        print(f'WARNING: only {len(chunks)} QA chunks at {LENGTH} tokens', flush=True)
    qa = torch.tensor([chunks[i] for i in np.random.default_rng(7412).permutation(len(chunks))[:min(N_QA, len(chunks))]])
    kept_texts = pq.read_table(args.kept, columns=['text'])['text'].to_pylist()
    removed_texts = pq.ParquetFile(args.removed).read_row_groups(list(range(s73.REMOVED_ROW_GROUPS)),
                                                                  columns=['text'])['text'].to_pylist()
    t = time.perf_counter()
    k_rows, k_chunks = s73.first_tokens(kept_texts, tok, length=LENGTH)
    r_rows, r_chunks = s73.first_tokens(removed_texts, tok, length=LENGTH)
    res['web'] = dict(kept_documents=len(kept_texts), removed_documents=len(removed_texts),
                      kept_with_147_tokens=len(k_rows), removed_with_147_tokens=len(r_rows),
                      stage76_kept_with_147=len(set(cal76['kept_documents']) & set(k_rows)),
                      stage76_removed_with_147=len(set(cal76['removed_documents']) & set(r_rows)),
                      tokenize_seconds=time.perf_counter() - t)
    print('supply', res['qa'], res['web'], flush=True)
    kpos, rpos = {d: i for i, d in enumerate(k_rows)}, {d: i for i, d in enumerate(r_rows)}
    kept = [k_chunks[kpos[d]] for d in cal76['kept_documents'] if d in kpos][:768]
    removed = [r_chunks[rpos[d]] for d in cal76['removed_documents'] if d in rpos][:768]
    sub = torch.cat([qa[:512], torch.tensor(kept), torch.tensor(removed)]).to(device)
    labels = np.r_[np.ones(512), np.zeros(len(kept) + len(removed))]

    # cost
    s73.PROBE_BATCH = SPEC['probe_batch']
    model = smoke1.load_model(SPEC, args.models_dir, SEED, device)
    res['c_r_147_calibration_rule'] = smoke1.step_cost(model, sub, device, n=24)
    res['c_r_147_200_steps'] = smoke1.step_cost(model, sub, device, n=204)
    probe = s73.build_items(dev, tok)
    for batch in (8, 4):
        s73.score_grad.__defaults__ = (batch,)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            scores, secs = s73.score(model, 'align_grad', sub, probe, device)
            break
        except torch.OutOfMemoryError:
            print(f'scoring batch {batch} out of memory', flush=True)
    c_r = res['c_r_147_calibration_rule']
    res['scoring'] = dict(chunks=len(sub), batch=batch, probe_batch=s73.PROBE_BATCH, seconds=secs,
                          random_steps_per_chunk=secs / len(sub) / c_r, auc_qa_vs_web=smoke1.auc(scores, labels),
                          top256_qa=float(np.mean(np.argsort(-scores, kind='stable')[:256] < 512)),
                          peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9)
    print('cost', res['c_r_147_calibration_rule'], res['c_r_147_200_steps'], res['scoring'], flush=True)
    del model
    torch.cuda.empty_cache()

    # replay
    order = np.random.default_rng(SEED).integers(len(sub), size=s73.DIGEST_STEP)
    digests = []
    for _ in range(2):
        m = smoke1.load_model(SPEC, args.models_dir, SEED, device)
        digests.append(s73.train(m, sub, order, s73.DIGEST_STEP, set(), None, device)['adapter_digest_50'])
        del m
        torch.cuda.empty_cache()
    res['replay_50_identical'] = digests[0] == digests[1]

    # estimate (seconds)
    ev = json.loads(Path(EVAL_SECONDS_SMOKE1).read_text())['candidates']['llm-7b']
    eval_s, load_s = ev['eval_seconds'], ev['load_seconds']
    per_chunk_s = secs / len(sub)
    S = per_chunk_s * M
    n = int(np.floor(B - S / c_r + 1e-9))
    step = res['c_r_147_200_steps']
    run = B * step + 3 * eval_s + load_s
    total = SEEDS * ARMS * run + ALIGN_RUNS * (S + load_s) + (24 * step + S + 2 * load_s) + 20 * 60
    res['estimate'] = dict(B=B, M=M, scoring_seconds=S, scoring_random_steps=S / c_r, scoring_share=S / c_r / B, N=n,
                           qa_supply_ok=len(chunks) >= N_QA and n <= N_QA, eval_seconds=eval_s, run_seconds=run,
                           total_hours=total / 3600,
                           per_seed_hours=(ARMS * run + S + load_s) / 3600,
                           note='total = 24 runs + 6 scorings + calibration + 20 min data, target eligibility and loading; '
                                'eval time from smoke 1 (512 items, batch 8), not re-measured')
    print('estimate', res['estimate'], 'replay identical', res['replay_50_identical'], flush=True)
    OUT.write_text(json.dumps(res, indent=2) + '\n')
    print('written', OUT)


if __name__ == '__main__':
    main()

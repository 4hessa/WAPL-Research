"""Stage 77 feasibility smoke test before any protocol: DeepSeek models as the trained model on the stage-76 task.

Exposed data only. Evaluated: the 512 stage-75 target items (exposed since stage 75). Probe: the 120 All/dev items.
Trained: the 4,946 ArabicMMLU items already recorded as training data (stage 74), packed into 65-token QA chunks in
each candidate's tokenization, and the first 65 tokens of the 3072 kept and 3072 removed FineWeb-2 documents of the
stage-76 pool (document indices from the stage-76 calibration). No unexposed ArabicMMLU item is read.

Candidates (all DeepSeek pretraining from scratch; none is a Qwen- or Llama-initialised distill):
  coder-1.3b   deepseek-ai/deepseek-coder-1.3b-base, float32
  vl-1.3b-lm   language model of deepseek-ai/deepseek-vl-1.3b-base (DeepSeek-LLM-1.3B base, then vision-language
               training), float32; same Arabic tokenization as deepseek-llm-7b-base (possible proxy)
  llm-7b       deepseek-ai/deepseek-llm-7b-base, bfloat16 (float32 weights, 27.6 GB, exceed the L4's memory);
               LoRA factors and logits in float32
Per candidate: memory, chars per token, item and prompt lengths, random-step cost at 65 and 147 tokens, alignment
scoring cost per chunk and QA-vs-web AUC on 2048 chunks, evaluation time, loss and accuracy before training, and after
1000 and 3000 random steps (seed 77001) on the 8192-chunk pool.

python -m tools.smoke_llm_deepseek --kept K --removed R --mmlu M [--only NAME ...]
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
from experiments import llm_qa_mix as s74
from experiments import llm_web_filter as s73
from experiments.llm_budget import PerExampleCapture

OUT = Path('results/llm-deepseek-smoke.json')
STAGE75_JOURNAL = Path('results/llm-once-filter-journal.jsonl')
STAGE76_JOURNAL = Path('results/llm-cheap-filter-journal.jsonl')
LEDGER = Path('data/evaluation_status.json')
SEED, STEPS, CHECKS = 77001, 3000, (1000, 3000)
LONG = 147  # 65 Qwen tokens x 2.56 / 1.13 chars per token: about the same text as a stage-76 chunk
CANDIDATES = {
    'coder-1.3b': dict(dir='deepseek-coder-1.3b-base', repo='deepseek-ai/deepseek-coder-1.3b-base',
                       revision='c919139c3a9b4070729c8b2cca4847ab29ca8d94', dtype='float32', eval_batch=32, probe_batch=8, grad_batch=16),
    'vl-1.3b-lm': dict(dir='deepseek-vl-1.3b-base', repo='deepseek-ai/deepseek-vl-1.3b-base',
                       revision='3729510359afff41411fc11fa4f487fd5b5a610c', dtype='float32', eval_batch=32, probe_batch=8,
                       grad_batch=16),
    'llm-7b': dict(dir='deepseek-llm-7b-base', repo='deepseek-ai/deepseek-llm-7b-base',
                   revision='7683fea62db869066ddaff6a41d032262c490d4f', dtype='bfloat16', eval_batch=8, probe_batch=2, grad_batch=8),
}


def _mixed_forward(self, x):  # identical to LoRALinear.forward for float32 bases
    return self.base(x) + ((x.to(self.A.dtype) @ self.A.t() @ self.B.t()) * self.scale).to(x.dtype)


class FloatCapture(PerExampleCapture):
    def per_example(self):
        self.inputs = {m: x.float() for m, x in self.inputs.items()}
        self.grads = {m: g.float() for m, g in self.grads.items()}
        return super().per_example()


def load_model(spec, models_dir, seed, device):
    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
    d = models_dir / spec['dir']
    if spec['dir'] == 'deepseek-vl-1.3b-base':
        from safetensors.torch import load_file
        cfg = json.loads((d / 'config.json').read_text())['language_config']
        cfg = {k: v for k, v in cfg.items() if k not in ('model_type', 'torch_dtype')}
        model = LlamaForCausalLM(LlamaConfig(**cfg))
        state = {k[len('language_model.'):]: v.float() for k, v in load_file(d / 'model.safetensors').items()
                 if k.startswith('language_model.')}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or [m for m in missing if 'rotary' not in m]:
            raise ValueError(f'VL language model keys: missing {missing}, unexpected {unexpected}')
        model = model.float()
    else:
        model = AutoModelForCausalLM.from_pretrained(d, torch_dtype=getattr(torch, spec['dtype']))
    model = llm.add_lora(model, seed).to(device)
    model.lm_head.register_forward_hook(lambda m, i, o: o.float())
    return model


def auc(scores, labels):
    s, y = np.asarray(scores), np.asarray(labels, bool)
    ranks = np.argsort(np.argsort(s, kind='stable'), kind='stable') + 1.
    return float((ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def step_cost(model, chunks, device, n=24):
    params = llm.adapter_params(model)
    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.)
    order = np.random.default_rng(76999).integers(len(chunks), size=n)
    secs = []
    saved = [p.detach().clone() for p in params]
    for i in order:
        s69.sync(device)
        t = time.perf_counter()
        opt.zero_grad()
        llm.lm_loss(model, chunks[i:i + 1]).backward()
        opt.step()
        s69.sync(device)
        secs.append(time.perf_counter() - t)
    with torch.no_grad():
        for p, s in zip(params, saved):
            p.copy_(s)
    return median(secs[4:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('--kept', '--removed', '--mmlu'):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--only', nargs='*', default=list(CANDIDATES))
    args = parser.parse_args()
    import pandas as pd
    import pyarrow.parquet as pq
    import transformers
    from transformers import AutoTokenizer
    s69.setup_numerics()
    device = torch.device('cuda')
    llm.LoRALinear.forward = _mixed_forward
    s73.PerExampleCapture = FloatCapture
    for path, digest in ((args.kept, s73.KEPT_SHA256), (args.removed, s73.REMOVED_SHA256),
                         (args.mmlu / 'dev.csv', s73.DEV_SHA256), (args.mmlu / 'test.csv', s73.TEST_SHA256)):
        if s69.sha256(path) != digest:
            raise ValueError(f'{path} differs from the protocol')
    dev = pd.read_csv(args.mmlu / 'dev.csv').to_dict('records')
    test = pd.read_csv(args.mmlu / 'test.csv').to_dict('records')
    by_id = {int(r['ID']): r for r in test}
    cal75 = next(r for r in s69.read_journal(STAGE75_JOURNAL) if r.get('kind') == 'calibration')['population']
    cal76 = next(r for r in s69.read_journal(STAGE76_JOURNAL) if r.get('kind') == 'calibration')['population']
    target_rows = [by_id[i] for i in cal75['target_items']]
    trained_ids = json.loads(LEDGER.read_text())['additional_sources']['ArabicMMLU']['stage_74']['used_for_training']['ids']
    if set(trained_ids) & set(cal75['target_items']):
        raise ValueError('Trained items overlap the evaluated items')
    trained_rows = [by_id[i] for i in sorted(trained_ids)]
    kept_texts = pq.read_table(args.kept, columns=['text'])['text'].to_pylist()
    removed_texts = pq.ParquetFile(args.removed).read_row_groups(list(range(s73.REMOVED_ROW_GROUPS)),
                                                                  columns=['text'])['text'].to_pylist()
    web_docs = [kept_texts[i] for i in cal76['kept_documents']] + [removed_texts[i] for i in cal76['removed_documents']]
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out.update(stage='77 smoke (before protocol)', torch=torch.__version__, transformers=transformers.__version__,
               cuda=torch.version.cuda, python=platform.python_version(), device=torch.cuda.get_device_name(0),
               gpu_memory_gb=torch.cuda.get_device_properties(0).total_memory / 1e9,
               data='exposed only: stage-75 target (512, evaluated), 120 dev (probe), 4,946 stage-74 trained items '
                    'and the stage-76 web documents (trained)')
    out.setdefault('candidates', {})
    sample = ' '.join(str(r['Question']) for r in dev)
    for name in args.only:
        spec = CANDIDATES[name]
        print('==', name, flush=True)
        tok = AutoTokenizer.from_pretrained(args.models_dir / spec['dir'])
        res = dict(repo=spec['repo'], revision=spec['revision'], dtype=spec['dtype'], vocab=len(tok),
                   chars_per_token_dev_questions=len(sample) / len(tok(sample, add_special_tokens=False)['input_ids']))
        items = [tok(s74.mmlu_qa_text(r), add_special_tokens=False)['input_ids'] for r in trained_rows]
        lens = np.array([len(x) for x in items])
        res['qa_item_tokens'] = dict(median=float(np.median(lens)), p90=float(np.quantile(lens, .9)),
                                     share_over_65=float(np.mean(lens > 65)))
        chunks = s74.pack(items, np.random.default_rng(7411).permutation(len(items)))
        res['qa_chunks_available'] = len(chunks)
        qa = torch.tensor([chunks[i] for i in np.random.default_rng(7412).permutation(len(chunks))[:2048]])
        web_ids = [tok(t[:4000], add_special_tokens=False)['input_ids'] for t in web_docs]
        res['web_docs_under_65_tokens'] = int(sum(len(w) < 65 for w in web_ids))
        web = torch.tensor([w[:65] + [tok.eos_token_id] * (65 - len(w)) for w in web_ids])
        web_long = torch.tensor([w[:LONG] for w in web_ids if len(w) >= LONG][:256])
        pool = torch.cat([qa, web]).to(device)
        probe = s73.build_items(dev, tok)
        target = s73.build_items(target_rows, tok)
        plens = [len(tok((str(r['Context']) + '\n' if s73._present(r.get('Context')) else '') + 'سؤال: '
                         + str(r['Question']) + '\nالجواب:', add_special_tokens=False)['input_ids']) for r in target_rows]
        res['target_prompt_tokens'] = dict(median=float(np.median(plens)),
                                           share_truncated_at_192=float(np.mean(np.array(plens) > s73.PROMPT_MAX)))
        s73.EVAL_BATCH, s73.PROBE_BATCH, s73.GRAD_BATCH = spec['eval_batch'], spec['probe_batch'], spec['grad_batch']
        s73.score_grad.__defaults__ = (spec['grad_batch'],)  # bound at definition time
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        model = load_model(spec, args.models_dir, SEED, device)
        res['load_seconds'] = time.perf_counter() - t
        res['memory_after_load_gb'] = torch.cuda.memory_allocated() / 1e9
        res['c_r_65'] = step_cost(model, pool, device)
        res['c_r_147'] = step_cost(model, web_long.to(device), device)
        res['peak_memory_training_gb'] = torch.cuda.max_memory_allocated() / 1e9
        sub = torch.cat([pool[:512], pool[2048:2048 + 768], pool[5120:5120 + 768]])
        torch.cuda.reset_peak_memory_stats()
        scores, secs = s73.score(model, 'align_grad', sub, probe, device)
        labels = np.r_[np.ones(512), np.zeros(1536)]
        res['scoring'] = dict(chunks=len(sub), seconds=secs, random_steps_per_chunk=secs / len(sub) / res['c_r_65'],
                              auc_qa_vs_web=auc(scores, labels),
                              auc_qa_vs_kept=auc(np.r_[scores[:512], scores[512:1280]], labels[:1280]),
                              top256_qa=float(np.mean(np.argsort(-scores, kind='stable')[:256] < 512)),
                              batch=s73.GRAD_BATCH, probe_batch=s73.PROBE_BATCH, peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9)
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        loss0, acc0 = s73.evaluate(model, target, device)
        res['eval_seconds'] = time.perf_counter() - t
        res['eval_peak_memory_gb'] = torch.cuda.max_memory_allocated() / 1e9
        res['eval_batch'] = spec['eval_batch']
        print(name, 'cost', round(res['c_r_65'], 4), 'scoring', round(res['scoring']['random_steps_per_chunk'], 3),
              'auc', round(res['scoring']['auc_qa_vs_web'], 3), 'loss0', round(loss0, 3), 'acc0', round(acc0, 3), flush=True)
        order = np.random.default_rng(SEED).integers(len(pool), size=STEPS)
        t = time.perf_counter()
        row = s73.train(model, pool, order, STEPS, set(CHECKS), lambda m: s73.evaluate(m, target, device), device)
        res['random_training'] = dict(seed=SEED, pool=len(pool), wall_seconds=time.perf_counter() - t,
                                      loss={'0': loss0, **row['losses']}, accuracy={'0': acc0, **row['accuracy']},
                                      median_step_seconds=median(row['step_seconds']))
        print(name, 'random', {k: round(v, 3) for k, v in res['random_training']['loss'].items()},
              {k: round(v, 3) for k, v in res['random_training']['accuracy'].items()}, flush=True)
        out['candidates'][name] = res
        OUT.write_text(json.dumps(out, indent=2) + '\n')
        del model, pool
        gc.collect()
        torch.cuda.empty_cache()
    print('written', OUT)


if __name__ == '__main__':
    main()

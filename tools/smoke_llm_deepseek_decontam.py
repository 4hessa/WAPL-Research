"""Stage 77 smoke 3 (data only; no model loaded, nothing evaluated): decontamination alternatives for amendment 1.

The protocol's rule (token 8-grams in DeepSeek tokenization) leaves 562 kept and 1,596 removed web chunks of 147
tokens, fewer than the 2,988 required (dry run of experiments.llm_deepseek_filter.load_inputs). DeepSeek tokens carry
about 1.13 Arabic characters, so 8 tokens span about 9 characters (about 20 in Qwen tokenization). Alternatives, each
applied to every n-gram criterion (web decontamination, target criteria (d) and (e)) with the training chunks
unchanged (first 147 DeepSeek tokens; the 1,992 QA chunks):
  current     DeepSeek tokens, n = 8 (the protocol as written)
  ds18        DeepSeek tokens, n = 18 (about the text span of 8 Qwen tokens: 8 x 2.56 / 1.13)
  qwen8       Qwen2.5 tokens, n = 8, on the decoded text of each chunk and on every item field (stage-76 rule)
  char20      characters, n = 20, after removing Arabic diacritics and tatweel and collapsing whitespace
Per alternative: eligible and dropped web chunks; if both kinds reach 2,988, the pool is drawn as in the protocol
(seeds 7711, 7712) and target eligibility under criteria (a)-(e) is counted (candidates, drops per criterion,
eligible). No target item is drawn or evaluated.

python -m tools.smoke_llm_deepseek_decontam --kept K --removed R --mmlu M
"""
import argparse
import json
from pathlib import Path
import re
import time


from experiments import llm_deepseek_filter as stage
from experiments import llm_once_filter as s75
from experiments import llm_qa_mix as s74
from experiments import llm_web_filter as s73

OUT = Path('results/llm-deepseek-decontam-smoke.json')
_MARKS = re.compile(r'[ً-ْٰـ]')
_SPACE = re.compile(r'\s+')


class CharTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return dict(input_ids=[ord(c) for c in _SPACE.sub(' ', _MARKS.sub('', str(text)))])


class Chunks(list):
    def tolist(self):
        return self


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('--kept', '--removed', '--mmlu'):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    args = parser.parse_args()
    import pandas as pd
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    for path, digest in ((args.kept, s73.KEPT_SHA256), (args.removed, s73.REMOVED_SHA256),
                         (args.mmlu / 'dev.csv', s73.DEV_SHA256), (args.mmlu / 'test.csv', s73.TEST_SHA256)):
        if stage.s69.sha256(path) != digest:
            raise ValueError(f'{path} differs from the protocol')
    ds = AutoTokenizer.from_pretrained(args.models_dir / stage.MODEL)
    qwen = AutoTokenizer.from_pretrained(args.models_dir / 'qwen2.5-1.5b')
    dev = pd.read_csv(args.mmlu / 'dev.csv').to_dict('records')
    test = pd.read_csv(args.mmlu / 'test.csv').to_dict('records')
    trained_ids = set(json.loads(s75.LEDGER.read_text())['additional_sources']['ArabicMMLU']['stage_74']
                      ['used_for_training']['ids'])
    by_id = {int(r['ID']): r for r in test}
    qa, _ = stage.qa_chunks([by_id[i] for i in sorted(trained_ids)], ds)
    kept_texts = pq.read_table(args.kept, columns=['text'])['text'].to_pylist()
    removed_texts = pq.ParquetFile(args.removed).read_row_groups(list(range(s73.REMOVED_ROW_GROUPS)),
                                                                  columns=['text'])['text'].to_pylist()
    t = time.perf_counter()
    kr, kc = s73.first_tokens(kept_texts, ds, length=stage.CHUNK)
    rr, rc = s73.first_tokens(removed_texts, ds, length=stage.CHUNK)
    print('web chunks', len(kc), len(rc), round(time.perf_counter() - t), 's', flush=True)
    excluded = stage.excluded_rows(test)
    decode = lambda chunks: [ds.decode(c) for c in chunks]
    texts = dict(qa=decode(qa.tolist()), kept=decode(kc), removed=decode(rc))
    alts = dict(current=(ds, 8, None), ds18=(ds, 18, None), qwen8=(qwen, 8, 'text'), char20=(CharTokenizer(), 20, 'text'))
    out = dict(stage='77 smoke 3 (data only, before amendment 1)', chunk_tokens=stage.CHUNK, required_per_kind=stage.N_WEB,
               web_documents_with_147_tokens=[len(kc), len(rc)], qa_chunks=len(qa), excluded_evaluated_rows=len(excluded),
               alternatives={})
    default = s73.ngrams.__defaults__
    try:
        for name, (tok, n, mode) in alts.items():
            t = time.perf_counter()
            s73.ngrams.__defaults__ = (n,)
            rep = (lambda key, chunks: chunks) if mode is None else \
                (lambda key, chunks: [tok(x, add_special_tokens=False)['input_ids'] for x in texts[key]])
            banned = s74.content_banned(dev + test, tok)
            k_rep, r_rep = rep('kept', kc), rep('removed', rc)
            k_keep = [i for i, c in enumerate(k_rep) if not (s73.ngrams(c) & banned)]
            r_keep = [i for i, c in enumerate(r_rep) if not (s73.ngrams(c) & banned)]
            res = dict(tokenizer=type(tok).__name__, n=n, web_eligible=[len(k_keep), len(r_keep)],
                       web_dropped=[len(kc) - len(k_keep), len(rc) - len(r_keep)],
                       web_eligible_share=[len(k_keep) / len(kc), len(r_keep) / len(rc)])
            if min(len(k_keep), len(r_keep)) >= stage.N_WEB:
                k_idx, r_idx = stage.web_draws(len(k_keep), len(r_keep))
                pool_rep = Chunks(rep('qa', qa.tolist()) + [k_rep[k_keep[i]] for i in k_idx]
                                  + [r_rep[r_keep[i]] for i in r_idx])
                counts, eligible = s75.target_candidates(test, tok, [pool_rep], trained_ids, excluded)
                res['target_pool'] = counts
                res['target_feasible'] = len(eligible) >= stage.TARGET_ITEMS
            res['seconds'] = time.perf_counter() - t
            print(name, json.dumps(res), flush=True)
            out['alternatives'][name] = res
            OUT.write_text(json.dumps(out, indent=2) + '\n')
    finally:
        s73.ngrams.__defaults__ = default
    print('written', OUT)


if __name__ == '__main__':
    main()

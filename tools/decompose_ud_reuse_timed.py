"""Post-hoc exploratory decomposition of stage-64 timed contrasts (not in protocol 64).

For a guided run with n accepted updates, replay the random schedule's first n
choices from the same warmed state. Then

  L_guided(T) - L_random(T) = [L_guided(T) - L_random@n]   (per-update selection)
                            + [L_random@n - L_random(T)]   (fewer updates in T)

Negative first term: the guided choices lower loss more than the same number of
random updates. Positive second term: loss forgone through lower throughput.
"""
import json
from pathlib import Path
from statistics import mean

import torch

from experiments import pud_policy as original
from experiments import ud_reuse_timed as timed
from experiments.ud_reuse_transfer import prepare_language
from tools.audit_pud_reuse import digest, read_json, read_rows

OUTPUT = Path('results/ud-reuse-timed-decomposition.json')


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    result = read_json('results/ud-reuse-timed.json')
    rows = read_rows('results/ud-reuse-timed-records.jsonl')
    plan = result['protocol']
    manifest = read_json(timed.MANIFEST)
    states, languages = [], {}
    for lang in plan['languages']:
        train, _, target, _ = prepare_language(manifest['languages'][lang]['splits'], plan, {})
        for seed in plan['seeds']:
            base, opt, ids, uniform, _ = timed.build_state(train, lang, seed, plan)
            runs = [r for r in rows if r['language'] == lang and r['seed'] == seed]
            random_final = mean(r['final_loss'] for r in runs if r['arm'] == 'random')
            curve, cache = {}, {}
            needed = sorted({r['updates'] for r in runs if r['arm'] != 'random'})
            model, o = original.clone(base, opt)
            done = 0
            for n in needed:
                while done < n:
                    original.update(model, o, train[ids[done]][int(uniform[done]):int(uniform[done]) + 1])
                    done += 1
                cache[n] = original.evaluate(model, target)
            state = dict(language=lang, seed=seed, random_final=random_final)
            for arm in ('reuse4', 'fresh'):
                guided = [r for r in runs if r['arm'] == arm]
                at_n = mean(cache[r['updates']] for r in guided)
                g = mean(r['final_loss'] for r in guided)
                state[arm] = dict(final=g, random_at_matched_updates=at_n,
                                  selection_term=g - at_n, throughput_term=at_n - random_final,
                                  total=g - random_final)
            states.append(state)
            print(lang, seed, {a: round(state[a]['selection_term'], 4) for a in ('reuse4', 'fresh')}, flush=True)
        group = [s for s in states if s['language'] == lang]
        languages[lang] = {arm: {k: mean(s[arm][k] for s in group) for k in group[0][arm]}
                           | dict(seeds_selection_negative=sum(s[arm]['selection_term'] < 0 for s in group))
                           for arm in ('reuse4', 'fresh')}
    out = dict(stage='64-posthoc', status='exploratory, not specified in protocol 64',
               source_sha256={'results/ud-reuse-timed.json': digest('results/ud-reuse-timed.json'),
                              'results/ud-reuse-timed-records.jsonl': digest('results/ud-reuse-timed-records.jsonl'),
                              'tools/decompose_ud_reuse_timed.py': digest(__file__)},
               definition=__doc__.strip(), languages=languages, states=states)
    with OUTPUT.open('x') as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()

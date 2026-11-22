import unittest

from experiments import llm_gpu as s69
from experiments import llm_gpu_indomain as stage
from tests.test_llm_gpu import FakeTokenizer


class InDomainTests(unittest.TestCase):
    def test_split_excludes_stage69_articles_and_is_disjoint(self):
        texts = [('x' * (70 if i % 4 else 5)) + chr(1000 + i) for i in range(1200)]
        tok = FakeTokenizer()
        _, _, old = s69.select_articles(texts, tok, {})
        eligible, chunks = stage.eligible_chunks(texts, tok)
        probe, target, cand, pop = stage.split(eligible, chunks)
        used69 = set(old['probe_articles']) | set(old['target_articles'])
        self.assertEqual(pop['excluded_stage69'], 272)
        self.assertFalse(used69 & (set(pop['probe_articles']) | set(pop['target_articles'])))
        self.assertFalse(set(pop['probe_articles']) & set(pop['target_articles']))
        self.assertEqual((len(probe), len(target), len(cand)), (16, 256, len(eligible) - 272 - 16 - 256))
        self.assertEqual(pop['candidate_articles'], len(cand))

    def test_summary_includes_no_update_baseline(self):
        plan = dict(rho=[0.0, 0.9], seeds=[1, 2])
        n = dict(random=400, loss4=200, graddot4_batched=300)
        rows = []
        for model in ('qwen2.5-0.5b', 'qwen2.5-1.5b'):
            for rho in plan['rho']:
                for seed in plan['seeds']:
                    rows.append(dict(model=model, rho=rho, seed=seed, arm='random', english_share=rho,
                                     losses={'0': 3., '200': 2.8, '300': 2.7, '400': 2.6}))
                    for arm, loss in (('loss4', 2.75), ('graddot4_batched', 2.65)):
                        rows.append(dict(model=model, rho=rho, seed=seed, arm=arm, english_share=0.,
                                         losses={'0': 3., str(n[arm]): loss}))
        cal = {m: dict(costs=dict(random=1., loss4=2., graddot4_batched=4 / 3), updates=n, device='L4')
               for m in ('qwen2.5-0.5b', 'qwen2.5-1.5b')}
        out = stage.summarize(rows, cal, plan)
        self.assertAlmostEqual(out['primary']['mean'], .05)
        e = out['models']['qwen2.5-1.5b']['per_rho']['0.0']
        self.assertAlmostEqual(e['random_minus_no_update']['mean'], -.4)
        self.assertAlmostEqual(e['arms']['graddot4_batched']['minus_no_update']['mean'], -.35)
        self.assertAlmostEqual(e['arms']['graddot4_batched']['selection_term'], -.05)

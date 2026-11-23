import unittest

from experiments import llm_gpu_indomain as s70
from experiments import llm_gpu_threshold as stage
from tests.test_llm_gpu import FakeTokenizer


class ThresholdTests(unittest.TestCase):
    def test_split_excludes_stage69_and_stage70_articles(self):
        texts = [('x' * (70 if i % 4 else 5)) + chr(1000 + i) for i in range(1500)]
        eligible, chunks = s70.eligible_chunks(texts, FakeTokenizer())
        _, _, _, pop70 = s70.split(eligible, chunks)
        probe, target, cand, pop = stage.split(eligible, chunks)
        used = set(pop70['probe_articles']) | set(pop70['target_articles'])
        self.assertEqual(pop['excluded_stages_69_70'], 272 + 272)
        self.assertFalse(used & (set(pop['probe_articles']) | set(pop['target_articles'])))
        self.assertEqual(len(cand), len(eligible) - 544 - 272)

    def test_crossover_interpolation(self):
        rhos = [0.0, .25, .5, .75, .9]
        self.assertAlmostEqual(stage.crossover(rhos, [.02, .01, -.01, -.05, -.09]), .375)
        self.assertIsNone(stage.crossover(rhos, [-.01, -.02, -.03, -.04, -.05]))
        self.assertIsNone(stage.crossover(rhos, [.01] * 5))

    def test_summary(self):
        plan = dict(rho=[0.0, .25, .5, .75, .9], seeds=[1, 2])
        n = dict(random=400, graddot4_batched=200)
        rows = []
        for rho, d in zip(plan['rho'], (.03, .01, -.02, -.05, -.08)):
            for seed in plan['seeds']:
                rows.append(dict(rho=rho, seed=seed, arm='random', english_share=rho,
                                 losses={'0': 3., '200': 2.0, '400': 1.9}))
                rows.append(dict(rho=rho, seed=seed, arm='graddot4_batched', english_share=rho / 2,
                                 losses={'0': 3., '200': 1.9 + d}))
        cal = dict(costs=dict(random=1., graddot4_batched=2.), updates=n, device='L4')
        out = stage.summarize(rows, cal, plan)
        self.assertAlmostEqual(out['primary']['crossover_rho'], .25 + .25 / 3)
        self.assertEqual(out['primary']['resamples_with_crossing'], 10000)
        self.assertTrue(all(out['prediction_signs_agree'].values()))
        self.assertAlmostEqual(out['per_rho']['0.9']['selection_term'], -.18)

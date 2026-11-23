import unittest
from collections import Counter

from experiments import llm_gpu_indomain as s70
from experiments import llm_gpu_noise as stage
from experiments import llm_gpu_threshold as s71
from tests.test_llm_gpu import FakeTokenizer


class NoiseTests(unittest.TestCase):
    def setUp(self):
        texts = [''.join(chr(1000 + (i * 7 + j) % 300) for j in range(70 if i % 4 else 5)) for i in range(2500)]
        self.eligible, self.chunks = s70.eligible_chunks(texts, FakeTokenizer())

    def test_split_excludes_stages_69_to_71_and_garbles_by_permutation(self):
        _, _, _, pop70 = s70.split(self.eligible, self.chunks)
        _, _, _, pop71 = s71.split(self.eligible, self.chunks)
        probe, target, clean, garbled, pop = stage.split(self.eligible, self.chunks)
        used = set().union(*(set(p['probe_articles']) | set(p['target_articles']) for p in (pop70, pop71)))
        self.assertEqual(pop['excluded_stages_69_71'], 3 * 272)
        self.assertFalse(used & (set(pop['probe_articles']) | set(pop['target_articles'])))
        self.assertEqual(len(clean) + len(garbled), len(self.eligible) - 4 * 272)
        self.assertTrue(abs(len(clean) - len(garbled)) <= 1)
        # each garbled row is a permutation of some eligible chunk, not the chunk itself
        originals = {tuple(sorted(c)): c for c in self.chunks}
        row = garbled[0].tolist()
        self.assertIn(tuple(sorted(row)), originals)
        self.assertNotEqual(row, originals[tuple(sorted(row))])

    def test_summary_and_prediction(self):
        plan = dict(rho=[0.5, 0.9], seeds=[1, 2])
        n = dict(random=400, graddot4_batched=200)
        rows = []
        for t in stage.TYPES:
            for rho in plan['rho']:
                for seed in plan['seeds']:
                    rows.append(dict(type=t, rho=rho, seed=seed, arm='random', contaminant_share=rho,
                                     losses={'0': 3., '200': 2.2, '400': 2.0}))
                    rows.append(dict(type=t, rho=rho, seed=seed, arm='graddot4_batched', contaminant_share=.1,
                                     losses={'0': 3., '200': 1.9}))
        cal = dict(costs=dict(random=1., graddot4_batched=2.), updates=n, device='L4')
        out = stage.summarize(rows, cal, plan)
        self.assertAlmostEqual(out['primary']['mean'], -.1)
        self.assertTrue(all(out['prediction_negative_holds'].values()))
        self.assertAlmostEqual(out['types']['news']['0.9']['selection_term'], -.3)

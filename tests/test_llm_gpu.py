import unittest

import numpy as np
import torch

from experiments import llm_gpu as stage
from tests.test_llm_selection import tiny


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': [ord(ch) % 40 for ch in text]}


class LLMGPUTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        g = torch.Generator().manual_seed(3)
        self.pool = torch.randint(0, 40, (30, 9), generator=g)
        self.probe, self.target = self.pool[:3], self.pool[3:7]
        self.device = torch.device('cpu')

    def test_article_sampling_is_disjoint_and_skips_short_articles(self):
        texts = ['x' * (70 if i % 3 else 10) for i in range(600)]
        probe, target, pop = stage.select_articles(texts, FakeTokenizer(), {})
        self.assertEqual((probe.shape, target.shape), ((16, 65), (256, 65)))
        self.assertEqual(pop['eligible_articles'], 400)
        self.assertFalse(set(pop['probe_articles']) & set(pop['target_articles']))
        self.assertTrue(all(i % 3 for i in pop['target_articles']))

    def test_arms_run_and_batched_matches_sequential_alignment(self):
        ids, choices, _ = stage.schedule(0, 5, .5, 20, 10, length=8)
        out = {arm: stage.run(tiny(3), arm, self.pool, self.probe, self.target, ids, choices, 8, [4, 8], self.device)
               for arm in stage.ARMS}
        self.assertEqual(out['random']['choices'], choices[:8].tolist())
        self.assertEqual(sorted(out['loss4']['losses']), ['0', '4', '8'])
        self.assertEqual(out['graddot4']['choices'][:3], out['graddot4_batched']['choices'][:3])
        again = stage.run(tiny(3), 'graddot4_batched', self.pool, self.probe, self.target, ids, choices, 8, [4, 8], self.device)
        self.assertEqual(again['losses'], out['graddot4_batched']['losses'])

    def test_calibration_runs_without_target(self):
        ids, choices, _ = stage.schedule(0, 5, .5, 20, 10, length=4)
        row = stage.run(tiny(3), 'loss4', self.pool, self.probe, None, ids, choices, 4, [], self.device)
        self.assertEqual((row['losses'], len(row['step_seconds'])), ({}, 4))

    def test_budgets_and_summary(self):
        costs = dict(random=.1, loss4=.2, graddot4=.4, graddot4_batched=.25)
        n = stage.budgets(costs)
        self.assertEqual(n, dict(random=400, loss4=200, graddot4=100, graddot4_batched=160))
        plan = dict(models={'qwen2.5-0.5b': {}, 'qwen2.5-1.5b': {}}, rho=[0.0, 0.9], seeds=[1, 2])
        rows = []
        for model in plan['models']:
            for rho in plan['rho']:
                for seed in plan['seeds']:
                    rows.append(dict(model=model, rho=rho, seed=seed, arm='random', english_share=rho,
                                     losses={'0': 4., '100': 3.8, '160': 3.7, '200': 3.6, '400': 3.4}))
                    for arm in stage.GUIDED:
                        rows.append(dict(model=model, rho=rho, seed=seed, arm=arm, english_share=0.,
                                         losses={'0': 4., str(n[arm]): 3.3}))
        cal = {m: dict(costs=costs, updates=n) for m in plan['models']}
        out = stage.summarize(rows, cal, plan)
        self.assertAlmostEqual(out['primary']['mean'], -.1)
        arm = out['models']['qwen2.5-1.5b']['per_rho']['0.9']['arms']['graddot4_batched']
        self.assertAlmostEqual(arm['selection_term'], -.4)
        self.assertAlmostEqual(arm['throughput_term'], .3)

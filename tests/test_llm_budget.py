import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import llm_budget as stage
from experiments import llm_common as llm
from experiments import llm_selection as s66
from tests.test_llm_selection import tiny


class LLMBudgetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        g = torch.Generator().manual_seed(2)
        self.train = torch.randint(0, 40, (30, 9), generator=g)
        self.probe, self.target = self.train[:3], self.train[3:7]

    def test_per_example_gradients_match_separate_backward(self):
        model = tiny(4)
        params = llm.adapter_params(model)
        opt = torch.optim.AdamW(params, lr=5e-3)
        for i in range(3):  # make B nonzero so both dA and dB terms are exercised
            opt.zero_grad()
            llm.lm_loss(model, self.train[10 + i:11 + i]).backward()
            opt.step()
        capture = stage.PerExampleCapture(model)
        batch = self.train[:4]
        opt.zero_grad()
        capture.active = True
        llm.lm_loss(model, batch, 'none').sum().backward()
        capture.active = False
        grads = capture.per_example()
        for i in range(4):
            opt.zero_grad()
            llm.lm_loss(model, batch[i:i + 1]).backward()
            torch.testing.assert_close(grads[i], s66.flat_grad(params), rtol=1e-4, atol=1e-6)

    def test_batched_arm_matches_graddot_choices_and_replays(self):
        ids, choices = stage.schedule(0, 5, .5, 20, 10, length=8)
        with patch.object(llm, 'load_model', tiny):
            a = stage.run('graddot4', 3, self.train, self.probe, self.target, ids, choices, 8, [0])
            b = stage.run('graddot4_batched', 3, self.train, self.probe, self.target, ids, choices, 8, [0])
            c = stage.run('graddot4_batched', 3, self.train, self.probe, self.target, ids, choices, 8, [0])
        self.assertEqual(b, dict(c, step_seconds=b['step_seconds']))
        self.assertEqual(a['choices'][:3], b['choices'][:3])
        self.assertEqual(sorted(b['losses']), ['0', '8'])

    def test_schedule_and_budgets(self):
        ids, choices = stage.schedule(1, 7, .9, 100, 50, length=4000)
        self.assertAlmostEqual(float((ids >= 100).mean()), .9, places=1)
        self.assertTrue(((ids >= 0) & (ids < 150)).all() and ((choices >= 0) & (choices < 4)).all())
        n = stage.budgets(dict(random=.6, loss4=1.3, graddot4=2.7, graddot4_batched=1.8), None)
        self.assertEqual(n, dict(random=96, loss4=44, graddot4=21, graddot4_batched=32))

    def test_summary_decomposition(self):
        plan = dict(rho=[0.0, 0.9], seeds=[1, 2])
        n = dict(random=8, loss4=6, graddot4=2, graddot4_batched=4)
        rows = []
        for rho in plan['rho']:
            for seed in plan['seeds']:
                rows.append(dict(rho=rho, seed=seed, arm='random', english_share=rho, choices=[0] * 8,
                                 losses={'0': 4., '2': 3.9, '4': 3.8, '6': 3.7, '8': 3.6}))
                for arm, loss in (('loss4', 3.75), ('graddot4', 3.5), ('graddot4_batched', 3.55)):
                    rows.append(dict(rho=rho, seed=seed, arm=arm, english_share=0., choices=[0, 1],
                                     losses={'0': 4., str(n[arm]): loss}))
        out = stage.summarize(rows, plan, n)['per_rho']['0.9']['arms']['graddot4']
        self.assertAlmostEqual(out['minus_random'], -.1)
        self.assertAlmostEqual(out['selection_term'] + out['throughput_term'], out['minus_random'])
        self.assertEqual(out['seeds_negative'], 2)

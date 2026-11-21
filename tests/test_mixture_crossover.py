import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import mixture_crossover as stage


PLAN = dict(rho=[0.0, 0.9], phase_A=dict(seeds=[1, 2]), phase_B=dict(seeds=[1, 2], timing_repeats=3))


class MixtureCrossoverTests(unittest.TestCase):
    def test_mixture_respects_language_ranges_and_rate(self):
        rng = np.random.default_rng(0)
        ids = stage.mixture(rng, .9, 100, 50, (20000, 4))
        self.assertTrue(((ids >= 0) & (ids < 150)).all())
        self.assertAlmostEqual(float((ids >= 100).mean()), .9, places=2)
        self.assertTrue((stage.mixture(rng, 0., 100, 50, (100, 4)) < 100).all())

    def test_stage66_lines_are_excluded(self):
        blob = b'\n'.join(bytes([97 + i]) * 40 for i in range(10))
        with patch.object(stage, 'exposed_dev_chunks', lambda n: {1}):
            touched = stage.stage66_line_chunks(blob)
        # Lines 0 (0-39) and 4.. are eligible in stage 66; line 0 touches chunk 0,
        # line 4 (164-203) chunks 2-3, and so on; chunk 1 itself is not added.
        self.assertIn(0, touched)
        self.assertNotIn(1, touched)
        self.assertIn(2, touched)

    def test_phase_a_prediction_rule(self):
        rows = []
        for rho, (rand, loss4, reuse4) in ((0.0, (3.0, 3.01, 2.99)), (0.9, (3.8, 3.85, 3.2))):
            for seed in (1, 2):
                for arm, final, t in (('random', rand, 1.), ('loss4', loss4, 1.2), ('reuse4', reuse4, 6.)):
                    rows.append(dict(rho=rho, seed=seed, arm=arm, loss0=4., loss_final=final,
                                     step_seconds=[t] * 4, english_share=rho))
        out = stage.summarize_a(rows, PLAN)
        self.assertAlmostEqual(out['r']['reuse4'], 6.)
        low, high = out['per_rho']['0.0']['arms']['reuse4'], out['per_rho']['0.9']['arms']['reuse4']
        self.assertFalse(low['predicted_guided_wins'])  # s/a = .01 < 5
        self.assertAlmostEqual(high['s_over_a'], 3.)  # .6 / .2
        self.assertFalse(high['predicted_guided_wins'])  # 3 < 5
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            stage.summarize_a(rows[1:], PLAN)

    def test_phase_b_agreement_counting(self):
        rows, curves = [], {}
        for rho in (0.0, 0.9):
            for seed in (1, 2):
                curves[str(rho), seed] = {10: 3.5, 50: 3.2}
                for repeat in range(3):
                    for arm, final, n in (('random', 3.0, 100), ('loss4', 3.1, 50), ('reuse4', 2.9 + seed * .01, 10)):
                        rows.append(dict(rho=rho, seed=seed, arm=arm, repeat=repeat, final_loss=final, updates=n))
        predictions = dict(per_rho={str(r): dict(arms=dict(loss4=dict(predicted_guided_wins=False),
                                                           reuse4=dict(predicted_guided_wins=r == .9)))
                                    for r in (0.0, 0.9)})
        out = stage.summarize_b(rows, curves, PLAN, predictions)
        self.assertEqual(out['primary'], dict(agreements=3, cases=4))
        arm = out['per_rho']['0.9']['arms']['reuse4']
        self.assertAlmostEqual(arm['selection_term'] + arm['throughput_term'], arm['minus_random'])

import unittest

import numpy as np
import torch

from experiments import pud_policy as original
from experiments import ud_cost_frontier as frontier
from experiments.multilingual_policy import scores


PLAN = dict(warmup_updates=2, warmup_batch_size=4, schedule_length=40, schedule_width=16,
            budget_seconds=3., candidates=dict(random=4, loss4=4, loss16=16, loss64=64, reuse4=4),
            target_seed=6501, target_chunks=50)


class CostFrontierTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(65)
        torch.set_num_threads(1)
        self.pool = torch.randint(0, 256, (40, 65))
        self.model = original.ByteTransformer()
        self.opt = original.optimizer(self.model)

    def test_five_repeats_balance_positions(self):
        for index in range(20):
            orders = [frontier.arm_order(index, j) for j in range(5)]
            for arm in frontier.ARMS:
                self.assertEqual(sorted(o.index(arm) for o in orders), list(range(5)))

    def test_target_excludes_previously_scored_dev_chunks(self):
        target, exposed = frontier.unscored_target(3771, PLAN)
        self.assertEqual(len(target), 50)
        self.assertFalse(set(target.tolist()) & set(exposed))
        self.assertEqual(len(exposed), 190)  # the PADT dev overlap of the two samples is 2

    def test_loss_decision_updates_highest_loss_candidate_on_a_clone(self):
        candidates = self.pool[:16]
        expected = int(torch.argmax(scores(self.model, candidates)))
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        winner, wo, choice = frontier.LossDecision()(self.model, self.opt, candidates, None, 'alignment', 0)
        self.assertEqual(choice, expected)
        torch.testing.assert_close(self.model.state_dict(), before, rtol=0, atol=0)
        reference, ro = original.clone(self.model, self.opt)
        original.update(reference, ro, candidates[choice:choice + 1])
        torch.testing.assert_close(winner.state_dict(), reference.state_dict(), rtol=0, atol=0)

    def test_loss_arm_uses_first_k_columns_and_the_deadline(self):
        ids = np.tile(np.arange(16), (40, 1))
        ids[:, 4:] = 39  # columns beyond K must never be chosen by loss4
        clock = iter([0., 0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1., 3.5, 3.6])
        _, row = frontier.run_arm(self.model, self.opt, self.pool, None, ids, np.zeros(40, int),
                                  'loss4', PLAN, clock=clock.__next__)
        self.assertEqual((row['updates'], row['attempted'], row['candidates']), (5, 6, 4))
        self.assertTrue(all(c < 4 for c in row['choices']))

    def test_random_curve_matches_random_arm_prefix(self):
        _, _, ids, choices, _ = frontier.build_state(self.pool, 'ar', 3, PLAN)
        clock = iter([0.] + [.1 * i for i in range(1, 8)] + [3.1, 3.2])
        model, row = frontier.run_arm(self.model, self.opt, self.pool, None, ids, choices,
                                      'random', PLAN, clock=clock.__next__)
        curve = frontier.random_curve(self.model, self.opt, self.pool, ids, choices, self.pool[:8],
                                      [row['updates']])
        self.assertEqual(curve[row['updates']], original.evaluate(model, self.pool[:8]))

    def test_decomposition_terms_add_to_timed_difference(self):
        plan = dict(PLAN, languages=['ar'], seeds=[1, 2], timing_repeats=5,
                    bootstrap_seed=1, bootstrap_resamples=200)
        rows, curves = [], {}
        for i, seed in enumerate(plan['seeds']):
            curves['ar', seed] = {10: 3.0, 20: 2.9, 30: 2.95}
            for j in range(5):
                for position, arm in enumerate(frontier.arm_order(i, j)):
                    n, loss = dict(random=(100, 2.0), loss4=(30, 2.5), loss16=(20, 2.6),
                                   loss64=(10, 2.9), reuse4=(10, 3.1))[arm]
                    rows.append(dict(language='ar', seed=seed, repeat=j, position=position, arm=arm,
                                     initial_loss=4., final_loss=loss + seed * .01, updates=n,
                                     attempted=n + 1, actual_seconds=3., overrun_seconds=0.))
        states = [dict(language='ar', seed=s) for s in plan['seeds']]
        out = frontier.summarize(rows, curves, states, plan)
        d = out['states'][0]['decomposition']['loss4']
        self.assertAlmostEqual(d['selection_term'] + d['throughput_term'],
                               out['states'][0]['arms']['loss4']['final_loss'] - out['states'][0]['arms']['random']['final_loss'])
        self.assertAlmostEqual(d['r'], 100 / 30)
        self.assertFalse(d['predicted_win'] or d['observed_win'])
        self.assertEqual(out['prediction_agreement']['cases'], 8)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            frontier.summarize(rows + rows[:1], curves, states, plan)

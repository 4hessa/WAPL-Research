import copy
import inspect
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.multilingual_readiness import ByteTransformer
from experiments.multilingual_policy import optimizer, clone, update
from experiments.pud_policy import alignment_score, decision, run_policy, summarize


class PudPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(57)
        self.m = ByteTransformer()
        self.o = optimizer(self.m)
        self.data = torch.randint(0, 256, (8, 65))
        update(self.m, self.o, self.data[:2])

    def test_alignment_sign_and_head_projection(self):
        d = {'other': torch.tensor([2.]), 'head.weight': torch.tensor([[1.], [3.]]), 'head.bias': torch.tensor([2., 4.])}
        pg = torch.tensor([3., 2., 5., 7., 11.])
        # Centered vector [2,-1,1,-1,1] has dot product 13.
        self.assertEqual(alignment_score(d, pg), -13.)
        with self.assertRaises(ValueError):
            alignment_score(d, pg * float('nan'))

    def test_trials_isolated_and_winner_optimizer_preserved(self):
        before = copy.deepcopy(self.m.state_dict())
        opt_before = copy.deepcopy(self.o.state_dict())
        expected, eo = clone(self.m, self.o)
        update(expected, eo, self.data[1:2])
        with patch('experiments.pud_policy.alignment_score', side_effect=[1., 3., 3., 2.]):
            actual, ao, chosen = decision(self.m, self.o, self.data[:4], self.data[4:], 'alignment', 0)
        self.assertEqual(chosen, 1)
        for k, v in before.items():
            torch.testing.assert_close(self.m.state_dict()[k], v, rtol=0, atol=0)
            torch.testing.assert_close(actual.state_dict()[k], expected.state_dict()[k], rtol=0, atol=0)
        for wanted, got in [(opt_before, self.o.state_dict()), (eo.state_dict(), ao.state_dict())]:
            self.assertEqual(wanted['param_groups'], got['param_groups'])
            for k, state in wanted['state'].items():
                for name, value in state.items():
                    torch.testing.assert_close(got['state'][k][name], value, rtol=0, atol=0)

    def test_deadline_accept_and_reject_for_both_arms(self):
        for arm in ['random', 'alignment']:
            for finish, accepted in [(3.01, 0), (3., 1)]:
                times = iter([0., 0., finish, 3.02, 3.03] if accepted else [0., 0., finish, 3.02])
                m, row = run_policy(self.m, self.o, self.data, self.data[4:], np.tile(np.arange(4), (2, 1)), np.array([0, 0]), arm, 'time', clock=lambda: next(times))
                self.assertEqual(row['updates'], accepted)
                if not accepted:
                    for k, v in self.m.state_dict().items():
                        torch.testing.assert_close(m.state_dict()[k], v, rtol=0, atol=0)
                self.assertGreater(row['overrun_seconds'], 0.)

    def test_target_not_in_selector_interface(self):
        self.assertEqual(list(inspect.signature(decision).parameters), ['model', 'opt', 'candidates', 'probe', 'arm', 'random_choice'])
        self.assertNotIn('target', inspect.signature(run_policy).parameters)

    def test_schedule_exhaustion_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Schedule exhausted'):
            run_policy(self.m, self.o, self.data, self.data[4:], np.arange(4)[None], np.array([0]), 'random', 'time', clock=lambda: 0.)

    def synthetic(self):
        plan = {'languages': ['ar'], 'folds': [0, 1], 'seeds': [10, 11], 'arms': ['random', 'alignment'], 'budgets': {'seconds': 3., 'time_repeats': 2, 'fixed_updates': 1}, 'bootstrap': {'seed': 5701, 'resamples': 100}}
        rows = []
        for seed in plan['seeds']:
            for fold in plan['folds']:
                for mode in ['time', 'updates']:
                    for repeat in range(2 if mode == 'time' else 1):
                        for arm in plan['arms']:
                            # Paired differences 2 and 4, irrespective of fold/repeat offsets.
                            loss = seed + 10*fold + repeat + (2*(seed-9) if arm == 'alignment' else 0)
                            rows.append(dict(language='ar', fold=fold, seed=seed, mode=mode, repeat=repeat, arm=arm, initial_loss=1., final_loss=loss, updates=1, attempted=1, choices=[0], actual_seconds=2., last_accepted_seconds=2., overrun_seconds=0., schedule_sha256=str((seed, fold))))
        return plan, rows

    def test_pairing_and_nested_aggregation(self):
        plan, rows = self.synthetic()
        out, comp = summarize(list(reversed(rows)), plan)
        self.assertEqual(comp['ar']['time']['per_seed_difference'], [2., 4.])
        self.assertEqual(comp['ar']['time']['difference'], 3.)
        self.assertEqual(out['ar']['time']['random']['per_seed_mean_loss'], [15.5, 16.5])

    def test_grid_nonfinite_and_pairing_failures(self):
        plan, rows = self.synthetic()
        for bad in [rows[:-1], rows+[rows[0]]]:
            with self.assertRaises(ValueError):
                summarize(bad, plan)
        for field, value in [('final_loss', float('nan')), ('schedule_sha256', 'wrong'), ('last_accepted_seconds', 3.1)]:
            bad = copy.deepcopy(rows)
            bad[0][field] = value
            with self.assertRaises(ValueError):
                summarize(bad, plan)

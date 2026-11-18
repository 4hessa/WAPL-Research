"""Checks for the disjoint timing boundaries and behavior of the wrapped decision."""
import unittest
from unittest.mock import patch

import torch

from experiments import pud_policy as policy
from experiments.pud_policy_profile import DecisionTimer, check_winners, timed_decision


class DecisionProfileTests(unittest.TestCase):
    def test_component_boundaries_include_delta_and_probe_preparation(self):
        timer = DecisionTimer(clock=iter([1, 3, 11, 15, 20, 25, 32, 35]).__next__)
        with patch.object(policy, 'clone', return_value=object()), \
             patch.object(policy, 'flatten', return_value=object()), \
             patch.object(policy, 'update', return_value=None), \
             patch.object(policy, 'alignment_score', return_value=2.0):
            timer.clone(None, None)
            timer.flatten({})
            timer.clone(None, None)
            timer.update(None, None, None)
            timer.alignment_score({}, None)
        self.assertEqual(timer.finish(40, 1), dict(probe_clone=2, probe_gradient=8,
                         candidate_clones=5, candidate_updates=7, candidate_delta_scores=3, residual=15))

    def test_wrapping_preserves_actual_winner_and_optimizer_and_local_globals(self):
        torch.manual_seed(59)
        torch.set_num_threads(1)
        model = policy.ByteTransformer()
        opt = policy.optimizer(model)
        data = torch.randint(0, 256, (8, 65))
        policy.update(model, opt, data[:2])
        inputs = (model, opt, data[:4], data[4:], 'alignment', 0)
        raw, _, _ = timed_decision(inputs, False)
        hooked, total, components = timed_decision(inputs, True)
        check_winners(raw, hooked)
        self.assertIs(policy.decision.__globals__['clone'], policy.clone)
        self.assertEqual(total, sum(components.values()))
        self.assertTrue(all(value >= 0 for value in components.values()))

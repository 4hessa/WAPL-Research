import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import pud_policy as original
from experiments import pud_reuse
from experiments.pud_policy_profile import check_winners


class ProbeReuseTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(60)
        torch.set_num_threads(1)
        self.model = original.ByteTransformer()
        self.opt = original.optimizer(self.model)
        self.pool = torch.randint(0, 256, (12, 65))
        self.probe = self.pool[4:8]
        original.update(self.model, self.opt, self.pool[8:])
        self.ids = np.tile(np.arange(4), (10, 1))
        self.choices = np.zeros(10, dtype=int)

    def test_refreshes_on_steps_zero_and_four_and_preserves_current_optimizer(self):
        current = self.model, self.opt
        cache = pud_reuse.CachedDecision()
        with patch.object(original, 'byte_loss', wraps=original.byte_loss) as probe_loss:
            for step in range(5):
                inputs = (*current, self.pool[self.ids[step]], self.probe, 'alignment', 0)
                cached = cache(*inputs)
                if step in (0, 4):
                    reference = original.decision(*inputs)
                    check_winners(reference, cached)
                current = cached[:2]
            # Two cached and two separately recomputed probes.
            self.assertEqual(probe_loss.call_count, 4)
        self.assertEqual((cache.attempts, cache.refreshes), (5, 2))

    def test_reject_late_refresh_and_start_next_run_with_new_cache(self):
        for _ in range(2):
            # Accept the first four decisions. The fifth refresh finishes late.
            clock = iter([0., 0., .1, .2, .3, .4, .5, .6, .7, .8, 3.1, 3.2])
            with patch.object(original, 'byte_loss', wraps=original.byte_loss) as loss:
                got, row = pud_reuse.timed_run(self.model, self.opt, self.pool, self.probe,
                                               self.ids, self.choices, 'reuse4',
                                               clock=clock.__next__)
            expected, op = original.clone(self.model, self.opt)
            for index in row['choices']:
                original.update(expected, op, self.pool[index:index+1])
            torch.testing.assert_close(got.state_dict(), expected.state_dict(), rtol=0, atol=0)
            self.assertEqual(row['updates'], 4)
            self.assertEqual((row['accepted_refreshes'], row['attempted_refreshes']), (1, 2))
            self.assertEqual(loss.call_count, 2)

    def test_stale_decisions_are_committed_to_diagnostic_trajectory(self):
        rows = list(pud_reuse.drift_checks(self.model, self.opt, self.pool, self.probe,
                                           self.ids, self.choices, 5))
        self.assertEqual([r['age'] for r in rows], [0, 1, 2, 3, 0])
        self.assertEqual([r['refresh_parity'] for r in rows], [True, None, None, None, True])
        cache = pud_reuse.CachedDecision()
        current = self.model, self.opt
        for row in rows:
            inputs = (*current, self.pool[:4], self.probe, 'alignment', 0)
            fresh = original.decision(*inputs)
            reused = cache(*inputs)
            self.assertEqual(row['fresh_choice'], fresh[2])
            self.assertEqual(row['reuse_choice'], reused[2])
            current = reused[:2]

import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import pud_policy as original
from experiments.pud_reuse_matched import run_arm, state_fingerprint


class MatchedBudgetTests(unittest.TestCase):
    def test_checkpoint_reads_do_not_change_reuse_trajectory_or_base(self):
        torch.set_num_threads(1)
        torch.manual_seed(61)
        base = original.ByteTransformer()
        opt = original.optimizer(base)
        pool = torch.randint(0, 256, (12, 65))
        original.update(base, opt, pool[8:])
        model_hash = state_fingerprint(base.state_dict())
        opt_hash = state_fingerprint(opt.state_dict())
        ids = np.tile(np.arange(4), (5, 1))
        inputs = (base, opt, pool, pool[4:8], ids, np.zeros(5, dtype=int), pool[8:], 'reuse4')
        result = run_arm(*inputs, {'updates': 5, 'checkpoints': [0, 2, 5], 'refresh_interval': 4})
        sparse = run_arm(*inputs, {'updates': 5, 'checkpoints': [0, 5], 'refresh_interval': 4})
        self.assertEqual(set(result['losses']), {'0', '2', '5'})
        self.assertEqual(len(result['choices']), 5)
        self.assertEqual(result['choices'], sparse['choices'])
        self.assertEqual(result['losses']['5'], sparse['losses']['5'])
        self.assertEqual(result['refreshes'], 2)
        self.assertEqual((model_hash, opt_hash),
                         (state_fingerprint(base.state_dict()), state_fingerprint(opt.state_dict())))

    def test_checkpoint_labels_refer_to_committed_updates(self):
        torch.set_num_threads(1)
        base = original.ByteTransformer()
        opt = original.optimizer(base)
        pool = torch.randint(0, 256, (4, 65))
        # Here the optimizer has performed one warmup update. Expose its actual
        # step count through the checkpoint call in lieu of a target loss.
        original.update(base, opt, pool)
        counter = []
        saved = original.decision
        def count_updates(*args):
            result = saved(*args)
            counter.append(int(next(iter(result[1].state.values()))['step']))
            return result
        with patch.object(original, 'decision', side_effect=count_updates), \
             patch.object(original, 'evaluate', side_effect=lambda model, target: counter[-1] if counter else 1):
            r = run_arm(base, opt, pool, pool, np.tile(np.arange(4), (4, 1)), [3, 2, 1, 0],
                        pool, 'random', {'updates': 4, 'checkpoints': [0, 1, 4], 'refresh_interval': 4})
        self.assertEqual(r['losses'], {'0': 1, '1': 2, '4': 5})
        self.assertEqual(r['choices'], [3, 2, 1, 0])
        self.assertEqual(r['refreshes'], 0)

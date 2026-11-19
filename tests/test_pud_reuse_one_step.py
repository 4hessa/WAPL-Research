import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import pud_policy as original
from experiments.pud_reuse_one_step import aggregate, paired_steps, read_loss, summarize


class OneStepTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(62)
        torch.set_num_threads(1)
        self.model = original.ByteTransformer()
        self.opt = original.optimizer(self.model)
        self.data = torch.randint(0, 256, (16, 65))
        original.update(self.model, self.opt, self.data[8:12])

    def test_same_choice_both_branches_scored_and_reuse_chain_preserved(self):
        # Four identical candidates force a tie and hence first-choice parity.
        ids = np.zeros((4, 4), dtype=int)
        with patch.object(original, 'evaluate', wraps=original.evaluate) as score:
            rows = list(paired_steps(self.model, self.opt, self.data, self.data[4:8], ids,
                                     [0]*4, self.data[12:], 4, 4))
            self.assertEqual(score.call_count, 8)
        for i, r in enumerate(rows):
            self.assertEqual(r['age'], i)
            self.assertEqual((r['fresh_choice'], r['reuse_choice'], r['delta']), (0, 0, 0.))
            for suffix in ('model_sha256', 'optimizer_sha256'):
                self.assertEqual(r['reuse_' + suffix], r['fresh_' + suffix])
                if i:
                    self.assertEqual(r['parent_' + suffix], rows[i-1]['reuse_' + suffix])

    def test_evaluation_cannot_change_target_branch(self):
        def mutate(model, target):
            with torch.no_grad():
                next(model.parameters()).add_(1)
            return 0.
        with patch.object(original, 'evaluate', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'mutated'):
                read_loss(self.model, self.opt, self.data)

    def test_null_conditional_excludes_no_agreements_from_primary(self):
        rows = []
        for seed in (1, 2):
            for step in range(4):
                delta = .75 if seed == 2 and step == 1 else 0.
                rows.append(dict(language='ar', fold=0, seed=seed, step=step, age=step,
                                 delta=delta, agreement=(delta == 0.)))
        result = summarize(rows, dict(languages=['ar'], folds=[0], seeds=[1, 2]), 4, 4)
        stale = result['languages']['ar']['stale']
        self.assertEqual((stale['n'], stale['changed'], stale['tied'], stale['mean_delta']),
                         (6, 1, 5, .125))
        self.assertEqual(stale['changed_only']['mean_delta'], .75)
        self.assertIsNone(aggregate(rows[:2])['changed_only']['mean_delta'])

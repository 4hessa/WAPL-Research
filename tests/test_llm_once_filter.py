import json
import unittest

import numpy as np
import torch

from experiments import llm_once_filter as stage
from experiments import llm_qa_mix as s74
from tests.test_llm_gpu import FakeTokenizer
from tests.test_llm_qa_mix import mmlu_row


class OnceFilterTests(unittest.TestCase):
    def test_protocol_matches_implementation(self):
        plan = json.loads(stage.PROTOCOL.read_text())
        self.assertEqual(plan['stage'], 75)
        self.assertEqual(tuple(plan['design']['sigma']), stage.SIGMAS)
        self.assertEqual(tuple(plan['design']['arms']), stage.ARMS)
        self.assertEqual(len(stage.run_grid(plan['design']['seeds'])), 30)

    def test_schedule_extends_stage74(self):
        a = stage.schedule(0, 37500, 1511)
        rand, filt, subset, _ = s74.schedule(0, 37500)
        self.assertTrue((a['rand'] == rand).all() and (a['filt'] == filt).all() and (a['subset512'] == subset).all())
        self.assertEqual(a['sha256'], stage.schedule(0, 37500, 1511)['sha256'])
        self.assertNotEqual(a['sha256'], stage.schedule(0, 37501, 1511)['sha256'])
        self.assertEqual(len(a['once']), stage.BUDGET)
        self.assertEqual(sorted(a['once'][:1511].tolist()), list(range(1511)))   # one pass by N
        self.assertEqual(len(set(a['subset_n'].tolist())), 1511)

    def test_orders_and_views(self):
        n = 1511
        sch = stage.schedule(0, 37502, n)
        kept = np.sort(np.random.default_rng(0).choice(2048, n, replace=False))
        for arm in stage.ARMS:
            order, members = stage.training_order(arm, sch, kept if arm in stage.FILTER_ARMS else None)
            self.assertEqual(len(order), stage.BUDGET)
            self.assertTrue(set(order.tolist()) <= set(np.asarray(members).tolist()))
            v = stage.views(order, members, n)
            if arm in ('align_once', 'random_once'):
                self.assertEqual(v, dict(max=1, mean=1.0))
                self.assertEqual(stage.views(order, members, stage.BUDGET)['max'], 2)
        order, members = stage.training_order('align_q', sch, kept[:512])
        self.assertAlmostEqual(stage.views(order, members, n)['mean'], n / 512)

    def test_arm_rotation(self):
        self.assertEqual(stage.arm_order(0), stage.ARMS)
        self.assertEqual(stage.arm_order(6)[0], 'align_q')
        grid = stage.run_grid([1, 2])
        self.assertEqual(len(set(grid)), 10)

    def test_target_criterion_e_uses_content_fields(self):
        tok = FakeTokenizer()
        trained = mmlu_row(1, 'zzzz qqqq')
        cands = [mmlu_row(10, 'xx abcdefghij yy'), mmlu_row(11, 'totally other', opts=('mmmm', 'nnnn'))]
        pool = torch.tensor([tok('abcdefghijkl')['input_ids'][:10]])
        counts, eligible = stage.target_candidates([trained] + cands, tok, [pool], {1}, set())
        self.assertEqual(eligible, [2])
        self.assertEqual(counts['decontamination']['trained_tokenization_8gram_after_a_to_d'], 1)
        self.assertEqual(len(stage.sample_target(list(range(600)))), stage.TARGET_ITEMS)

    def test_summary(self):
        plan = dict(design=dict(sigma=[0.25], seeds=[1, 2]))
        L = dict(random=(2.8, 2.7), align_q=(2.6, 2.9), random512=(2.9, 3.), align_once=(2.5, 2.6),
                 random_once=(2.75, 2.8))
        rows = []
        for seed in (1, 2):
            for arm, (ln, lb) in L.items():
                r = dict(sigma=0.25, seed=seed, arm=arm, losses={'0': 3., '1500': ln, '2400': lb},
                         accuracy={'0': .3, '1500': .3, '2400': .3})
                if arm != 'random':
                    r['qa_share'] = .35 if arm != 'random512' else .25
                if arm in stage.FILTER_ARMS:
                    r['scoring_seconds'] = 100.
                rows.append(r)
        cal = dict(costs=dict(random=.1, align_grad=90.), updates=1500, device='L4')
        out = stage.summarize(rows, cal, plan)
        s = out['sigmas']['0.25']
        self.assertAlmostEqual(out['primary']['mean'], -.2)
        self.assertAlmostEqual(s['matched_once']['mean'], -.25)
        self.assertAlmostEqual(s['fixing_repetition']['mean'], -.1)
        self.assertAlmostEqual(s['charged_q']['mean'], -.1)
        self.assertAlmostEqual(s['throughput_term_once'], .1)
        self.assertFalse(out['prediction_holds']['replication: align_q charged positive'])
        with self.assertRaises(ValueError):
            stage.summarize(rows[:-1], cal, plan)


if __name__ == '__main__':
    unittest.main()

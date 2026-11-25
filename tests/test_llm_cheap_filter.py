import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments import llm_cheap_filter as stage


class CheapFilterTests(unittest.TestCase):
    def test_protocol_matches_implementation(self):
        plan = json.loads(stage.PROTOCOL.read_text())
        self.assertEqual(plan['stage'], 76)
        self.assertEqual(tuple(plan['design']['arms']), stage.ARMS)
        self.assertEqual(plan['design']['seeds'], list(range(37600, 37606)))
        self.assertEqual(len(stage.run_grid(plan['design']['seeds'])), 24)
        self.assertIn('B = 3000', plan['design']['budget'])
        self.assertEqual(stage.BUDGET, 3000)
        self.assertEqual(stage.POOL_SIZE, stage.N_QA + 2 * stage.N_WEB)

    def test_web_draws(self):
        k, r = stage.web_draws(*stage.WEB_ELIGIBLE)
        self.assertEqual((len(set(k.tolist())), len(set(r.tolist()))), (3072, 3072))
        self.assertTrue((k == np.random.default_rng(7611).choice(41443, 3072, replace=False)).all())
        with self.assertRaises(ValueError):
            stage.web_draws(41442, 18831)

    def test_schedule(self):
        for n in (1000, 1510, 2048):
            a = stage.schedule(37600, n)
            self.assertEqual(a['sha256'], stage.schedule(37600, n)['sha256'])
            self.assertEqual(len(a['once']), stage.BUDGET)
            self.assertEqual(sorted(a['once'][:n].tolist()), list(range(n)))
            self.assertEqual(len(set(a['subset_n'].tolist())), n)
            self.assertTrue((a['qa_n'] < stage.N_QA).all() and len(set(a['qa_n'].tolist())) == n)
            self.assertTrue((a['rand'] < stage.POOL_SIZE).all())
        self.assertNotEqual(stage.schedule(37600, 1510)['sha256'], stage.schedule(37601, 1510)['sha256'])
        a = stage.schedule(37600, 1000)
        self.assertEqual(sorted(a['once'][2000:3000].tolist()), list(range(1000)))  # third pass

    def test_orders_views_and_shares(self):
        n = 1510
        sch = stage.schedule(37602, n)
        kept = np.sort(np.random.default_rng(0).choice(stage.POOL_SIZE, n, replace=False))
        for arm in stage.ARMS:
            order, members = stage.training_order(arm, sch, kept if arm == 'align05_once' else None)
            self.assertEqual(len(order), stage.BUDGET)
            self.assertTrue(set(order.tolist()) <= set(np.asarray(members).tolist()))
            if arm != 'random':
                self.assertEqual(stage.views(order, members, n), dict(max=1, mean=1.0))
                self.assertEqual(stage.views(order, members, stage.BUDGET)['max'], 2)
        _, members = stage.training_order('qa_once', sch)
        self.assertEqual(stage.qa_share(members), 1.0)
        self.assertEqual(stage.qa_share([0, 2047, 2048, 8191]), 0.5)
        q = stage.random_qa(np.array([0, 0, 5, 3000, 9]), 4)
        self.assertEqual(q, dict(drawn=3, distinct=2))

    def test_arm_rotation(self):
        self.assertEqual(stage.arm_order(0), stage.ARMS)
        self.assertEqual(stage.arm_order(5)[0], 'align05_once')
        self.assertEqual(len(set(stage.run_grid([1, 2, 3]))), 12)

    def test_summary(self):
        plan = dict(design=dict(seeds=[1, 2]))
        L = dict(random=(2.8, 2.7), align05_once=(2.5, 2.45), random_once=(2.75, 2.7), qa_once=(2.45, 2.4))
        rows = []
        for seed in (1, 2):
            for arm, (ln, lb) in L.items():
                r = dict(seed=seed, arm=arm, losses={'0': 3., '1500': ln, '3000': lb},
                         accuracy={'0': .3, '1500': .3, '3000': .3})
                if arm == 'random':
                    r['random_qa'] = {'1500': dict(drawn=375, distinct=340), '3000': dict(drawn=750, distinct=630)}
                else:
                    r['qa_share'] = {'align05_once': .9, 'random_once': .25, 'qa_once': 1.}[arm]
                if arm == 'align05_once':
                    r['scoring_seconds'] = 160.
                rows.append(r)
        cal = dict(costs=dict(random=.11, proxy_align_grad=165.), updates=1500, device='L4')
        out = stage.summarize(rows, cal, plan)
        a = out['arms']
        self.assertAlmostEqual(out['primary']['mean'], -.2)
        self.assertAlmostEqual(a['matched']['mean'], -.25)
        self.assertAlmostEqual(a['ranking_vs_oracle']['mean'], .05)
        self.assertAlmostEqual(a['oracle_charged']['mean'], -.25)
        self.assertAlmostEqual(a['selection_term'], -.3)
        self.assertAlmostEqual(a['throughput_term'], .1)
        self.assertAlmostEqual(out['scoring_in_random_steps'], 1500.)
        self.assertTrue(all(out['prediction_holds'].values()))
        with self.assertRaises(ValueError):
            stage.summarize(rows[:-1], cal, plan)

    def test_backup_retries_push_and_logs_stderr(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            journal = d / 'j.jsonl'
            journal.write_text(json.dumps(dict(kind='run')) + '\n')
            calls = []

            def fake(cmd, check=False, capture_output=False, text=False):
                calls.append(cmd[3])
                if cmd[3] == 'diff':
                    return subprocess.CompletedProcess(cmd, 1)
                if cmd[3] == 'push' and calls.count('push') == 1:
                    raise subprocess.CalledProcessError(1, cmd, stderr='remote hung up')
                return subprocess.CompletedProcess(cmd, 0)

            import contextlib
            import io
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                stage.backup(journal, None, d / 'wt', 'b', run=fake)
            self.assertEqual(calls.count('push'), 2)
            self.assertIn('remote hung up', buf.getvalue())
            self.assertIn('backup pushed: 1 finished runs (retry)', buf.getvalue())
            self.assertTrue((d / 'wt' / stage.INTERIM / 'journal.jsonl').exists())


if __name__ == '__main__':
    unittest.main()

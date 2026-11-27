import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments import llm_supply_filter as stage


class SupplyFilterTests(unittest.TestCase):
    def test_protocol_matches_implementation(self):
        plan = json.loads(stage.PROTOCOL.read_text())
        self.assertEqual(plan['stage'], 78)
        self.assertEqual(tuple(plan['design']['arms']), stage.ARMS)
        self.assertEqual(tuple(plan['design']['levels']), stage.LEVELS)
        self.assertEqual(plan['design']['seeds'], list(range(37800, 37806)))
        self.assertIn('amendment_1', plan)
        self.assertIn('7822', plan['target']['items'])
        self.assertEqual(len(stage.run_grid(plan['design']['seeds'])), 54)
        self.assertEqual((stage.BUDGET, stage.N, stage.POOL_SIZE), (3000, 1633, 8192))

    def test_qa_subsets_nested(self):
        s = stage.qa_subsets()
        self.assertEqual({q: len(set(v.tolist())) for q, v in s.items()}, {1024: 1024, 1633: 1633, 2048: 2048})
        self.assertTrue(set(s[1024].tolist()) <= set(s[1633].tolist()) <= set(s[2048].tolist()))
        self.assertTrue((np.diff(s[1024]) > 0).all() and (np.diff(s[1633]) > 0).all())
        self.assertTrue((stage.qa_subsets()[1633] == s[1633]).all())

    def test_web_counts_and_members(self):
        self.assertEqual({q: stage.web_counts(q) for q in stage.LEVELS},
                         {1024: (3584, 3584), 1633: (3280, 3279), 2048: (3072, 3072)})
        k76, r76 = np.arange(0, 3072), np.arange(0, 3072)
        k_extra = stage.extra_draws(5000, k76, 7811, 512)
        r_extra = stage.extra_draws(4000, r76, 7812, 512)
        self.assertFalse(set(k_extra.tolist()) & set(k76.tolist()))
        self.assertEqual(len(set(k_extra.tolist())), 512)
        prev = None
        for q in (2048, 1633, 1024):
            km, rm = stage.level_members(q, k76, r76, k_extra, r_extra)
            self.assertEqual(q + len(km) + len(rm), stage.POOL_SIZE)
            self.assertEqual((len(km), len(rm)), stage.web_counts(q))
            self.assertTrue((km[:3072] == k76).all())
            if prev is not None:  # nested: fewer QA, more web, earlier web kept
                self.assertTrue(set(prev[0].tolist()) <= set(km.tolist()) and set(prev[1].tolist()) <= set(rm.tolist()))
            prev = (km, rm)

    def test_schedule_and_orders(self):
        a = stage.schedule(37800)
        self.assertEqual(a['sha256'], stage.schedule(37800)['sha256'])
        self.assertNotEqual(a['sha256'], stage.schedule(37801)['sha256'])
        self.assertEqual(len(a['rand']), stage.BUDGET)
        self.assertEqual(sorted(a['once'].tolist()), list(range(stage.N)))
        kept = np.sort(np.random.default_rng(0).choice(stage.POOL_SIZE, stage.N, replace=False))
        for arm in stage.ARMS:
            order, members = stage.training_order(arm, a, kept if arm == 'align05_once' else None)
            self.assertGreaterEqual(len(order), stage.updates(arm))
            if arm != 'random':
                self.assertEqual(stage.views(order, members, stage.N), dict(max=1, mean=1.0))
                self.assertEqual(stage.checkpoints(arm), {stage.N})
        self.assertEqual(stage.checkpoints('random'), {0, stage.N, stage.BUDGET})
        self.assertEqual(stage.qa_share([0, 1023, 1024, 5000], 1024), .5)
        self.assertEqual(stage.random_qa(np.array([0, 0, 5, 3000, 9]), 4, 1024), dict(drawn=3, distinct=2))

    def test_grid_rotation(self):
        g = stage.run_grid([1, 2, 3])
        self.assertEqual(len(set(g)), 27)
        self.assertEqual(g[0], (1, 1024, 'random'))
        self.assertEqual(g[9], (2, 1633, 'align05_once'))
        self.assertEqual(g[18], (3, 2048, 'random_once'))

    def test_summary(self):
        plan = dict(design=dict(seeds=[1, 2]))
        n, B = str(stage.N), str(stage.BUDGET)
        A = {1024: 3.15, 1633: 3.10, 2048: 3.05}
        rows = []
        for seed in (1, 2):
            for q in stage.LEVELS:
                rows.append(dict(seed=seed, level=q, arm='random', losses={'0': 3.9, n: 3.25, B: 3.25},
                                 accuracy={'0': .3, n: .35, B: .35},
                                 random_qa={n: dict(drawn=300, distinct=280), B: dict(drawn=600, distinct=500)}))
                rows.append(dict(seed=seed, level=q, arm='align05_once', losses={n: A[q]}, accuracy={n: .36},
                                 qa_share={1024: .6, 1633: .85, 2048: .99}[q], scoring_seconds=157.))
                rows.append(dict(seed=seed, level=q, arm='random_once', losses={n: 3.3}, accuracy={n: .34},
                                 qa_share=q / 8192))
        cal = dict(costs=dict(random=.11, proxy_align_grad=150.), N_cal=1633, device='L4')
        out = stage.summarize(rows, cal, plan)
        self.assertAlmostEqual(out['primary']['mean'], .10)
        self.assertEqual(out['primary']['verdict'], 'supported')
        r = out['results']
        self.assertAlmostEqual(r['levels']['2048']['charged']['mean'], -.20)
        self.assertAlmostEqual(r['levels']['1024']['matched']['mean'], -.15)
        self.assertAlmostEqual(r['levels']['2048']['throughput_term'], 0.)
        self.assertTrue(all(out['prediction_holds'].values()))
        with self.assertRaises(ValueError):
            stage.summarize(rows[:-1], cal, plan)
        bad = [dict(r, losses=dict(r['losses'], **{'0': 4.0})) if r['arm'] == 'random' and r['seed'] == 2 else r
               for r in rows]
        with self.assertRaises(ValueError):
            stage.summarize(bad, cal, plan)

    def test_backup_retries_push(self):
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

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                stage.backup(journal, None, d / 'wt', 'b', run=fake)
            self.assertEqual(calls.count('push'), 2)
            self.assertIn('backup pushed: 1 finished runs (retry)', buf.getvalue())
            self.assertTrue((d / 'wt' / stage.INTERIM / 'journal.jsonl').exists())


if __name__ == '__main__':
    unittest.main()

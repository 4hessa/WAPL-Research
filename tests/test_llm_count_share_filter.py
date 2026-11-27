import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments import llm_count_share_filter as stage
from experiments import llm_supply_filter as s78


class CountShareFilterTests(unittest.TestCase):
    def test_protocol_matches_implementation(self):
        plan = json.loads(stage.PROTOCOL.read_text())
        stage.check_protocol(plan)
        self.assertEqual(plan['design']['seeds'], list(range(37900, 37906)))
        self.assertIn('7922', plan['target']['items'])
        self.assertIn('7903', plan['primary']['definition'])
        self.assertIn('0.0125', plan['primary']['definition'])
        self.assertEqual(len(stage.run_grid(plan['design']['seeds'])), 72)
        for c, (m, s, q) in stage.CELLS.items():
            self.assertEqual(m * s, q)
        bad = json.loads(stage.PROTOCOL.read_text())
        bad['design']['N'] = 1648
        with self.assertRaises(ValueError):
            stage.check_protocol(bad)

    def test_qa_subsets_nested_and_q1024_is_stage78(self):
        s = stage.qa_subsets()
        self.assertEqual({q: len(set(v.tolist())) for q, v in s.items()}, {256: 256, 512: 512, 1024: 1024})
        self.assertTrue(set(s[256].tolist()) <= set(s[512].tolist()) <= set(s[1024].tolist()))
        self.assertTrue(all((np.diff(v) > 0).all() for v in s.values()))
        self.assertTrue((s[1024] == s78.qa_subsets()[1024]).all())
        q1024 = s78.qa_subsets()[1024]
        self.assertTrue((s[512] == np.sort(q1024[np.random.default_rng(7901).choice(1024, 512, replace=False)])).all())
        self.assertTrue((stage.qa_subsets()[256] == s[256]).all())

    def test_stage78_q1024_subset_digest(self):
        import hashlib
        rec = stage.stage78_population()['levels']['1024']
        self.assertEqual(hashlib.sha256(stage.qa_subsets()[1024].tobytes()).hexdigest(), rec['qa_subset_sha256'])
        self.assertEqual(len(stage.stage78_population()['target_items']), 512)

    def test_web_counts_lists_and_nesting(self):
        self.assertEqual({c: stage.web_counts(c) for c in stage.CELLS},
                         {'a': (1920, 1920), 'b': (1792, 1792), 'c': (3840, 3840), 'd': (3584, 3584)})
        k76 = np.random.default_rng(1).choice(6000, 3072, replace=False)
        r76 = np.random.default_rng(2).choice(5000, 3072, replace=False)
        kl, rl, nk, nr = stage.web_lists(k76, r76, 6000, 5000)
        self.assertEqual((nk, nr, len(kl), len(rl)), (768, 768, 3840, 3840))
        self.assertTrue((kl[:3072] == k76).all() and (rl[:3072] == r76).all())
        self.assertEqual(len(set(kl.tolist())), 3840)
        # the first 512 extras are those of stage 78
        self.assertTrue((kl[3072:3584] == s78.extra_draws(6000, k76, 7811, 512)).all())
        self.assertTrue((rl[3072:3584] == s78.extra_draws(5000, r76, 7812, 512)).all())
        km78, rm78 = s78.level_members(1024, k76, r76, s78.extra_draws(6000, k76, 7811, 512),
                                       s78.extra_draws(5000, r76, 7812, 512))
        mem = {c: stage.cell_members(c, kl, rl) for c in stage.CELLS}
        self.assertTrue((mem['d'][0] == km78).all() and (mem['d'][1] == rm78).all())
        sets = {c: (set(k.tolist()), set(r.tolist())) for c, (k, r) in mem.items()}
        for small, big in (('b', 'a'), ('a', 'd'), ('d', 'c')):
            self.assertTrue(sets[small][0] <= sets[big][0] and sets[small][1] <= sets[big][1])
        with self.assertRaises(ValueError):
            stage.cell_members('c', kl[:3000], rl)

    def test_schedule_and_orders(self):
        for m in stage.SIZES:
            a = stage.schedule(37900, m)
            self.assertEqual(a['sha256'], stage.schedule(37900, m)['sha256'])
            self.assertEqual(len(a['rand']), stage.BUDGET)
            self.assertLess(a['rand'].max(), m)
            self.assertLess(a['subset_n'].max(), m)
            self.assertEqual(sorted(a['once'].tolist()), list(range(stage.N)))
            kept = np.sort(np.random.default_rng(0).choice(m, stage.N, replace=False))
            for arm in stage.ARMS:
                order, members = stage.training_order(arm, a, m, kept if arm == 'align05_once' else None)
                self.assertGreaterEqual(len(order), stage.updates(arm))
                if arm == 'random':
                    self.assertEqual(len(members), m)
                else:
                    self.assertEqual(stage.views(order, members, stage.N), dict(max=1, mean=1.0))
        self.assertNotEqual(stage.schedule(37900, 4096)['sha256'], stage.schedule(37900, 8192)['sha256'])
        self.assertNotEqual(stage.schedule(37900, 4096)['sha256'], stage.schedule(37901, 4096)['sha256'])
        # the M = 8192 schedule equals the stage-78 construction
        self.assertTrue((stage.schedule(37800, 8192)['rand'] == s78.schedule(37800)['rand']).all())
        d = stage.schedule_digests([1, 2])
        self.assertEqual(set(d), {'1', '2'})
        self.assertEqual(set(d['1']), {'4096', '8192'})
        self.assertEqual(stage.qa_share([0, 255, 256, 3000], 256), .5)
        self.assertEqual(stage.qa_count([0, 255, 256, 3000], 256), 2)
        self.assertEqual(stage.random_qa(np.array([0, 0, 5, 3000, 9]), 4, 512), dict(drawn=3, distinct=2))

    def test_checkpoints_and_calibration(self):
        self.assertEqual(stage.checkpoints('random', 'a', 2300), {0, stage.N, 2300, stage.BUDGET})
        self.assertEqual(stage.checkpoints('random', 'b', 2300), {0, stage.N, 2300, stage.BUDGET})
        self.assertEqual(stage.checkpoints('random', 'c', 2300), {0, stage.N, stage.BUDGET})
        self.assertEqual(stage.checkpoints('random', 'd', 2300), {0, stage.N, stage.BUDGET})
        for c in stage.CELLS:
            self.assertEqual(stage.checkpoints('align05_once', c, 2300), {stage.N})
            self.assertEqual(stage.checkpoints('random_once', c, 2300), {stage.N})
        n_cal, b_eq = stage.calibrate(0.11482, 155.23, 77.6)
        self.assertEqual(n_cal, 1648)
        self.assertEqual(b_eq, stage.N + 676)
        stage.check_calibration_bounds(n_cal, b_eq)
        for bad in ((1469, 2300), (1797, 2300), (1633, 3000), (1633, 3100)):
            with self.assertRaises(SystemExit):
                stage.check_calibration_bounds(*bad)

    def test_grid_rotation(self):
        g = stage.run_grid([1, 2, 3, 4, 5])
        self.assertEqual(len(set(g)), 60)
        self.assertEqual(g[:3], [(1, 'a', 'random'), (1, 'a', 'align05_once'), (1, 'a', 'random_once')])
        self.assertEqual(g[12], (2, 'b', 'align05_once'))
        self.assertEqual(g[24], (3, 'c', 'random_once'))
        self.assertEqual(g[36], (4, 'd', 'random'))
        self.assertEqual(g[48], (5, 'a', 'align05_once'))
        self.assertEqual([c for s, c, a in g[:12:3]], ['a', 'b', 'c', 'd'])

    def _rows(self, A, R_B, R_eq):
        n, B, e = str(stage.N), str(stage.BUDGET), '2300'
        rows = []
        for seed in (1, 2, 3):
            jitter = .001 * seed
            for c, (m, _, q) in stage.CELLS.items():
                lb = {'0': 3.9, n: 3.3, B: R_B[c] + jitter}
                ab = {'0': .3, n: .33, B: .35}
                if m == 4096:
                    lb[e], ab[e] = R_eq[c] + jitter, .34
                rows.append(dict(seed=seed, cell=c, arm='random', losses=lb, accuracy=ab,
                                 random_qa={k: dict(drawn=100, distinct=90) for k in lb if k != '0'}))
                rows.append(dict(seed=seed, cell=c, arm='align05_once', losses={n: A[c]}, accuracy={n: .37},
                                 qa_share=.9 * q / stage.N, qa_count=int(.9 * q), scoring_seconds=150.))
                rows.append(dict(seed=seed, cell=c, arm='random_once', losses={n: 3.4}, accuracy={n: .33},
                                 qa_share=q / m, qa_count=int(stage.N * q / m)))
        return rows

    def test_summary(self):
        plan = dict(design=dict(seeds=[1, 2, 3]))
        cal = dict(costs=dict(random=.11), N_cal=1648, B_eq_4096=2300, device='NVIDIA L4')
        A = dict(a=3.20, b=3.10, c=3.05, d=3.00)
        R_B = dict(a=3.30, b=3.20, c=3.30, d=3.20)
        R_eq = dict(a=3.31, b=3.22)
        out = stage.summarize(self._rows(A, R_B, R_eq), cal, plan)
        r = out['results']
        # D: a -0.10, b -0.10, c -0.25, d -0.20 (+ seed jitter, identical across cells)
        self.assertAlmostEqual(r['cells']['c']['charged']['mean'], -.25 - .002)
        self.assertAlmostEqual(out['primary']['E_count']['mean'], .5 * ((-.25 + .10) + (-.20 + .10)))
        self.assertAlmostEqual(out['primary']['E_share']['mean'], -.15)
        self.assertEqual(out['primary']['E_share']['verdict'], 'supported')
        self.assertEqual(len(out['primary']['E_count']['ci975']), 2)
        self.assertAlmostEqual(r['cells']['a']['charged_exact_equal_cost']['mean'], 3.20 - 3.31 - .002)
        self.assertNotIn('charged_exact_equal_cost', r['cells']['c'])
        dec = r['secondary']['decomposition']['E_share']
        self.assertAlmostEqual(dec['filter_part']['mean'], 3.05 - 3.10)
        self.assertAlmostEqual(dec['random_part']['mean'], -(3.30 - 3.20))
        self.assertAlmostEqual(dec['filter_part']['mean'] + dec['random_part']['mean'],
                               out['primary']['E_share']['mean'])
        self.assertAlmostEqual(r['secondary']['exact_equal_cost']['E_share']['mean'],
                               (3.05 - 3.30) - (3.10 - 3.22))
        self.assertAlmostEqual(r['cells']['a']['kept_qa']['count_over_Q'], int(.9 * 256) / 256)
        self.assertEqual(r['cells']['a']['random_qa'].keys(), {str(stage.N), '3000', '2300'})
        self.assertTrue(all(out['prediction_holds'].values()))
        self.assertEqual(r['loss0'], 3.9)

    def test_summary_rejects_incomplete_or_inconsistent(self):
        plan = dict(design=dict(seeds=[1, 2, 3]))
        cal = dict(costs=dict(random=.11), N_cal=1648, B_eq_4096=2300, device='NVIDIA L4')
        rows = self._rows(dict(a=3.2, b=3.1, c=3.0, d=3.0), dict(a=3.3, b=3.2, c=3.3, d=3.2), dict(a=3.3, b=3.2))
        with self.assertRaises(ValueError):
            stage.summarize(rows[:-1], cal, plan)
        bad = [dict(r, losses=dict(r['losses'], **{'0': 4.0})) if r['arm'] == 'random' and r['seed'] == 2 else r
               for r in rows]
        with self.assertRaises(ValueError):
            stage.summarize(bad, cal, plan)

    def test_verdict_and_bootstrap(self):
        self.assertEqual(stage.verdict([-.2, -.01]), 'supported')
        self.assertEqual(stage.verdict([.01, .2]), 'refuted')
        self.assertEqual(stage.verdict([-.1, .1]), 'not established')
        v = [-.3, -.1, -.2, -.25, -.15, -.05]
        wide, narrow = stage.bootstrap(v, stage.PRIMARY_Q), stage.bootstrap(v, stage.SECONDARY_Q)
        self.assertLessEqual(wide[0], narrow[0])
        self.assertGreaterEqual(wide[1], narrow[1])
        self.assertEqual(stage.bootstrap(v), stage.bootstrap(v))

    def test_backup_retries_push(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            journal = d / 'j.jsonl'
            journal.write_text(json.dumps(dict(kind='run')) + '\n')
            calls, msgs = [], []

            def fake(cmd, check=False, capture_output=False, text=False):
                calls.append(cmd[3])
                if cmd[3] == 'commit':
                    msgs.append(cmd[-1])
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
            self.assertTrue(msgs[0].startswith('Stage 79 interim journal snapshot: 1 finished runs'))
            self.assertTrue((d / 'wt' / 'results' / 'stage79-interim' / 'journal.jsonl').exists())


if __name__ == '__main__':
    unittest.main()

import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments import llm_count_share_filter as s79
from experiments import llm_exact_cost_count as stage

N = {8192: 1574, 4096: 2268}
CAL = dict(costs=dict(random=.1086), N={'8192': 1574, '4096': 2268}, device='NVIDIA L4')


class ExactCostCountTests(unittest.TestCase):
    def test_protocol_matches_implementation(self):
        plan = json.loads(stage.PROTOCOL.read_text())
        stage.check_protocol(plan)
        self.assertEqual(stage.SEEDS, tuple(range(38000, 38024)))
        self.assertEqual(stage.BLOCKS[1], tuple(range(38000, 38012)))
        self.assertEqual(stage.BLOCKS[2], tuple(range(38012, 38024)))
        self.assertIn('[1417, 1731]', plan['design']['calibration'])
        self.assertIn('[2041, 2495]', plan['design']['calibration'])
        self.assertIn('quantiles 0.025 and 0.975', plan['primary']['definition'])
        self.assertEqual(stage.CELLS, s79.CELLS)
        self.assertNotIn('random_once', plan['design']['arms'])
        for key in ('stage', 'B'):
            bad = json.loads(stage.PROTOCOL.read_text())
            if key == 'stage':
                bad['stage'] = 79
            else:
                bad['design']['B'] = 3100
            with self.assertRaises(ValueError):
                stage.check_protocol(bad)

    def test_stage79_population_available(self):
        pop = stage.stage79_population()
        self.assertEqual(set(pop['cells']), set(stage.CELLS))
        self.assertEqual(len(pop['target_items']), 512)
        for c, (m, share, q) in stage.CELLS.items():
            self.assertEqual((pop['cells'][c]['M'], pop['cells'][c]['qa']), (m, q))

    def test_pool_check(self):
        rec = {c: dict(pool_sha256=c * 4, kept_documents=[1]) for c in stage.CELLS}
        stage.check_pools_equal_stage79(rec, json.loads(json.dumps(rec)))
        bad = json.loads(json.dumps(rec))
        bad['b']['kept_documents'] = [2]
        with self.assertRaises(ValueError):
            stage.check_pools_equal_stage79(rec, bad)

    def test_target_draw(self):
        eligible = list(range(10, 3810))
        a, b = stage.draw_target(eligible), stage.draw_target(eligible)
        self.assertTrue((a == b).all())
        self.assertEqual(len(set(a.tolist())), 512)
        self.assertTrue((np.diff(a) > 0).all())
        ref = np.sort(np.asarray(eligible)[np.random.default_rng(8022).choice(3800, 512, replace=False)])
        self.assertTrue((a == ref).all())

    def test_calibration(self):
        n = stage.calibrate(0.10862121399998159, 154.8706990579999, 79.52535539600012)  # stage-79 costs
        self.assertEqual(n, {8192: 1574, 4096: 2267})
        stage.check_calibration_bounds(n)
        for bad in ({8192: 1416, 4096: 2268}, {8192: 1732, 4096: 2268}, {8192: 1574, 4096: 2040},
                    {8192: 1574, 4096: 2496}):
            with self.assertRaises(SystemExit):
                stage.check_calibration_bounds(bad)
        self.assertEqual(stage.n_of(CAL), N)

    def test_cost_points_and_checkpoints(self):
        self.assertEqual(stage.cost_steps('random', 1574), {2400: 2400, 2700: 2700, 3000: 3000})
        self.assertEqual(stage.cost_steps('align05_once', 1574), {2400: 974, 2700: 1274, 3000: 1574})
        self.assertEqual(stage.cost_steps('align05_once', 2268), {2400: 1668, 2700: 1968, 3000: 2268})
        self.assertEqual(stage.checkpoints('random', 1574), {2400, 2700, 3000})
        self.assertEqual(stage.checkpoints('random', 1574, step0=True), {0, 2400, 2700, 3000})
        self.assertEqual(stage.checkpoints('align05_once', 2268, step0=True), {1668, 1968, 2268})
        self.assertEqual(stage.updates('random', 2268), 3000)
        self.assertEqual(stage.updates('align05_once', 2268), 2268)

    def test_schedule_and_orders(self):
        for m in stage.SIZES:
            a = stage.schedule(38000, m, N[m])
            self.assertEqual(a['sha256'], stage.schedule(38000, m, N[m])['sha256'])
            self.assertEqual(len(a['rand']), 3000)
            self.assertLess(a['rand'].max(), m)
            self.assertEqual(sorted(a['once'].tolist()), list(range(N[m])))
            # rand is drawn first, as in stage 79
            self.assertTrue((a['rand'] == s79.schedule(38000, m)['rand']).all())
            kept = np.sort(np.random.default_rng(0).choice(m, N[m], replace=False))
            order, members = stage.training_order('align05_once', a, m, kept)
            self.assertEqual(len(order), N[m])
            self.assertEqual(stage.views(order, members, N[m]), dict(max=1, mean=1.0))
            order, members = stage.training_order('random', a, m)
            self.assertEqual(len(members), m)
        self.assertNotEqual(stage.schedule(38000, 4096, 2268)['sha256'], stage.schedule(38000, 4096, 2267)['sha256'])
        d = stage.schedule_digests([1, 2], N)
        self.assertEqual(set(d['1']), {'4096', '8192'})

    def test_grid_blocks_and_step0(self):
        g = stage.run_grid()
        self.assertEqual(len(g), 192)
        self.assertEqual(len(set(g)), 192)
        self.assertEqual(g[:2], [(38000, 'a', 'random'), (38000, 'a', 'align05_once')])
        self.assertEqual(g[8:10], [(38001, 'b', 'align05_once'), (38001, 'b', 'random')])
        self.assertEqual(g[16], (38002, 'c', 'random'))
        self.assertEqual(g[24], (38003, 'd', 'align05_once'))
        self.assertEqual([c for s, c, a in g[:8:2]], ['a', 'b', 'c', 'd'])
        self.assertEqual(sum(stage.block_of(s) == 1 for s, _, _ in g), 96)
        self.assertEqual(stage.first_random(38000), (38000, 'a', 'random'))
        self.assertEqual(stage.first_random(38001), (38001, 'b', 'random'))
        with self.assertRaises(SystemExit):
            stage.check_block(2, set(g[:95]))
        stage.check_block(2, set(g[:96]))
        stage.check_block(1, set())

    def _rows(self, A, R, jitter_block2=0.):
        rows = []
        for i, seed in enumerate(stage.SEEDS):
            j = .001 * (i % 5) + (jitter_block2 if seed in stage.BLOCKS[2] else 0.)
            for c, (m, _, q) in stage.CELLS.items():
                n = N[m]
                lr = {'2400': R[c] + .05 + j, '2700': R[c] + .02 + j, '3000': R[c] + j}
                if c == 'a':
                    lr['0'] = 3.9
                rows.append(dict(seed=seed, cell=c, arm='random', losses=lr,
                                 accuracy={k: .35 for k in lr}, step_seconds=[.11] * 3000,
                                 random_qa={str(t): dict(drawn=100, distinct=90) for t in stage.COST_POINTS}))
                steps = stage.cost_steps('align05_once', n)
                la = {str(steps[2400]): A[c] + .04, str(steps[2700]): A[c] + .01, str(n): A[c]}
                rows.append(dict(seed=seed, cell=c, arm='align05_once', losses=la, accuracy={k: .37 for k in la},
                                 step_seconds=[.11] * n, scoring_seconds=150. if m == 8192 else 78.,
                                 qa_share=.9 * q / n, qa_count=int(.9 * q)))
        return rows

    def test_summary(self):
        A = dict(a=3.20, b=3.10, c=3.05, d=3.00)
        R = dict(a=3.30, b=3.20, c=3.30, d=3.20)
        out = stage.summarize(self._rows(A, R), CAL)
        r = out['results']
        # D: a -0.10, b -0.10, c -0.25, d -0.20, minus a per-seed jitter on random that cancels in E_count
        self.assertAlmostEqual(out['primary']['E_count']['mean'], .5 * ((-.25 + .10) + (-.20 + .10)))
        self.assertEqual(out['primary']['E_count']['verdict'], 'supported')
        self.assertEqual(len(out['primary']['E_count']['ci95']), 2)
        self.assertAlmostEqual(r['secondary']['E_share']['mean'], -.15)
        dec = r['secondary']['decomposition']['E_count']
        self.assertAlmostEqual(dec['filter_part']['mean'] + dec['random_part']['mean'],
                               out['primary']['E_count']['mean'])
        self.assertAlmostEqual(dec['filter_part']['mean'], .5 * ((3.05 - 3.20) + (3.00 - 3.10)))
        # D at cost 2400: filter +0.04, random +0.05 above their endpoints
        self.assertAlmostEqual(r['cells']['a']['charged_at']['2400']['mean'] - r['cells']['a']['charged']['mean'],
                               -.01)
        self.assertAlmostEqual(r['cells']['a']['charged_mean_of_cost_points']['mean']
                               - r['cells']['a']['charged']['mean'], (.05 - .07) / 3)
        self.assertEqual(r['cells']['c']['filter_steps'], {'2400': 974, '2700': 1274, '3000': 1574})
        self.assertAlmostEqual(r['cells']['a']['kept_qa']['count_over_Q'], int(.9 * 256) / 256)
        self.assertAlmostEqual(r['cells']['a']['realized_cost_seconds']['filter_minus_random'],
                               78. + 2268 * .11 - 3000 * .11)
        self.assertEqual(r['loss0'], dict(values=[3.9], identical=True, seeds=list(stage.SEEDS)))
        self.assertEqual(r['cells']['a']['spikes'], {'random': 0, 'align05_once': 0})
        self.assertEqual(set(r['secondary']['blocks']), {'1', '2'})
        self.assertTrue(all(out['prediction_holds'].values()))

    def test_summary_blocks_and_rejects(self):
        A = dict(a=3.20, b=3.10, c=3.05, d=3.00)
        R = dict(a=3.30, b=3.20, c=3.30, d=3.20)
        rows = self._rows(A, R)
        with self.assertRaises(ValueError):
            stage.summarize(rows[:-1], CAL)
        no0 = [dict(r, losses={k: v for k, v in r['losses'].items() if k != '0'}) if r['seed'] == 38005 else r
               for r in rows]
        with self.assertRaises(ValueError):
            stage.summarize(no0, CAL)
        diff0 = [dict(r, losses=dict(r['losses'], **{'0': 4.0})) if r['seed'] == 38013 and '0' in r['losses']
                 else r for r in rows]
        self.assertFalse(stage.summarize(diff0, CAL)['results']['loss0']['identical'])

    def test_spike(self):
        row = dict(losses={'2400': 3.0, '2700': 3.2, '3000': 3.05})
        self.assertTrue(stage.spike(row, 'random', 1574))
        row = dict(losses={'2400': 3.0, '2700': 3.08, '3000': 3.05})
        self.assertFalse(stage.spike(row, 'random', 1574))
        row = dict(losses={'974': 3.0, '1274': 3.3, '1574': 3.1})
        self.assertTrue(stage.spike(row, 'align05_once', 1574))

    def test_backup_message(self):
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
            self.assertTrue(msgs[0].startswith('Stage 80 interim journal snapshot: 1 finished runs'))
            self.assertTrue((d / 'wt' / 'results' / 'stage80-interim' / 'journal.jsonl').exists())


if __name__ == '__main__':
    unittest.main()

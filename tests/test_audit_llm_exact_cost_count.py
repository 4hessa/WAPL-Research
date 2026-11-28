import hashlib
import unittest

import numpy as np

from experiments import llm_exact_cost_count as stage
from tools import audit_llm_exact_cost_count as audit

COSTS = dict(random=0.10862121399998159, proxy_align_grad_8192=154.8706990579999,
             proxy_align_grad_4096=79.52535539600012)


def calibration():
    n = stage.calibrate(COSTS['random'], COSTS['proxy_align_grad_8192'], COSTS['proxy_align_grad_4096'])
    return dict(kind='calibration', model=stage.MODEL, proxy=stage.PROXY, device='NVIDIA L4', costs=dict(COSTS),
                N={str(m): v for m, v in n.items()}, schedules=stage.schedule_digests(stage.SEEDS, n)), n


def fake_run(seed, cell, arm, n, step0=False):
    m, _, q = stage.CELLS[cell]
    n_m = n[m]
    sch = stage.schedule(seed, m, n_m)
    ckpts = stage.checkpoints(arm, n_m, step0)
    row = dict(kind='run', model=stage.MODEL, seed=seed, cell=cell, M=m, Q=q, arm=arm, N_M=n_m,
               block=stage.block_of(seed), device='NVIDIA L4', gpu_uuid='g1', schedule_sha256=sch['sha256'],
               step_seconds=[.1] * stage.updates(arm, n_m), adapter_digest_50='0' * 64,
               losses={str(t): 3. for t in ckpts}, accuracy={str(t): .3 for t in ckpts})
    kept = None
    if arm == 'align05_once':
        scores = np.random.default_rng(seed).normal(size=m)
        kept = s = stage.s73.keep_top(scores, n_m)
        row.update(scores=scores.tolist(), kept=s.tolist(), kept_sha256=hashlib.sha256(s.tobytes()).hexdigest(),
                   scoring_seconds=80.)
    order, members = stage.training_order(arm, sch, m, kept)
    if arm == 'random':
        row['random_qa'] = {str(t): stage.random_qa(sch['rand'], t, q) for t in stage.COST_POINTS}
    else:
        row['qa_share'], row['qa_count'] = stage.qa_share(members, q), stage.qa_count(members, q)
    row['views'] = {str(t): stage.views(order, members, t) for t in sorted(ckpts - {0})}
    return row


class AuditExactCostCountTests(unittest.TestCase):
    def test_calibration(self):
        cal, n = calibration()
        self.assertEqual(audit.check_calibration(cal), n)
        bad = dict(cal, N={'8192': n[8192] + 1, '4096': n[4096]})
        with self.assertRaises(Exception):
            audit.check_calibration(bad)
        bad = dict(cal, schedules=stage.schedule_digests(stage.SEEDS, {8192: 1500, 4096: 2268}))
        with self.assertRaises(Exception):
            audit.check_calibration(bad)
        bad = dict(cal, device='NVIDIA A100-SXM4-40GB')
        with self.assertRaises(Exception):
            audit.check_calibration(bad)

    def test_check_run(self):
        _, n = calibration()
        for cell in ('a', 'd'):
            for arm in stage.ARMS:
                row = fake_run(38000, cell, arm, n, step0=arm == 'random' and cell == 'a')
                audit.check_run(row, n, arm == 'random' and cell == 'a')
        row = fake_run(38000, 'a', 'random', n, step0=True)
        with self.assertRaises(Exception):
            audit.check_run(row, n, False)  # step 0 where none is expected
        row = fake_run(38013, 'b', 'align05_once', n)
        audit.check_run(row, n, False)
        for key, value in (('block', 1), ('N_M', n[4096] + 1), ('qa_count', -1)):
            with self.assertRaises(Exception):
                audit.check_run(dict(row, **{key: value}), n, False)
        bad = dict(row, kept=row['kept'][::-1])
        with self.assertRaises(Exception):
            audit.check_run(bad, n, False)

    def test_step0_expected(self):
        grid = stage.run_grid()
        base = {stage.first_random(s, grid) for s in stage.SEEDS}
        env = lambda u, b=1: dict(kind='environment', gpu_uuid=u, block=b)
        run = lambda k: dict(kind='run', seed=k[0], cell=k[1], arm=k[2])
        journal = [env('g1'), dict(kind='calibration')] + [run(k) for k in grid[:5]]
        self.assertEqual(audit.step0_expected(journal), base)
        # resumption on the same GPU: nothing extra
        journal2 = journal + [env('g1')] + [run(k) for k in grid[5:10]]
        self.assertEqual(audit.step0_expected(journal2), base)
        # new GPU: the next random run carries step 0
        journal3 = journal + [env('g2')] + [run(k) for k in grid[5:10]]
        first_random_after = next(k for k in grid[5:10] if k[2] == 'random')
        self.assertEqual(audit.step0_expected(journal3), base | {first_random_after})

    def test_environments(self):
        envs = [dict(kind='environment', device='NVIDIA L4', gpu_uuid='g1', block=1, torch='t', transformers='x',
                     tokenizers='y', cuda='c', python='p', driver='d1'),
                dict(kind='environment', device='NVIDIA L4', gpu_uuid='g2', block=2, torch='t', transformers='x',
                     tokenizers='y', cuda='c', python='p', driver='d2')]
        info = audit.check_environments(envs, [dict(gpu_uuid='g1'), dict(gpu_uuid='g2')])
        self.assertEqual((info['count'], info['gpus'], info['software_identical']), (2, 2, True))
        with self.assertRaises(Exception):
            audit.check_environments(envs[::-1], [])
        with self.assertRaises(Exception):
            audit.check_environments([dict(envs[0], device='NVIDIA T4')], [])
        with self.assertRaises(Exception):
            audit.check_environments(envs, [dict(gpu_uuid='g3')])


if __name__ == '__main__':
    unittest.main()

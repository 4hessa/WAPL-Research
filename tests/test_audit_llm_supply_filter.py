import hashlib
import unittest

import numpy as np

from experiments import llm_supply_filter as stage
from tools import audit_llm_supply_filter as audit

N = stage.N


def fake_row(arm, q=1024, seed=37800):
    sch = stage.schedule(seed)
    keys = [str(t) for t in stage.checkpoints(arm)]
    row = dict(seed=seed, level=q, arm=arm, model=stage.MODEL, schedule_sha256=sch['sha256'],
               step_seconds=[.11] * stage.updates(arm), losses={k: 2. for k in keys}, accuracy={k: .3 for k in keys},
               adapter_digest_50='0' * 64)
    kept = None
    if arm == 'align05_once':
        scores = np.random.default_rng(1).normal(size=stage.POOL_SIZE)
        kept = stage.s73.keep_top(scores, N)
        row.update(scores=scores.tolist(), kept=kept.tolist(), scoring_seconds=155.,
                   kept_sha256=hashlib.sha256(kept.tobytes()).hexdigest())
    order, members = stage.training_order(arm, sch, kept)
    if arm == 'random_once':
        row['subset_sha256'] = hashlib.sha256(np.asarray(members).tobytes()).hexdigest()
    if arm == 'random':
        row['random_qa'] = {str(t): stage.random_qa(sch['rand'], t, q) for t in (N, stage.BUDGET)}
    else:
        row['qa_share'] = stage.qa_share(members, q)
    row['views'] = {str(t): stage.views(order, members, t) for t in sorted(stage.checkpoints(arm) - {0})}
    return row


class AuditSupplyFilterTests(unittest.TestCase):
    def test_check_run_accepts_consistent_and_rejects_tampered_rows(self):
        for q in stage.LEVELS:
            for arm in stage.ARMS:
                audit.check_run(fake_row(arm, q))
        tampered = []
        bad = fake_row('align05_once'); bad['kept'] = bad['kept'][::-1]; tampered.append(bad)
        bad = fake_row('align05_once'); bad['qa_share'] = stage.qa_share(bad['kept'], 2048); tampered.append(bad)
        bad = fake_row('random'); bad['random_qa'][str(N)]['distinct'] += 1; tampered.append(bad)
        bad = fake_row('random'); bad['level'] = 2048; tampered.append(bad)  # QA draws depend on the level
        bad = fake_row('random_once'); bad['schedule_sha256'] = '1' * 64; tampered.append(bad)
        bad = fake_row('random_once'); bad['views'][str(N)]['max'] = 2; tampered.append(bad)
        bad = fake_row('random'); bad['scores'] = [0.] * stage.POOL_SIZE; tampered.append(bad)
        bad = fake_row('random_once'); bad['step_seconds'] = bad['step_seconds'][:-1]; tampered.append(bad)
        bad = fake_row('align05_once'); bad['losses']['3000'] = 2.; tampered.append(bad)
        bad = fake_row('random'); del bad['accuracy']['0']; tampered.append(bad)
        for bad in tampered:
            with self.assertRaises(ValueError):
                audit.check_run(bad)

    def test_check_calibration(self):
        cal = dict(model=stage.MODEL, proxy=stage.PROXY, costs=dict(random=0.11482005099878734,
                                                                   proxy_align_grad=155.22979166599907),
                   N_cal=1648, updates=1633)
        self.assertEqual(audit.check_calibration(cal), 1648)
        for bad in (dict(cal, N_cal=1647), dict(cal, updates=1648), dict(cal, proxy='x'),
                    dict(cal, costs=dict(random=.1, proxy_align_grad=200.), N_cal=1000)):
            with self.assertRaises(ValueError):
                audit.check_calibration(bad)

    def test_independent_extras_match_the_stage(self):
        drawn = np.random.default_rng(3).choice(5000, 3072, replace=False)
        for seed, k in ((7811, 512), (7812, 512)):
            self.assertTrue(np.array_equal(audit.independent_extras(5000, drawn, seed, k),
                                           stage.extra_draws(5000, drawn, seed, k)))


if __name__ == '__main__':
    unittest.main()

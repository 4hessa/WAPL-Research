import hashlib
import unittest

import numpy as np

from experiments import llm_count_share_filter as stage
from tools import audit_llm_count_share_filter as audit

N, B_EQ = stage.N, 2366


def fake_row(arm, cell='a', seed=37900):
    m, _, q = stage.CELLS[cell]
    sch = stage.schedule(seed, m)
    ckpts = stage.checkpoints(arm, cell, B_EQ)
    keys = [str(t) for t in ckpts]
    row = dict(seed=seed, cell=cell, M=m, Q=q, arm=arm, model=stage.MODEL, schedule_sha256=sch['sha256'],
               step_seconds=[.11] * stage.updates(arm), losses={k: 2. for k in keys}, accuracy={k: .3 for k in keys},
               adapter_digest_50='0' * 64)
    kept = None
    if arm == 'align05_once':
        scores = np.random.default_rng(1).normal(size=m)
        kept = stage.s73.keep_top(scores, N)
        row.update(scores=scores.tolist(), kept=kept.tolist(), scoring_seconds=80.,
                   kept_sha256=hashlib.sha256(kept.tobytes()).hexdigest())
    order, members = stage.training_order(arm, sch, m, kept)
    if arm == 'random_once':
        row['subset_sha256'] = hashlib.sha256(np.asarray(members).tobytes()).hexdigest()
    if arm == 'random':
        row['random_qa'] = {str(t): stage.random_qa(sch['rand'], t, q) for t in sorted(ckpts - {0})}
    else:
        row['qa_share'] = stage.qa_share(members, q)
        row['qa_count'] = stage.qa_count(members, q)
    row['views'] = {str(t): stage.views(order, members, t) for t in sorted(ckpts - {0})}
    return row


class AuditCountShareFilterTests(unittest.TestCase):
    def test_check_run_accepts_consistent_and_rejects_tampered_rows(self):
        for cell in stage.CELLS:
            for arm in stage.ARMS:
                audit.check_run(fake_row(arm, cell), B_EQ)
        tampered = []
        bad = fake_row('align05_once'); bad['kept'] = bad['kept'][::-1]; tampered.append(bad)
        bad = fake_row('align05_once'); bad['qa_count'] += 1; tampered.append(bad)
        bad = fake_row('random'); bad['random_qa'][str(N)]['distinct'] += 1; tampered.append(bad)
        bad = fake_row('random', 'b'); bad['cell'] = 'c'; tampered.append(bad)  # M and Q no longer match
        bad = fake_row('random', 'b'); bad['cell'] = 'a'; bad['Q'] = 256; tampered.append(bad)  # QA draws differ
        bad = fake_row('random'); del bad['losses'][str(B_EQ)]; del bad['accuracy'][str(B_EQ)]; tampered.append(bad)
        bad = fake_row('random_once'); bad['schedule_sha256'] = '1' * 64; tampered.append(bad)
        bad = fake_row('random_once'); bad['views'][str(N)]['max'] = 2; tampered.append(bad)
        bad = fake_row('random'); bad['scores'] = [0.] * 4096; tampered.append(bad)
        bad = fake_row('random_once'); bad['step_seconds'] = bad['step_seconds'][:-1]; tampered.append(bad)
        bad = fake_row('align05_once'); bad['losses']['3000'] = 2.; tampered.append(bad)
        for bad in tampered:
            with self.assertRaises(ValueError):
                audit.check_run(bad, B_EQ)
        with self.assertRaises(ValueError):
            audit.check_run(fake_row('random', 'a'), 2400)

    def test_check_calibration(self):
        cal = dict(model=stage.MODEL, proxy=stage.PROXY,
                   costs=dict(random=0.10862121399998159, proxy_align_grad_8192=154.8706990579999,
                              proxy_align_grad_4096=79.52535539600012), N_cal=1574, B_eq_4096=2366, updates=1633)
        self.assertEqual(audit.check_calibration(cal), (1574, 2366))
        for bad in (dict(cal, N_cal=1575), dict(cal, B_eq_4096=2365), dict(cal, updates=1574), dict(cal, proxy='x'),
                    dict(cal, costs=dict(cal['costs'], proxy_align_grad_8192=200.), N_cal=1158)):
            with self.assertRaises(ValueError):
                audit.check_calibration(bad)


if __name__ == '__main__':
    unittest.main()

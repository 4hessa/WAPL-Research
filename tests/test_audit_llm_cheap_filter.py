import hashlib
import unittest

import numpy as np

from experiments import llm_cheap_filter as stage
from tools import audit_llm_cheap_filter as audit


def fake_row(arm, n=1500, seed=37600):
    sch = stage.schedule(seed, n)
    row = dict(seed=seed, arm=arm, schedule_sha256=sch['sha256'], step_seconds=[.1] * stage.BUDGET,
               losses={k: 3. for k in ('0', str(n), '3000')}, accuracy={k: .3 for k in ('0', str(n), '3000')},
               adapter_digest_50='0' * 64)
    kept = None
    if arm == 'align05_once':
        scores = np.random.default_rng(1).normal(size=stage.POOL_SIZE)
        kept = stage.s73.keep_top(scores, n)
        row.update(scores=scores.tolist(), kept=kept.tolist(), scoring_seconds=160.,
                   kept_sha256=hashlib.sha256(kept.tobytes()).hexdigest())
    order, members = stage.training_order(arm, sch, kept)
    if arm in ('random_once', 'qa_once'):
        row['subset_sha256'] = hashlib.sha256(np.asarray(members).tobytes()).hexdigest()
    if arm == 'random':
        row['random_qa'] = {str(t): stage.random_qa(sch['rand'], t) for t in (n, stage.BUDGET)}
    else:
        row['qa_share'] = stage.qa_share(members)
    row['views'] = {str(t): stage.views(order, members, t) for t in (n, stage.BUDGET)}
    return row


class AuditCheapFilterTests(unittest.TestCase):
    def test_check_run_accepts_consistent_and_rejects_tampered_rows(self):
        for arm in stage.ARMS:
            audit.check_run(fake_row(arm), 1500)
        bad = fake_row('align05_once')
        bad['kept'] = bad['kept'][::-1]
        with self.assertRaises(ValueError):
            audit.check_run(bad, 1500)
        bad = fake_row('random')
        bad['random_qa']['1500']['distinct'] += 1
        with self.assertRaises(ValueError):
            audit.check_run(bad, 1500)
        bad = fake_row('qa_once')
        bad['schedule_sha256'] = '1' * 64
        with self.assertRaises(ValueError):
            audit.check_run(bad, 1500)
        bad = fake_row('random_once')
        bad['views']['1500']['max'] = 2
        with self.assertRaises(ValueError):
            audit.check_run(bad, 1500)


if __name__ == '__main__':
    unittest.main()

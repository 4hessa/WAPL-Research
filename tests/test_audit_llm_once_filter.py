import hashlib
import unittest

import numpy as np

from experiments import llm_once_filter as stage
from experiments import llm_qa_mix as s74
from tools import audit_llm_once_filter as audit


def fake_row(arm, n=1500, seed=37500):
    sch = stage.schedule(0, seed, n)
    row = dict(sigma=0.25, seed=seed, arm=arm, schedule_sha256=sch['sha256'], step_seconds=[.1] * stage.BUDGET,
               losses={k: 3. for k in ('0', str(n), '2400')}, accuracy={k: .3 for k in ('0', str(n), '2400')},
               adapter_digest_50='0' * 64)
    kept = None
    if arm in stage.FILTER_ARMS:
        scores = np.random.default_rng(1).normal(size=stage.POOL_SIZE)
        kept = stage.s73.keep_top(scores, 512 if arm == 'align_q' else n)
        row.update(scores=scores.tolist(), kept=kept.tolist(), scoring_seconds=100.,
                   kept_sha256=hashlib.sha256(kept.tobytes()).hexdigest())
    order, members = stage.training_order(arm, sch, kept)
    if arm in ('random512', 'random_once'):
        row['subset_sha256'] = hashlib.sha256(np.asarray(members).tobytes()).hexdigest()
    if arm != 'random':
        row['qa_share'] = s74.qa_share(members, 0.25)
    row['views'] = {str(t): stage.views(order, members, t) for t in (n, stage.BUDGET)}
    return row


class AuditOnceFilterTests(unittest.TestCase):
    def test_check_run_accepts_consistent_and_rejects_tampered_rows(self):
        for arm in stage.ARMS:
            audit.check_run(fake_row(arm), 1500)
        bad = fake_row('align_once')
        bad['kept'] = bad['kept'][::-1]
        with self.assertRaises(ValueError):
            audit.check_run(bad, 1500)
        bad = fake_row('random_once')
        bad['views']['1500']['max'] = 2
        with self.assertRaises(ValueError):
            audit.check_run(bad, 1500)


if __name__ == '__main__':
    unittest.main()

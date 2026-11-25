import hashlib
import unittest

import numpy as np

from experiments import llm_deepseek_filter as stage
from tools import audit_llm_deepseek_filter as audit

N = 1813


def fake_row(arm, n=N, seed=37700, batch=8):
    sch = stage.schedule(seed, n)
    row = dict(seed=seed, arm=arm, model=stage.MODEL, schedule_sha256=sch['sha256'], step_seconds=[.16] * stage.BUDGET,
               losses={k: 2. for k in ('0', str(n), '7400')}, accuracy={k: .3 for k in ('0', str(n), '7400')},
               adapter_digest_50='0' * 64)
    kept = None
    if arm == 'align_once':
        scores = np.random.default_rng(1).normal(size=stage.POOL_SIZE)
        kept = stage.s73.keep_top(scores, n)
        row.update(scores=scores.tolist(), kept=kept.tolist(), scoring_seconds=900., scoring_batch=batch,
                   kept_sha256=hashlib.sha256(kept.tobytes()).hexdigest())
    order, members = stage.training_order(arm, sch, kept)
    if arm == 'random_once':
        row['subset_sha256'] = hashlib.sha256(np.asarray(members).tobytes()).hexdigest()
    if arm == 'random':
        row['random_qa'] = {str(t): stage.random_qa(sch['rand'], t) for t in (n, stage.BUDGET)}
    else:
        row['qa_share'] = stage.qa_share(members)
    row['views'] = {str(t): stage.views(order, members, t) for t in (n, stage.BUDGET)}
    return row


class AuditDeepseekFilterTests(unittest.TestCase):
    def test_check_run_accepts_consistent_and_rejects_tampered_rows(self):
        for arm in stage.ARMS:
            audit.check_run(fake_row(arm), N, 8)
        bad = fake_row('align_once')
        bad['kept'] = bad['kept'][::-1]
        with self.assertRaises(ValueError):
            audit.check_run(bad, N, 8)
        with self.assertRaises(ValueError):  # scoring batch differs from the calibration
            audit.check_run(fake_row('align_once'), N, 4)
        bad = fake_row('random')
        bad['random_qa'][str(N)]['distinct'] += 1
        with self.assertRaises(ValueError):
            audit.check_run(bad, N, 8)
        bad = fake_row('random_once')
        bad['schedule_sha256'] = '1' * 64
        with self.assertRaises(ValueError):
            audit.check_run(bad, N, 8)
        bad = fake_row('random_once')
        bad['views'][str(N)]['max'] = 2
        with self.assertRaises(ValueError):
            audit.check_run(bad, N, 8)
        bad = fake_row('random')
        bad['scores'] = [0.] * stage.POOL_SIZE
        with self.assertRaises(ValueError):
            audit.check_run(bad, N, 8)
        bad = fake_row('random_once')
        bad['step_seconds'] = bad['step_seconds'][:-1]
        with self.assertRaises(ValueError):
            audit.check_run(bad, N, 8)

    def test_check_calibration(self):
        cal = dict(model=stage.MODEL, revision=stage.REVISION, costs=dict(random=.1617, self_align_grad=903.5),
                   updates=1812, scoring_batch=8, scoring_oom_batches=[])
        self.assertEqual(audit.check_calibration(cal), (1812, 8))
        self.assertEqual(audit.check_calibration(dict(cal, scoring_batch=4, scoring_oom_batches=[8])), (1812, 4))
        for bad in (dict(cal, updates=1813), dict(cal, scoring_batch=4), dict(cal, scoring_batch=2),
                    dict(cal, costs=dict(random=.1, self_align_grad=100.), updates=6400), dict(cal, revision='x')):
            with self.assertRaises(ValueError):
                audit.check_calibration(bad)

    def test_exposed_ids_cover_all_earlier_sets(self):
        import pandas as pd
        from pathlib import Path
        mmlu = Path('/content/stage76_data/mmlu_repo/All/test.csv')
        if not mmlu.exists():
            self.skipTest('ArabicMMLU test file not downloaded')
        test = pd.read_csv(mmlu).to_dict('records')
        ids = audit.exposed_ids(test)
        self.assertEqual(len(ids), 4946 + 512 + 256 + 512 + 512 + 512)  # trained + 73, 74 pilot, 74, 75, 76 targets


if __name__ == '__main__':
    unittest.main()

import hashlib
import json
import unittest

import numpy as np

from experiments import llm_deepseek_filter as s77
from experiments import llm_web_filter as s73
from tools import verify_llm_deepseek_filter_resume as verify


def fake_run(seed, arm, n, device='NVIDIA L4'):
    sch = s77.schedule(seed, n)
    row = dict(kind='run', model=s77.MODEL, seed=seed, arm=arm, device=device, schedule_sha256=sch['sha256'],
               losses={k: 0.0 for k in ('0', str(n), str(s77.BUDGET))},
               accuracy={k: 0.0 for k in ('0', str(n), str(s77.BUDGET))})
    kept = None
    if arm == 'align_once':
        scores = np.random.default_rng(seed).normal(size=s77.POOL_SIZE)
        kept = s73.keep_top(scores, n)
        row.update(scores=scores.tolist(), kept=kept.tolist(), kept_sha256=hashlib.sha256(kept.tobytes()).hexdigest())
    order, members = s77.training_order(arm, sch, kept)
    if arm == 'random_once':
        row['subset_sha256'] = hashlib.sha256(np.asarray(members).tobytes()).hexdigest()
    if arm == 'random':
        row['random_qa'] = {str(t): s77.random_qa(sch['rand'], t) for t in (n, s77.BUDGET)}
    else:
        row['qa_share'] = s77.qa_share(members)
    row['views'] = {str(t): s77.views(order, members, t) for t in (n, s77.BUDGET)}
    return json.loads(json.dumps(row))  # as read back from the journal


class VerifyResumeTest(unittest.TestCase):
    n = 1992
    grid = s77.run_grid([37700, 37701, 37702, 37703])

    def collect(self):
        results = []
        return results, lambda name, ok: results.append((name, bool(ok)))

    def test_valid_prefix_passes(self):
        runs = [fake_run(s, a, self.n) for s, a in self.grid[:4]]
        results, check = self.collect()
        remaining = verify.check_runs(runs, self.n, self.grid, 'NVIDIA L4', check)
        self.assertTrue(all(ok for _, ok in results), results)
        self.assertEqual(remaining, self.grid[4:])

    def test_out_of_order_or_tampered_runs_fail(self):
        runs = [fake_run(s, a, self.n) for s, a in self.grid[:3]]
        results, check = self.collect()
        verify.check_runs(runs[::-1], self.n, self.grid, 'NVIDIA L4', check)
        self.assertFalse(all(ok for _, ok in results))
        runs[1]['kept'] = runs[1]['kept'][::-1]
        results, check = self.collect()
        verify.check_runs(runs, self.n, self.grid, 'NVIDIA L4', check)
        self.assertFalse(all(ok for _, ok in results))
        results, check = self.collect()
        verify.check_runs([fake_run(37700, 'random', self.n)], 2044, self.grid, 'NVIDIA L4', check)
        self.assertFalse(all(ok for _, ok in results))

    def test_calibration_uses_capped_n(self):
        journal = [dict(kind='environment'),
                   dict(kind='calibration', model=s77.MODEL, revision=s77.REVISION, device='NVIDIA L4',
                        costs=dict(random=0.16287156099951972, self_align_grad=872.3100562769996), updates=2044)]
        results, check = self.collect()
        _, n = verify.check_calibration(journal, 'NVIDIA L4', check)
        self.assertEqual(n, 1992)
        self.assertTrue(all(ok for _, ok in results), results)
        results, check = self.collect()
        verify.check_calibration(journal, 'Tesla T4', check)
        self.assertFalse(all(ok for _, ok in results))


if __name__ == '__main__':
    unittest.main()

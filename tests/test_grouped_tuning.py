import unittest
import numpy as np
from experiments.grouped_tuning import fit_candidate

class GroupedTuningTests(unittest.TestCase):
    def test_tune_and_validation_excluded_from_grouped_fit(self):
        rng=np.random.default_rng(51);rows=[dict(split=s,seed=seed,baseline=rng.normal(size=5).tolist(),utility=float(rng.normal())) for s in ['fit','tune','validate'] for seed in range(4) for _ in range(4)]
        a,m,selection=fit_candidate(rows,'baseline','grouped')
        for r in rows:
            if r['split']!='fit':r['baseline']=[1e10]*5;r['utility']=-1e10
        b,n,selection2=fit_candidate(rows,'baseline','grouped')
        self.assertEqual(selection,selection2);np.testing.assert_array_equal(a.mean_,b.mean_);np.testing.assert_array_equal(m.coef_,n.coef_)
        for f in selection['folds']:
            self.assertNotIn(f['held_out_seed'],f['train_seeds']);self.assertEqual(f['held_out_rows'],4);self.assertEqual(f['train_rows'],12)

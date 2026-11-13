import unittest
import numpy as np
from experiments.regret_tuning import state_regret,fit_regret

class RegretTuningTests(unittest.TestCase):
    def test_regret_is_within_state(self):
        rows=[dict(seed=1,checkpoint=cp,utility=float(y)) for cp,ys in [(16,[0,1,2,3]),(48,[100,101,102,103])] for y in ys]
        self.assertEqual(state_regret(rows,np.array([0,1,2,3,3,2,1,0])),1.)
    def test_validation_and_tuning_excluded(self):
        rng=np.random.default_rng(52);rows=[dict(split=s,seed=seed,checkpoint=cp,baseline=rng.normal(size=5).tolist(),utility=float(rng.normal())) for s in ['fit','tune','validate'] for seed in range(3) for cp in [16,48] for _ in range(4)]
        sc,m,a=fit_regret(rows,'baseline')
        for r in rows:
            if r['split']!='fit':r['utility']=1e10;r['baseline']=[1e10]*5
        sc2,m2,b=fit_regret(rows,'baseline');self.assertEqual(a,b);np.testing.assert_array_equal(sc.mean_,sc2.mean_);np.testing.assert_array_equal(m.coef_,m2.coef_)

import unittest
import numpy as np
from experiments.pairwise_logistic import pairs,fit_ranker

class PairwiseLogisticTests(unittest.TestCase):
    def test_pairs_remain_within_state_and_ignore_offsets(self):
        rows=[dict(seed=1,checkpoint=cp,utility=float(u)) for cp in [16,48] for u in [0,1,2]];x=np.arange(12).reshape(6,2);a,y,t=pairs(rows,x)
        self.assertEqual(len(y),12);self.assertEqual(t,0)
        for r in rows:r['utility']+=r['checkpoint']*10
        b,z,_=pairs(rows,x);np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(y,z)
        np.testing.assert_array_equal(a[::2],-a[1::2]);np.testing.assert_array_equal(y[::2],1-y[1::2])
    def test_validation_and_tune_do_not_change_ranker(self):
        rng=np.random.default_rng(53);rows=[dict(split=s,seed=seed,checkpoint=cp,baseline=rng.normal(size=5).tolist(),utility=float(rng.normal())) for s in ['fit','tune','validate'] for seed in range(3) for cp in [16,48] for _ in range(4)]
        a,m,sel=fit_ranker(rows,'baseline')
        for r in rows:
            if r['split']!='fit':r['utility']=1e10;r['baseline']=[1e10]*5
        b,n,sel2=fit_ranker(rows,'baseline');self.assertEqual(sel,sel2);np.testing.assert_array_equal(a.mean_,b.mean_);np.testing.assert_array_equal(m.coef_,n.coef_)

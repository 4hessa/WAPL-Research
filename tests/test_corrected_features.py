import copy,unittest
import numpy as np
from experiments.corrected_features import fit_models

class CorrectedFeatureLeakageTests(unittest.TestCase):
    def test_validation_cannot_change_fit_or_alpha(self):
        rng=np.random.default_rng(48);rows=[]
        for split in ['fit','tune','validate']:
            for _ in range(16):rows.append(dict(split=split,baseline=rng.normal(size=5).tolist(),corrected=rng.normal(size=3).tolist(),raw=rng.normal(size=3).tolist(),noise=rng.normal(size=3).tolist(),utility=float(rng.normal())))
        a,sa=fit_models(rows)
        for r in rows:
            if r['split']=='validate':
                for key in ['baseline','corrected','raw','noise']:r[key]=[1e10]*len(r[key])
                r['utility']=-1e10
        b,sb=fit_models(rows);self.assertEqual(sa,sb)
        for arm in a:
            np.testing.assert_array_equal(a[arm][0].mean_,b[arm][0].mean_);np.testing.assert_array_equal(a[arm][1].coef_,b[arm][1].coef_);self.assertEqual(a[arm][1].intercept_,b[arm][1].intercept_)

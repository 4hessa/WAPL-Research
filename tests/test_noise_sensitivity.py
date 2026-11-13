import unittest
import numpy as np
from experiments.noise_sensitivity import fit_noise

class NoiseSensitivityTests(unittest.TestCase):
    def test_validation_labels_do_not_select_alpha(self):
        rng=np.random.default_rng(50);rows=[dict(split=s,baseline=rng.normal(size=5).tolist(),utility=float(rng.normal())) for s in ['fit','tune','validate'] for _ in range(12)]
        a=fit_noise(rows,50000,10.)
        for r in rows:
            if r['split']=='validate':r['utility']=1e10
        b=fit_noise(rows,50000,10.)
        self.assertEqual(a,b);self.assertEqual(a['fixed']['alpha'],10.)

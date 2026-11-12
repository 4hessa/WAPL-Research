import unittest
import numpy as np
from experiments.utility_student import fit_models
class StudentLeakageTests(unittest.TestCase):
 def test_test_teacher_and_targets_excluded(self):
  rng=np.random.default_rng(12);rows=[{'split':s,'baseline':rng.normal(size=5).tolist(),'weight':rng.normal(size=3).tolist(),'teacher':float(rng.normal()),'target':None} for s in ['train','dev','test'] for _ in range(12)]
  a,_=fit_models(rows)
  for r in rows:
   r['target']=1e9
   if r['split']=='test':r['teacher']=-1e9;r['baseline']=[1e9]*5;r['weight']=[1e9]*3
  b,_=fit_models(rows)
  for arm in a:np.testing.assert_array_equal(a[arm][1].coef_,b[arm][1].coef_);np.testing.assert_array_equal(a[arm][0].mean_,b[arm][0].mean_)

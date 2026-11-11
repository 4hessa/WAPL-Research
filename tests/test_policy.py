import unittest
import json
from pathlib import Path
import numpy as np
from experiments.policy import choose,fit_predictors
from experiments.text_pilot import V,counts
class PolicyTests(unittest.TestCase):
 def test_choice_does_not_mutate_state(self):
  w=np.zeros((V,V));v=np.zeros_like(w);c=counts([65,66,65]);saved=w.copy()
  self.assertEqual(choose(w,v,[c,c],c,'alignment',{},0),0)
  np.testing.assert_array_equal(w,saved);np.testing.assert_array_equal(v,saved)
 def test_test_labels_cannot_change_fit(self):
  records=json.loads(Path('results/centered-records.json').read_text());a,_=fit_predictors(records)
  for r in records:
   if r['seed']>=3000:r['utility']=[1e8]*3
  b,_=fit_predictors(records)
  for arm in a:np.testing.assert_array_equal(a[arm][1].coef_,b[arm][1].coef_)

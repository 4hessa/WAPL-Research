import unittest
import numpy as np
from experiments.reuse import shared_scores,reference_scores,V,counts
class ReuseTests(unittest.TestCase):
 def test_scores_match(self):
  rng=np.random.default_rng(21);w=rng.normal(size=(V,V));v=rng.normal(size=(V,V))*.03
  cs=[counts(rng.integers(0,V,size=65)) for _ in range(8)];probe=sum(cs)
  np.testing.assert_allclose(shared_scores(w,v,cs,probe)[0],reference_scores(w,v,cs,probe),atol=1e-12,rtol=1e-12)
 def test_tie_order(self):
  w=np.zeros((V,V));c=counts([65,66,65])
  self.assertEqual(np.argmax(shared_scores(w,w,[c]*8,c)[0]),0)

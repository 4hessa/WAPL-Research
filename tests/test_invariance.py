import unittest
import numpy as np
from experiments.invariance import features
from experiments.text_pilot import V,counts,loss_grad,LR,effects
class InvarianceTests(unittest.TestCase):
 def test_centered_features_and_utility_survive_row_shift(self):
  rng=np.random.default_rng(51);w=rng.normal(size=(V,V))*.01;v=rng.normal(size=(V,V))*.001
  ws=w+rng.normal(size=(V,1));c=counts([65,66,67,65]);g=loss_grad(w,c)[1];gs=loss_grad(ws,c)[1]
  d=-LR*(.9*v+g);ds=-LR*(.9*v+gs)
  np.testing.assert_allclose(features(w,g,d,True),features(ws,gs,ds,True),atol=1e-12)
  self.assertGreater(np.max(abs(features(w,g,d)-features(ws,gs,ds))),.001)
  np.testing.assert_allclose(effects(w,v,c,[c]*15,c),effects(ws,v,c,[c]*15,c),atol=1e-12)

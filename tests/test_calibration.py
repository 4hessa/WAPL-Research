import unittest
import torch
from experiments.transformer_calibration import TinyTransformer,optimizer,update,vec,loss,signals,clone
class CalibrationTests(unittest.TestCase):
 def test_signal_isolation_and_exact_step(self):
  torch.set_num_threads(1);torch.manual_seed(111);m=TinyTransformer();o=optimizer(m);b=torch.randint(128,(2,33));probe=torch.randint(128,(3,33));update(m,o,b);before=vec(m).clone()
  a,p,mi,oi,_=signals(m,o,b,probe);torch.testing.assert_close(vec(m),before,atol=0,rtol=0)
  ref,ro=clone(m,o);update(ref,ro,b);torch.testing.assert_close(vec(mi),vec(ref),atol=0,rtol=0)
  with torch.no_grad():expected=(loss(m,probe)-loss(ref,probe)).item()
  self.assertEqual(p,expected)

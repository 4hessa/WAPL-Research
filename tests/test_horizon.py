import unittest
import torch
from experiments.transformer_horizon import TinyTransformer,optimizer,update,clone,vec,loss,rollout
class HorizonTests(unittest.TestCase):
 def test_rollout_initial_gain_and_parent(self):
  torch.set_num_threads(1);torch.manual_seed(14);m=TinyTransformer();o=optimizer(m);b=torch.randint(128,(2,33));update(m,o,b);before=vec(m).clone()
  r=rollout(m,o,b,[b]*3,{'probe':b});mi,oi=clone(m,o);update(mi,oi,b)
  with torch.no_grad():expected=(loss(m,b)-loss(mi,b)).item()
  self.assertEqual(r['1']['probe'],expected);self.assertEqual(set(r),{'1','4'});torch.testing.assert_close(vec(m),before,rtol=0,atol=0)

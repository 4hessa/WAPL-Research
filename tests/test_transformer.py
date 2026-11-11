import unittest
import torch
from experiments.transformer_pilot import TinyTransformer,optimizer,clone,update,vec,loss
class TransformerTests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(1);torch.manual_seed(9);self.m=TinyTransformer();self.o=optimizer(self.m);self.b=torch.randint(0,128,(2,33))
 def test_causality(self):
  x=self.b[:,:-1].clone();y=x.clone();y[:,16:]=(y[:,16:]+1)%128
  torch.testing.assert_close(self.m(x)[:,:16],self.m(y)[:,:16],rtol=0,atol=0)
 def test_shifted_targets(self):
  expected=torch.nn.functional.cross_entropy(self.m(self.b[:,:-1]).reshape(-1,128),self.b[:,1:].reshape(-1))
  torch.testing.assert_close(loss(self.m,self.b),expected)
 def test_adam_clone_and_restoration(self):
  update(self.m,self.o,self.b);m,o=clone(self.m,self.o);parent=vec(self.m).clone()
  for old,new in zip(self.o.state.values(),o.state.values()):
   for k in ['exp_avg','exp_avg_sq','step']:
    torch.testing.assert_close(old[k],new[k]);self.assertNotEqual(old[k].data_ptr(),new[k].data_ptr())
  update(m,o,self.b);torch.testing.assert_close(vec(self.m),parent,rtol=0,atol=0)
  update(self.m,self.o,self.b);torch.testing.assert_close(vec(self.m),vec(m),rtol=0,atol=0)

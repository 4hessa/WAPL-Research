import unittest
import numpy as np
from experiments.text_pilot import V,counts,loss_grad,step,effects,prepare
class TextTests(unittest.TestCase):
 def setUp(self):
  self.w=np.random.default_rng(8).normal(0,.01,(V,V));self.v=np.random.default_rng(19).normal(0,.001,(V,V));self.c=counts([65,66,65,67])
 def test_next_character_alignment(self):
  self.assertEqual(self.c.sum(),3);self.assertEqual(self.c[65,66],1);self.assertEqual(self.c[67,65],0)
 def test_gradient_finite_difference(self):
  g=loss_grad(self.w,self.c)[1];eps=1e-5
  for a,b in [(65,66),(66,65),(67,65)]:
   wp=self.w.copy();wm=self.w.copy();wp[a,b]+=eps;wm[a,b]-=eps
   fd=(loss_grad(wp,self.c)[0]-loss_grad(wm,self.c)[0])/(2*eps)
   self.assertAlmostEqual(g[a,b],fd,places=7)
 def test_one_step_effect_and_state(self):
  w,v=self.w.copy(),self.v.copy();wi,_=step(w,v,self.c)
  y=effects(w,v,self.c,[self.c]*15,self.c)
  self.assertAlmostEqual(y[0],loss_grad(w,self.c)[0]-loss_grad(wi,self.c)[0])
  np.testing.assert_array_equal(w,self.w);np.testing.assert_array_equal(v,self.v)
  self.assertEqual(y,effects(w,v,self.c,[self.c]*15,self.c));self.assertEqual(len(y),3)
 def test_momentum_matters(self):
  self.assertNotEqual(effects(self.w,self.v,self.c,[],self.c),effects(self.w,np.zeros_like(self.v),self.c,[],self.c))
 def test_uniform_row_shift_invariance(self):
  shift=np.random.default_rng(9).normal(size=(V,1))
  self.assertAlmostEqual(loss_grad(self.w,self.c)[0],loss_grad(self.w+shift,self.c)[0],places=12)
 def test_deduplication(self):
  pools,_=prepare('a'*6500)
  chunks=[tuple(s) for pool in pools for s in pool]
  self.assertEqual(len(chunks),len(set(chunks)))
if __name__=='__main__':unittest.main()

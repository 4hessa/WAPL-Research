import unittest
import numpy as np
from experiments.optimized_time import feature_batch,V,LR,counts,loss_grad,features,step
class OptimizedTests(unittest.TestCase):
 def test_features_and_update_match_reference(self):
  rng=np.random.default_rng(24);w=rng.normal(size=(V,V));v=rng.normal(size=(V,V))*.02;cs=[counts(rng.integers(0,V,65)) for _ in range(8)];probe=sum(cs);gp=loss_grad(w,probe)[1]
  x,gs=feature_batch(w,v,cs,probe,'centered');ref=[]
  for i,c in enumerate(cs):
   l,g=loss_grad(w,c);d=-LR*(.9*v+g);ref.append([l,np.linalg.norm(g),np.sum(g*gp),-np.sum(d*gp),np.linalg.norm(d)]+features(w,g,d,True).tolist())
   wn,vn=step(w,v,c);np.testing.assert_allclose(vn,.9*v+gs[i],atol=1e-12);np.testing.assert_allclose(wn,w-LR*(.9*v+gs[i]),atol=1e-12)
  np.testing.assert_allclose(x,ref,atol=1e-12,rtol=1e-12)

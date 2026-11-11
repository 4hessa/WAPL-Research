import unittest
import numpy as np
from experiments.pilot import utility, update, loss

class CounterfactualTests(unittest.TestCase):
    def setUp(self):
        self.w=np.array([.1,-.2]);self.v=np.array([.03,.02])
        self.c=(np.array([[1.,2.]]),np.array([1.]))
    def test_one_step_matches_direct_difference(self):
        wi,_=update(self.w,self.v,*self.c)
        self.assertAlmostEqual(utility(self.w,self.v,self.c,[],self.c),loss(self.w,*self.c)-loss(wi,*self.c))
    def test_parent_unchanged_and_repeatable(self):
        w,v=self.w.copy(),self.v.copy()
        a=utility(w,v,self.c,[self.c]*3,self.c)
        self.assertEqual(a,utility(w,v,self.c,[self.c]*3,self.c))
        np.testing.assert_array_equal(w,self.w);np.testing.assert_array_equal(v,self.v)
    def test_optimizer_state_matters(self):
        a=utility(self.w,self.v,self.c,[self.c]*3,self.c)
        b=utility(self.w,np.zeros(2),self.c,[self.c]*3,self.c)
        self.assertNotAlmostEqual(a,b,places=8)
if __name__=='__main__':unittest.main()

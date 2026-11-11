import unittest
import numpy as np
from experiments.time_policy import timed_run,V,counts
class DeadlineTests(unittest.TestCase):
 def test_late_update_is_not_accepted(self):
  w=np.zeros((V,V));p=[np.array([65,66,65])];times=iter([0.,0.,2.,2.])
  wn,r=timed_run(w,w,p,counts(p[0]),'random',{},np.zeros((1,8),dtype=int),[0],budget=1.,clock=lambda:next(times))
  np.testing.assert_array_equal(wn,w);self.assertEqual(r['updates'],0);self.assertEqual(r['attempted'],1)
 def test_in_time_update_accepted(self):
  w=np.zeros((V,V));p=[np.array([65,66,65])];times=iter([0.,0.,.5,.6])
  wn,r=timed_run(w,w,p,counts(p[0]),'random',{},np.zeros((1,8),dtype=int),[0],budget=1.,clock=lambda:next(times))
  self.assertEqual(r['updates'],1);self.assertGreater(np.linalg.norm(wn),0)

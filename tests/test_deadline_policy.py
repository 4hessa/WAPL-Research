import unittest
import numpy as np
import torch
from experiments.deadline_policy import timed_policy
from experiments.transformer_pilot import TinyTransformer,optimizer,update,vec

class TransformerDeadlineTests(unittest.TestCase):
    def check_deadline(self,completion,expected):
        torch.set_num_threads(1);torch.manual_seed(34);m=TinyTransformer();o=optimizer(m);pool=torch.randint(0,128,(8,33));update(m,o,pool[:2]);before=vec(m).clone()
        clock=iter([0.,0.,completion,completion+.01])
        result,record=timed_policy(m,o,pool,pool[:2],'random',None,np.arange(8)[None],np.array([0]),clock=lambda:next(clock))
        torch.testing.assert_close(vec(m),before,rtol=0,atol=0);self.assertEqual(record['updates'],expected);self.assertEqual(record['attempted'],1)
        if expected:self.assertGreater(float((vec(result)-before).norm()),0)
        else:torch.testing.assert_close(vec(result),before,rtol=0,atol=0)
    def test_late_update_discarded(self):self.check_deadline(1.1,0)
    def test_in_time_update_accepted(self):self.check_deadline(.8,1)

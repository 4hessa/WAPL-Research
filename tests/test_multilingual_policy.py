import unittest
import numpy as np
import torch
from experiments.multilingual_readiness import ByteTransformer,byte_loss
from experiments.multilingual_policy import optimizer,run_policy,evaluate

class MultilingualPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(44);self.model=ByteTransformer();self.opt=optimizer(self.model);self.data=torch.randint(0,256,(8,65))
    def test_batched_evaluation_matches_all(self):
        with torch.no_grad():expected=float(byte_loss(self.model,self.data))
        # Float32 reduction order changes slightly with batch size.
        np.testing.assert_allclose(evaluate(self.model,self.data,batch_size=3),expected,rtol=1e-6,atol=1e-6)
    def test_late_update_is_discarded(self):
        before={k:v.clone() for k,v in self.model.state_dict().items()};clock=iter([0.,0.,3.1,3.2])
        m,r=run_policy(self.model,self.opt,self.data,np.arange(8)[None],np.array([0]),'random','time',clock=lambda:next(clock))
        self.assertEqual(r['updates'],0)
        for k,v in before.items():torch.testing.assert_close(m.state_dict()[k],v,rtol=0,atol=0);torch.testing.assert_close(self.model.state_dict()[k],v,rtol=0,atol=0)
    def test_in_time_update_is_accepted(self):
        clock=iter([0.,0.,2.,2.1]);m,r=run_policy(self.model,self.opt,self.data,np.arange(8)[None],np.array([0]),'random','time',clock=lambda:next(clock))
        self.assertEqual(r['updates'],1);self.assertFalse(torch.equal(m.head.weight,self.model.head.weight))
    def test_fixed_budget_counts_updates(self):
        _,r=run_policy(self.model,self.opt,self.data,np.tile(np.arange(8),(2,1)),np.array([0,1]),'random','updates',clock=lambda:0.,fixed_updates=2)
        self.assertEqual(r['updates'],2);self.assertEqual(r['choices'],[0,1])

import unittest
from unittest.mock import patch
import numpy as np
import torch
from experiments.deadline_policy import timed_policy
from experiments.transformer_pilot import TinyTransformer,optimizer

class SparsePolicyTests(unittest.TestCase):
    def test_only_steps_zero_and_sixteen_guided(self):
        torch.set_num_threads(1);torch.manual_seed(36);m=TinyTransformer();o=optimizer(m);pool=torch.randint(0,128,(8,33));ids=np.tile(np.arange(8),(17,1));choices=np.zeros(17,dtype=int)
        with patch('experiments.deadline_policy.score_candidates',return_value=np.arange(8)) as score:
            _,r=timed_policy(m,o,pool,pool[:2],'pairwise',None,ids,choices,clock=lambda:0.,guidance_interval=16)
        self.assertEqual(score.call_count,2);self.assertEqual(r['guided_accepted'],2);self.assertEqual(r['guided_attempted'],2)
        self.assertEqual(r['choices'],[7]+[0]*15+[7]);self.assertEqual(r['updates'],17)

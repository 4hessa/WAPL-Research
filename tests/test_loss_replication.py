import unittest
from unittest.mock import patch
import numpy as np
import torch
from experiments.deadline_policy import timed_policy
from experiments.transformer_pilot import TinyTransformer,optimizer

class ShamLossTests(unittest.TestCase):
    def test_sham_computes_scores_but_chooses_random(self):
        torch.set_num_threads(1);torch.manual_seed(40);m=TinyTransformer();o=optimizer(m);pool=torch.randint(0,128,(8,33));ids=np.tile(np.arange(8),(17,1));choices=np.ones(17,dtype=int)
        with patch('experiments.deadline_policy.score_candidates',return_value=np.arange(8)) as scorer:
            _,r=timed_policy(m,o,pool,pool[:2],'loss_sham',None,ids,choices,clock=lambda:0.,guidance_interval=16)
        self.assertEqual(scorer.call_count,2);self.assertEqual(r['choices'],choices.tolist());self.assertEqual(r['guided_accepted'],2)
        for call in scorer.call_args_list:self.assertEqual(call.args[4],'loss')

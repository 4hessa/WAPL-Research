import copy,unittest
import numpy as np
import torch
from experiments.sequential_policy import score_candidates
from experiments.transformer_pilot import TinyTransformer,optimizer,update,clone,vec,loss

class SequentialPolicyTests(unittest.TestCase):
    def test_scoring_preserves_parent_and_matches_direct_alignment(self):
        torch.set_num_threads(1);torch.manual_seed(32);m=TinyTransformer();o=optimizer(m)
        data=torch.randint(0,128,(6,33));update(m,o,data[:2]);weights=vec(m).clone();grads=vec(m,True).clone();state=copy.deepcopy(o.state_dict())
        scores=score_candidates(m,o,data[2:4],data[4:],'alignment')
        torch.testing.assert_close(vec(m),weights,rtol=0,atol=0);torch.testing.assert_close(vec(m,True),grads,rtol=0,atol=0)
        for key,value in state['state'].items():
            for field,tensor in value.items():torch.testing.assert_close(o.state_dict()['state'][key][field],tensor,rtol=0,atol=0)
        pm,po=clone(m,o);po.zero_grad();loss(pm,data[4:]).backward();pg=vec(pm,True)
        expected=[]
        for c in data[2:4]:
            cm,co=clone(m,o);update(cm,co,c[None]);expected.append(float(-(vec(cm)-weights).dot(pg)))
        np.testing.assert_array_equal(scores,expected)

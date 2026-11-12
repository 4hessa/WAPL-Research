import copy,unittest
import numpy as np
import torch
from experiments.sequential_policy import score_candidates
from experiments.transformer_pilot import TinyTransformer,optimizer,update,loss,vec

class LossProxyTests(unittest.TestCase):
    def test_batched_losses_match_individual_and_preserve_state(self):
        torch.set_num_threads(1);torch.manual_seed(38);m=TinyTransformer();o=optimizer(m);data=torch.randint(0,128,(8,33));update(m,o,data[:2]);w=vec(m).clone();g=vec(m,True).clone();state=copy.deepcopy(o.state_dict())
        scores=score_candidates(m,o,data,None,'loss')
        with torch.no_grad():expected=np.array([float(loss(m,c[None])) for c in data])
        np.testing.assert_allclose(scores,expected,rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(vec(m),w,rtol=0,atol=0);torch.testing.assert_close(vec(m,True),g,rtol=0,atol=0)
        for key,value in state['state'].items():
            for field,tensor in value.items():torch.testing.assert_close(o.state_dict()['state'][key][field],tensor,rtol=0,atol=0)

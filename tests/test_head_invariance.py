import unittest
import torch
from experiments.multilingual_readiness import ByteTransformer
from experiments.multilingual_policy import optimizer
from experiments.head_invariance import canonical_head,shifted_copy

class HeadInvarianceTests(unittest.TestCase):
    def test_projection_only_changes_head_and_preserves_original(self):
        torch.manual_seed(46);m=ByteTransformer().double();w={k:p.detach().clone() for k,p in m.named_parameters()};original={k:v.clone() for k,v in w.items()};c=canonical_head(w)
        for k in w:
            torch.testing.assert_close(w[k],original[k],rtol=0,atol=0)
            if not k.startswith('head.'):torch.testing.assert_close(c[k],w[k],rtol=0,atol=0)
        torch.testing.assert_close(c['head.weight'].mean(0),torch.zeros(32,dtype=torch.float64),rtol=0,atol=1e-15)
        self.assertLess(abs(float(c['head.bias'].mean())),1e-15)
    def test_head_shift_preserves_probabilities(self):
        torch.set_num_threads(1);torch.manual_seed(46);m=ByteTransformer().double();o=optimizer(m);shift=torch.randn(32,dtype=torch.float64);sm,_=shifted_copy(m,o,shift);x=torch.randint(0,256,(2,64))
        with torch.no_grad():a=m(x).softmax(-1);b=sm(x).softmax(-1)
        torch.testing.assert_close(a,b,rtol=0,atol=1e-12)
        torch.testing.assert_close(sm.head.weight-m.head.weight,shift.expand(256,-1),rtol=0,atol=1e-15)

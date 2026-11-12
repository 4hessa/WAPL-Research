import unittest
import numpy as np
import torch
from experiments.multilingual_readiness import ByteTransformer,byte_chunks,byte_loss
from tools.prepare_multilingual import extract_text

class MultilingualReadinessTests(unittest.TestCase):
    def test_utf8_roundtrip_and_shifted_targets(self):
        text='العربية والإنجليزية: اختبار UTF-8. '*8;blob=text.encode('utf-8');self.assertEqual(bytes(blob).decode('utf-8'),text)
        chunks=byte_chunks(blob);self.assertGreater(int(chunks.max()),127)
        np.testing.assert_array_equal(chunks.numpy().ravel(),np.frombuffer(blob[:len(chunks)*65],dtype=np.uint8))
        self.assertEqual(chunks[:,1:].shape[1],64)
    def test_causal_mask_and_finite_arabic_loss(self):
        torch.set_num_threads(1);torch.manual_seed(43);model=ByteTransformer();x=torch.randint(0,256,(2,64));changed=x.clone();changed[:,32:]=(changed[:,32:]+1)%256
        with torch.no_grad():a=model(x);b=model(changed)
        torch.testing.assert_close(a[:,:32],b[:,:32],rtol=0,atol=0)
        batch=byte_chunks(('نص عربي للتجربة فقط. '*20).encode())[:2];self.assertTrue(torch.isfinite(byte_loss(model,batch)))
        with self.assertRaises(ValueError):model(torch.zeros(1,65,dtype=torch.long))
    def test_extracts_text_not_annotations(self):
        blob='# sent_id = 1\n# text = نص عربي\n1\tنص\tlemma\n\n# text = Hello.\n'.encode()
        self.assertEqual(extract_text(blob),['نص عربي','Hello.'])

import json,unittest
from pathlib import Path
import numpy as np
from experiments.pud_transfer import frozen_scores,partition

class PUDTransferTests(unittest.TestCase):
    def test_frozen_prediction_matches_historical_ranker(self):
        rows=json.loads(Path('results/corrected-feature-records.json').read_text());frozen=json.loads(Path('research/pud-frozen-rankers.json').read_text());old=json.loads(Path('results/pairwise-logistic.json').read_text())
        for lang in ['ar','en']:
            rr=[r for r in rows if r['language']==lang and r['split']=='validate']
            for arm in ['baseline','corrected','raw']:
                expected=next(r['predictions'] for r in old['results'][lang]['runs'] if r['name']==arm);np.testing.assert_allclose(frozen_scores(rr,arm,frozen['models'][lang]),expected,rtol=0,atol=1e-12)
    def test_partition_respects_held_out_sentences(self):
        rows=[{'source_index':i,'text':chr(65+i)*5000} for i in range(10)];folds=[[i] for i in range(10)]
        pool,probe,target=partition(rows,folds,0)
        self.assertNotIn(65,pool.numpy());self.assertNotIn(66,pool.numpy());self.assertNotIn(65,probe.numpy());self.assertEqual(int(target[0,0]),65)

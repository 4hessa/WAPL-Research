import copy,json,unittest
from pathlib import Path
import numpy as np
from experiments.pairwise_student import fit_rankers,regret

class PairwiseStudentTests(unittest.TestCase):
    def test_test_data_cannot_change_fit_or_selection(self):
        rows=json.loads(Path('results/student-records.json').read_text());a,sa=fit_rankers(rows)
        for r in rows:
            r['target']=1e10
            if r['split']=='test':r['teacher']=-1e10;r['baseline']=[1e10]*5
        b,sb=fit_rankers(rows)
        self.assertEqual(sa,sb)
        for arm in a:
            np.testing.assert_array_equal(a[arm][0].mean_,b[arm][0].mean_)
            np.testing.assert_array_equal(a[arm][1].coef_,b[arm][1].coef_)
            self.assertEqual(a[arm][1].intercept_,b[arm][1].intercept_)

    def test_pairwise_invariant_to_state_label_offsets(self):
        rows=json.loads(Path('results/student-records.json').read_text());a,_=fit_rankers(rows)
        for r in rows:
            if r['split'] in ['train','dev']:r['teacher']+=r['checkpoint']*.01
        b,_=fit_rankers(rows)
        np.testing.assert_allclose(a['pairwise'][1].coef_,b['pairwise'][1].coef_,rtol=1e-10,atol=1e-12)
        self.assertEqual(regret(np.array([1.,2.,3.,4.]),np.array([1.,2.,3.,4.])),0)

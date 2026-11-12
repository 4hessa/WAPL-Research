import unittest
import numpy as np
from experiments.ranking_diagnostic import metrics

class RankingDiagnosticTests(unittest.TestCase):
    def test_offset_preserves_selection_and_decomposes_error(self):
        y=np.array([0.,1.,2.,3.]);p=y+5
        a=metrics(y,p)
        self.assertEqual(a['within_mse'],0)
        self.assertEqual(a['offset_mse'],25)
        self.assertEqual(a['mse'],25)
        self.assertEqual(a['pair_accuracy'],1)
        self.assertEqual(a['regret'],0)
        b=metrics(y,-y)
        self.assertEqual(b['pair_accuracy'],0)
        self.assertEqual(b['top2_overlap'],0)
        self.assertEqual(b['regret'],2)

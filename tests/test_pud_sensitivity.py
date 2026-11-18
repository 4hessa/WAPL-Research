import unittest
import numpy as np
from experiments.pud_sensitivity import metric_matrix,sensitivity

class PUDSensitivityTests(unittest.TestCase):
    def test_keyed_grid_and_duplicate_rejection(self):
        rows=[dict(seed=s,fold=f,top2_regret=float(s+f)) for s in [1,2] for f in [0,1]]
        np.testing.assert_array_equal(metric_matrix(rows[::-1],[1,2],[0,1]),[[1,2],[2,3]])
        with self.assertRaises(ValueError):metric_matrix(rows+[rows[0]],[1,2],[0,1])
        with self.assertRaises(ValueError):metric_matrix(rows[:-1],[1,2],[0,1])
    def test_single_fold_can_reverse_mean(self):
        result=sensitivity(np.array([[-1.,-1.,10.],[-1.,-1.,10.]]))
        self.assertGreater(result['mean_difference'],0);self.assertEqual(result['leave_one_fold_out'],[4.5,4.5,-1.]);self.assertEqual(result['state_counts'],{'corrected_better':4,'corrected_worse':2,'exact_tie':0})

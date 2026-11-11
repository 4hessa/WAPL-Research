import unittest
from experiments.state_coverage import training_rows
class CoverageTests(unittest.TestCase):
 def test_equal_training_budget_and_no_eval_rows(self):
  records=[{'split':s,'seed':seed,'checkpoint':cp,'candidate':i} for s in ['train','dev','test'] for seed in range(12) for cp in [24,120,480] for i in range(48)]
  early=training_rows(records,'early');multi=training_rows(records,'multi')
  self.assertEqual(len(early),576);self.assertEqual(len(multi),576)
  self.assertEqual({r['checkpoint'] for r in early},{24})
  self.assertEqual({r['checkpoint'] for r in multi},{24,120,480})
  self.assertTrue(all(r['split']=='train' for r in early+multi))

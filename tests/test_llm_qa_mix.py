import unittest

import numpy as np
import torch

from experiments import llm_qa_mix as stage
from experiments import llm_web_filter as s73
from tests.test_llm_gpu import FakeTokenizer


def mmlu_row(i, q, answer='A', opts=('x', 'y')):
    return {'ID': i, 'Question': q, 'Context': float('nan'), 'Answer Key': answer,
            'Option 1': opts[0], 'Option 2': opts[1], 'Option 3': float('nan'), 'Option 4': float('nan'),
            'Option 5': float('nan')}


class QAMixTests(unittest.TestCase):
    def setUp(self):
        self.tok = FakeTokenizer()

    def test_content_decontamination_ignores_template(self):
        banned = stage.content_banned([mmlu_row(1, 'abcdefghijkl')], self.tok)
        items, dropped = stage.clean_qa([('zz abcdefghij zz', 'ok'), ('different text', 'fine answer')], self.tok, banned)
        self.assertEqual(dropped, 1)
        self.assertEqual(items[0], self.tok(stage.qa_text('different text', 'fine answer'))['input_ids'])

    def test_pack_starts_chunks_at_item_starts(self):
        items = [[1] * 70, [2] * 30, [3] * 40, [4] * 10]
        chunks = stage.pack(items, [0, 1, 2, 3], length=65)
        self.assertEqual(chunks[0], [1] * 65)
        self.assertEqual(chunks[1], [2] * 30 + [3] * 35)
        self.assertEqual(len(chunks), 2)  # the short last item is discarded

    def test_near_duplicate_criteria_and_members(self):
        evals = [mmlu_row(1, 'ما هي عاصمة المملكة العربية السعودية'), mmlu_row(2, 'كم عدد سور القرآن الكريم في المصحف')]
        cands = [mmlu_row(10, 'ما هِيَ عاصمة المملكة العربية السعودية؟'),          # identical after normalisation
                 mmlu_row(11, 'ما هي عاصمة المملكة العربية السعودية حاليا'),      # high Jaccard / containment
                 mmlu_row(12, 'كم عدد سور القرآن الكريم'),                        # contained in an eval question
                 mmlu_row(13, 'اشرح دورة الماء في الطبيعة بالتفصيل', opts=('مطر', 'ثلج'))]
        keep, counts = stage.near_duplicates(cands, evals, self.tok)
        self.assertEqual(keep, [3])
        self.assertEqual(counts['dropped_total'], 3)
        self.assertGreaterEqual(counts['identical'], 1)
        self.assertGreaterEqual(counts['containment'], 1)
        members = []
        chunks = stage.pack([[1] * 70, [2] * 30, [3] * 40], [0, 1, 2], length=65, members=members)
        self.assertEqual((len(chunks), members), (2, [[0], [1, 2]]))
        text = stage.mmlu_qa_text(mmlu_row(5, 'سؤال قصير', answer='B', opts=('خطأ', 'صواب')))
        self.assertTrue(text.startswith('سؤال: سؤال قصير\nالجواب: صواب'))

    def test_composition_and_nested_pools(self):
        self.assertEqual(stage.composition(0.25), (512, 768, 768))
        self.assertEqual(stage.composition(0.1), (205, 922, 921))
        self.assertEqual(stage.composition(1.0), (2048, 0, 0))
        qa = torch.zeros(2048, 3, dtype=torch.long)
        kept, removed = torch.ones(1843, 3, dtype=torch.long), 2 * torch.ones(922, 3, dtype=torch.long)
        p = stage.build_pool(qa, kept, removed, 0.1)
        self.assertEqual([int((p[:, 0] == v).sum()) for v in (0, 1, 2)], [205, 922, 921])
        self.assertEqual(stage.qa_share(np.arange(205), 0.1), 1.)
        self.assertEqual(stage.qa_share([205, 0], 0.1), .5)

    def test_item_splits_are_disjoint_from_stage73(self):
        pilot, target = stage.item_splits(14455)
        used = set(s73.sample_target(14455).tolist())
        self.assertEqual((len(pilot), len(target)), (256, 512))
        self.assertFalse(used & set(pilot.tolist()) or used & set(target.tolist()) or set(pilot) & set(target))

    def test_schedule(self):
        a = stage.schedule(1, 37400)
        b = stage.schedule(1, 37400)
        self.assertEqual(a[3], b[3])
        self.assertNotEqual(a[3], stage.schedule(0, 37400)[3])
        self.assertEqual((len(a[2]), len(set(a[2].tolist())), a[1].max() < 512), (512, 512, True))

    def test_gate_and_summary(self):
        self.assertTrue(stage.gate([dict(sigma=1.0, losses={'0': 3., '2400': 3.5}),
                                    dict(sigma=0.25, losses={'0': 3., '2400': 2.9})]))
        self.assertFalse(stage.gate([dict(sigma=0.25, losses={'0': 3., '2400': 3.1})]))
        plan = dict(phase_B_main=dict(sigma=[0.25, 0.1], seeds=[1, 2]))
        rows = []
        for s in (0.25, 0.1):
            for seed in (1, 2):
                acc = {'0': .3, '1500': .3, '2400': .3}
                rows += [dict(sigma=s, seed=seed, arm='random', losses={'0': 3., '1500': 2.8, '2400': 2.7}, accuracy=acc),
                         dict(sigma=s, seed=seed, arm='align_grad', losses={'0': 3., '1500': 2.5, '2400': 2.4},
                              accuracy=acc, qa_share=.8, scoring_seconds=100.),
                         dict(sigma=s, seed=seed, arm='random512', losses={'0': 3., '1500': 2.9, '2400': 3.}, accuracy=acc,
                              qa_share=.25)]
        cal = dict(costs=dict(random=.1, align_grad=90.), updates=1500, device='L4')
        out = stage.summarize(rows, cal, plan)
        s = out['sigmas']['0.25']
        self.assertAlmostEqual(out['primary']['mean'], -.2)
        self.assertAlmostEqual(s['matched_selection']['mean'], -.4)
        self.assertAlmostEqual(s['repetition']['mean'], .1)
        self.assertAlmostEqual(s['selection_term'], -.3)
        self.assertAlmostEqual(s['throughput_term'], .1)
        self.assertTrue(all(out['prediction_holds'].values()))
        with self.assertRaises(ValueError):
            stage.summarize(rows[:-1], cal, plan)


if __name__ == '__main__':
    unittest.main()

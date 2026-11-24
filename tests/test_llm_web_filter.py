import math
import unittest

import numpy as np
import torch

from experiments import llm_common as llm
from experiments import llm_web_filter as stage
from tests.test_llm_gpu import FakeTokenizer
from tests.test_llm_selection import tiny


def row(i, answer='B', context=None, options=('aa', 'bbb', 'c', None, None)):
    out = {'ID': i, 'Question': f'question {i}', 'Context': context, 'Answer Key': answer}
    for j, o in enumerate(options, 1):
        out[f'Option {j}'] = o if o is not None else float('nan')
    return out


class WebFilterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tok = FakeTokenizer()
        self.device = torch.device('cpu')

    def test_items_cloze_format_and_answer_index(self):
        items = stage.build_items([row(1), row(2, 'A', context='ctx', options=('x', 'y', 'z', 'w', 'v'))], self.tok)
        self.assertEqual(items[0]['correct'], 1)
        self.assertEqual(len(items[0]['options']), 3)
        self.assertEqual(items[0]['options'][1], self.tok(' bbb')['input_ids'])
        self.assertEqual(items[1]['prompt'][:4], self.tok('ctx\n')['input_ids'])
        long = stage.build_items([row(3, options=('a' * 100, 'b' * 100, None, None, None), context='q' * 500)], self.tok)
        self.assertEqual(len(long[0]['prompt']), stage.PROMPT_MAX)
        self.assertEqual(len(long[0]['options'][0]), stage.CONT_MAX)
        with self.assertRaises(ValueError):
            stage.build_items([row(4, 'E')], self.tok)

    def test_first_tokens_and_decontamination(self):
        texts = ['', 'ab' * 40, 'short', 'cd' * 40]
        rows, chunks = stage.first_tokens(texts, self.tok)
        self.assertEqual(rows, [1, 3])
        self.assertTrue(all(len(c) == stage.CHUNK for c in chunks))
        banned = stage.ngrams(self.tok('dcdcdcdcdc')['input_ids'])
        r, c, dropped = stage.decontaminate(rows, chunks, banned)
        self.assertEqual((r, dropped), ([1], 1))

    def test_pools_composition_and_determinism(self):
        kc = [[i % 40] * stage.CHUNK for i in range(5000)]
        rc = [[(i + 7) % 40] * stage.CHUNK for i in range(3000)]
        pools, pop = stage.build_pools(list(range(5000)), kc, list(range(3000)), rc)
        self.assertEqual(pools['kept'].shape, (stage.POOL_SIZE, 65))
        self.assertTrue(torch.equal(pools['raw'][:stage.RAW_KEPT], pools['kept'][:stage.RAW_KEPT]))
        self.assertEqual(len(set(pop['removed_documents'])), stage.POOL_SIZE - stage.RAW_KEPT)
        again, pop2 = stage.build_pools(list(range(5000)), kc, list(range(3000)), rc)
        self.assertEqual(pop, pop2)

    def test_schedule_and_keep_top(self):
        a, b, d = stage.schedule(1, 37300)
        a2, b2, d2 = stage.schedule(1, 37300)
        self.assertTrue((a == a2).all() and (b == b2).all() and d == d2)
        self.assertNotEqual(d, stage.schedule(0, 37300)[2])
        self.assertEqual((a.max() < stage.POOL_SIZE, b.max() < stage.KEEP, len(a)), (True, True, stage.BUDGET))
        scores = np.array([1., 3., 3., 0., 2.])
        self.assertEqual(stage.keep_top(scores, 3).tolist(), [1, 2, 4])
        self.assertEqual(stage.keep_top(np.ones(5), 2).tolist(), [0, 1])

    def test_continuation_loss_ignores_right_padding(self):
        model = tiny(3)
        items = stage.build_items([row(1), row(2, 'C', context='longer context text here')], self.tok)
        seqs = stage._sequences(items, True)
        together = stage.continuation_nll(model, seqs, self.device)
        alone = torch.cat([stage.continuation_nll(model, [s], self.device) for s in seqs])
        self.assertTrue(torch.allclose(together, alone, atol=1e-5))
        loss, acc = stage.evaluate(model, items, self.device)
        self.assertAlmostEqual(loss, float(alone.mean()), places=5)
        self.assertIn(acc, (0., .5, 1.))

    def test_fd_scores_track_exact_gradient_scores(self):
        model = tiny(4)
        with torch.no_grad():  # move off the B = 0 start so both LoRA factors carry gradient
            for p in llm.adapter_params(model):
                p.add_(0.05 * torch.randn(p.shape, generator=torch.Generator().manual_seed(p.numel())))
        pool = torch.randint(0, 40, (24, 9), generator=torch.Generator().manual_seed(2))
        items = stage.build_items([row(i) for i in range(4)], self.tok)
        exact, _ = stage.score(model, 'align_grad', pool, items, self.device)
        before = [p.detach().clone() for p in llm.adapter_params(model)]
        fd, _ = stage.score(model, 'align_fd', pool, items, self.device)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, llm.adapter_params(model))))
        g = stage.probe_gradient(model, items, self.device)
        np.testing.assert_allclose(fd * float(g.norm()), exact, rtol=2e-2, atol=1e-4)
        # exact scores equal dot products of separately computed gradients
        params = llm.adapter_params(model)
        for p in params:
            p.grad = None
        llm.lm_loss(model, pool[5:6]).backward()
        single = torch.cat([p.grad.reshape(-1) for p in params]) @ g
        self.assertAlmostEqual(float(single), exact[5], places=4)

    def test_train_records_digest_and_checkpoints(self):
        data = torch.randint(0, 40, (10, 9), generator=torch.Generator().manual_seed(1))
        order = np.arange(60) % 10
        evaluator = lambda m: (float(llm.lm_loss(m, data[:3])), 0.)
        out = stage.train(tiny(2), data, order, 60, {0, 20, 60}, evaluator, self.device)
        self.assertEqual(sorted(out['losses']), ['0', '20', '60'])
        again = stage.train(tiny(2), data, order, 60, {0, 20, 60}, evaluator, self.device)
        self.assertEqual((out['losses'], out['adapter_digest_50']), (again['losses'], again['adapter_digest_50']))

    def test_budgets_agreement_and_summary(self):
        n = stage.budgets(dict(random=.1, align_grad=100., align_fd=90.))
        self.assertEqual(n, dict(align_grad=1400))
        self.assertLess(stage.budgets(dict(random=.1, align_grad=300.))['align_grad'], 1)
        x = np.arange(stage.POOL_SIZE, dtype=float)
        self.assertEqual(stage.agreement(x, 2 * x), dict(spearman=1., kept_overlap=1.))
        self.assertEqual(stage.agreement(x, -x)['kept_overlap'], 0.)
        plan = dict(seeds=[1, 2])
        rows = []
        for pool in stage.POOLS:
            for seed in plan['seeds']:
                rows.append(dict(pool=pool, seed=seed, arm='random', losses={'0': 3., '1400': 2.3, '2400': 2.},
                                 accuracy={'0': .4, '1400': .4, '2400': .5}))
                rows.append(dict(pool=pool, seed=seed, arm='align_grad', losses={'0': 3., '1400': 2.1, '2400': 1.9},
                                 accuracy={'0': .4, '1400': .45, '2400': .5}, removed_share=.1, scoring_seconds=5.))
        cal = dict(costs=dict(random=.1, align_grad=100., align_fd=90.), updates=n, device='L4',
                   fd_vs_grad=dict(spearman=.9, kept_overlap=.8))
        out = stage.summarize(rows, cal, plan)
        self.assertAlmostEqual(out['primary']['mean'], .1)
        a = out['pools']['raw']['arms']['align_grad']
        self.assertAlmostEqual(a['selection_term'], -.2)
        self.assertAlmostEqual(a['throughput_term'], .3)
        self.assertAlmostEqual(a['amortized'], -.1)
        self.assertAlmostEqual(a['accuracy_charged'], -.05)
        self.assertEqual(out['prediction_holds'], {'raw align_grad negative': False, 'kept align_grad positive': True})
        self.assertAlmostEqual(out['scoring_in_random_steps']['align_fd'], 900.)
        self.assertTrue(math.isclose(out['pools']['raw']['random_minus_no_update'], -1.))
        with self.assertRaises(ValueError):
            stage.summarize(rows[:-1], cal, plan)

if __name__ == '__main__':
    unittest.main()

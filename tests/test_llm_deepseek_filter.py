import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments import llm_common as llm
from experiments import llm_deepseek_filter as stage
from experiments import llm_gpu as s69
from experiments import llm_web_filter as s73


def tiny_llama(seed=0, dtype=torch.float32):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64)
    return LlamaForCausalLM(cfg).to(dtype)


def lora_nonzero(model, seed=1):
    """Give the B factors nonzero values so that dA and the adapter path are exercised."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, llm.LoRALinear):
                m.B.copy_(torch.randn(m.B.shape, generator=g) * .1)


def items(seed=0, n=6):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append(dict(prompt=rng.integers(1, 64, size=int(rng.integers(3, 9))).tolist(),
                        options=[rng.integers(1, 64, size=int(rng.integers(1, 4))).tolist() for _ in range(3)],
                        correct=int(rng.integers(3))))
    return out


class ProtocolAndScheduleTests(unittest.TestCase):
    def test_protocol_matches_implementation(self):
        plan = json.loads(stage.PROTOCOL.read_text())
        self.assertEqual(plan['stage'], 77)
        self.assertEqual(tuple(plan['design']['arms']), stage.ARMS)
        self.assertEqual(plan['design']['seeds'], [37700, 37701, 37702, 37703])
        self.assertEqual(len(stage.run_grid(plan['design']['seeds'])), 12)
        self.assertIn('B = 7400', plan['design']['budget'])
        self.assertIn('N = min(N_uncapped, 1992)', plan['design']['budget'])
        self.assertIn('N = min(N_uncapped, 1992) = 1992', plan['amendment_2']['changes'][0])
        self.assertEqual((stage.BUDGET, stage.POOL_SIZE, stage.N_QA, stage.CHUNK), (7400, 7968, 1992, 147))
        self.assertIn(stage.REVISION, plan['model']['trained_and_scorer'])
        for name, digest in stage.WEIGHTS_SHA256.items():
            self.assertIn(f'{name} {digest}', plan['model']['trained_and_scorer'])
        self.assertIn('default_rng(7711)', plan['data']['web_chunks'])
        self.assertIn('default_rng(7722)', plan['task']['target'])
        self.assertIn('seed 7703', plan['primary'])

    def test_web_draws(self):
        k, r = stage.web_draws(41000, 18000)
        self.assertEqual((len(set(k.tolist())), len(set(r.tolist()))), (2988, 2988))
        self.assertTrue((k == np.random.default_rng(7711).choice(41000, 2988, replace=False)).all())
        self.assertTrue((r == np.random.default_rng(7712).choice(18000, 2988, replace=False)).all())
        with self.assertRaises(SystemExit):
            stage.web_draws(41000, 2987)

    def test_schedule(self):
        for n in (1000, 1813, 1992):
            a = stage.schedule(37700, n)
            self.assertEqual(a['sha256'], stage.schedule(37700, n)['sha256'])
            self.assertEqual(len(a['once']), stage.BUDGET)
            self.assertEqual(len(a['rand']), stage.BUDGET)
            self.assertEqual(sorted(a['once'][:n].tolist()), list(range(n)))
            self.assertEqual(len(set(a['subset_n'].tolist())), n)
            self.assertTrue((a['rand'] < stage.POOL_SIZE).all() and (a['subset_n'] < stage.POOL_SIZE).all())
        self.assertNotEqual(stage.schedule(37700, 1813)['sha256'], stage.schedule(37701, 1813)['sha256'])
        a = stage.schedule(37700, 1000)
        self.assertEqual(len(set(a['once'][7000:7400].tolist())), 400)  # partial eighth pass, no repeats
        self.assertEqual(sorted(a['once'][6000:7000].tolist()), list(range(1000)))  # seventh pass

    def test_orders_views_and_shares(self):
        n = 1813
        sch = stage.schedule(37702, n)
        kept = np.sort(np.random.default_rng(0).choice(stage.POOL_SIZE, n, replace=False))
        for arm in stage.ARMS:
            order, members = stage.training_order(arm, sch, kept if arm == 'align_once' else None)
            self.assertEqual(len(order), stage.BUDGET)
            self.assertTrue(set(order.tolist()) <= set(np.asarray(members).tolist()))
            if arm != 'random':
                self.assertEqual(stage.views(order, members, n), dict(max=1, mean=1.0))
                self.assertEqual(stage.views(order, members, stage.BUDGET)['max'], 5)  # ceil(7400 / 1813)
        self.assertEqual(stage.qa_share([0, 1991, 1992, 7967]), 0.5)
        self.assertEqual(stage.random_qa(np.array([0, 0, 5, 3000, 9]), 4), dict(drawn=3, distinct=2))

    def test_arm_rotation(self):
        self.assertEqual(stage.arm_order(0), stage.ARMS)
        self.assertEqual(stage.arm_order(1)[0], 'align_once')
        self.assertEqual(stage.arm_order(3), stage.ARMS)
        self.assertEqual(len(set(stage.run_grid([1, 2, 3, 4]))), 12)

    def test_updates(self):
        self.assertEqual(stage.updates(dict(random=.1617, self_align_grad=903.5)), 1812)
        self.assertEqual(stage.updates(dict(random=.1, self_align_grad=640.)), 1000)
        cal = dict(random=0.16287156099951972, self_align_grad=872.3100562769996)  # the journaled calibration
        self.assertEqual((stage.uncapped_updates(cal), stage.updates(cal)), (2044, 1992))  # amendment 2

    def test_summary(self):
        plan = dict(design=dict(seeds=[1, 2]))
        L = dict(random=(2.8, 2.7), align_once=(2.5, 2.45), random_once=(2.75, 2.7))
        rows = []
        for seed in (1, 2):
            for arm, (ln, lb) in L.items():
                r = dict(seed=seed, arm=arm, losses={'0': 3., '1800': ln, '7400': lb},
                         accuracy={'0': .3, '1800': .32, '7400': .31})
                if arm == 'random':
                    r['random_qa'] = {'1800': dict(drawn=450, distinct=400), '7400': dict(drawn=1850, distinct=1200)}
                else:
                    r['qa_share'] = {'align_once': .95, 'random_once': .25}[arm]
                if arm == 'align_once':
                    r['scoring_seconds'] = 900.
                rows.append(r)
        cal = dict(costs=dict(random=.16, self_align_grad=896.), updates=1800, device='L4', scoring_batch=8)
        out = stage.summarize(rows, cal, plan)
        a = out['arms']
        self.assertAlmostEqual(out['primary']['mean'], -.2)
        self.assertAlmostEqual(a['matched']['mean'], -.25)
        self.assertAlmostEqual(a['one_view']['mean'], -.05)
        self.assertAlmostEqual(a['selection_term'], -.3)
        self.assertAlmostEqual(a['throughput_term'], .1)
        self.assertAlmostEqual(a['amortized']['mean'], -.25)
        self.assertAlmostEqual(a['random_minus_no_update'], -.3)
        self.assertAlmostEqual(out['scoring_in_random_steps'], 5600.)
        self.assertEqual(a['charged']['seeds_negative'], 2)
        self.assertTrue(all(out['prediction_holds'].values()))
        with self.assertRaises(ValueError):
            stage.summarize(rows[:-1], cal, plan)
        with self.assertRaises(ValueError):  # journaled N_uncapped must match the costs
            stage.summarize(rows, dict(cal, updates=1801), plan)
        rows[1]['losses']['0'] = 3.1
        with self.assertRaises(ValueError):
            stage.summarize(rows, cal, plan)

    def test_summary_uses_capped_n(self):
        plan = dict(design=dict(seeds=[1]))
        rows = []
        for arm, ln in dict(random=2.8, align_once=2.5, random_once=2.75).items():
            r = dict(seed=1, arm=arm, losses={'0': 3., '1992': ln, '7400': 2.7}, accuracy={'0': .3, '1992': .3, '7400': .3})
            if arm == 'random':
                r['random_qa'] = {k: dict(drawn=1, distinct=1) for k in ('1992', '7400')}
            else:
                r['qa_share'] = 1. if arm == 'align_once' else .25
            if arm == 'align_once':
                r['scoring_seconds'] = 872.
            rows.append(r)
        cal = dict(costs=dict(random=0.16287156099951972, self_align_grad=872.3100562769996), updates=2044,
                   device='L4', scoring_batch=8)
        out = stage.summarize(rows, cal, plan)
        self.assertEqual((out['updates'], out['updates_uncapped'], out['unused_charged_steps']), (1992, 2044, 52))
        self.assertAlmostEqual(out['primary']['mean'], -.2)

    def test_backup_retries_push_and_logs_stderr(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            journal = d / 'j.jsonl'
            journal.write_text(json.dumps(dict(kind='run')) + '\n')
            calls = []

            def fake(cmd, check=False, capture_output=False, text=False):
                calls.append(cmd[3])
                if cmd[3] == 'diff':
                    return subprocess.CompletedProcess(cmd, 1)
                if cmd[3] == 'push' and calls.count('push') == 1:
                    raise subprocess.CalledProcessError(1, cmd, stderr='remote hung up')
                if cmd[3] == 'commit':
                    self.assertIn('Stage 77 interim journal snapshot: 1 finished runs', cmd[-1])
                return subprocess.CompletedProcess(cmd, 0)

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                stage.backup(journal, None, d / 'wt', 'b', run=fake)
            self.assertEqual(calls.count('push'), 2)
            self.assertIn('remote hung up', buf.getvalue())
            self.assertIn('backup pushed: 1 finished runs (retry)', buf.getvalue())
            self.assertTrue((d / 'wt' / stage.INTERIM / 'journal.jsonl').exists())


class DataTests(unittest.TestCase):
    class Tok:
        def __call__(self, text, add_special_tokens=False):
            return dict(input_ids=[ord(c) % 50 + 1 for c in text])

    def test_qa_chunks_count_is_enforced(self):
        rows = [dict(Context=None, Question='س' * 40, **{'Option 1': 'ج' * 30, 'Answer Key': 'A'}) for _ in range(10)]
        with self.assertRaises(ValueError):
            stage.qa_chunks(rows, self.Tok())
        old = stage.N_QA
        try:
            stage.N_QA = 5
            chunks, members = stage.qa_chunks(rows, self.Tok())
        finally:
            stage.N_QA = old
        self.assertEqual(tuple(chunks.shape), (5, stage.CHUNK))
        self.assertEqual(sorted(i for m in members for i in m), list(range(10)))

    def test_excluded_rows_cover_stage_75_and_76_targets(self):
        ids = set()
        for path in (stage.STAGE75_JOURNAL, stage.STAGE76_JOURNAL):
            ids |= set(next(r for r in s69.read_journal(path) if r.get('kind') == 'calibration')
                       ['population']['target_items'])
        test = [dict(ID=i) for i in range(max(ids) + 1)]
        rows = stage.excluded_rows(test)
        self.assertEqual(len(ids), 1024)
        self.assertTrue(all(i in rows for i in ids))  # row index equals ID in this synthetic table
        self.assertGreater(len(rows), len(ids))  # plus the stage-73 target and stage-74 pilot and target


class AmendmentOneTests(unittest.TestCase):
    class Ds:  # 'DeepSeek': one token per character
        def __call__(self, text, add_special_tokens=False):
            return dict(input_ids=[ord(c) for c in text])

        def decode(self, ids):
            return ''.join(chr(i) for i in ids)

    class Qw:  # 'Qwen': one token per pair of characters
        def __call__(self, text, add_special_tokens=False):
            return dict(input_ids=[hash(text[i:i + 2]) % 997 for i in range(0, len(text), 2)])

    def test_protocol_records_amendment_1(self):
        plan = json.loads(stage.PROTOCOL.read_text())
        a = plan['amendment_1']
        self.assertIn(stage.QWEN_TOKENIZER_SHA256, ' '.join(a['changes']))
        self.assertIn('Qwen2.5-1.5B tokenizer', plan['data']['web_chunks'])
        self.assertIn('Qwen2.5-1.5B tokenizer', plan['task']['target_eligibility'])
        self.assertEqual(s73.NGRAM, 8)
        self.assertEqual(s73.ngrams.__defaults__, (8,))  # the 8-gram rule is used unchanged

    def test_decon_tokenizer_is_pinned(self):
        path = Path('models') / stage.QWEN_TOKENIZER / 'tokenizer.json'
        if not path.exists():
            self.skipTest('Qwen tokenizer not downloaded')
        self.assertEqual(stage.check_decon_tokenizer(Path('models')), stage.QWEN_TOKENIZER_SHA256)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / stage.QWEN_TOKENIZER).mkdir()
            (Path(d) / stage.QWEN_TOKENIZER / 'tokenizer.json').write_text('{}')
            with self.assertRaises(ValueError):
                stage.check_decon_tokenizer(Path(d))

    def test_view_and_decontamination_use_the_qwen_tokens(self):
        ds, qw = self.Ds(), self.Qw()
        texts = ['abcdefghijklmnopqrstuvwxyz', 'zyxwvutsrqponmlkjihgfedcba', 'abcdefghijklmnopqrZZZZZZZZ']
        chunks = [ds(x)['input_ids'] for x in texts]
        view = stage.qwen_view(chunks, ds, qw)
        self.assertEqual(view.tolist(), [qw(x)['input_ids'] for x in texts])
        banned = s73.ngrams(qw('abcdefghijklmnop')['input_ids'])  # one 8-gram of Qwen tokens = 16 characters
        rows, kept, dropped = stage.decontaminate([10, 11, 12], chunks, view, banned)
        self.assertEqual((rows, dropped), ([11], 2))
        self.assertEqual(kept, [chunks[1]])
        self.assertTrue(s73.ngrams(chunks[0]) & s73.ngrams(ds('abcdefgh')['input_ids']))  # DeepSeek 8-grams would differ


class NumericsTests(unittest.TestCase):
    def test_mixed_lora_equals_original_for_float32(self):
        base = torch.nn.Linear(16, 8)
        g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
        a, b = llm.LoRALinear(base, 8, 16, g1), stage.MixedLoRALinear(base, 8, 16, g2)
        with torch.no_grad():
            a.B.normal_()
            b.B.copy_(a.B)
        x = torch.randn(4, 5, 16)
        self.assertTrue(torch.equal(a(x), b(x)))

    def test_bfloat16_model_has_float32_adapters_and_logits(self):
        model = stage.add_lora(tiny_llama(dtype=torch.bfloat16), 5)
        params = llm.adapter_params(model)
        self.assertEqual(len(params), 2 * 2 * 2)
        self.assertTrue(all(p.dtype == torch.float32 for p in params))
        batch = torch.randint(1, 64, (1, 12))
        loss = llm.lm_loss(model, batch)
        self.assertEqual(model(input_ids=batch).logits.dtype, torch.float32)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params))

    def test_add_lora_initialisation_matches_stage_76(self):
        a = llm.add_lora(tiny_llama(), 11)
        b = stage.add_lora(tiny_llama(), 11)
        for p, q in zip(llm.adapter_params(a), llm.adapter_params(b)):
            self.assertTrue(torch.equal(p, q))

    def test_evaluate_matches_stage_73(self):
        model = stage.add_lora(tiny_llama(), 2)
        lora_nonzero(model)
        its = items()
        old = s73.EVAL_BATCH
        try:
            s73.EVAL_BATCH = 4
            ref = s73.evaluate(model, its, torch.device('cpu'))
        finally:
            s73.EVAL_BATCH = old
        got = stage.evaluate(model, its, torch.device('cpu'), batch=4)
        self.assertAlmostEqual(got[0], ref[0], places=6)
        self.assertEqual(got[1], ref[1])

    def test_score_matches_stage_73_in_float32(self):
        dev = torch.device('cpu')
        pool = torch.randint(1, 64, (20, 10))
        m1 = stage.add_lora(tiny_llama(), 4)
        lora_nonzero(m1)
        ref, _ = s73.score(m1, 'align_grad', pool, items(1), dev)
        m2 = stage.add_lora(tiny_llama(), 4)
        lora_nonzero(m2)
        got, secs = stage.score(m2, pool, items(1), dev, batch=8)
        self.assertGreater(secs, 0)
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-7 * np.abs(ref).max())

    def test_score_runs_in_bfloat16(self):
        model = stage.add_lora(tiny_llama(dtype=torch.bfloat16), 4)
        lora_nonzero(model)
        before = [p.detach().clone() for p in llm.adapter_params(model)]
        scores, _ = stage.score(model, torch.randint(1, 64, (12, 10)), items(2), torch.device('cpu'), batch=4)
        self.assertEqual(scores.shape, (12,))
        self.assertTrue(np.isfinite(scores).all())
        for p, q in zip(llm.adapter_params(model), before):  # scoring leaves the adapter unchanged
            self.assertTrue(torch.equal(p, q))
            self.assertIsNone(p.grad)


if __name__ == '__main__':
    unittest.main()

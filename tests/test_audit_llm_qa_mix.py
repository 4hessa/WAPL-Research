import hashlib
import json
import unittest

import numpy as np

from experiments import llm_qa_mix as stage
from experiments import llm_web_filter as s73
from tools import audit_llm_qa_mix as audit

PLAN = {'phase_B_main': {'seeds': [37400, 37401, 37402, 37403, 37404, 37405]}}


def lines(rows):
    return ''.join(json.dumps(r) + '\n' for r in rows)


def run(key):
    return dict(kind='run', sigma=key[0], seed=key[1], arm=key[2])


class SpaceMergeTokenizer:
    """A space joins the following character into one token, as in byte-level BPE (' ج' differs from 'ج')."""
    def toks(self, text):
        out, i = [], 0
        while i < len(text):
            if text[i] == ' ' and i + 1 < len(text):
                out.append(text[i:i + 2]); i += 2
            else:
                out.append(text[i]); i += 1
        return out

    def __call__(self, text, add_special_tokens=False):
        self.vocab = getattr(self, 'vocab', {})
        return {'input_ids': [self.vocab.setdefault(s, len(self.vocab)) for s in self.toks(text)]}

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return ''.join(inv[i] for i in ids)


def mmlu(i, q, answer):
    return {'ID': i, 'Question': q, 'Context': float('nan'), 'Answer Key': 'A', 'Option 1': answer, 'Option 2': 'no'}


class TemplatedOverlapTests(unittest.TestCase):
    def test_finds_overlap_that_fields_alone_miss(self):
        tok = SpaceMergeTokenizer()
        train = [mmlu(7, 'qqqq', 'abcdefgh')]
        evals = [('target', mmlu(9, 'zz abcdefgh', 'yes'))]
        self.assertFalse(stage.content_banned(train, tok) & stage.content_banned([r for _, r in evals], tok))
        hits = audit.templated_overlap(train, evals, tok)
        self.assertEqual([(h['training_id'], h['eval_set'], h['eval_id'], h['eval_field']) for h in hits],
                         [(7, 'target', 9, 'Question')])
        self.assertEqual(hits[0]['ngram'], ' abcdefgh')

    def test_no_overlap(self):
        tok = SpaceMergeTokenizer()
        self.assertEqual(audit.templated_overlap([mmlu(7, 'qqqq', 'abcdefgh')], [('dev', mmlu(9, 'zzzz', 'yes'))], tok), [])


class ResumptionTests(unittest.TestCase):
    def setUp(self):
        order = audit.expected_order(PLAN)
        self.before = [dict(kind='environment', resumed_runs=0), dict(kind='calibration')] + [run(k) for k in order[:3]]
        self.after = [dict(kind='environment', resumed_runs=3)] + [run(k) for k in order[3:5]]
        self.snapshot = lines(self.before)
        self.final = self.snapshot + lines(self.after)
        self.log = 'noise\n1/36 runs complete\n2/36 runs complete\n3/36 runs complete\n'
        self.message = 'Stage 74 interim journal snapshot: 5 finished runs (x)'

    def test_reconciles_line_count_with_run_count(self):
        out = audit.reconcile_resumption(self.final, self.snapshot, self.message, self.log, PLAN)
        self.assertEqual(out['snapshot_commit_message_count'], 5)
        self.assertEqual(out['finished_runs_before_loss'], 3)
        self.assertEqual(out['snapshot_line_kinds'], {'environment': 1, 'calibration': 1, 'run': 3})
        self.assertEqual(out['restarted_run'], dict(zip(('sigma', 'seed', 'arm'), audit.expected_order(PLAN)[3])))
        self.assertIn('counted every journal line', out['explanation'])

    def test_rejects_altered_snapshot(self):
        altered = self.snapshot.replace('"run"', '"run" ', 1)
        with self.assertRaises(ValueError):
            audit.reconcile_resumption(self.final, altered, self.message, self.log, PLAN)

    def test_rejects_unexplained_message_count(self):
        with self.assertRaises(ValueError):
            audit.reconcile_resumption(self.final, self.snapshot, 'snapshot: 16 finished runs', self.log, PLAN)

    def test_rejects_wrong_resumed_count_or_log(self):
        bad = self.snapshot + lines([dict(self.after[0], resumed_runs=2)] + self.after[1:])
        with self.assertRaises(ValueError):
            audit.reconcile_resumption(bad, self.snapshot, self.message, self.log, PLAN)
        with self.assertRaises(ValueError):
            audit.reconcile_resumption(self.final, self.snapshot, self.message, '1/36 runs complete\n', PLAN)

    def test_rejects_skipped_run_after_resumption(self):
        order = audit.expected_order(PLAN)
        bad = self.snapshot + lines([self.after[0], run(order[4])])
        with self.assertRaises(ValueError):
            audit.reconcile_resumption(bad, self.snapshot, self.message, self.log, PLAN)


class RunCheckTests(unittest.TestCase):
    def row(self, arm, k=0, n=1412):
        sigma = stage.SIGMAS[k]
        _, _, subset, digest = stage.schedule(k, 37400)
        r = dict(sigma=sigma, seed=37400, arm=arm, schedule_sha256=digest, step_seconds=[0.1] * stage.BUDGET,
                 losses={'0': 1., str(n): 1., '2400': 1.}, accuracy={'0': .5, str(n): .5, '2400': .5},
                 adapter_digest_50='0' * 64)
        if arm == 'align_grad':
            scores = np.random.default_rng(1).normal(size=stage.POOL_SIZE)
            kept = s73.keep_top(scores, stage.KEEP)
            r.update(scores=scores.tolist(), kept=kept.tolist(), kept_sha256=hashlib.sha256(kept.tobytes()).hexdigest(),
                     qa_share=stage.qa_share(kept, sigma), scoring_seconds=100.)
        if arm == 'random512':
            r.update(subset_sha256=hashlib.sha256(subset.tobytes()).hexdigest(), qa_share=stage.qa_share(subset, sigma))
        return r

    def test_valid_rows_pass(self):
        for arm in stage.ARMS:
            for k in (0, 1):
                audit.check_run(self.row(arm, k), k, 1412)

    def test_detects_mismatches(self):
        r = self.row('align_grad')
        r['kept'] = r['kept'][:-1] + [r['kept'][-1] + 1]
        with self.assertRaises(ValueError):
            audit.check_run(r, 0, 1412)
        r = self.row('random512')
        r['qa_share'] += 0.01
        with self.assertRaises(ValueError):
            audit.check_run(r, 0, 1412)
        with self.assertRaises(ValueError):
            audit.check_run(self.row('random'), 1, 1412)  # schedule of the other sigma
        r = self.row('random')
        del r['losses']['1412']
        with self.assertRaises(ValueError):
            audit.check_run(r, 0, 1412)

    def test_session_costs(self):
        journal = [dict(kind='environment', time='t0'), dict(kind='calibration'),
                   dict(self.row('random'), kind='run'), dict(self.row('align_grad'), kind='run'),
                   dict(kind='environment', time='t1'), dict(self.row('random512'), kind='run')]
        out = audit.session_costs(journal, dict(costs=dict(random=0.1, align_grad=100.)))
        self.assertEqual([s['runs'] for s in out['sessions']], [2, 1])
        self.assertAlmostEqual(out['sessions'][0]['scoring_in_random_steps'], 1000.)
        self.assertIsNone(out['sessions'][1]['random_step_seconds_median'])


if __name__ == '__main__':
    unittest.main()

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from experiments.ud_reuse_transfer import prepare_language, summarize


class ReuseSourceTransferTests(unittest.TestCase):
    def test_roles_indices_tails_and_digest(self):
        # Every chunk encodes its split and index. Crossed split roles or
        # sampling dev & test from a shared RNG stream change the answers.
        with TemporaryDirectory() as loc:
            splits = {}
            for split, offset in [('train', 1), ('dev', 41), ('test', 121)]:
                path = Path(loc) / split
                blob = b''.join(bytes([offset + i]) * 65 for i in range(10)) + b'abc'
                path.write_bytes(blob)
                splits[split] = {'path': str(path), 'sha256': hashlib.sha256(blob).hexdigest()}
            plan = dict(probe_seed=6301, target_seed=6302, probe_chunks=3, target_chunks=6)
            train, dev, test, population = prepare_language(splits, plan, {})
            self.assertEqual(population['discarded_tail_bytes'], dict(train=3, dev=3, test=3))
            self.assertEqual(train[:, 0].tolist(), list(range(1, 11)))
            for tensor, label, key, seed, count in [(dev, 41, 'probe_ids', 6301, 3),
                                                     (test, 121, 'target_ids', 6302, 6)]:
                selected = sorted(np.random.default_rng(seed).choice(10, count, replace=False))
                self.assertEqual(population[key], selected)
                torch.testing.assert_close(tensor[:, 0], torch.tensor([i + label for i in selected]))
            Path(splits['dev']['path']).write_bytes(b'x' * 650)
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                prepare_language(splits, plan, {})

    def test_seed_weighted_primary_and_uneven_decision_changes(self):
        rows = []
        for seed, effects in [(1, [0., .3, 0., 0.]), (2, [0., -.1, -.1, -.1])]:
            for step, d in enumerate(effects):
                rows.append(dict(language='ar', seed=seed, step=step, age=step,
                                 delta=d, agreement=d == 0))
        plan = dict(languages=['ar'], seeds=[1, 2], steps=4, refresh_interval=4)
        result = summarize(rows, plan)['languages']['ar']['stale']
        self.assertEqual((result['n'], result['changed'], result['tied']), (6, 4, 2))
        self.assertAlmostEqual(result['mean_delta'], 0.)
        self.assertAlmostEqual(result['mean_abs_delta'], .1)
        self.assertAlmostEqual(result['changed_only']['mean_abs_delta'], .15)
        with self.assertRaisesRegex(ValueError, 'Incomplete decisions'):
            summarize(rows[1:], plan)

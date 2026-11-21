import unittest

import numpy as np
import torch

from experiments import ud_reuse_timed as timed


class TimedReuseTransferTests(unittest.TestCase):
    def test_rotation_places_each_arm_once_per_position(self):
        for index in range(16):
            orders = [timed.arm_order(index, repeat) for repeat in range(3)]
            for arm in timed.ARMS:
                self.assertEqual(sorted(o.index(arm) for o in orders), [0, 1, 2])
        self.assertEqual(timed.arm_order(0, 0), ('reuse4', 'fresh', 'random'))
        self.assertEqual(timed.arm_order(1, 1), ('random', 'reuse4', 'fresh'))

    def test_schedule_prefix_matches_short_stage63_draw(self):
        # Stage 63 drew a 16-by-4 matrix; stage 64 draws 20000-by-4 from the
        # same generator state. The first rows must coincide.
        plan = dict(warmup_updates=2, warmup_batch_size=4, schedule_length=50, candidates_per_step=4)
        train = torch.randint(0, 256, (30, 65))
        torch.set_num_threads(1)
        _, _, ids, choices, info = timed.build_state(train, 'ar', 7, plan)
        rng = np.random.default_rng(7)
        for _ in range(2):
            rng.integers(30, size=4)
        short = rng.integers(30, size=(16, 4))
        self.assertEqual(ids[:16].tolist(), short.tolist())
        reference = {('ar', 7): dict(initial_model_sha256=info['initial_model_sha256'],
                                     initial_optimizer_sha256=info['initial_optimizer_sha256'])}
        rows = {('ar', 7): [dict(step=i, candidate_ids=short[i].tolist()) for i in range(16)]}
        timed.check_against_stage63(info, ids, rows, reference)
        rows[('ar', 7)][3]['candidate_ids'] = [0, 0, 0, 0]
        with self.assertRaisesRegex(ValueError, 'prefix'):
            timed.check_against_stage63(info, ids, rows, reference)

    def test_repeats_averaged_within_seed_before_seed_average(self):
        plan = dict(languages=['ar', 'en'], seeds=[1, 2], timing_repeats=3)
        rows, states = [], []
        index = 0
        for lang in plan['languages']:
            for seed in plan['seeds']:
                states.append(dict(language=lang, seed=seed))
                for repeat in range(3):
                    for position, arm in enumerate(timed.arm_order(index, repeat)):
                        loss = {'reuse4': 2., 'fresh': 2.5, 'random': 1.}[arm] + seed * repeat * .1
                        rows.append(dict(language=lang, seed=seed, repeat=repeat, position=position,
                                         arm=arm, initial_loss=5., final_loss=loss, updates=10,
                                         attempted=11, actual_seconds=3., overrun_seconds=0.,
                                         accepted_refreshes=0, attempted_refreshes=0))
                index += 1
        summary = timed.summarize(rows, states, plan)
        self.assertAlmostEqual(summary['primary']['final_loss'], -.5)
        c = summary['languages']['ar']['contrasts']['reuse4_minus_random']
        self.assertEqual(c['seeds_positive'], 2)
        self.assertAlmostEqual(c['final_loss'], 1.)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            timed.summarize(rows + rows[:1], states, plan)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            timed.summarize(rows[1:], states, plan)

import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments import llm_common as llm
from experiments import llm_selection as stage


def tiny(seed):
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=40, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, tie_word_embeddings=True)
    return llm.add_lora(Qwen2ForCausalLM(config), seed, rank=2, alpha=4)


PLAN = dict(updates=6, checkpoints=[0, 2, 6], candidates_per_step=4)


class LLMSelectionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        g = torch.Generator().manual_seed(1)
        self.train = torch.randint(0, 40, (30, 9), generator=g)
        self.probe, self.target = self.train[:3], self.train[3:7]
        self.ids, self.choices = stage.schedule(5, len(self.train), PLAN)

    def run_arm(self, arm):
        with patch.object(llm, 'load_model', tiny):
            return stage.run_arm(3, self.train, self.probe, self.target, self.ids, self.choices, arm, PLAN)

    def test_only_adapters_train_and_b_starts_at_zero(self):
        model = tiny(3)
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(names and all(n.endswith(('.A', '.B')) for n in names))
        self.assertTrue(all(float(p.abs().sum()) == 0 for n, p in model.named_parameters() if n.endswith('.B')))

    def test_graddot_reused_gradient_equals_recomputed_update(self):
        row = self.run_arm('graddot4')
        model = tiny(3)
        opt = torch.optim.AdamW(llm.adapter_params(model), lr=5e-4, weight_decay=0.)
        for step, c in enumerate(row['choices']):
            opt.zero_grad()
            llm.lm_loss(model, self.train[self.ids[step]][c:c + 1]).backward()
            opt.step()
        self.assertEqual(stage.evaluate(model, self.target), row['losses']['6'])

    def test_random_and_loss_choices(self):
        row = self.run_arm('random')
        self.assertEqual(row['choices'], self.choices.tolist())
        self.assertEqual(sorted(row['losses']), ['0', '2', '6'])
        row = self.run_arm('loss4')
        model = tiny(3)
        with torch.no_grad():
            first = int(torch.argmax(llm.lm_loss(model, self.train[self.ids[0]], 'none')))
        self.assertEqual(row['choices'][0], first)

    def test_dev_lines_touching_scored_chunks_are_dropped(self):
        blob = b'\n'.join(bytes([97 + i]) * 40 for i in range(10))  # 41-byte line stride
        with patch.object(stage, 'exposed_dev_chunks', lambda n: {1}):  # bytes 65..129
            text, kept, dropped = stage.eligible_dev_text(blob)
        # lines start at 0,41,82,123,...: lines 1 (41-80), 2 (82-121) and 3 (123-162) meet chunk 1
        self.assertEqual((kept, dropped), (7, 3))
        self.assertNotIn('b', text)
        self.assertNotIn('d', text)

    def test_summary_break_even(self):
        plan = dict(PLAN, languages=['ar'], seeds=[1, 2])
        rows = []
        for seed in (1, 2):
            for arm, final, t in (('random', 3.0, .5), ('loss4', 3.05, 1.), ('graddot4', 2.9, 2.)):
                rows.append(dict(language='ar', seed=seed, arm=arm, step_seconds=[t] * 6,
                                 losses={'0': 4.0, '2': 3.5, '6': final}))
        d = stage.summarize(rows, plan)['languages']['ar']['decomposition']
        self.assertAlmostEqual(d['graddot4']['s_over_a'], .1)
        self.assertAlmostEqual(d['graddot4']['r'], 4.)
        self.assertFalse(d['graddot4']['pays_off_first_order'])
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            stage.summarize(rows[1:], plan)

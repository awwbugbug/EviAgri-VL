import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from static_qlora_model import language_attention_targets


class FakeQwen:
    def named_modules(self):
        names = [
            "",
            "visual.blocks.0.attn.q_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.gate_proj",
            "lm_head",
        ]
        return [(name, object()) for name in names]


class StaticQloraModelTest(unittest.TestCase):
    def test_language_attention_targets_exclude_visual_modules(self):
        names = language_attention_targets(FakeQwen())

        self.assertEqual(
            names,
            [
                "model.layers.0.self_attn.k_proj",
                "model.layers.0.self_attn.o_proj",
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.self_attn.v_proj",
            ],
        )
        self.assertTrue(all("visual" not in name for name in names))

    def test_language_attention_targets_reject_empty_target_set(self):
        class NoLanguageAttention:
            def named_modules(self):
                return [("visual.blocks.0.attn.q_proj", object())]

        with self.assertRaisesRegex(RuntimeError, "empty LoRA target set"):
            language_attention_targets(NoLanguageAttention())


if __name__ == "__main__":
    unittest.main()

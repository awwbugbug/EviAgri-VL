import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from static_qlora_config import load_training_config


CONFIG_PATH = ROOT / "server" / "configs" / "static_qlora_v1.json"


class StaticQloraConfigTest(unittest.TestCase):
    def test_static_config_freezes_approved_values(self):
        config = load_training_config(CONFIG_PATH)

        self.assertEqual(config["seed"], 20260714)
        self.assertEqual(config["lora"], {"r": 16, "alpha": 32, "dropout": 0.05})
        self.assertEqual(
            config["quantization"],
            {"type": "nf4", "double_quant": True, "compute_dtype": "bfloat16"},
        )
        self.assertEqual(config["vision"], {"min_pixels": 200704, "max_pixels": 401408})
        self.assertEqual(config["data"]["train_positive"], 13652)
        self.assertEqual(config["data"]["train_null"], 6826)
        self.assertEqual(config["training"]["gradient_accumulation_steps"], 16)
        self.assertEqual(config["training"]["num_train_epochs"], 1)
        self.assertEqual(config["training"]["max_length"], 1024)

    def test_rejects_non_two_to_one_training_ratio(self):
        config = {
            "seed": 20260714,
            "model_path": "/model",
            "source_data_root": "/source",
            "mixed_data_root": "/mixed",
            "experiment_root": "/experiments",
            "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
            "quantization": {
                "type": "nf4",
                "double_quant": True,
                "compute_dtype": "bfloat16",
            },
            "vision": {"min_pixels": 200704, "max_pixels": 401408},
            "data": {"train_positive": 4, "train_null": 3},
            "training": {
                "max_length": 1024,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 16,
                "num_train_epochs": 1,
                "learning_rate": 0.0002,
                "warmup_ratio": 0.03,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "logging_steps": 10,
                "eval_steps": 250,
                "save_steps": 250,
                "save_total_limit": 2,
            },
        }
        with tempfile.TemporaryDirectory() as tmp_string:
            path = Path(tmp_string) / "invalid.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "2:1"):
                load_training_config(path)

    def test_rejects_missing_required_section(self):
        with tempfile.TemporaryDirectory() as tmp_string:
            path = Path(tmp_string) / "invalid.json"
            path.write_text(json.dumps({"seed": 20260714}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model_path"):
                load_training_config(path)


if __name__ == "__main__":
    unittest.main()

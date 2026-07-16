import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from train_static_qlora import (
    deterministic_smoke_subset,
    ensure_empty_output,
    extract_logged_losses,
    resolve_mode_paths,
    training_argument_values,
)


def record(index: int, kind: str) -> dict:
    return {
        "id": f"{kind}-{index}",
        "task_type": "pest_evidence_grounding" if kind == "positive" else "prompt_conflict_null_evidence",
    }


def fixture_config(root: Path) -> dict:
    return {
        "seed": 20260714,
        "mixed_data_root": str(root / "mixed"),
        "experiment_root": str(root / "experiments"),
        "training": {
            "max_length": 1024,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "num_train_epochs": 1,
            "learning_rate": 0.0002,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "optim": "paged_adamw_8bit",
            "bf16": True,
            "tf32": True,
            "logging_steps": 10,
            "eval_steps": 250,
            "save_steps": 250,
            "save_total_limit": 2,
            "dataloader_num_workers": 2,
        },
        "smoke": {
            "positive": 24,
            "null": 8,
            "max_steps": 2,
            "gradient_accumulation_steps": 2,
        },
    }


class TrainStaticQloraTest(unittest.TestCase):
    def test_smoke_subset_has_fixed_positive_and_null_counts(self):
        rows = [record(index, "positive") for index in range(30)] + [
            record(index, "null") for index in range(12)
        ]

        first = deterministic_smoke_subset(rows, positive=24, null=8, seed=20260714)
        second = deterministic_smoke_subset(list(reversed(rows)), positive=24, null=8, seed=20260714)

        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])
        self.assertEqual(sum(row["task_type"] == "pest_evidence_grounding" for row in first), 24)
        self.assertEqual(sum(row["task_type"] == "prompt_conflict_null_evidence" for row in first), 8)

    def test_training_values_freeze_smoke_and_formal_modes(self):
        with tempfile.TemporaryDirectory() as tmp_string:
            config = fixture_config(Path(tmp_string))

            smoke = training_argument_values(config, "smoke")
            formal = training_argument_values(config, "formal")

            self.assertEqual(smoke["max_steps"], 2)
            self.assertEqual(smoke["gradient_accumulation_steps"], 2)
            self.assertEqual(smoke["eval_strategy"], "no")
            self.assertFalse(smoke["logging_nan_inf_filter"])
            self.assertEqual(formal["num_train_epochs"], 1)
            self.assertEqual(formal["gradient_accumulation_steps"], 16)
            self.assertEqual(formal["eval_strategy"], "steps")
            self.assertEqual(formal["optim"], "paged_adamw_8bit")
            self.assertEqual(formal["label_names"], ["labels"])

    def test_logged_losses_preserve_nonfinite_values_for_gate_detection(self):
        history = [{"loss": 2.5, "step": 1}, {"loss": float("nan"), "step": 2}, {"eval_loss": 1.0}]

        losses = extract_logged_losses(history)

        self.assertEqual(losses[0], 2.5)
        self.assertTrue(losses[1] != losses[1])

    def test_paths_and_output_guard_are_mode_scoped(self):
        with tempfile.TemporaryDirectory() as tmp_string:
            root = Path(tmp_string)
            config = fixture_config(root)
            paths = resolve_mode_paths(config, "smoke")

            self.assertEqual(paths.train_jsonl, root / "mixed" / "train.jsonl")
            self.assertEqual(paths.val_jsonl, root / "mixed" / "val.jsonl")
            self.assertEqual(paths.output_dir, root / "experiments" / "smoke")
            paths.output_dir.mkdir(parents=True)
            (paths.output_dir / "sentinel").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                ensure_empty_output(paths.output_dir)


if __name__ == "__main__":
    unittest.main()

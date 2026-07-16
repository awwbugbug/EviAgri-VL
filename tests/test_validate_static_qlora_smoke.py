import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from validate_static_qlora_smoke import validate_smoke


def structured_text(present: bool) -> str:
    value = {
        "evidence_present": present,
        "evidence_bbox": [1, 2, 3, 4] if present else None,
        "visible_attributes": [],
        "diagnosis": {"pest_id": 0, "pest_name": "fixture"} if present else "uncertain",
        "reliability": "supported" if present else "insufficient_visual_evidence",
    }
    return json.dumps(value, separators=(",", ":"))


def smoke_fixture(path: Path, losses: list[float], peak_vram_gb: float) -> None:
    path.mkdir(parents=True)
    (path / "environment.json").write_text(
        json.dumps(
            {
                "torch": "2.5.1+cu121",
                "transformers": "4.51.3",
                "peft": "0.15.2",
                "bitsandbytes": "0.45.5",
                "cuda_available": True,
            }
        ),
        encoding="utf-8",
    )
    (path / "run_summary.json").write_text(
        json.dumps(
            {
                "completed": True,
                "losses": losses,
                "peak_vram_reserved_bytes": int(peak_vram_gb * 1024**3),
            }
        ),
        encoding="utf-8",
    )
    (path / "trainable_parameters.json").write_text(
        json.dumps(["base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"]),
        encoding="utf-8",
    )
    (path / "lora_targets.json").write_text(
        json.dumps(["model.layers.0.self_attn.q_proj"]), encoding="utf-8"
    )
    (path / "reload_and_generation.json").write_text(
        json.dumps(
            {
                "base_loaded_in_4bit": True,
                "adapter_reload": True,
                "generations": {
                    "positive": structured_text(True),
                    "null": structured_text(False),
                },
            }
        ),
        encoding="utf-8",
    )


class ValidateStaticQloraSmokeTest(unittest.TestCase):
    def test_gate_rejects_nonfinite_loss_and_excess_vram(self):
        with tempfile.TemporaryDirectory() as tmp_string:
            output = Path(tmp_string) / "smoke"
            smoke_fixture(output, losses=[2.1, float("nan")], peak_vram_gb=30.1)

            report = validate_smoke(output)

            self.assertFalse(report["passed"])
            self.assertFalse(report["gates"]["finite_loss"])
            self.assertFalse(report["gates"]["peak_vram_below_30gb"])

    def test_gate_passes_all_six_independent_checks(self):
        with tempfile.TemporaryDirectory() as tmp_string:
            output = Path(tmp_string) / "smoke"
            smoke_fixture(output, losses=[2.1, 1.9], peak_vram_gb=12.0)

            report = validate_smoke(output)

            self.assertTrue(report["passed"])
            self.assertEqual(len(report["gates"]), 6)
            self.assertTrue(all(report["gates"].values()))


if __name__ == "__main__":
    unittest.main()

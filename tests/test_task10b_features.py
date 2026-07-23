import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from extract_task10b_features import (
    assert_frozen,
    build_feature_matrix,
    extract_features,
    mean_pool_l2,
    prepare_visual_inputs,
    write_feature_outputs,
)


def test_mean_pool_l2_matches_exact_float32_result():
    tokens = torch.tensor([[3.0, 0.0], [0.0, 4.0]], dtype=torch.float16)

    feature = mean_pool_l2(tokens)

    expected = torch.tensor([1.5, 2.0], dtype=torch.float32)
    expected = expected / torch.linalg.vector_norm(expected)
    assert feature.dtype == torch.float32
    assert torch.allclose(feature, expected)
    assert torch.linalg.vector_norm(feature).item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "tokens, message",
    [
        (torch.zeros(2, 3), "zero-norm"),
        (torch.tensor([[float("nan"), 1.0]]), "non-finite"),
        (torch.ones(3), "two-dimensional"),
        (torch.ones(0, 3), "non-empty"),
    ],
)
def test_mean_pool_l2_rejects_invalid_tokens(tokens, message):
    with pytest.raises(ValueError, match=message):
        mean_pool_l2(tokens)


def test_assert_frozen_rejects_any_trainable_parameter():
    frozen = torch.nn.Linear(2, 2)
    frozen.requires_grad_(False)
    assert_frozen(frozen)

    frozen.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="trainable parameters"):
        assert_frozen(frozen)


def test_build_feature_matrix_preserves_row_identity_and_index():
    rows = [
        {"id": "a", "split": "train", "class_id": 1},
        {"id": "b", "split": "dev", "class_id": 2},
    ]
    tokens = {
        "a": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        "b": torch.tensor([[0.0, 2.0], [0.0, 2.0]]),
    }

    matrix, feature_rows = build_feature_matrix(rows, lambda row: tokens[row["id"]])

    assert matrix.dtype == np.float32
    assert matrix.shape == (2, 2)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)
    assert [row["id"] for row in feature_rows] == ["a", "b"]
    assert [row["feature_index"] for row in feature_rows] == [0, 1]


def test_build_feature_matrix_rejects_inconsistent_dimensions():
    rows = [{"id": "a"}, {"id": "b"}]
    tokens = {
        "a": torch.ones(2, 3),
        "b": torch.ones(2, 4),
    }

    with pytest.raises(ValueError, match="feature dimensions"):
        build_feature_matrix(rows, lambda row: tokens[row["id"]])


def test_write_feature_outputs_signs_files_and_refuses_overwrite(tmp_path):
    matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    rows = [
        {"id": "a", "feature_index": 0},
        {"id": "b", "feature_index": 1},
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "features"

    write_feature_outputs(
        matrix=matrix,
        feature_rows=rows,
        output_root=output,
        manifest_path=manifest,
        config={"min_pixels": 200704, "max_pixels": 401408},
        model_identity={"config.json": "a" * 64},
    )

    summary = json.loads((output / "run_summary.json").read_text())
    assert summary["state"] == "completed"
    assert summary["feature_count"] == 2
    assert summary["feature_dimension"] == 2
    assert summary["version"] == "task10b-v2-feature-summary-1"
    assert (output / "completion.sha256").is_file()
    with pytest.raises(FileExistsError):
        write_feature_outputs(
            matrix=matrix,
            feature_rows=rows,
            output_root=output,
            manifest_path=manifest,
            config={},
            model_identity={},
        )


def test_extract_features_records_runtime_loader_failure(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"id": "x", "image": "/missing.jpg"}) + "\n")
    output = tmp_path / "failed"

    def broken_runtime_loader():
        raise ModuleNotFoundError("transformers unavailable")

    with pytest.raises(ModuleNotFoundError, match="transformers unavailable"):
        extract_features(
            manifest_path=manifest,
            model_path=tmp_path / "model",
            output_root=output,
            limit=1,
            runtime_loader=broken_runtime_loader,
        )

    assert json.loads((output / "status.json").read_text(encoding="utf-8"))["state"] == "failed"
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["state"] == "failed"
    assert "transformers unavailable" in failure["error"]


def test_prepare_visual_inputs_uses_only_the_processor_image_token():
    class FakeProcessor:
        image_token = "<image-token>"

        def __init__(self):
            self.call = None

        def __call__(self, **kwargs):
            self.call = kwargs
            return {"pixel_values": torch.ones(2, 3), "image_grid_thw": torch.ones(1, 3)}

    processor = FakeProcessor()
    image = object()

    result = prepare_visual_inputs(processor, image)

    assert result["pixel_values"].shape == (2, 3)
    assert processor.call == {
        "text": ["<image-token>"],
        "images": [image],
        "return_tensors": "pt",
    }

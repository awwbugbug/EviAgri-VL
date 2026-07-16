import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from run_task9d_inference import (
    ensure_prediction_output_new,
    generation_contract,
    inference_messages,
    verify_expected_ids,
)


def test_base_and_adapter_share_one_generation_contract():
    base = generation_contract()
    adapter = generation_contract()
    assert base == adapter == {
        "do_sample": False, "max_new_tokens": 512,
        "min_pixels": 200704, "max_pixels": 401408,
        "parser_version": "task9d-json-parser-v1",
    }


def test_inference_uses_only_message_pixels_and_prompt_not_private_metadata():
    row = {
        "id": "opaque", "source_path": "/class_name/image.jpg", "query_class_id": 9,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {"role": "user", "content": [
                {"type": "image", "image": "/opaque/abc.png"},
                {"type": "text", "text": "neutral prompt"},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "{}"}]},
        ],
    }
    messages = inference_messages(row)
    assert [message["role"] for message in messages] == ["system", "user"]
    serialized = repr(messages)
    assert "class_name" not in serialized and "query_class_id" not in serialized
    assert "/opaque/abc.png" in serialized


def test_expected_ids_and_output_are_fail_closed(tmp_path):
    verify_expected_ids([{"id": "a"}, {"id": "b"}], [{"id": "b"}, {"id": "a"}])
    with pytest.raises(ValueError, match="ID mismatch"):
        verify_expected_ids([{"id": "a"}], [{"id": "b"}])
    output = tmp_path / "group"
    output.mkdir()
    (output / "sentinel").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing"):
        ensure_prediction_output_new(output)

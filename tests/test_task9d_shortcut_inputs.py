import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from task9d_shortcut_inputs import probe_views


def test_probe_views_are_aligned_and_do_not_expose_private_labels_or_paths():
    envelope = {
        "id": "opaque", "family_id": "family-private", "role": "semantic_negative",
        "model": {"messages": [
            {"role": "system", "content": [{"type": "text", "text": "system text"}]},
            {"role": "user", "content": [
                {"type": "image", "image": "images/class_name/secret.jpg"},
                {"type": "text", "text": "neutral query"},
            ]},
        ]},
    }
    views = probe_views(envelope, split="train", index=2)
    signatures = {(row["id"], row["family_id"], row["split"], row["label"])
                  for row in views.values()}
    assert len(signatures) == 1
    assert next(iter(signatures))[-1] == 0
    metadata_text = views["prompt_nonimage_metadata"]["text"]
    assert "semantic_negative" not in metadata_text
    assert "class_name" not in metadata_text
    assert "secret.jpg" not in metadata_text
    assert "family-private" not in metadata_text
    assert "image_content=present" in metadata_text

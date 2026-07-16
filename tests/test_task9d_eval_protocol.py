import json
import sys
from pathlib import Path

from PIL import Image, ImageChops


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from task9d_eval_protocol import (
    BLANK_RGB,
    BLUR_RADIUS_FRACTION,
    SHUFFLE_GRID,
    UNSEEN_TEMPLATE_IDS,
    apply_heldout_transform,
    build_eval_manifest_from_records,
    build_paired_conditions,
)
from task9b_protocol import build_prompt, build_target


def _gradient(size=(70, 56)):
    image = Image.new("RGB", size)
    image.putdata([(x * 3 % 256, y * 5 % 256, (x + y) * 7 % 256)
                   for y in range(size[1]) for x in range(size[0])])
    return image


def test_heldout_transform_contract_is_distinct_and_dimension_preserving():
    source = _gradient()
    blank = apply_heldout_transform(source, "blank", key="family-1")
    blur = apply_heldout_transform(source, "blur", key="family-1")
    shuffle = apply_heldout_transform(source, "shuffle", key="family-1")
    assert BLANK_RGB == (91, 107, 123)
    assert BLANK_RGB != (127, 127, 127)
    assert BLUR_RADIUS_FRACTION == 0.16
    assert BLUR_RADIUS_FRACTION not in {0.05, 0.08, 0.12}
    assert SHUFFLE_GRID == 7
    assert all(image.size == source.size for image in (blank, blur, shuffle))
    assert blank.getpixel((0, 0)) == BLANK_RGB
    assert ImageChops.difference(source, blur).getbbox() is not None
    assert ImageChops.difference(source, shuffle).getbbox() is not None
    assert apply_heldout_transform(source, "shuffle", key="family-1").tobytes() == shuffle.tobytes()


def test_every_challenge_family_has_complete_paired_conditions(tmp_path):
    source = _gradient()
    image_path = tmp_path / "source.jpg"
    source.save(image_path)
    families = [
        {"family_id": "f1", "image_path": str(image_path), "query_class_id": 1},
        {"family_id": "f2", "image_path": str(image_path), "query_class_id": 2},
    ]
    rows = build_paired_conditions(families, tmp_path / "out")
    assert len(rows) == 8
    for family_id in ("f1", "f2"):
        conditions = {row["condition"] for row in rows if row["family_id"] == family_id}
        assert conditions == {"original", "blank", "blur", "shuffle"}
    assert len({row["pair_id"] for row in rows}) == 2
    assert all(Path(row["image_path"]).is_file() for row in rows)
    assert all(Path(row["image_path"]).name.split(".")[0] == row["image_sha256"] for row in rows)


def test_eval_prompts_are_heldout_and_do_not_name_task8_transforms():
    assert len(UNSEEN_TEMPLATE_IDS) == 2
    joined = json.dumps(UNSEEN_TEMPLATE_IDS).lower()
    assert "task8" not in joined
    assert not any(token in joined for token in ("blank", "blur", "shuffle"))


def test_complete_eval_manifest_has_shared_core_prompt_and_paired_views(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "opaque.png"
    _gradient().save(image_path)
    models, provenance = {}, []
    for role, query_id, present in (("positive", 1, True), ("semantic_negative", 2, False),
                                    ("visual_counterfactual", 1, False)):
        identifier = f"f1-{role}"
        target = build_target(present, [0, 0, 10, 10] if present else None,
                              1 if present else None, "pest 1" if present else None)
        models[identifier] = {"id": identifier, "messages": [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {"role": "user", "content": [{"type": "image", "image": "images/opaque.png"},
                                           {"type": "text", "text": build_prompt("dev_neutral_0", f"pest {query_id}")}]},
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps(target)}]},
        ]}
        provenance.append({"id": identifier, "family_id": "f1", "role": role,
                           "query_class_id": query_id, "split": "dev"})
    rows = build_eval_manifest_from_records(models, provenance, ["f1"], tmp_path, tmp_path / "eval_images")
    assert len(rows) == 11  # 3 core + 5 alternate prompt views + 3 transformed pairs
    assert len({row["id"] for row in rows}) == len(rows)
    assert {row["condition"] for row in rows if row["family_id"] == "f1"} >= {
        "original", "semantic_null", "source_visual_null", "blank", "blur", "shuffle"
    }
    positive_views = {row["prompt_view"] for row in rows if row["role"] == "positive"}
    assert positive_views == {"canonical", "native_0", "native_1", "native_2",
                              "unseen_alpha", "unseen_beta"}
    assert all(len(row["messages"]) == 2 for row in rows)

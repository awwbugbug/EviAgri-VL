import sys
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from task9d_prepare import build_variant_schedule, prepare_task9d, select_families
from task9b_protocol import build_prompt


def _fixture(per_split=12):
    provenance, models = [], {}
    for split in ("train", "val", "dev"):
        for index in range(per_split):
            family = f"{split}-fam-{index:03d}"
            class_id = index % 4
            template = ("dev_neutral_0", "dev_neutral_1")[index % 2] if split == "dev" else f"train_neutral_{index % 3}"
            for role in ("positive", "semantic_negative", "visual_counterfactual"):
                identifier = f"{family}-{role}"
                query_id = class_id if role != "semantic_negative" else (class_id + 1) % 4
                models[identifier] = {"id": identifier, "messages": [
                    {"role": "system", "content": [{"type": "text", "text": "system"}]},
                    {"role": "user", "content": [{"type": "image", "image": "images/" + "a" * 64 + ".jpg"}, {"type": "text", "text": build_prompt(template, f"pest {query_id}")}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "{}"}]},
                ]}
                provenance.append({
                    "id": identifier, "family_id": family, "role": role,
                    "split": split, "template_id": template,
                    "query_class_id": query_id,
                    "present_class_ids": [class_id], "source_image_id": f"img-{split}-{index}",
                    "source_image_sha256": f"{index + 1:064x}",
                    "near_duplicate_component_id": f"component-{split}-{index}",
                })
    return models, provenance


def test_select_families_is_deterministic_class_stratified_and_split_local():
    _, provenance = _fixture()
    first = select_families(provenance, "train", 8, seed=20260716)
    second = select_families(list(reversed(provenance)), "train", 8, seed=20260716)
    assert first == second
    assert len(first) == len(set(first)) == 8
    positive = {row["family_id"]: row for row in provenance if row["role"] == "positive"}
    assert Counter(positive[f]["query_class_id"] for f in first) == {0: 2, 1: 2, 2: 2, 3: 2}
    assert all(positive[f]["split"] == "train" for f in first)


def test_abc_only_change_roles_and_templates_and_share_family_pool():
    models, provenance = _fixture()
    families = select_families(provenance, "train", 8, seed=20260716)
    schedules = {
        variant: build_variant_schedule(models, provenance, families, variant, seed=17, target_rows=24)
        for variant in "ABC"
    }
    assert all(set(value["family_ids"]) == set(families) for value in schedules.values())
    assert Counter(row["role"] for row in schedules["A"]["schedule"]) == {"positive": 12, "semantic_negative": 12}
    assert Counter(row["role"] for row in schedules["B"]["schedule"]) == {
        "positive": 8, "semantic_negative": 8, "visual_counterfactual": 8
    }
    assert Counter(row["role"] for row in schedules["C"]["schedule"]) == {
        "positive": 8, "semantic_negative": 8, "visual_counterfactual": 8
    }
    assert set(schedules["A"]["template_ids"]) == {"train_neutral_0"}
    assert set(schedules["B"]["template_ids"]) == {"train_neutral_0"}
    assert len(set(schedules["C"]["template_ids"])) == 3


def test_c_template_distribution_is_role_symmetric():
    models, provenance = _fixture()
    families = select_families(provenance, "train", 12, seed=1)
    result = build_variant_schedule(models, provenance, families, "C", seed=17, target_rows=36)
    by_role = defaultdict(Counter)
    for row in result["schedule"]:
        by_role[row["role"]][row["template_id"]] += 1
    assert len({tuple(sorted(value.items())) for value in by_role.values()}) == 1


def test_locked_overlap_and_wrong_target_size_fail_closed():
    models, provenance = _fixture()
    families = select_families(provenance, "train", 8, seed=1)
    selected_image_id = next(
        row["source_image_id"]
        for row in provenance
        if row["family_id"] == families[0] and row["role"] == "positive"
    )
    with pytest.raises(ValueError, match="locked"):
        build_variant_schedule(models, provenance, families, "B", seed=17, target_rows=24,
                               locked_image_ids={selected_image_id})
    with pytest.raises(ValueError, match="target_rows"):
        build_variant_schedule(models, provenance, families, "B", seed=17, target_rows=23)


def test_prepare_task9d_writes_frozen_equal_family_outputs(tmp_path):
    models, provenance = _fixture()
    source = tmp_path / "source"
    (source / "model").mkdir(parents=True)
    (source / "dev_audit").mkdir()
    (source / "private").mkdir()
    (source / "images").mkdir()
    image_name = "a" * 64 + ".jpg"
    (source / "images" / image_name).write_bytes(b"image")
    by_id = {row["id"]: row for row in provenance}
    split_paths = {"train": source / "model/train.jsonl", "val": source / "model/val.jsonl",
                   "dev": source / "dev_audit/model.jsonl"}
    for split, path in split_paths.items():
        rows = [model for identifier, model in models.items() if by_id[identifier]["split"] == split]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (source / "private/provenance.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in provenance), encoding="utf-8"
    )
    (source / "private/locked_exclusion.json").write_text(
        json.dumps({"image_ids": [], "image_sha256": []}), encoding="utf-8"
    )
    (source / "freeze_report.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    output = tmp_path / "prepared"
    report = prepare_task9d(source, output, train_families=8, val_families=8,
                            dev_families=8, challenge_families=4, train_rows=24)
    assert report["passed"] is True
    for variant in "ABC":
        rows = [json.loads(line) for line in
                (output / f"variants/{variant}/train_schedule.jsonl").read_text().splitlines()]
        assert len(rows) == 24
        assert len({row["family_id"] for row in rows}) == 8
        assert (output / f"variants/{variant}/val.jsonl").is_file()
    assert (output / "images" / image_name).is_file()
    assert (output / "private/selections.json").is_file()
    assert (output / "completion.sha256").is_file()
    with pytest.raises(FileExistsError, match="refusing"):
        prepare_task9d(source, output, train_families=8, val_families=8,
                       dev_families=8, challenge_families=4, train_rows=24)

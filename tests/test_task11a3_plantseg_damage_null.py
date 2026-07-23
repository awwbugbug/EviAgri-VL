import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from prepare_task11a3_plantseg_damage_null import (
    HTTPRangeReader,
    PLANTS,
    deterministic_selection,
    parse_resolution,
)


def _row(plant, index, *, split="Validation", ratio="0.1", resolution="640x480"):
    return {
        "Name": f"{plant}_{index}.jpg",
        "Plant": plant,
        "Disease": f"{plant} disease",
        "Resolution": resolution,
        "Label file": f"{plant}_{index}.png",
        "Mask ratio": ratio,
        "URL": "https://example.invalid",
        "License": "CC-BY-NC",
        "Split": split,
    }


def test_resolution_is_strict():
    assert parse_resolution("640x480") == (640, 480)
    with pytest.raises(ValueError):
        parse_resolution("640 X 480")


def test_selection_is_deterministic_and_balanced():
    rows = [_row(plant, index) for plant in PLANTS for index in range(6)]
    first = deterministic_selection(rows, images_per_plant=3)
    second = deterministic_selection(rows, images_per_plant=3)
    assert first == second
    assert len(first) == 24
    assert {plant: sum(row["Plant"] == plant for row in first) for plant in PLANTS} == {
        plant: 3 for plant in PLANTS
    }


def test_selection_rejects_unfrozen_cardinality_and_ineligible_rows():
    rows = [_row(plant, index) for plant in PLANTS for index in range(3)]
    with pytest.raises(ValueError):
        deterministic_selection(rows, images_per_plant=2)
    rows = [row for row in rows if row["Plant"] != "Rice"] + [
        _row("Rice", 0, ratio="0.5")
    ]
    with pytest.raises(ValueError):
        deterministic_selection(rows, images_per_plant=1)


def test_http_range_reader_seek_and_block_cache_without_network():
    payload = bytes(range(100))
    calls = []

    def fetch(start, end):
        calls.append((start, end))
        return payload[start:end]

    reader = HTTPRangeReader("unused", len(payload), block_size=16, fetcher=fetch)
    assert reader.read(5) == payload[:5]
    assert reader.seek(-4, 2) == 96
    assert reader.read() == payload[96:]
    reader.seek(2)
    assert reader.read(2) == payload[2:4]
    assert calls == [(0, 16), (96, 100), (0, 16)]

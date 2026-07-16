import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageChops


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "server"))

from task9b_transforms import (
    DEV_TRANSFORMS,
    TRAIN_TRANSFORMS,
    apply_transform,
    transform_kind_for_index,
)


def gradient_image(size=(60, 48)) -> Image.Image:
    image = Image.new("RGB", size)
    for x in range(size[0]):
        for y in range(size[1]):
            image.putpixel((x, y), ((x * 3) % 256, (y * 5) % 256, (x + y) % 256))
    return image


def test_train_dev_registries_and_parameter_domains_are_disjoint_from_task8():
    assert set(TRAIN_TRANSFORMS).isdisjoint(DEV_TRANSFORMS)
    assert TRAIN_TRANSFORMS["train_blur"]["radius_fraction"] == [0.08, 0.11]
    assert DEV_TRANSFORMS["dev_blur"]["radius_fraction"] == [0.14, 0.18]
    assert 0.05 not in TRAIN_TRANSFORMS["train_blur"]["radius_fraction"]
    assert 0.05 not in DEV_TRANSFORMS["dev_blur"]["radius_fraction"]
    assert TRAIN_TRANSFORMS["train_patch_permute"]["grids"] == [3, 4]
    assert DEV_TRANSFORMS["dev_patch_permute"]["grids"] == [5, 6]
    assert TRAIN_TRANSFORMS["train_low_information"]["mean_range"] != [127, 127]
    assert DEV_TRANSFORMS["dev_low_information"]["mean_range"] != [127, 127]


def test_all_transforms_are_deterministic_nonidentity_and_preserve_dimensions():
    image = gradient_image()
    bbox = [20, 14, 40, 34]
    for surface, registry in (("train", TRAIN_TRANSFORMS), ("dev", DEV_TRANSFORMS)):
        for kind in registry:
            first = apply_transform(surface, kind, image, bbox, seed=17)
            second = apply_transform(surface, kind, image, bbox, seed=17)
            assert first.size == second.size == image.size
            assert ImageChops.difference(first, second).getbbox() is None
            assert ImageChops.difference(first, image).getbbox() is not None


def test_low_information_transforms_are_not_uniform_task8_gray():
    image = gradient_image()
    task8_blank = Image.new("RGB", image.size, (127, 127, 127))
    for surface, kind in (("train", "train_low_information"), ("dev", "dev_low_information")):
        transformed = apply_transform(surface, kind, image, [20, 14, 40, 34], seed=19)
        assert ImageChops.difference(transformed, task8_blank).getbbox() is not None
        assert len(set(transformed.getdata())) > 1


def test_transform_rotation_is_exactly_balanced_over_complete_cycles():
    for surface, registry in (("train", TRAIN_TRANSFORMS), ("dev", DEV_TRANSFORMS)):
        kinds = [transform_kind_for_index(index, surface) for index in range(len(registry) * 7)]
        assert set(kinds) == set(registry)
        assert set(Counter(kinds).values()) == {7}

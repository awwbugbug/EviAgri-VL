from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _index(paths: list[Path]) -> dict[tuple[str, str], Path]:
    result = {}
    for split, path in zip(("train", "val", "test"), paths):
        for row in _load_jsonl(path):
            image = Path(row["image"])
            result[(split, image.stem)] = image
    return result


def make_sheet(split_paths: list[Path], review_path: Path, output: Path, count: int = 24) -> None:
    index = _index(split_paths)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    rows = review["high_confidence_candidates"]
    ranked = sorted(rows, key=lambda row: (-row["structural_correlation"], row["phash_distance"]))
    rng = random.Random(20260715)
    selected = ranked[: count // 2]
    remaining = [row for row in rows if row not in selected]
    selected.extend(rng.sample(remaining, min(count - len(selected), len(remaining))))
    pair_width, pair_height, columns = 400, 190, 3
    canvas = Image.new(
        "RGB",
        (pair_width * columns, pair_height * ((len(selected) + columns - 1) // columns)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    manifest = []
    for position, row in enumerate(selected):
        x = (position % columns) * pair_width
        y = (position // columns) * pair_height
        left = index[(row["left_split"], row["left_id"])]
        right = index[(row["right_split"], row["right_id"])]
        for offset, path in ((0, left), (200, right)):
            with Image.open(path) as image:
                thumb = ImageOps.pad(image.convert("RGB"), (196, 150), color=(230, 230, 230))
            canvas.paste(thumb, (x + offset, y))
        label = (
            f"{position}: pH={row['phash_distance']} corr={row['structural_correlation']:.3f}\n"
            f"{row['left_split']}:{row['left_id']} | {row['right_split']}:{row['right_id']}"
        )
        draw.multiline_text((x + 2, y + 153), label, fill="black", font=font, spacing=2)
        manifest.append({"position": position, "left_path": str(left), "right_path": str(right), **row})
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    output.with_suffix(".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--val-jsonl", required=True, type=Path)
    parser.add_argument("--test-jsonl", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", default=24, type=int)
    args = parser.parse_args()
    make_sheet(
        [args.train_jsonl, args.val_jsonl, args.test_jsonl],
        args.review,
        args.output,
        args.count,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
from pathlib import Path

MANIFEST = Path(os.environ["MODEL_MANIFEST"])
REQUIRED = {
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
}
MINIMUM_BYTES = 7_000_000_000


def main() -> None:
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))
    model_path = Path(record["resolved_path"])
    missing = sorted(name for name in REQUIRED if not (model_path / name).is_file())
    shards = sorted(model_path.glob("model-*.safetensors"))
    total_bytes = sum(path.stat().st_size for path in model_path.rglob("*") if path.is_file())
    if missing:
        raise SystemExit(f"missing files: {missing}")
    if len(shards) != 2:
        raise SystemExit(f"expected 2 shards, found {len(shards)}")
    if total_bytes < MINIMUM_BYTES:
        raise SystemExit(f"snapshot too small: {total_bytes}")
    print(
        json.dumps(
            {
                "status": "MODEL_OK",
                "model_path": str(model_path),
                "total_bytes": total_bytes,
                "shard_count": len(shards),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

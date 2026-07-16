from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from modelscope import snapshot_download

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
REVISION = "master"
MODEL_ROOT = Path(os.environ["MODEL_ROOT"])
MODEL_MANIFEST = Path(os.environ["MODEL_MANIFEST"])


def main() -> None:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    resolved = Path(
        snapshot_download(
            MODEL_ID,
            revision=REVISION,
            cache_dir=str(MODEL_ROOT),
        )
    ).resolve()
    record = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "resolved_path": str(resolved),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    MODEL_MANIFEST.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()

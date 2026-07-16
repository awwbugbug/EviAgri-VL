from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_MANIFEST = Path(os.environ["MODEL_MANIFEST"])
OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"])
IMAGE_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
IMAGE_PATH = OUTPUT_DIR / "demo.jpeg"
RESULT_PATH = OUTPUT_DIR / "smoke_results.json"


def download_demo_image() -> None:
    if IMAGE_PATH.is_file() and IMAGE_PATH.stat().st_size > 10_000:
        return
    response = requests.get(IMAGE_URL, timeout=30)
    response.raise_for_status()
    IMAGE_PATH.write_bytes(response.content)


def extract_json(text: str) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def run_prompt(model: Any, processor: Any, prompt: str) -> tuple[str, float]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{IMAGE_PATH}"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
    output = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output, elapsed


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    download_demo_image()
    record = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    model_path = record["resolved_path"]
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=256 * 28 * 28,
        max_pixels=1024 * 28 * 28,
        use_fast=False,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.generation_config.temperature = None
    torch.cuda.reset_peak_memory_stats()
    description, description_seconds = run_prompt(
        model, processor, "Describe this image concisely."
    )
    structured_prompt = """Return JSON only with this schema:
{"evidence_present": true, "evidence_bbox": [x1, y1, x2, y2], "visible_attributes": [], "diagnosis": "", "reliability": "supported|insufficient_evidence"}.
Coordinates must use the image coordinate system. If evidence is insufficient, set evidence_present=false, evidence_bbox=null, diagnosis="uncertain", reliability="insufficient_evidence"."""
    structured, structured_seconds = run_prompt(model, processor, structured_prompt)
    result = {
        "model_path": model_path,
        "image_path": str(IMAGE_PATH),
        "description": description,
        "description_seconds": description_seconds,
        "structured_raw": structured,
        "structured_parsed": extract_json(structured),
        "structured_seconds": structured_seconds,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
    RESULT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

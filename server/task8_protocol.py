from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


GROUPS = ("B0", "B1", "B2", "B3")
EXPECTED_SCHEMA_KEYS = (
    "evidence_present",
    "evidence_bbox",
    "visible_attributes",
    "diagnosis",
    "reliability",
)

SCHEMA_INSTRUCTION = (
    "Return one JSON object in this exact order: evidence_present, evidence_bbox, "
    "visible_attributes, diagnosis, reliability. evidence_present must be boolean; "
    "evidence_bbox must be [x1,y1,x2,y2] in image pixels or null; visible_attributes "
    "must be an array of strings; diagnosis must be either "
    '{"pest_id":<integer>,"pest_name":"<string>"} or "uncertain"; reliability must '
    "be a string. "
    "If visible evidence is insufficient, set evidence_present=false, "
    'evidence_bbox=null, visible_attributes=[], diagnosis="uncertain", and '
    'reliability="insufficient_visual_evidence".'
)


@dataclass(frozen=True)
class ProtocolSpec:
    prompt_policy: str
    min_pixels: int = 200704
    max_pixels: int = 401408
    quantization: str = "nf4-double-quant"
    compute_dtype: str = "bfloat16"
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: None = None
    schema_keys: tuple[str, ...] = EXPECTED_SCHEMA_KEYS
    parser_id: str = "strict-evidence-json-v2"
    evaluator_id: str = "task8-condition-aware-v2"


_GROUP_PROMPT_POLICIES = {
    "B0": "ordinary-diagnosis-v3",
    "B1": "neutral-evidence-first-v3",
    "B2": "neutral-evidence-first-v3",
    "B3": "static-qlora-training-template-v1",
}


def generation_kwargs() -> dict[str, object]:
    return {"max_new_tokens": 128, "do_sample": False, "temperature": None}


def protocol_spec(group: str) -> ProtocolSpec:
    if group not in GROUPS:
        raise ValueError(f"unknown Task 8 group: {group}")
    return ProtocolSpec(prompt_policy=_GROUP_PROMPT_POLICIES[group])


def protocol_hash(group: str) -> str:
    payload = json.dumps(
        asdict(protocol_spec(group)), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_prompt(
    group: str,
    queried_pest_name: str,
    condition: str,
    queried_pest_id: int | None = None,
) -> str:
    if group not in GROUPS:
        raise ValueError(f"unknown Task 8 group: {group}")
    if not queried_pest_name or not queried_pest_name.strip():
        raise ValueError("queried_pest_name must be non-empty")
    if group == "B0":
        return (
            "Diagnose the pest visible in this image. Base the diagnosis only on image pixels. "
            + SCHEMA_INSTRUCTION
        )
    if group in {"B1", "B2"}:
        identity = queried_pest_name.strip()
        conditional_identity = ""
        if queried_pest_id is not None:
            identity += f" (pest_id={int(queried_pest_id)})"
            conditional_identity = (
                " The queried identity object is diagnosis="
                f'{{"pest_id":{int(queried_pest_id)},"pest_name":"{queried_pest_name.strip()}"}}; '
                "use this object if and only if the queried pest is supported by pixels."
            )
        return (
            f"Is {identity} visibly present in this image? "
            "Base the decision only on image pixels. "
            + conditional_identity
            + " "
            + SCHEMA_INSTRUCTION
        )
    if condition == "original_correct":
        return (
            "Identify the pest supported by visible evidence. Return one JSON object in this "
            "exact order: evidence_present, evidence_bbox, visible_attributes, diagnosis, "
            "reliability. Use image-pixel coordinates and do not infer attributes that are not "
            "annotated."
        )
    return (
        f"Is {queried_pest_name.strip()} visibly present in this image? Return one JSON object "
        "in this exact order: evidence_present, evidence_bbox, visible_attributes, diagnosis, "
        "reliability. If the queried pest is not supported, use null evidence and an uncertain "
        "diagnosis."
    )

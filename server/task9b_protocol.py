from __future__ import annotations

import hashlib
import json
from typing import Any


OUTPUT_KEYS = (
    "evidence_present",
    "evidence_region",
    "visible_attributes",
    "diagnosis",
    "reliability",
)
DIAGNOSIS_KEYS = ("status", "pest_id", "pest_name", "species", "stage")
TRAIN_TEMPLATE_IDS = ("train_neutral_0", "train_neutral_1", "train_neutral_2")
DEV_TEMPLATE_IDS = ("dev_neutral_0", "dev_neutral_1")
LENGTH_BUCKETS = (320, 352, 384)
SYSTEM_PROMPT = (
    "You are a visual evidence assessor. Follow the requested JSON contract and base every "
    "decision only on image pixels and the queried condition."
)


_TEMPLATE_PREFIXES = {
    "train_neutral_0": "Determine whether the queried pest '{name}' is supported by visible image evidence.",
    "train_neutral_1": "Check the image pixels for visual support of the queried pest '{name}'.",
    "train_neutral_2": "Assess whether image evidence supports the queried pest '{name}'.",
    "dev_neutral_0": "Using only the pixels, decide whether the image supports the queried pest '{name}'.",
    "dev_neutral_1": "Evaluate the visual evidence for the queried pest '{name}' without relying on the query wording.",
}
_COMMON_INSTRUCTION = (
    " Use pixels only. Return one JSON object with keys in this exact order: "
    "evidence_present, evidence_region, visible_attributes, diagnosis, reliability. "
    "When evidence is insufficient, use evidence_present=false, evidence_region=null, "
    "and diagnosis.status set to uncertain or abstain without a pest, species, or stage."
)


def build_prompt(template_id: str, queried_name: str) -> str:
    if template_id not in _TEMPLATE_PREFIXES:
        raise ValueError(f"unknown template_id: {template_id}")
    if not isinstance(queried_name, str) or not queried_name.strip():
        raise ValueError("queried_name must be non-empty")
    name = " ".join(queried_name.split())
    return _TEMPLATE_PREFIXES[template_id].format(name=name) + _COMMON_INSTRUCTION


def build_target(
    evidence_present: bool,
    evidence_region: list[int | float] | None,
    pest_id: int | None,
    pest_name: str | None,
    abstention_status: str = "abstain",
) -> dict[str, Any]:
    if evidence_present:
        diagnosis = {
            "status": "supported",
            "pest_id": pest_id,
            "pest_name": pest_name,
            "species": None,
            "stage": None,
        }
        reliability = "supported"
    else:
        diagnosis = {
            "status": abstention_status,
            "pest_id": None,
            "pest_name": None,
            "species": None,
            "stage": None,
        }
        reliability = "insufficient_visual_evidence"
    target = {
        "evidence_present": evidence_present,
        "evidence_region": evidence_region,
        "visible_attributes": [],
        "diagnosis": diagnosis,
        "reliability": reliability,
    }
    validate_target_semantics(target)
    return target


def validate_target_semantics(target: dict[str, Any]) -> None:
    if not isinstance(target, dict) or tuple(target) != OUTPUT_KEYS:
        raise ValueError("invalid target key order")
    if not isinstance(target["evidence_present"], bool):
        raise ValueError("evidence_present must be boolean")
    if not isinstance(target["visible_attributes"], list):
        raise ValueError("visible_attributes must be a list")
    diagnosis = target.get("diagnosis")
    if not isinstance(diagnosis, dict) or tuple(diagnosis) != DIAGNOSIS_KEYS:
        raise ValueError("invalid diagnosis key order")
    if target["evidence_present"]:
        region = target.get("evidence_region")
        if (
            not isinstance(region, list)
            or len(region) != 4
            or not all(isinstance(value, (int, float)) for value in region)
        ):
            raise ValueError("positive evidence_region must contain four coordinates")
        if (
            diagnosis.get("status") != "supported"
            or not isinstance(diagnosis.get("pest_id"), int)
            or not isinstance(diagnosis.get("pest_name"), str)
            or not diagnosis["pest_name"].strip()
            or diagnosis.get("species") is not None
            or diagnosis.get("stage") is not None
        ):
            raise ValueError("invalid supported diagnosis")
        if target.get("reliability") != "supported":
            raise ValueError("invalid supported reliability")
        return
    if target.get("evidence_region") is not None:
        raise ValueError("null evidence_region must be null")
    if (
        diagnosis.get("status") not in {"uncertain", "abstain"}
        or any(diagnosis.get(key) is not None for key in ("pest_id", "pest_name", "species", "stage"))
    ):
        raise ValueError("null diagnosis must abstain without pest, species, or stage")
    if target.get("reliability") != "insufficient_visual_evidence":
        raise ValueError("invalid null reliability")


def length_bucket_for_family(family_id: str) -> int:
    digest = hashlib.sha256(f"task9b-length:{family_id}".encode("utf-8")).digest()
    return LENGTH_BUCKETS[int.from_bytes(digest[:8], "big") % len(LENGTH_BUCKETS)]


def serialize_target(target: dict[str, Any], length_bucket: int) -> str:
    validate_target_semantics(target)
    if length_bucket not in LENGTH_BUCKETS:
        raise ValueError(f"unsupported length bucket: {length_bucket}")
    payload = json.dumps(target, ensure_ascii=False, separators=(",", ":"))
    if len(payload) > length_bucket:
        raise ValueError(f"target length {len(payload)} exceeds bucket {length_bucket}")
    return payload + " " * (length_bucket - len(payload))


def opaque_id(seed: str | int, family_id: str, role: str) -> str:
    return hashlib.sha256(f"task9b:{seed}:{family_id}:{role}".encode("utf-8")).hexdigest()[:32]

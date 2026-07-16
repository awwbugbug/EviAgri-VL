# Task 8 Causal Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed, paired counterfactual audit that compares B0-B3 under registered protocols and decides whether Static QLoRA v1 may advance to complex adaptation modules.

**Architecture:** An immutable audit manifest separates pixels and out-of-band labels from prompts. Protocol construction, leakage checks, inference, metrics, and paired statistics are independent modules joined by SHA256/protocol hashes. GPU inference is forbidden until CPU-only leakage and smoke gates pass.

**Tech Stack:** Python 3.10+, Pillow, NumPy, PyTorch/Transformers/PEFT/bitsandbytes, standard-library `statistics`, `hashlib`, `random`, `math`, pytest.

## Global Constraints

- B1/B2 must share byte-identical prompts, images, processor settings, decoding settings, schema, parser, and evaluator.
- B0/B3 may differ only in their registered prompt treatment; every other protocol field remains fixed.
- Seed is `20260715`; bootstrap repetitions are `1000`; generation is greedy with `max_new_tokens=128`.
- Positive localization metrics use only GT-positive `original_correct`; null rows never receive IoU or Pointing Game credit.
- All transformed image paths are opaque and never enter prompts.
- Existing non-empty experiment output is never overwritten.
- This workspace has no Git repository; verification uses tests and SHA256 manifests instead of commits.

---

### Task 1: Freeze Task 7 and define protocol contracts

**Files:**
- Create: `server/task8_protocol.py`
- Create: `tests/test_task8_protocol.py`
- Create: `artifacts/2026-07-15_task8/task7_freeze/`

**Interfaces:**
- Produces `ProtocolSpec`, `build_prompt(group, row)`, `generation_kwargs()`, `protocol_hash(group)` and `EXPECTED_SCHEMA_KEYS`.

- [ ] Write failing tests asserting B1/B2 prompts and protocol hashes are identical, B0/B3 hashes differ only through prompt IDs, forbidden leakage tokens are absent, and generation kwargs equal `{"max_new_tokens": 128, "do_sample": False, "temperature": None}`.
- [ ] Run `python -m pytest tests/test_task8_protocol.py -q`; expect import failure for `task8_protocol`.
- [ ] Implement frozen dataclass protocol objects and prompt builders. The neutral prompt accepts only `queried_pest_name`; no function accepts an image path, split, task type, or positive/null flag.
- [ ] Run the protocol tests; expect all pass.
- [ ] Download Task 7 summary, metrics, prediction JSONL, completion manifest, adapter config/weights, and run summary into the freeze directory; generate `artifact.sha256` and verify it locally.

### Task 2: Build paired audit families and transformed pixels

**Files:**
- Create: `server/build_task8_audit.py`
- Create: `tests/test_task8_audit_builder.py`

**Interfaces:**
- Consumes Static QLoRA test JSONL and IP102 class names.
- Produces `audit_manifest.jsonl`, `family_manifest.jsonl`, `images/*.png`, `manifest.sha256`, and `build_summary.json`.

- [ ] Write failing tests with three tiny classes and fixture images. Assert deterministic stratified selection, six conditions per family, wrong-query class differs, no-target image has a different GT class, transformed dimensions equal the original, opaque names exclude class/path tokens, and all four group records reuse the same image SHA.
- [ ] Run `python -m pytest tests/test_task8_audit_builder.py -q`; expect missing-module failure.
- [ ] Implement `select_families(records, per_class, seed)`, `derive_conditions(family, pool, output_dir)`, and atomic JSONL/summary writers. Use `GaussianBlur(radius=max(width,height)/20)` for strong blur and RGB `(127,127,127)` for blank images.
- [ ] Run tests; expect all pass.
- [ ] Build a CPU-only smoke manifest with four families and verify exactly `4 * 6 = 24` audit rows before expanding them into B0-B3 inference jobs.

### Task 3: Implement hidden-leakage gates

**Files:**
- Create: `server/audit_task8_leakage.py`
- Create: `tests/test_task8_leakage.py`

**Interfaces:**
- Produces `leakage_report.json` with `passed`, hard failures, warnings, exact duplicate groups, near-duplicate candidates, template fingerprints, and answer-length/fixed-phrase summaries.

- [ ] Write failing tests for class-bearing derived filenames, forbidden prompt tokens, B1/B2 hash mismatch, repeated audit IDs, cross-split exact SHA duplicate, source ID crossing splits, and dHash distance `<=4` candidate discovery.
- [ ] Run the leakage tests; expect missing-module failure.
- [ ] Implement 64-bit dHash, Hamming distance, exact SHA grouping, eight 8-bit candidate buckets, prompt/answer fingerprint summaries, and hard-gate evaluation. Near duplicates are warnings requiring explicit review; exact cross-split duplicates are failures.
- [ ] Run tests; expect all pass.
- [ ] Run the leakage auditor on full train/val/test plus the smoke manifest. Do not start GPU inference unless `passed=true`; archive candidate near-duplicate rows for review.

### Task 4: Implement same-protocol recoverable inference

**Files:**
- Create: `server/run_task8_inference.py`
- Create: `server/run_task8_smoke.sh`
- Create: `tests/test_task8_inference.py`

**Interfaces:**
- Consumes immutable audit manifest, group list, model/adapter paths, and passed leakage report.
- Produces one append-only `predictions.jsonl` per group plus `status.json`, `run_summary.json`, and failure reports.

- [ ] Write failing tests proving completed audit IDs are skipped, duplicates/unknown IDs/invalid protocol hashes fail, B1/B2 input payloads are byte-identical apart from model identity, and all groups call the same parser and generation kwargs.
- [ ] Run inference tests; expect missing-module failure.
- [ ] Implement manifest validation, resumable prediction writing, one base load for B0/B1, one PEFT load for B2/B3, `model.eval()`, `torch.inference_mode()`, and atomic statuses. Do not include source paths in generated prompt text.
- [ ] Run tests; expect all pass.
- [ ] Run four-family GPU smoke (`4 families * 6 conditions * 4 groups = 96 predictions`). Require 96 unique IDs, zero malformed lines, exact group protocol hashes, and no failure report.

### Task 5: Implement condition-aware metrics

**Files:**
- Create: `server/evaluate_task8.py`
- Create: `tests/test_task8_metrics.py`

**Interfaces:**
- Produces per-group/per-condition metrics and row-level binary outcomes for paired statistics.

- [ ] Write failing tests showing a null empty prediction does not enter IoU, a missing positive bbox contributes IoU 0, false boxes on null raise Predicted-Box-on-Null Rate, wrong-query echo raises Prompt Compliance Error/EBHR, and Supported Diagnosis requires correct class plus IoU `>=0.5`.
- [ ] Run metrics tests; expect missing-module failure.
- [ ] Implement positive, null/counterfactual, and overall metric calculators using the existing strict JSON parser. Preserve row-level `diagnosis_correct`, `presence_correct`, `refusal_correct`, `supported`, and `prompt_compliant` booleans.
- [ ] Run tests; expect all pass.

### Task 6: Implement paired uncertainty and significance

**Files:**
- Modify: `server/evaluate_task8.py`
- Create: `tests/test_task8_statistics.py`

**Interfaces:**
- Produces percentile bootstrap CIs, B1-B2 paired differences, exact McNemar counts/p-values, and condition-separated tables.

- [ ] Write failing tests with deterministic synthetic families for 1,000 family-level resamples, paired delta CI, and exact McNemar p-value `2 * sum(C(n,k))/2^n` over the smaller discordant tail, capped at 1.
- [ ] Run statistics tests; expect missing functions.
- [ ] Implement `bootstrap_ci`, `paired_bootstrap_delta`, and `exact_mcnemar` with seed `20260715`. Resample unique family IDs and retain all conditions belonging to each sampled family.
- [ ] Run tests; expect all pass.
- [ ] Add gate output `decision=A|B|inconclusive` with explicit evidence rather than a hidden weighted score.

### Task 7: Verify smoke and authorize or block formal audit

**Files:**
- Create: `server/validate_task8_smoke.py`
- Create: `tests/test_task8_smoke_gate.py`
- Create: `关键记忆/对话信息_2026_7_15/05_Task8协议与Smoke.md`

**Interfaces:**
- Produces `smoke_gate.json` and a concise permanent research record.

- [ ] Write failing tests for all required gate fields: manifest counts, protocol hashes, leakage passed, prediction counts, unique IDs, schema parsing, image SHA equality, no failures, and peak VRAM below 48GB.
- [ ] Run smoke-gate tests; expect missing-module failure.
- [ ] Implement the validator and run the entire local suite with `python -m pytest -q`.
- [ ] Run server smoke validation. If any gate is false, stop without formal inference and write the blocker. If every gate is true, estimate formal screening cost for 204 families and request/record the formal-run decision before launching the long job.

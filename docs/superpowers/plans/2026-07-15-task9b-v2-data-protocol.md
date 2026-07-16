# Task 9B v2 Data Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and freeze a leakage-resistant Static QLoRA v2 data protocol plus an independent `task9_dev_audit`, without training or consulting Task 8 predictions.

**Architecture:** `task9b_protocol.py` owns the model-visible prompt/schema contract; `build_task9b_v2.py` owns cluster-disjoint family construction and train/dev-only visual transforms; `validate_task9b_freeze.py` is a fail-closed protocol validator. Model-visible JSONL contains opaque IDs and messages only; labels, source IDs, null provenance, transform IDs, and cluster membership live in a separate non-model sidecar.

**Tech Stack:** Python 3.10+, Pillow, standard-library JSON/hash/random/statistics, pytest.

## Global Constraints

- Task 8's 186 families are a prospectively locked confirmatory set; 9B may consume only a blinded source-ID/SHA exclusion list, never predictions or metrics.
- Train, validation, `task9_dev_audit`, and locked confirmatory sources must be disjoint by original image ID, SHA256, and reviewed near-duplicate connected component.
- Every family has exactly `1 positive + 1 semantic-negative query + 1 rotating visual counterfactual`.
- Semantic-negative rows are `real_null`; transformed rows are `synthetic_null`; their counts and metrics remain separate.
- Positive/null rows use the same template-ID distribution, system prompt, JSON keys, key order, decoding contract, and target-length bucket distribution.
- Model-visible records must not expose task type, split, source/class path, class-bearing filename, positive/null token, family role, or provenance.
- `evidence_present=false` requires `diagnosis.status in {uncertain,abstain}`, null pest/species/stage fields, and `evidence_region=null`.
- Train and dev template IDs and transform implementations/parameter ranges are mutually exclusive and differ from Task 8 confirmatory transforms.
- Do not download a backbone, train a model, implement dynamic LoRA/Gating or SAM2, or start 7B.
- The workspace has no Git repository; freeze integrity uses tests, manifests, and SHA256.

---

### Task 1: Freeze the model-visible protocol contract

**Files:**
- Create: `server/task9b_protocol.py`
- Create: `tests/test_task9b_protocol.py`

**Interfaces:**
- Produces `OUTPUT_KEYS`, `TRAIN_TEMPLATE_IDS`, `DEV_TEMPLATE_IDS`, `build_prompt(template_id, queried_name)`, `build_target(...)`, `serialize_target(target, length_bucket)`, and `opaque_id(...)`.

- [ ] Write failing tests asserting identical positive/null template distributions, byte-identical instructions for a given template/query, disjoint train/dev template IDs, no Task 8 prompt string, exact key order `(evidence_present,evidence_region,visible_attributes,diagnosis,reliability)`, null semantic constraints, fixed per-family serialized length buckets, and opaque IDs with no role/class/path tokens.
- [ ] Run `python -m pytest tests/test_task9b_protocol.py -q`; expect missing-module failure.
- [ ] Implement three train templates and two dev templates that all ask the same conditional visual question and give the same abstention rule. Use a fixed-shape diagnosis object and valid trailing JSON whitespace to reach deterministic label-independent length buckets `320/352/384` selected only from family hash.
- [ ] Re-run targeted tests; expect all pass.

### Task 2: Create blinded exclusions and cluster-disjoint splits

**Files:**
- Create: `server/task9b_split.py`
- Create: `tests/test_task9b_split.py`

**Interfaces:**
- Produces `connected_components(image_ids, reviewed_pairs)`, `locked_exclusion(family_manifest)`, and `assign_components(records, exclusions, seed)`.

- [ ] Write failing tests proving transitive near-duplicate pairs stay in one component, every component goes to exactly one split, locked IDs/SHA and their component neighbors are excluded, class-stratified dev selection is deterministic, and the exclusion output contains no labels/prompts/predictions.
- [ ] Run the tests and confirm missing-module failure.
- [ ] Implement deterministic union-find assignment using only the reviewed high-confidence pair list. Generate a blinded exclusion manifest from locked `source_id/source_image_sha256` fields and record its source SHA.
- [ ] Re-run tests; expect all pass.

### Task 3: Implement disjoint train/dev visual counterfactuals

**Files:**
- Create: `server/task9b_transforms.py`
- Create: `tests/test_task9b_transforms.py`

**Interfaces:**
- Produces deterministic `train_transform(kind, image, seed)` and `dev_transform(kind, image, seed)` plus immutable registries.

- [ ] Write failing tests that registries and parameter domains are disjoint; neither reproduces Task 8 uniform `(127,127,127)` blank or Gaussian radius `max(size)/20`; output dimensions are preserved; results are deterministic; transform type rotates evenly across families.
- [ ] Implement train transforms using mild/medium blur bands, block permutation, occluding crop/reframe, and seeded low-information noise canvas. Implement dev transforms with non-overlapping blur bands, different block grids/crop ratios/noise distributions.
- [ ] Run tests; expect all pass.

### Task 4: Build sanitized three-row families and sidecars

**Files:**
- Create: `server/build_task9b_v2.py`
- Create: `tests/test_task9b_builder.py`

**Interfaces:**
- Consumes clean positive detection records, component split map, template registry, transform registry, and exclusions.
- Produces `model/train.jsonl`, `model/val.jsonl`, `dev_audit/model.jsonl`, `private/provenance.jsonl`, opaque derived images, and build summary.

- [ ] Write failing tests for exactly three rows per family, 1:1:1 roles, real/synthetic null separation, absent semantic query class, same per-role template distribution, identical JSON key order and length-bucket distribution, model-visible key allowlist, opaque image names, and zero source/cluster overlap across outputs.
- [ ] Implement deterministic family assembly. Keep `role`, `evidence_present`, source IDs, query IDs, transform parameters, and split solely in the private sidecar; messages contain only an opaque image reference, neutral prompt, and assistant JSON.
- [ ] Run tests; expect all pass.

### Task 5: Implement freeze validator and 9C-ready exports

**Files:**
- Create: `server/validate_task9b_freeze.py`
- Create: `tests/test_task9b_freeze.py`

**Interfaces:**
- Produces `freeze_report.json`, `protocol_manifest.json`, input/output SHA256, and three label-bearing probe exports stored outside model JSONL.

- [ ] Write failing tests for prompt/metadata leakage, role imbalance, target-length imbalance, JSON syntax/schema/semantic/task compliance, train/dev transform collision, cluster overlap, locked-source overlap, and non-empty output overwrite refusal.
- [ ] Implement fail-closed checks and export the three 9C views without adding label fields to model-visible data.
- [ ] Run targeted and full test suites; expect all pass.

### Task 6: Build, audit, and freeze v2 protocol on the server

**Files:**
- Create remotely: `/root/autodl-tmp/EviAgriDiag/datasets/derived/static_qlora_v2_protocol/2026-07-15/`
- Create locally: `关键记忆/对话信息_2026_7_15/10_Task9B协议冻结结果.md`

**Interfaces:**
- Produces a non-overwriting immutable dataset/protocol directory and `completion.sha256`.

- [ ] Upload tested scripts with matching local/remote SHA256.
- [ ] Build blinded locked exclusions and near-duplicate components without reading prediction/metric files.
- [ ] Build CPU-only model JSONL, private sidecars, and independent `task9_dev_audit`; run the freeze validator.
- [ ] Require zero prompt/path/task-type/role leakage, zero component overlap, exact 1:1:1 families, equal template/length-bucket distributions, and 100% four-level JSON quality.
- [ ] Verify `completion.sha256`, download compact reports/manifests, record 9B key memory, and stop before 9C.

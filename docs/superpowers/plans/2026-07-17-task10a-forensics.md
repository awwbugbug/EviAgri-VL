# Task 10A Forensic Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a fail-closed, no-training forensic audit for Qwen bbox coordinates, token-level PDM-H visual dependence, and family-level causal consistency on the frozen 32-family v2.2 dev audit.

**Architecture:** Three independent audit modules share small JSON/SHA utilities but never mutate Task 9 outputs. Bbox and pair audits are CPU-only; PDM-H loads the existing Base plus the three frozen Control adapters and performs teacher-forced conditioned/unconditioned forward passes without gradients. A final orchestrator validates all inputs, combines reports, and emits a signed Task 10A decision without authorizing Task 10B automatically.

**Tech Stack:** Python 3.10/3.13, pytest, Pillow, PyTorch, Transformers 4.51.3, PEFT, qwen-vl-utils, NumPy, existing Qwen2.5-VL-3B-Instruct.

## Global Constraints

- Never read or reference Task 8 locked 186 families.
- Never train, update weights, restart an existing task, overwrite an output, or shut down the server.
- Use only `/root/autodl-tmp/EviAgriDiag/experiments/task9d_v22_micro/2026-07-16` and its six completed inference outputs as historical inputs.
- New output root is `/root/autodl-tmp/EviAgriDiag/experiments/task10_micro_first/2026-07-17/task10a` and must not exist before launch.
- Bbox round-trip gate: 32/32 valid records, max coordinate error <=1 pixel, synthetic IoU >=0.999, and a single declared frame across prompt/target/evaluator.
- PDM quality gate: token span coverage >=95%, finite distributions with normalization error <=1e-5, and all 32 families for Control seeds 17/29/43.
- PDM visual gate: original-minus-blank or original-minus-shuffle paired mean PDM-H must be positive with 1,000-family-bootstrap 95% CI lower bound >0 for taxonomy or `evidence_present` tokens.
- Never launch Task 10B/C/D from this plan.

---

### Task 1: Shared Task 10 audit primitives

**Files:**
- Create: `server/task10_audit_common.py`
- Test: `tests/test_task10_audit_common.py`

**Interfaces:**
- Produces: `sha256_file(path: Path) -> str`, `write_json_new(path: Path, value: Any) -> None`, `ensure_new_directory(path: Path) -> None`, `family_bootstrap_delta(values: list[tuple[str,float,float]], repetitions: int, seed: int) -> dict[str,float]`.
- Consumers: all later Task 10A modules.

- [ ] **Step 1: Write failing tests for fail-closed writes and deterministic family bootstrap**

```python
def test_write_json_new_refuses_existing_file(tmp_path):
    target = tmp_path / "report.json"
    target.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_json_new(target, {"passed": True})
    assert target.read_text(encoding="utf-8") == "keep"

def test_family_bootstrap_delta_is_deterministic_and_paired():
    rows = [("f1", 0.8, 0.2), ("f2", 0.6, 0.4), ("f3", 0.7, 0.3)]
    left = family_bootstrap_delta(rows, repetitions=1000, seed=20260717)
    right = family_bootstrap_delta(rows, repetitions=1000, seed=20260717)
    assert left == right
    assert left["estimate"] == pytest.approx(0.4)
    assert left["low"] > 0
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `pytest -q tests/test_task10_audit_common.py`
Expected: FAIL because `task10_audit_common` does not exist.

- [ ] **Step 3: Implement minimal shared primitives**

```python
def family_bootstrap_delta(values, repetitions=1000, seed=20260717):
    by_family = {str(fid): (float(a), float(b)) for fid, a, b in values}
    families = sorted(by_family)
    if len(families) != len(values) or not families:
        raise ValueError("family bootstrap requires one non-empty paired row per family")
    rng = random.Random(seed)
    observed = mean(by_family[f][0] - by_family[f][1] for f in families)
    samples = []
    for _ in range(repetitions):
        draw = [families[rng.randrange(len(families))] for _ in families]
        samples.append(mean(by_family[f][0] - by_family[f][1] for f in draw))
    samples.sort()
    return {"estimate": observed, "low": percentile(samples, 0.025),
            "high": percentile(samples, 0.975), "repetitions": repetitions,
            "seed": seed, "unit": "family_id"}
```

`write_json_new` must create parent directories but refuse an existing target; `ensure_new_directory` must refuse any pre-existing path.

- [ ] **Step 4: Run focused tests**

Run: `pytest -q tests/test_task10_audit_common.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add server/task10_audit_common.py tests/test_task10_audit_common.py
git commit -m "feat(task10): add fail-closed audit primitives"
```

### Task 2: Bbox coordinate-chain audit

**Files:**
- Create: `server/audit_task10_bbox_coordinates.py`
- Test: `tests/test_task10_bbox_coordinates.py`

**Interfaces:**
- Consumes: v2.2 `evaluation_manifest.jsonl`, Qwen processor, `process_vision_info`.
- Produces: `scale_box(box, from_size, to_size)`, `box_iou(a,b)`, `audit_coordinate_record(...)`, CLI report `bbox_coordinate_report.json`.

- [ ] **Step 1: Write failing geometry tests**

```python
def test_bbox_roundtrip_preserves_original_coordinates():
    original = (1000, 600)
    processed = (616, 392)
    box = [100, 50, 900, 550]
    restored = scale_box(scale_box(box, original, processed), processed, original)
    assert max(abs(a-b) for a, b in zip(box, restored)) <= 1.0
    assert box_iou(box, restored) >= 0.999

def test_invalid_or_degenerate_box_is_rejected():
    with pytest.raises(ValueError):
        validate_box([5, 5, 5, 10], image_size=(100, 100))
    with pytest.raises(ValueError):
        validate_box([-1, 5, 10, 10], image_size=(100, 100))
```

- [ ] **Step 2: Run and observe failure**

Run: `pytest -q tests/test_task10_bbox_coordinates.py`
Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement pure coordinate helpers and record validation**

```python
def scale_box(box, from_size, to_size):
    fw, fh = map(float, from_size); tw, th = map(float, to_size)
    if min(fw, fh, tw, th) <= 0:
        raise ValueError("image dimensions must be positive")
    x1, y1, x2, y2 = map(float, box)
    return [x1 * tw/fw, y1 * th/fh, x2 * tw/fw, y2 * th/fh]
```

The CLI must select exactly the 32 canonical positive/original rows, open only their image paths, capture original PIL size, qwen-vl-utils image size, processor `image_grid_thw`, and derive grid dimensions as `(grid_w*patch_size, grid_h*patch_size)`. It must compare qwen-vl-utils and processor sizes rather than assume they are identical. For every GT and predicted box, record all three fixed interpretations—original-image absolute pixels, processor-input absolute pixels, and 0–1000 normalized coordinates—plus their conversion matrices. The primary interpretation must be the single Qwen frame declared consistently by prompt, target builder, and evaluator; the other two are diagnostic alternatives only.

- [ ] **Step 4: Add fail-closed protocol tests**

Test that duplicate families, non-32 inputs, missing GT bbox, non-multiple-of-28 Qwen dimensions, inconsistent prompt frame, or missing `image_grid_thw` produce `blocked` reasons and never `passed=true`.

- [ ] **Step 5: Run focused tests**

Run: `pytest -q tests/test_task10_bbox_coordinates.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add server/audit_task10_bbox_coordinates.py tests/test_task10_bbox_coordinates.py
git commit -m "feat(task10): add bbox coordinate-chain audit"
```

### Task 3: Token-level PDM-H core and teacher-forced runner

**Files:**
- Create: `server/audit_task10_pdm.py`
- Test: `tests/test_task10_pdm.py`

**Interfaces:**
- Produces: `hellinger_from_logits(logits_p, logits_q)`, `assistant_token_spans(...)`, `summarize_pdm_records(...)`, CLI `pdm_token_report.json` and `pdm_observations.jsonl`.
- Consumes: Base Qwen, Control adapters 17/29/43, 32 family positive targets, original/blank/shuffle pixels.

- [ ] **Step 1: Write failing numerical and token-group tests**

```python
def test_hellinger_identical_distributions_is_zero():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    assert hellinger_from_logits(logits, logits).item() == pytest.approx(0.0, abs=1e-7)

def test_hellinger_is_symmetric_and_bounded():
    p = torch.tensor([[8.0, 0.0, -2.0]])
    q = torch.tensor([[-2.0, 0.0, 8.0]])
    hpq = hellinger_from_logits(p, q).item()
    assert hpq == pytest.approx(hellinger_from_logits(q, p).item())
    assert 0.0 <= hpq <= 1.0

def test_json_spans_identify_taxonomy_evidence_and_bbox_tokens(fake_tokenizer):
    target = '{"evidence_present":true,"evidence_region":[1,2,3,4],"diagnosis":{"pest_id":12}}'
    spans = assistant_token_spans(fake_tokenizer, target)
    assert spans.coverage >= 0.95
    assert spans.groups["evidence_present"]
    assert spans.groups["bbox_value"]
    assert spans.groups["taxonomy_value"]
```

- [ ] **Step 2: Run and observe failure**

Run: `pytest -q tests/test_task10_pdm.py`
Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement stable Hellinger computation**

```python
def hellinger_from_logits(logits_p, logits_q):
    p = torch.softmax(logits_p.float(), dim=-1)
    q = torch.softmax(logits_q.float(), dim=-1)
    value = torch.sqrt(torch.clamp(((torch.sqrt(p)-torch.sqrt(q))**2).sum(-1), min=0.0)) / math.sqrt(2.0)
    if not torch.isfinite(value).all():
        raise FloatingPointError("non-finite PDM-H")
    return value
```

- [ ] **Step 4: Implement tokenizer-offset grouping and quality gates**

Serialize canonical JSON without whitespace. Obtain fast-tokenizer offset mappings for the assistant target and map only value spans—not field-name tokens—to groups. Report mapped active tokens divided by all assistant target tokens; BLOCK below95%.

- [ ] **Step 5: Implement no-gradient dual-forward runner**

For each Control seed and family, use the canonical positive target for all three pixel conditions. Construct:

- conditioned inputs: identical system/user text plus original, blank, or shuffle image;
- unconditioned inputs: identical system/user text with image content removed;
- teacher-forced assistant tokens: always the original positive canonical target.

Run under `torch.inference_mode()`, batch size1, and never save raw logits. Store per-token-group mean PDM-H, token counts, seed, family, condition, model/input hashes, and finite/normalization checks.

- [ ] **Step 6: Implement family bootstrap decision**

Aggregate each seed and pooled families. For taxonomy and `evidence_present`, compute original-minus-blank and original-minus-shuffle deltas with1,000 repetitions. Set `visual_dependency_passed=true` only if at least one group/intervention has positive estimate and CI lower bound>0, while all quality gates pass.

- [ ] **Step 7: Run focused and regression tests**

Run: `pytest -q tests/test_task10_pdm.py tests/test_task9d_inference.py`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add server/audit_task10_pdm.py tests/test_task10_pdm.py
git commit -m "feat(task10): add token-level PDM-H audit"
```

### Task 4: Hallusion-style family and pair audit

**Files:**
- Create: `server/evaluate_task10_pairs.py`
- Test: `tests/test_task10_pairs.py`

**Interfaces:**
- Produces: `evaluate_family_pairs(manifest, predictions) -> dict`, CLI `pair_metrics.json`.
- Consumes: existing six v2.2 prediction files; Task 10A primary report uses Control17/29/43 separately and pooled.

- [ ] **Step 1: Write failing causal-consistency tests**

```python
def test_strict_family_success_requires_original_and_all_nulls():
    rows, predictions = complete_family_fixture()
    report = evaluate_family_pairs(rows, predictions)
    assert report["strict_family_success"] == 1.0
    predictions["f1-shuffle"] = supported_json()
    report = evaluate_family_pairs(rows, predictions)
    assert report["strict_family_success"] == 0.0
    assert report["by_condition"]["shuffle"]["pair_success"] == 0.0

def test_missing_condition_blocks_report():
    rows, predictions = complete_family_fixture()
    rows = [row for row in rows if row["condition"] != "blank"]
    with pytest.raises(ValueError, match="condition set"):
        evaluate_family_pairs(rows, predictions)
```

- [ ] **Step 2: Run and observe failure**

Run: `pytest -q tests/test_task10_pairs.py`
Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement exact family joins and metrics**

Require one canonical original positive and one each of semantic_null, source_visual_null, blank, blur, shuffle per family. Invalid/missing JSON counts as failure, never silent exclusion. Original success requires both acceptance and the correct canonical diagnosis ID. Compute original positive TPR, null FPR by condition, pair success for each intervention, strict family success across all five nulls, original-to-intervention concrete-diagnosis drop, and the fraction of families containing causally opposite decisions as contradiction rate.

- [ ] **Step 4: Run focused tests**

Run: `pytest -q tests/test_task10_pairs.py tests/test_task9d_v22_evaluation.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add server/evaluate_task10_pairs.py tests/test_task10_pairs.py
git commit -m "feat(task10): add family causal-consistency audit"
```

### Task 5: Task 10A orchestrator, preflight, and signed decision

**Files:**
- Create: `server/run_task10a_forensics.py`
- Create: `server/run_task10a_forensics.sh`
- Test: `tests/test_task10a_forensics.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `task10a_decision_report.json`, `status.json`, `completion.sha256`, `failure.json` on failure.
- Consumes: outputs from Tasks2–4 and frozen v2.2 hashes.

- [ ] **Step 1: Write failing decision tests**

```python
def test_decision_blocks_verifier_when_pdm_fails_but_preserves_diagnosis_path():
    report = decide_task10a(bbox=passing_bbox(), pdm=failing_pdm(), pairs=pair_report())
    assert report["authorize_existing_verifier_for_task10d"] is False
    assert report["authorize_task10b_planning"] is True
    assert report["authorize_training"] is False

def test_decision_never_authorizes_training_or_task10b_execution():
    report = decide_task10a(bbox=passing_bbox(), pdm=passing_pdm(), pairs=pair_report())
    assert report["authorize_training"] is False
    assert report["authorize_task10b_execution"] is False
```

- [ ] **Step 2: Run and observe failure**

Run: `pytest -q tests/test_task10a_forensics.py`
Expected: FAIL because the orchestrator does not exist.

- [ ] **Step 3: Implement preflight and fail-closed orchestration**

Preflight must verify:

- six v2.2 groups each contain 352 predictions and valid `completion.sha256`;
- one manifest hash and one decoding contract across groups;
- input manifest has exactly 32 target families and no path/reference containing the Task 8 locked-set location;
- output root does not exist;
- Base and all three Control adapter files exist and their hashes match run summaries.

Run bbox and pair audits first. Run PDM-H only after CPU outputs pass integrity tests. Any exception writes `failure.json`, leaves existing inputs untouched, and exits non-zero.

- [ ] **Step 4: Implement decision/report contract**

The decision report must separately state:

- `bbox_coordinate_status`;
- `existing_verifier_visual_dependency_status`;
- `pair_forensic_findings`;
- `authorize_existing_verifier_for_task10d`;
- `authorize_task10b_planning=true`;
- `authorize_task10b_execution=false`;
- `authorize_training=false`;
- `task8_locked_set_read=false`.

- [ ] **Step 5: Run complete local test suite**

Run: `pytest -q`
Expected: existing194 tests plus all new Task10A tests PASS.

- [ ] **Step 6: Commit implementation**

```bash
git add server/run_task10a_forensics.py server/run_task10a_forensics.sh tests/test_task10a_forensics.py README.md
git commit -m "feat(task10): orchestrate no-training forensic gate"
```

### Task 6: Server sync, smoke, formal Task 10A run, and archival

**Files:**
- Create locally after run: `artifacts/2026-07-17_task10/10A_forensics/` (ignored by Git)
- Create key memory: `关键记忆/对话信息_2026_7_17/05_Task10A法医审计结果.md`

**Interfaces:**
- Produces the only formal Task10A report and a Git-tracked concise conclusion.

- [ ] **Step 1: Verify Git state and server idle state**

Run local: `git status --short`
Expected: empty.

Run remote read-only: `screen -ls`, `ps`, `nvidia-smi`
Expected: no active training/inference/evaluation process that would conflict with the audit.

- [ ] **Step 2: Sync only the new committed Task10A files**

Use the existing SSH key. Upload new files to `/root/EviAgri-VL/server/`; do not upload `.git`, data, artifacts, or credentials. Compare local and remote SHA256 for every uploaded file.

- [ ] **Step 3: Run server unit tests before formal audit**

Run: `/root/miniconda3/envs/eviagri/bin/python -m pytest -q tests/test_task10_*.py tests/test_task10a_forensics.py`
Expected: PASS. If pytest is unavailable in the server project layout, upload the exact test files to a new isolated test directory and run them there; do not install packages during an active experiment.

- [ ] **Step 4: Run a 1-family smoke in a separate output**

Smoke may validate model loading, message construction, token span coverage, and finite PDM-H only. It cannot be used for scientific metrics. Delete nothing after smoke; preserve its status and hashes.

- [ ] **Step 5: Launch the formal 32-family audit once**

Run in detached screen `task10a_forensics`. The wrapper must write progress by seed/family, never overwrite, and end with signed reports. No automatic rerun on failure.

- [ ] **Step 6: Verify and download formal outputs**

Check `sha256sum -c completion.sha256`, report family counts, input hashes, finite values, and decision fields. Download compact reports—not logits or images—to `artifacts/2026-07-17_task10/10A_forensics/` and verify local hashes.

- [ ] **Step 7: Write and commit concise key memory**

Record bbox status, PDM-H effect/CI, pair metrics, explicit blocks, and the next authorized design step. Then:

```bash
git add 关键记忆/对话信息_2026_7_17/05_Task10A法医审计结果.md
git commit -m "docs(task10): record Task 10A forensic decision"
```

- [ ] **Step 8: Stop at the gate**

Do not design or execute Task 10B automatically. Report Task 10A evidence to the user and request confirmation before planning the next subsystem.

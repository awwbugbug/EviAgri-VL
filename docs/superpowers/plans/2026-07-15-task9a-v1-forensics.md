# Task 9A v1 Data Forensics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible, read-only forensic audit of Static QLoRA v1 training data without accessing the Task 8 locked confirmatory set.

**Architecture:** A standard-library Python auditor reads only the frozen v1 train/val/test JSONL files, profiles label-correlated prompt/output/metadata signals, and evaluates train-fitted text-only probes on held-out splits. A fail-closed scope guard rejects Task 8, locked-set, and Task 9 development-audit paths. Tests use synthetic JSONL fixtures; the real run writes a new immutable Task 9A artifact directory.

**Tech Stack:** Python 3.10+ standard library, pytest, JSONL, SHA256.

## Global Constraints

- Never read Task 8 predictions or the 186 locked families for development decisions.
- Read only `/root/autodl-tmp/EviAgriDiag/datasets/derived/static_qlora_v1/{train,val,test}.jsonl` plus v1 manifest/config/generator code.
- Do not start training, inference, model download, dynamic LoRA/Gating, SAM2, or 7B work.
- Do not overwrite non-empty output directories; record input and output SHA256.
- Report real/synthetic null separately whenever the source records permit it.
- This workspace has no Git repository; verification uses tests and SHA256 manifests instead of commits.

---

### Task 1: Scope guard and immutable input inventory

**Files:**
- Create: `server/task9a_v1_forensics.py`
- Create: `tests/test_task9a_v1_forensics.py`

**Interfaces:**
- Produces `assert_allowed_input(path: Path)`, `sha256_file(path: Path)`, and `load_split(path: Path, split: str)`.

- [ ] Write failing tests that accept `static_qlora_v1/train.jsonl`, reject path components containing `task8`, `formal_clean_v2`, `locked_confirmatory`, or `task9_dev_audit`, reject duplicate IDs, split mismatches, malformed JSON, and non-boolean `target.evidence_present`.
- [ ] Run `python -m pytest tests/test_task9a_v1_forensics.py -q`; expect import failure for `task9a_v1_forensics`.
- [ ] Implement the minimal scope guard, strict loader, and SHA256 helper.
- [ ] Re-run the targeted tests; expect all tests pass.

### Task 2: Profile template, answer, path, and metadata leakage

**Files:**
- Modify: `server/task9a_v1_forensics.py`
- Modify: `tests/test_task9a_v1_forensics.py`

**Interfaces:**
- Produces `profile_records(rows_by_split) -> dict` with label counts, template fingerprints, canonical target lengths, fixed targets/phrases, image reuse, field order, path/prompt exposure, task-type association, and class counts.

- [ ] Add failing fixtures where prompt prefix, target length, task type, record ID, and image directory reveal the label; assert each appears in the profile with per-label counts/rates.
- [ ] Run the targeted tests and confirm expected assertion failures.
- [ ] Implement canonical JSON serialization, normalized prompt signatures, non-image metadata extraction, positive/null image-ID overlap, and exact association tables.
- [ ] Re-run tests; expect all pass.

### Task 3: Held-out text-only forensic probes

**Files:**
- Modify: `server/task9a_v1_forensics.py`
- Modify: `tests/test_task9a_v1_forensics.py`

**Interfaces:**
- Produces `probe_view(train_rows, eval_rows, view) -> dict` for `user_prompt`, `system_user_prompt`, and `prompt_metadata`, reporting confusion counts, Balanced Accuracy, and AUROC.

- [ ] Add failing tests for a train-fitted Bernoulli Naive Bayes probe that generalizes a prompt-template token to held-out rows; assert Balanced Accuracy/AUROC are 1.0 on a separable fixture and 0.5 on a constant-score fixture.
- [ ] Run the tests and confirm missing probe functions cause failure.
- [ ] Implement deterministic tokenization, Laplace-smoothed Bernoulli Naive Bayes, probability scores, Balanced Accuracy, tie-aware AUROC, and the three views. Exclude target/label fields from features.
- [ ] Re-run targeted tests and then `python -m pytest -q`; expect all pass.

### Task 4: Run 9A on the server and freeze evidence

**Files:**
- Create remotely: `/root/autodl-tmp/EviAgriDiag/experiments/task9/9A_v1_forensics/2026-07-15/`
- Create locally after verified run: `关键记忆/对话信息_2026_7_15/09_Task9A法医审计结果.md`

**Interfaces:**
- Produces `forensic_report.json`, `forensic_report.md`, `input_manifest.json`, `completion.sha256`, and `run_status.json`.

- [ ] Upload only the tested auditor to `/root/EviAgri-VL/server/` and run it against the three frozen v1 JSONL files; refuse a non-empty output directory.
- [ ] Verify report counts against the v1 manifest: train `13652 positive + 6826 null`, val `1526 + 1526`, test `3798 + 3798`.
- [ ] Verify all output hashes with `sha256sum -c completion.sha256`; require no failure report.
- [ ] Download the compact report locally, independently check headline counts/metrics, and write the concise key-memory result.
- [ ] Stop after 9A. Do not begin 9B until the forensic findings and protocol implications are reported.

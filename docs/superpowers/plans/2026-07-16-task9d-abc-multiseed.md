# Task 9D A/B/C Multi-Seed Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, train, evaluate, and decide among three small-scale Static QLoRA data protocols using only frozen train/val/task9_dev_audit data and three fixed seeds.

**Architecture:** A fail-closed preparation layer creates immutable A/B/C manifests from v2.1 while preserving family identity and exact semantic-query matching. A dedicated v2 collator/trainer runs nine identical-budget q/v-only Static QLoRA jobs; a shared inference/evaluation layer evaluates Base once and every adapter on the same paired dev conditions, then applies frozen elimination and lexicographic selection rules.

**Tech Stack:** Python 3.10+, PyTorch, Transformers, PEFT, bitsandbytes, Pillow, scikit-learn/scipy locally for statistical checks, pytest, SHA256 manifests, screen/SSH.

## Global Constraints

- Inputs are limited to v2.1 train, val, and task9_dev_audit; never read Task 8 locked images, predictions, or metrics.
- Variants are fixed: A=positive+exact semantic null/single neutral template; B=A+rotating visual counterfactual; C=B+symmetric three-template rewrites.
- Seeds are exactly `17,29,43`; Base, splits, family pools, rank, optimizer, resolution, parser, decoder, steps, and metrics are identical.
- Static QLoRA targets only language `q_proj/v_proj`; r=16, alpha=32, dropout=0.05, LR=1e-4; vision and all non-LoRA parameters remain frozen.
- Train budget is 192 optimizer steps, batch 1, gradient accumulation 8, exactly 1,536 micro-samples per run; early stopping is disabled.
- Any shortcut gate >0.55, non-finite training, overwrite attempt, locked-source overlap, or schema/semantic regression fails closed.
- Do not start Task 9E, dynamic LoRA/Gating, SAM2, 7B, a new backbone, shutdown, or Task 8 confirmation.
- The workspace has no Git repository; every review checkpoint uses tests, immutable directories, and local/remote SHA256 equality instead of commits.

---

### Task 1: Freeze A/B/C family pools and training schedules

**Files:**
- Create: `server/task9d_prepare.py`
- Test: `tests/test_task9d_prepare.py`

**Interfaces:**
- Consumes v2.1 `model/*.jsonl`, `dev_audit/model.jsonl`, and `private/provenance.jsonl`.
- Produces `prepare_task9d(source_root, output_root, seed=20260716)` with 512 train, 192 val, 512 dev, 128 paired-challenge families; A/B/C model JSONL, private role schedules, class bands, and input/output hashes.

- [ ] Write failing tests proving the family sets are identical across A/B/C, A has only positive/semantic rows, B/C have all three roles, A/B use one neutral template, C preserves symmetric template distributions, every train schedule has exactly 1,536 rows, and no source/split/family is deleted or moved.
- [ ] Run `python -m pytest tests/test_task9d_prepare.py -q`; expect missing-module failure.
- [ ] Implement deterministic class-stratified family selection and schedule expansion. A repeats 256 positive and 256 semantic rows deterministically to reach 768/768 exposures; B/C expose each of 512 roles once.
- [ ] Add fail-closed checks for locked source IDs/SHA, near-duplicate component overlap, exact query-class marginals, role counts, model/private ID alignment, and non-empty destination refusal.
- [ ] Re-run the targeted test; expect all pass.

### Task 2: Build held-out paired dev conditions

**Files:**
- Create: `server/task9d_eval_protocol.py`
- Test: `tests/test_task9d_eval_protocol.py`

**Interfaces:**
- Produces `build_eval_manifest(prepared_root)` with canonical neutral, native training, two unseen rewrites, positive/semantic/visual rows, and paired blank/blur/shuffle rows.

- [ ] Write failing tests that every challenge family has original+blank+blur+shuffle, dimensions are preserved, blank is not RGB127, blur avoids 0.05 and training bands, shuffle uses a held-out grid, and Task 8 prompt text/transforms are absent.
- [ ] Run the targeted test and verify missing-module failure.
- [ ] Implement deterministic RGB(91,107,123) blank, Gaussian radius fraction 0.16, and 7x7 patch permutation for the 128 challenge families; store images under SHA-named paths.
- [ ] Implement native/canonical/unseen prompt manifests with identical images, resolution, max tokens, decoding contract, and output schema.
- [ ] Re-run tests; expect all pass.

### Task 3: Add v2.1 training data and q/v-only model adapters

**Files:**
- Create: `server/task9d_data.py`
- Create: `server/task9d_model.py`
- Create: `server/task9d_config.py`
- Test: `tests/test_task9d_training_core.py`

**Interfaces:**
- `Task9dDataset(path, image_root)` resolves opaque image references.
- `AssistantOnlyV2Collator(processor,max_length)` supports system/user/assistant and masks system, prompt, image, and padding tokens.
- `language_qv_targets(model)` returns only language q/v projections.
- `load_task9d_config(path)` enforces every frozen constant.

- [ ] Write failing tests for three-role messages, `evidence_region` schema, null semantic constraints, assistant-only masking, max-length failure, q/v-only target regex, unsafe trainable-module rejection, exact seeds, LR=1e-4, r=16, 192 steps, accumulation=8, and disabled early stopping.
- [ ] Run the targeted tests; expect missing symbols/modules.
- [ ] Implement minimal dataset/collator/schema validation and q/v-only PEFT construction with all visual/non-LoRA parameters frozen.
- [ ] Implement strict config validation that rejects unknown/mismatched A/B/C constants.
- [ ] Re-run targeted tests; expect all pass.

### Task 4: Implement immutable smoke and nine-run trainer

**Files:**
- Create: `server/train_task9d.py`
- Create: `server/run_task9d_smoke.sh`
- Create: `server/run_task9d_matrix.sh`
- Test: `tests/test_task9d_trainer.py`

**Interfaces:**
- `run_training(config, variant, seed, mode)` writes adapter, SHA256, log history, final-checkpoint rationale, config/environment, sample exposure, peak VRAM, duration, status, and failure JSON.

- [ ] Write failing tests for immutable output refusal, seed/variant path isolation, fixed final-step checkpoint selection, finite-loss gate, adapter hash report, and accurate role-exposure counts.
- [ ] Run targeted tests; expect missing module failure.
- [ ] Implement Trainer configuration with batch=1, accumulation=8, max_steps=192, eval at 64/128/192, save only final step, no early stopping, q/v adapter, and atomic status files.
- [ ] Implement smoke mode: one batch per role plus 3 optimizer steps for A/B/C seed17; require finite loss, safe trainables, adapter reload, JSON generation, and peak VRAM <40GB.
- [ ] Implement sequential matrix shell runner ordered A17,A29,A43,B17,...,C43; never auto-restart or delete failed output.
- [ ] Re-run tests; expect all pass.

### Task 5: Implement shared inference and parser

**Files:**
- Create: `server/run_task9d_inference.py`
- Test: `tests/test_task9d_inference.py`

**Interfaces:**
- Runs Base or one adapter on a frozen eval manifest and writes one prediction row per expected ID plus run summary and SHA256.

- [ ] Write failing tests that Base/adapter share processor, pixels, prompt, resolution, decoder, max tokens, parser input, and expected IDs; resume/overwrite is refused; generation contains pixels only and never source path metadata.
- [ ] Run tests and confirm missing implementation.
- [ ] Implement deterministic decoding (`do_sample=false`, fixed max_new_tokens), adapter load/unload, per-row raw text/latency/error capture, atomic completion, and exact-count verification.
- [ ] Re-run tests; expect all pass.

### Task 6: Implement metrics, statistics, and frozen decision rules

**Files:**
- Create: `server/evaluate_task9d.py`
- Create: `server/decide_task9d.py`
- Test: `tests/test_task9d_evaluation.py`

**Interfaces:**
- `evaluate_predictions(manifest,predictions,class_bands)` returns the complete per-seed metric schema.
- `decide_task9d(base,variants,bootstrap_seed=20260716)` returns PASS/BLOCK for A/B/C, a unique selected protocol when eligible, and `authorize_9e_recommendation` without starting 9E.

- [ ] Write failing tests for Accuracy/Macro-F1/head-medium-tail, overall/semantic/visual FPR, blank/blur/shuffle refusal, concrete-null diagnosis, supported diagnosis, IoU/pointing, four JSON levels, native/canonical/unseen gaps, family-paired deltas, worst seed, sample SD, 1,000-bootstrap CI, paired bootstrap, and McNemar.
- [ ] Write elimination tests: any seed Base Accuracy delta<-3pp, Macro-F1 CI upper<0, blank/blur FPR>=0.10, prompt gap>=0.05, shortcut/schema leak, or semantic/task compliance<0.99 eliminates the group.
- [ ] Run tests; expect missing implementation.
- [ ] Implement metrics with positive-only localization and separate null denominators; never score empty null boxes as correct localization.
- [ ] Implement seed aggregation and lexicographic selection: eligible → higher Macro-F1 → lower mean/worst Null FPR → higher Supported Diagnosis → lower seed variance.
- [ ] Re-run targeted tests; expect all pass.

### Task 7: Deploy, freeze inputs, and pass pre-training gates

**Files:**
- Create remotely: `/root/autodl-tmp/EviAgriDiag/experiments/task9d/2026-07-16/preparation/`
- Create locally: `关键记忆/对话信息_2026_7_16/01_Task9D设计与准备.md`

**Interfaces:**
- Produces immutable prepared manifests, three shortcut reports, preflight report, smoke adapters, and `pretraining_gate.json`.

- [ ] Run full local `python -m pytest -q`; require zero failures.
- [ ] Upload scripts and verify local/remote SHA256 equality.
- [ ] Build A/B/C and eval manifests on server; verify family counts, role schedules, locked exclusions, image conditions, and completion hashes.
- [ ] Download A/B/C shortcut probes; run the unchanged Task 9C gate locally; require BA/AUROC<=0.55 for all views and variants.
- [ ] Run full tokenizer/image preflight and A/B/C smoke; require finite loss, safe q/v targets, adapter reload, valid JSON, and VRAM gate.
- [ ] Record hashes/results in the 2026-07-16 key memory. Stop and report if any gate fails.

### Task 8: Run nine small trainings and unified evaluation

**Files:**
- Create remotely: `/root/autodl-tmp/EviAgriDiag/experiments/task9d/2026-07-16/runs/{A,B,C}/seed_{17,29,43}/`
- Create remotely: `/root/autodl-tmp/EviAgriDiag/experiments/task9d/2026-07-16/evaluation/`

**Interfaces:**
- Produces nine adapters/checkpoints and ten inference groups (Base+9 adapters), each with completion hashes.

- [ ] Start the sequential matrix only after `pretraining_gate.json` passes; keep server on.
- [ ] For every run verify adapter SHA, 192 steps, 1,536 micro-samples, finite train/val losses, duration, peak VRAM, and final-step checkpoint rationale.
- [ ] Run Base once and nine adapters on exactly the frozen eval IDs; verify prediction counts and SHA256 before metrics.
- [ ] Evaluate each seed, aggregate three seeds, run paired statistics, and apply the frozen decision rules.

### Task 9: Freeze Task 9D decision and stop before 9E

**Files:**
- Create locally: `关键记忆/对话信息_2026_7_16/02_Task9D决策结果.md`
- Download to: `artifacts/2026-07-16_task9/9D_decision/`

**Interfaces:**
- Produces `task9d_decision_report.json`, a concise Markdown decision, all critical hashes, and an explicit 9E recommendation flag.

- [ ] Verify all input/output hashes, nine run summaries, prediction counts, metric files, bootstrap configuration, and elimination evidence.
- [ ] Write A/B/C pass/fail, reasons, unique best protocol if any, and `authorize_9e_recommendation`.
- [ ] Run full local tests and remote completion verification immediately before claiming completion.
- [ ] Save concise key memory and stop; never start Task 9E without a new explicit user approval.

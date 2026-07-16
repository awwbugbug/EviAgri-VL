# Task 9D Output Forensics and v2.2 Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine the dominant cause of Task 9D failure from frozen outputs, then freeze one single-variable, minimal v2.2 validation experiment without starting full training.

**Architecture:** A read-only forensic analyzer will join the frozen Task9D manifest, ten prediction files, full metrics, and v2.1 train schedules by opaque IDs. It will separate strict-format failure, tolerant JSON recovery, query-name echo, numeric-ID mapping, positive abstention, null hallucination, class-collapse, and label-map consistency. A deterministic cause gate will select exactly one v2.2 variable; no Task8 locked family is read and no new inference or training is started during forensics.

**Tech Stack:** Python 3.10+, pytest, JSON/JSONL, SHA256, existing Task9D artifacts.

## Global Constraints

- Read only Task9D train/val/dev-audit artifacts and Task9D inference outputs.
- Never read Task8 locked 186 families or run the confirmatory audit.
- Do not start Task9E, dynamic LoRA/Gating, SAM2, 7B, or a new backbone.
- Preserve all existing training, inference, and evaluation outputs.
- v2.2 changes exactly one causal variable; images, family split, model, LoRA, optimizer, steps, resolution, seeds, parser, and evaluation subset remain fixed.
- v2.2 remains a design until the forensic report identifies one dominant cause and its falsifiable acceptance gate.

---

### Task 1: Frozen-output forensic analyzer

**Files:**
- Create: `server/audit_task9d_outputs.py`
- Create: `tests/test_task9d_output_forensics.py`

**Interfaces:**
- Consumes: Task9D `manifest.jsonl`, ten `predictions.jsonl`, A/B/C `train_schedule.jsonl`, and `task9d_decision_report.json`.
- Produces: `forensics_report.json`, `forensics_cases.jsonl`, `run_summary.json`, and `completion.sha256`.

- [ ] **Step 1: Write failing tests for tolerant JSON extraction and failure taxonomy**

  Test fenced Base JSON recovery separately from strict schema validity, and classify missing/extra keys, invalid evidence region, invalid diagnosis keys, and reliability mismatch.

- [ ] **Step 2: Run the focused test and confirm RED**

  Run: `python -m pytest tests/test_task9d_output_forensics.py -q`

  Expected: FAIL because `audit_task9d_outputs` does not exist.

- [ ] **Step 3: Implement the minimum parser and taxonomy**

  Preserve strict metrics; tolerant extraction is secondary and uses the same deterministic extractor for every group.

- [ ] **Step 4: Add failing tests for positive behavior and class collapse**

  Require canonical-positive rates for evidence present, valid specific diagnosis, exact pest ID, normalized query-name echo, conditional ID accuracy given evidence, unique predicted IDs, top-1/top-5 share, and normalized entropy.

- [ ] **Step 5: Implement positive and collapse metrics**

  Extract queried names only from prompt text; label them explicitly as `query_name_echo`, never as visual diagnosis accuracy.

- [ ] **Step 6: Add failing tests for null behavior and label-map consistency**

  Require semantic/source-visual/blank/blur/shuffle refusal and concrete-diagnosis rates, wrong-query echo, and one-to-one query-name↔class-ID checks across manifest and train schedules.

- [ ] **Step 7: Implement null and label-map analysis**

  Report real null and synthetic null separately and retain representative opaque-ID cases without paths or filenames.

- [ ] **Step 8: Add failing tests for deterministic dominant-cause gating**

  Gate precedence:
  1. `label_map_corruption` if source targets are inconsistent;
  2. `numeric_id_generation_bottleneck` if source mapping is clean, query-name echo ≥80%, exact ID accuracy <10%, and predicted name→ID consistency <50%;
  3. `positive_over_abstention` if canonical-positive evidence-present rate <50%;
  4. `format_bottleneck` if tolerant recovery improves usable outputs by ≥20pp;
  5. `visual_or_objective_bottleneck` otherwise.

- [ ] **Step 9: Implement the gate and atomic artifact writer**

  Include input hashes, code hash, `task8_locked_set_read=false`, and output SHA256.

- [ ] **Step 10: Run focused and full tests**

  Run: `python -m pytest tests/test_task9d_output_forensics.py -q`

  Expected: PASS.

  Run: `python -m pytest -q`

  Expected: all tests pass with no warnings or errors.

### Task 2: Execute the read-only forensic audit

**Files:**
- Server output: `/root/autodl-tmp/EviAgriDiag/experiments/task9d/2026-07-16/evaluation/output_forensics_v1/`
- Local mirror: `artifacts/2026-07-16_task9/9D_output_forensics/`

**Interfaces:**
- Consumes: frozen Task9D artifacts whose completion SHA256 already passed.
- Produces: verified local forensic report and exact dominant-cause decision.

- [ ] **Step 1: Upload analyzer through a temporary path and verify SHA256**
- [ ] **Step 2: Run `py_compile` and a no-write input preflight**
- [ ] **Step 3: Execute once into a new output directory**
- [ ] **Step 4: Run `sha256sum -c completion.sha256`**
- [ ] **Step 5: Download report, cases, summary, and completion manifest**
- [ ] **Step 6: Verify downloaded report hashes locally**

### Task 3: Freeze one single-variable v2.2 validation design

**Files:**
- Create: `docs/task9d_v22_single_variable_design_2026-07-16.md`

**Interfaces:**
- Consumes: verified `forensics_report.json` and its `dominant_cause`.
- Produces: a preregistered two-arm micro experiment design; does not start it.

- [ ] **Step 1: Select one variable from the dominant-cause gate**

  Keep the current v2.1 arm as control. Change only the causal variable identified by the report.

- [ ] **Step 2: Freeze size and seeds**

  Use a class-balanced subset small enough for a short validation, the same families in both arms, and seeds 17/29/43. Do not tune on Task8.

- [ ] **Step 3: Freeze primary and guardrail metrics**

  Primary metric directly tests the suspected cause; guardrails retain canonical diagnosis, semantic/source-visual/blank/blur/shuffle behavior, four-level JSON validity, and seed variance.

- [ ] **Step 4: Freeze falsifiable pass/fail rules**

  Require a material paired improvement on the causal metric with no >3pp canonical diagnosis regression and no null-FPR/schema/task-compliance regression.

- [ ] **Step 5: State the next branch explicitly**

  PASS authorizes a larger Static QLoRA v2.2 ablation only; FAIL rejects that causal hypothesis and returns to forensics. Neither outcome authorizes Task9E or dynamic modules.

### Task 4: Research memory and handoff

**Files:**
- Create: `关键记忆/对话信息_2026_7_16/03_Task9D输出法医与v2.2设计.md`

- [ ] **Step 1: Record only the dominant cause, decisive numbers, v2.2 single variable, and prohibition state**
- [ ] **Step 2: Link the local forensic report and v2.2 design**
- [ ] **Step 3: Confirm the server remains on and no training was started**

## Self-Review

- Spec coverage: full-output forensics precedes v2.2 design; v2.2 is single-variable and minimal; no training is authorized here.
- Placeholder scan: no deferred implementation or unspecified gate remains.
- Type consistency: analyzer inputs/outputs and design inputs use the exact filenames defined above.

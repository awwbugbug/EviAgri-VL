# Task 10B Protocol Feasibility Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在加载Qwen或提取视觉特征前，验证冻结的Task 10B v1类别与样本配额是否可行；不可行时正式BLOCK并停止。

**Architecture:** 仅聚合既有JSON/JSONL元数据，不读取Task8 locked图像/标签，也不运行GPU。候选图像先限制为IP102 detection官方trainval，再排除Task8 locked ID/SHA与Task9D/v2.2使用过的source SHA，最后按冻结32类和`20/5/10`配额计算逐类可用数。

**Tech Stack:** Python 3.10、JSON/JSONL、SHA256、Git。

## Global Constraints

- 类别池固定为v2.2已冻结32类；head/medium/tail类别配额固定为`6/5/5`。
- 每类固定`20 train + 5 val + 10 dev`，即至少35个未排除source SHA。
- 图像只可来自IP102 detection官方trainval。
- Task9D/v2.2 train、val、dev使用过的source SHA全部排除。
- Task8 locked set只读取已有`locked_exclusion.json`中的ID/SHA边界，不读取图像、标签或预测。
- 任一band可行类别数不足时必须`BLOCKED_CLASS_QUOTA`；不得减少样本、扩大类别池或近似匹配。
- 本计划不加载模型、不提取特征、不训练分类器、不启动10C/10D。
- 正式BLOCK结论写入关键记忆并提交GitHub；配额修订必须另起版本并在观察特征/指标前冻结。

---

### Task 1: Read-only server feasibility audit

**Files:**
- Read: `/root/autodl-tmp/EviAgriDiag/datasets/derived/eviagridiag_detection_v1/vlm_sft/{train,val,test}_evidence_positive.jsonl`
- Read: `/root/autodl-tmp/EviAgriDiag/datasets/derived/static_qlora_v2_1_protocol/2026-07-15/private/provenance.jsonl`
- Read boundary only: `/root/autodl-tmp/EviAgriDiag/datasets/derived/static_qlora_v2_1_protocol/2026-07-15/private/locked_exclusion.json`
- Read: `/root/autodl-tmp/EviAgriDiag/experiments/task9d/2026-07-16/preparation/protocol/private/provenance.jsonl`
- Read: `/root/autodl-tmp/EviAgriDiag/experiments/task9d_v22_micro/2026-07-16/protocol/selected_classes.json`
- Read: `/root/autodl-tmp/EviAgriDiag/experiments/task9d/2026-07-16/preparation/protocol/class_bands.json`

**Interfaces:**
- Consumes: official split marker、source image ID/SHA、near-duplicate component ID、旧实验使用SHA、冻结class ID与band。
- Produces: 逐类`raw_trainval`、`source_sha_safe`和每band满足35张的类别数。

- [x] **Step 1: Confirm server and GPU without mutation**

Evidence: server online；RTX 4090 48GB；GPU utilization 0%；无Task9/10实验进程。

- [x] **Step 2: Confirm immutable input schemas and counts**

Evidence: official trainval positive images=`13,652 + 1,526`；official test=`3,798`；v2.1 provenance=`18,790` unique sources；Task9D used source SHA=`1,216`；frozen classes=`32`。

- [x] **Step 3: Apply the frozen exclusions and count eligible classes**

Observed eligible classes with `source_sha_safe >= 35`:

```json
{"head": 11, "medium": 11, "tail": 1}
```

Required:

```json
{"head": 6, "medium": 5, "tail": 5}
```

- [x] **Step 4: Verify that stricter component-level exclusion cannot rescue feasibility**

Evidence: excluding every component touching official test、Task8 locked或Task9D used source仍只有1个tail类达到35张，因此不存在通过加强隔离获得5个tail类的可能。

- [x] **Step 5: Apply the fail-closed decision**

Decision: `BLOCKED_CLASS_QUOTA`。不得加载Qwen、提取特征、训练逻辑回归或启动Task10C。

### Task 2: Record and publish the decision

**Files:**
- Create: `关键记忆/对话信息_2026_7_17/07_Task10B配额可行性BLOCK.md`
- Commit: this plan and the compact memory.

- [ ] **Step 1: Record exact blocker and scientifically valid amendment options**

The memory must contain the frozen requirement, observed availability, the no-GPU decision, and two pre-metric amendment options: retain the32-class pool but reduce per-class quota, or retain`20/5/10` while expanding the predeclared class pool beyond the frozen32. Neither option may be selected using feature performance.

- [ ] **Step 2: Validate repository diff**

Run: `git diff --check && git status --short`

Expected: only this plan and the new compact memory are modified/untracked.

- [ ] **Step 3: Commit and push**

Run:

```bash
git add docs/superpowers/plans/2026-07-17-task10b-linear-probe.md 关键记忆/对话信息_2026_7_17/07_Task10B配额可行性BLOCK.md
git commit -m "docs(task10): block infeasible Task 10B v1 quota"
git push
```

Expected: local `main` and `origin/main` have the same commit hash.

## Handoff Gate

Task 10B v1 is blocked before model access. A revised v2 protocol requires explicit user approval; after approval, write a separate feature-extraction/evaluation implementation plan and run a metadata-only exact-split preflight before any GPU work.

# Task 10B v2 Frozen-Vision Linear Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用320张严格隔离的IP102官方trainval图像，验证冻结Qwen2.5-VL-3B视觉特征能否支持16类害虫线性分类。

**Architecture:** 元数据构建器先完成排除、选类与exact split并签名；通过后，特征提取器以单图batch从Qwen视觉塔取得post-merge tokens，mean-pool并L2归一化；最后由CPU评估器运行三seed逻辑回归、无图对照、标签置换和family bootstrap。每层输出独立状态与SHA256，前层不通过时后层不得运行。

**Tech Stack:** Python 3.10、PyTorch、Transformers/Qwen2.5-VL、qwen-vl-utils、Pillow、NumPy、scikit-learn、pytest、Bash、Git。

## Global Constraints

- 类别池只使用v2.2冻结32类；选择head/medium/tail=`6/5/5`。
- 每类固定`12 train + 3 val + 5 dev`；总计`192/48/80=320`。
- 选类规则在特征前冻结：band内按官方trainval原始数量降序、class ID升序，且必须具有至少20个合格单类独立component。
- 每个selected component只取一个稳定哈希排序的代表图；同一component不得跨split。
- 仅用IP102 detection官方trainval；排除Task8 locked ID/SHA和Task9D/v2.2全部已用source SHA。
- Task8 locked set只读取已有ID/SHA边界；不读取其图像、标签或预测。
- Backbone固定现有官方`Qwen/Qwen2.5-VL-3B-Instruct`；所有参数`requires_grad=False`。
- processor固定`min_pixels=200704`、`max_pixels=401408`；单图batch。
- 特征固定为visual tower返回的post-merge tokens逐图mean pooling后L2归一化。
- 分类器固定`LogisticRegression(C=1.0,class_weight="balanced",max_iter=2000,solver="lbfgs",random_state=seed)`；seeds=`17/29/43`。
- 每seed增加训练标签置换对照与无图stratified-prior对照；不得使用val选择超参数、checkpoint或seed。
- 通过条件沿用Task10规格：mean Macro-F1≥25%、worst seed≥20%、pooled 95% CI下界>12.5%、permutation mean≤10%、source/component overlap=0。
- 任一输入不完整、哈希不一致、配额不足、非有限特征或输出目录已存在均fail-closed。
- 先8图engineering smoke，验证模型接口与特征契约；通过后才运行320图正式提取。
- 不启动10C/10D、Task9E、动态LoRA/Gating、SAM2、7B、新backbone或官方test评测；服务器不关机。

## File Structure

- Create `server/task10b_protocol.py`: 320图协议构建与签名。
- Create `server/extract_task10b_features.py`: 冻结Qwen视觉特征提取。
- Create `server/evaluate_task10b_probe.py`: 三seed探针、对照、bootstrap与决策。
- Create `server/run_task10b_v2.sh`: smoke和formal的fail-closed入口。
- Create `tests/test_task10b_protocol.py`: 配额、排除、选类、split、覆盖拒绝测试。
- Create `tests/test_task10b_features.py`: pooling、归一化、冻结与特征契约测试。
- Create `tests/test_task10b_evaluation.py`: 指标、对照、bootstrap与决策门槛测试。

---

### Task 1: Immutable v2 protocol

**Files:**
- Create: `server/task10b_protocol.py`
- Create: `tests/test_task10b_protocol.py`

**Interfaces:**
- `build_protocol(positive_rows, provenance_rows, used_sha256, locked_ids, locked_sha256, selected_classes, class_bands, split_quotas) -> dict`
- `write_protocol(result, output_root, input_paths) -> None`
- Produces `manifest.jsonl`, `selected_classes.json`, `protocol_report.json`, `input_sha256.json`, `config.snapshot.json`, `completion.sha256` only on PASS.

- [ ] Write tests first for official-trainval filtering, direct SHA/ID exclusions, mono-class component selection, deterministic class order, exact`192/48/80`, zero source/component overlap, quota BLOCK, and existing-output refusal.
- [ ] Run `pytest -q tests/test_task10b_protocol.py` and confirm RED because the module is absent.
- [ ] Implement only the tested metadata path; image bytes and model libraries must not be imported.
- [ ] Run `pytest -q tests/test_task10b_protocol.py tests/test_task10_audit_common.py` and confirm GREEN.
- [ ] Run `python -m py_compile server/task10b_protocol.py` and `git diff --check`.
- [ ] Commit with `feat(task10): freeze Task 10B v2 protocol`.

### Task 2: Frozen feature contract and 8-image smoke

**Files:**
- Create: `server/extract_task10b_features.py`
- Create: `tests/test_task10b_features.py`

**Interfaces:**
- `mean_pool_l2(tokens: torch.Tensor) -> torch.Tensor`
- `assert_frozen(model) -> None`
- `extract_features(manifest_path, model_path, output_root, limit=None) -> dict`
- Produces `features.npy`, `feature_rows.jsonl`, `config.snapshot.json`, `run_summary.json`, `completion.sha256`.

- [ ] Write tests first for 2D token input, exact mean pooling, unit norm, zero-vector rejection, non-finite rejection, and any trainable-parameter rejection.
- [ ] Run `pytest -q tests/test_task10b_features.py` and confirm RED because the module is absent.
- [ ] Implement pure pooling/freeze helpers and lazy imports for model-only dependencies.
- [ ] Run focused tests and confirm GREEN without loading the real model.
- [ ] Implement real single-image extraction using `AutoProcessor` and `Qwen2_5_VLForConditionalGeneration`; call `model.visual(pixel_values, grid_thw=image_grid_thw)` under `torch.inference_mode()` and reject output cardinality mismatches.
- [ ] Run local syntax tests, then sync committed code and run a new immutable8-image server smoke.
- [ ] Verify 8/8 finite unit-norm vectors、all parameters frozen、processor pixel bounds、model config/revision/hash记录与completion SHA256。
- [ ] Commit with `feat(task10): extract frozen Qwen visual features`.

### Task 3: Preregistered probe evaluation

**Files:**
- Create: `server/evaluate_task10b_probe.py`
- Create: `tests/test_task10b_evaluation.py`

**Interfaces:**
- `evaluate_seed(features, rows, seed) -> dict`
- `bootstrap_pooled_macro_f1(rows, predictions_by_seed, repetitions=1000, seed=20260717) -> dict`
- `decide_task10b(seed_metrics, bootstrap, overlap) -> dict`

- [ ] Write tests first for classifier parameters、deterministic label permutation、Accuracy/Macro-F1、band Macro-F1、无图对照、sample standard deviation、worst seed、1000次paired family bootstrap和每条PASS/FAIL门槛。
- [ ] Run `pytest -q tests/test_task10b_evaluation.py` and confirm RED because the module is absent.
- [ ] Implement the three fixed seeds over identical splits; decision reads dev only and val remains descriptive.
- [ ] Save per-seed predictions and metrics plus pooled report; reject feature/manifest SHA mismatch.
- [ ] Run focused tests, then full `pytest -q` and confirm all tests pass.
- [ ] Commit with `feat(task10): evaluate Task 10B linear probe`.

### Task 4: Formal server run and decision archival

**Files:**
- Create: `server/run_task10b_v2.sh`
- Create after result: `关键记忆/对话信息_2026_7_17/09_Task10B线性探针结果.md`

**Interfaces:**
- Consumes committed scripts and frozen local server assets.
- Produces `/root/autodl-tmp/EviAgriDiag/experiments/task10_micro_first/2026-07-17/task10b_v2/` and local `artifacts/2026-07-17_task10/10B_v2_linear_probe/`.

- [ ] Add shell contract tests proving existing outputs are refused and protocol/smoke failure prevents formal extraction.
- [ ] Sync only committed Task10B files to `/root/EviAgri-VL/server` and verify local/server SHA256 equality.
- [ ] Run protocol builder; verify exact class/split counts, zero overlaps, input hashes and completion hash.
- [ ] Run and verify 8-image smoke; only then start 320-image formal extraction.
- [ ] Run CPU probe evaluation and verify report hashes, three seeds, three controls and 1000 bootstrap repetitions.
- [ ] Download compact reports/hashes, verify locally, and write the concise key memory.
- [ ] Do not start Task10C regardless of PASS/FAIL; report result for user approval.
- [ ] Commit result memory, merge the feature branch after verification, push `main`, and remove the temporary worktree.

## Frozen v2 Selection Rule

Based only on pre-feature metadata, the expected classes are:

- head: `101, 24, 50, 16, 22, 45`
- medium: `68, 10, 99, 87, 71`
- tail: `82, 9, 64, 17, 83`

The builder must recompute and verify this list from immutable inputs rather than accepting it as an unchecked argument. Any mismatch is a protocol failure.

# Task 10C C2 64-Step Learning-Curve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建并执行一个三种子、单变量、64-step 的 diagnosis-only Static QLoRA 微实验，用相同 smoke-train 上的 8/16/32/64 checkpoints 判断规范输出、类别学习和图像依赖是否随训练量恢复。

**Architecture:** 保留 C1 代码和结果不变，新增 C2 专用训练、推理、候选条件似然和评测模块。训练每个 seed 只加载一次 Base 并连续运行 64 steps，checkpoint 仅保存 adapter；各 checkpoint 在同一 smoke-dev 上做四条件生成，最终 step 64 额外运行完整 dev、Base 同协议推理、候选字符串评分和冻结决策。所有阶段用 completion SHA256 串联，并在结果汇报后强制停止。

**Tech Stack:** Python 3.10、PyTorch 2.5.1、Transformers 4.51.3、PEFT 0.15.2、bitsandbytes 0.45.5、Qwen2.5-VL-3B-Instruct、pytest、Bash、SHA256。

## Global Constraints

- 规格来源：`docs/superpowers/specs/2026-07-17-task10c-c2-learning-curve-design.md`；冲突时该补充规格只覆盖上位规格的 C2 细节。
- manifest SHA256 必须为 `84d2d1b20d4a781bc6fca8c4e9c41dd57051b6201287193681409451095edc90`。
- 训练只使用 C1 已签名的同一 64-row `smoke_train.jsonl`；不得改用完整 192-row train。
- smoke-dev 固定 16 张；完整 dev 固定 80 张；source SHA 和 near-duplicate component 跨 split 重叠必须为 0。
- seeds 固定 `17,29,43`；每 seed 从同一 Base 独立训练，禁止加载或续跑 C1 adapter。
- 训练固定 64 optimizer steps、512 exposures；checkpoint 固定 `8,16,32,64`，不得早停、挑 checkpoint 或按结果调参。
- QLoRA、processor、loss、prompt、target、严格 parser 和 greedy 解码逐项复用 C1 冻结值。
- Task 8 locked set、official test、AGE、新数据、7B、新 backbone、动态 LoRA/Gating 和 SAM2 保持不可读、不可实现。
- 严格指标是唯一决策指标；格式法医和 conditional Top-k 只能解释失败，不得替代严格生成结果。
- 每个写入器拒绝非空既有输出；异常保留 `failure.json`，成功才写 `completion.sha256`；绝不关机。

---

### Task 1: 冻结 C2 训练与 checkpoint 契约

**Files:**
- Create: `server/task10c_c2_contract.py`
- Create: `tests/test_task10c_c2_contract.py`

**Interfaces:**
- Consumes: C1 `protocol/`、`task10c_contract.EXPECTED_MANIFEST_SHA256`、`train_task10c_smoke.verify_protocol_gate`。
- Produces: `C2_STEPS`、`C2_EXPOSURES`、`c2_training_arguments(seed)`、`verify_c2_protocol(protocol_root)`、`checkpoint_path(training_root, step)`、`validate_checkpoint_summary(summary, seed, step)`。

- [ ] **Step 1: 写失败测试，锁死单变量数据和四个 checkpoint**

```python
def test_c2_contract_uses_same_smoke_train_and_exact_schedule(tmp_path):
    protocol = frozen_protocol(tmp_path, smoke_train=64, smoke_dev=16)
    report = verify_c2_protocol(protocol)
    assert report["training_file"] == "smoke_train.jsonl"
    assert report["training_rows"] == 64
    assert C2_STEPS == (8, 16, 32, 64)
    assert C2_EXPOSURES == {8: 64, 16: 128, 32: 256, 64: 512}
    args = c2_training_arguments(17)
    assert args["max_steps"] == 64
    assert args["gradient_accumulation_steps"] == 8
    assert args["learning_rate"] == 1e-4
    assert args["lr_scheduler_type"] == "cosine"
    assert args["warmup_ratio"] == 0.03
    assert args["save_strategy"] == "no"


def test_c2_protocol_rejects_full_train_or_overlap(tmp_path):
    protocol = frozen_protocol(tmp_path, smoke_train=192, smoke_dev=16)
    with pytest.raises(ValueError, match="same 64-row smoke-train"):
        verify_c2_protocol(protocol)
```

- [ ] **Step 2: 运行测试并确认 RED**

Run: `python -m pytest tests/test_task10c_c2_contract.py -q`  
Expected: FAIL with `ModuleNotFoundError: No module named 'task10c_c2_contract'`.

- [ ] **Step 3: 实现冻结常量和协议门控**

```python
C2_STEPS = (8, 16, 32, 64)
C2_EXPOSURES = {step: step * 8 for step in C2_STEPS}


def c2_training_arguments(seed: int) -> dict[str, Any]:
    values = smoke_training_arguments(seed)
    values.update({
        "max_steps": 64,
        "gradient_accumulation_steps": 8,
        "eval_strategy": "no",
        "save_strategy": "no",
        "logging_steps": 1,
    })
    return values


def verify_c2_protocol(protocol_root: str | Path) -> dict[str, Any]:
    root = Path(protocol_root)
    gate = verify_protocol_gate(root)
    rows = _read_jsonl(root / "smoke_train.jsonl")
    dev = _read_jsonl(root / "smoke_dev.jsonl")
    if len(rows) != 64 or len(dev) != 16:
        raise ValueError("C2 requires the same 64-row smoke-train and 16-row smoke-dev")
    if Counter(int(row["class_id"]) for row in rows) != Counter({x: 4 for x in CLASS_IDS}):
        raise ValueError("C2 same 64-row smoke-train class quota mismatch")
    if set(row["source_image_sha256"] for row in rows) & set(row["source_image_sha256"] for row in dev):
        raise ValueError("C2 source SHA overlap")
    if set(row["near_duplicate_component_id"] for row in rows) & set(row["near_duplicate_component_id"] for row in dev):
        raise ValueError("C2 component overlap")
    return {**gate, "training_file": "smoke_train.jsonl", "training_rows": 64}
```

`validate_checkpoint_summary` 必须要求 seed 属于冻结集合、step 属于 `C2_STEPS`、exposures 等于 `C2_EXPOSURES[step]`、adapter SHA 为 64 位十六进制、loss/grad 全部有限、trainable names 只含安全 LoRA，并拒绝 `authorize_reuse=true`。

- [ ] **Step 4: 运行 Task 1 测试并确认 GREEN**

Run: `python -m pytest tests/test_task10c_c2_contract.py -q`  
Expected: all tests PASS.

- [ ] **Step 5: 提交 Task 1**

```bash
git add server/task10c_c2_contract.py tests/test_task10c_c2_contract.py
git commit -m "feat(task10): freeze C2 learning curve contract"
```

---

### Task 2: 实现一次连续训练与四个 adapter checkpoint

**Files:**
- Create: `server/train_task10c_c2.py`
- Create: `tests/test_train_task10c_c2.py`

**Interfaces:**
- Consumes: `verify_c2_protocol`、`c2_training_arguments`、`Task10CDataset`、`DiagnosisOnlyCollator`、`Task10CTrainer`、`build_task9d_model`。
- Produces: `save_c2_checkpoint(...)`、`C2CheckpointCallback`、`validate_c2_run_summary(summary)`、`run_c2_training(...)`。

- [ ] **Step 1: 写 checkpoint 原子保存和训练摘要失败测试**

```python
def test_checkpoint_writer_saves_exact_step_hash_and_refuses_overwrite(tmp_path):
    model = FakeAdapterModel(b"adapter-step-8")
    state = FakeState(global_step=8, log_history=[{"step": 8, "loss": 2.0, "grad_norm": 1.0}])
    result = save_c2_checkpoint(model, state, tmp_path, seed=17, step=8)
    assert result["optimizer_steps"] == 8
    assert result["actual_exposures"] == 64
    assert len(result["adapter"]["sha256"]) == 64
    assert (tmp_path / "checkpoints/step_008/completion.sha256").is_file()
    with pytest.raises(FileExistsError):
        save_c2_checkpoint(model, state, tmp_path, seed=17, step=8)


def test_run_summary_requires_all_four_checkpoints():
    summary = valid_c2_summary(steps=(8, 16, 32, 64))
    assert validate_c2_run_summary(summary)["passed"] is True
    summary["checkpoints"].pop("32")
    with pytest.raises(ValueError, match="checkpoint set"):
        validate_c2_run_summary(summary)
```

- [ ] **Step 2: 运行测试并确认 RED**

Run: `python -m pytest tests/test_train_task10c_c2.py -q`  
Expected: FAIL because `train_task10c_c2` does not exist.

- [ ] **Step 3: 实现 checkpoint callback**

```python
class C2CheckpointCallback(TrainerCallback):
    def __init__(self, training_root: Path, seed: int):
        self.training_root = Path(training_root)
        self.seed = seed
        self.saved: dict[int, dict[str, Any]] = {}

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = int(state.global_step)
        if step in C2_STEPS and step not in self.saved:
            self.saved[step] = save_c2_checkpoint(
                model=model,
                state=state,
                training_root=self.training_root,
                seed=self.seed,
                step=step,
            )
        return control
```

`save_c2_checkpoint` 使用临时目录 `step_XXX.tmp` 写 adapter、`adapter.sha256.json`、`checkpoint_summary.json`、`trainer_state.json`、`status.json` 和 `completion.sha256`；全部成功后用同文件系统原子 rename 为 `step_XXX`。adapter summary 必须记录 C2 protocol SHA、seed、step、exposures、loss reduction 和 `authorize_reuse=false`。

- [ ] **Step 4: 实现 64-step 单 seed 训练**

```python
dataset = Task10CDataset(protocol_root / "smoke_train.jsonl")
if len(dataset) != 64:
    raise ValueError("C2 must train on the same 64-row smoke-train")
arguments = c2_training_arguments(seed)
callback = C2CheckpointCallback(output, seed)
trainer = Task10CTrainer(
    model=model,
    args=TrainingArguments(**arguments),
    train_dataset=dataset,
    data_collator=collator,
    processing_class=processor,
    callbacks=[callback],
)
result = trainer.train()
if trainer.state.global_step != 64:
    raise ValueError("C2 must finish exactly 64 optimizer steps")
if set(callback.saved) != set(C2_STEPS):
    raise ValueError("C2 checkpoint set mismatch")
```

训练根目录保存 `config.snapshot.json`、`run_summary.json`、`trainer_state.json`、`status.json` 和 completion；摘要记录 512 exposures、完整 log history、四个 checkpoint SHA、peak VRAM、耗时、环境版本、协议门控和 `continued_from_c1=false`。异常写 failure 并退出，不删除已完成 checkpoint。

- [ ] **Step 5: 运行 Task 2 与相关回归测试**

Run: `python -m pytest tests/test_train_task10c_c2.py tests/test_task10c_training.py tests/test_train_task10c_smoke.py -q`  
Expected: all tests PASS.

- [ ] **Step 6: 提交 Task 2**

```bash
git add server/train_task10c_c2.py tests/test_train_task10c_c2.py
git commit -m "feat(task10): train C2 checkpoint curve"
```

---

### Task 3: 实现 smoke/full-dev 与 Base/adapter 的统一生成推理

**Files:**
- Create: `server/run_task10c_c2_inference.py`
- Create: `tests/test_task10c_c2_inference.py`

**Interfaces:**
- Consumes: C1 `generation_contract`、冻结 protocol、C2 checkpoint summary。
- Produces: `build_four_condition_rows(...)`、`build_c2_conditions(rows, split)`、`verify_c2_adapter(...)`、`run_c2_inference(...)`。

- [ ] **Step 1: 写统一条件矩阵和身份门控失败测试**

```python
def test_c2_conditions_are_exact_for_smoke_and_full_dev():
    smoke = build_c2_conditions(smoke_rows(), split="smoke_dev")
    full = build_c2_conditions(dev_rows(), split="dev")
    assert len(smoke) == 64
    assert len(full) == 320
    assert Counter(x["condition"] for x in full) == {name: 80 for name in CONDITIONS}
    assert all("source_image_id" not in json.dumps(x["messages"]) for x in full)


def test_adapter_gate_requires_declared_seed_and_checkpoint(tmp_path):
    checkpoint = signed_checkpoint(tmp_path, seed=17, step=16)
    assert verify_c2_adapter(checkpoint, seed=17, step=16)["optimizer_steps"] == 16
    with pytest.raises(ValueError, match="checkpoint step"):
        verify_c2_adapter(checkpoint, seed=17, step=32)
```

- [ ] **Step 2: 运行测试并确认 RED**

Run: `python -m pytest tests/test_task10c_c2_inference.py -q`  
Expected: FAIL because the C2 inference module does not exist.

- [ ] **Step 3: 实现条件构造和统一模型身份**

```python
def build_c2_conditions(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    expected_per_class = {"smoke_dev": 1, "dev": 5}[split]
    counts = Counter(int(row["class_id"]) for row in rows)
    if counts != Counter({class_id: expected_per_class for class_id in CLASS_IDS}):
        raise ValueError(f"C2 {split} class quota mismatch")
    return build_four_condition_rows(rows, CONDITIONS, TRAIN_PROMPT, UNSEEN_PROMPT)
```

Base 身份固定 `model_id="D0_base"`、`seed=None`、`checkpoint_step=0`；adapter 身份固定 `D1_seed_{seed}_step_{step:03d}`。adapter 模式必须先验证 checkpoint completion 和 adapter SHA；Base 模式不得接受 adapter 参数。

- [ ] **Step 4: 实现逐行 greedy 推理和完整写入**

`run_c2_inference` 参数固定为 `protocol_root, model_path, output_root, split, model_kind, seed=None, checkpoint_step=None, adapter_root=None`。processor、量化和 generation kwargs 逐字段等于 C1；每 16 行更新 status；prediction 保存 model identity、condition、GT、source SHA、component、raw text、严格 parsed 和 latency。成功摘要要求 smoke=64 或 dev=320、四条件均衡、模型/adapter/protocol SHA 完整；失败保留现场。

- [ ] **Step 5: 运行 Task 3 与 C1 推理回归测试**

Run: `python -m pytest tests/test_task10c_c2_inference.py tests/test_task10c_smoke_inference.py -q`  
Expected: all tests PASS.

- [ ] **Step 6: 提交 Task 3**

```bash
git add server/run_task10c_c2_inference.py tests/test_task10c_c2_inference.py
git commit -m "feat(task10): add C2 unified inference"
```

---

### Task 4: 实现候选 canonical 字符串条件似然 Top-k

**Files:**
- Create: `server/score_task10c_c2_candidates.py`
- Create: `tests/test_task10c_c2_candidates.py`

**Interfaces:**
- Consumes: 最终 D0 或 D1 step-64 模型、80-row dev 的两个有图 prompt。
- Produces: `candidate_targets()`、`mean_active_token_logprob(...)`、`rank_candidate_scores(...)`、`run_candidate_scoring(...)`。

- [ ] **Step 1: 写候选集、active-token reduction 和 Top-k 失败测试**

```python
def test_candidate_targets_are_exact_frozen_json():
    assert candidate_targets()[0] == '{"pest_id":"IP009"}'
    assert len(candidate_targets()) == 16
    assert all(strict_parse_pest_json(x)["schema_valid"] for x in candidate_targets())


def test_active_token_mean_excludes_prompt_and_padding():
    token_log_probs = torch.tensor([[-9.0, -9.0, -0.2, -0.4, -9.0]])
    active = torch.tensor([[False, False, True, True, False]])
    assert mean_active_token_logprob(token_log_probs, active).item() == pytest.approx(-0.3)


def test_rank_reports_top_1_3_5_without_generation():
    scores = {f"IP{x:03d}": -10.0 - index for index, x in enumerate(CLASS_IDS)}
    scores.update({"IP009": -0.1, "IP010": -0.2, "IP016": -0.3,
                   "IP017": -0.4, "IP022": -0.5})
    ranked = rank_candidate_scores(scores, truth="IP016")
    assert ranked["top1_correct"] is False
    assert ranked["top3_correct"] is True
    assert ranked["top5_correct"] is True
```

- [ ] **Step 2: 运行测试并确认 RED**

Run: `python -m pytest tests/test_task10c_c2_candidates.py -q`  
Expected: FAIL because the scorer does not exist.

- [ ] **Step 3: 实现冻结候选评分**

对每个有图 prompt 先构造 `add_generation_prompt=True` 的相同前缀，再追加 16 个合法紧凑 JSON target。按冻结 `candidate_batch_size=4` 分批前向；复制同一图像到批内四个文本实例。只聚合 answer suffix 的非 padding token：

```python
shift_logits = logits[:, :-1].log_softmax(dim=-1)
shift_labels = input_ids[:, 1:]
token_log_probs = shift_logits.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
active = answer_token_mask[:, 1:] & attention_mask[:, 1:].bool()
scores = mean_active_token_logprob(token_log_probs, active)
```

输出每个 source/prompt 的 16 个分数、排序、truth rank、Top-1/3/5；每个模型恰好 160 行。先运行 1 张图的 16-candidate resource preflight，记录耗时与 peak VRAM；batch=4 OOM、非有限分数或预计单模型耗时超过 30 分钟时写 `blocked` 并禁止宣称 C2 科学 PASS，不自动改变 batch 或省略指标。

- [ ] **Step 4: 运行 Task 4 测试并确认 GREEN**

Run: `python -m pytest tests/test_task10c_c2_candidates.py -q`  
Expected: all tests PASS.

- [ ] **Step 5: 提交 Task 4**

```bash
git add server/score_task10c_c2_candidates.py tests/test_task10c_c2_candidates.py
git commit -m "feat(task10): score C2 canonical candidates"
```

---

### Task 5: 实现学习曲线法医、配对 bootstrap 和冻结决策

**Files:**
- Create: `server/evaluate_task10c_c2.py`
- Create: `tests/test_evaluate_task10c_c2.py`

**Interfaces:**
- Consumes: 12 份 smoke checkpoint predictions、D0 与三 D1 的 full-dev predictions、四份 candidate scores、训练与协议 completion，以及已签名的 Task 10B v2 evaluation。
- Produces: `forensic_parse`、`forensic_metrics`、`condition_metrics`、`seed_learning_signal`、`aggregate_learning_signal`、`paired_source_bootstrap`、`pooled_source_bootstrap`、`decide_c2`、`run_c2_evaluation`。

- [ ] **Step 1: 写严格/法医分离和学习信号失败测试**

```python
def test_forensics_never_changes_strict_prediction():
    raw = '```json\n{"pest_id":"IP009"}\n```'
    strict = strict_parse_pest_json(raw)
    forensic = forensic_parse(raw)
    assert strict["schema_valid"] is False
    assert forensic["fence_stripped_schema_valid"] is True
    assert forensic["canonical_id_mentioned"] is True


def test_learning_signal_requires_two_metrics_in_two_seeds():
    curves = {
        17: curve(delta_schema=.30, delta_macro=.06, delta_gain=.00),
        29: curve(delta_schema=.30, delta_macro=.00, delta_gain=.06, final_gain=.06),
        43: curve(delta_schema=.10, delta_macro=.01, delta_gain=.01),
    }
    assert seed_learning_signal(curves[17]) is True
    assert seed_learning_signal(curves[29]) is True
    assert aggregate_learning_signal(curves)["passed_seed_count"] == 2
```

- [ ] **Step 2: 写 source 聚类 bootstrap 和九项 PASS 失败测试**

```python
def test_bootstrap_resamples_80_sources_not_prompt_or_seed_rows():
    base, models = paired_full_dev_rows(source_count=80, prompts=2, seeds=(17, 29, 43))
    result = pooled_source_bootstrap(base, models, repetitions=1000, seed=20260717)
    assert result["unit"] == "source_image_sha256"
    assert result["source_count"] == 80
    assert result["repetitions"] == 1000


@pytest.mark.parametrize("failed_gate", SCIENTIFIC_GATES)
def test_decision_fails_when_any_preregistered_gate_fails(failed_gate):
    evidence = passing_c2_evidence()
    evidence["gates"][failed_gate] = False
    assert decide_c2(evidence)["status"] != "PASS"
```

测试文件和实现共同冻结下列 gate 名称，顺序不得改变：

```python
SCIENTIFIC_GATES = (
    "d1_minus_d0_mean_macro_f1_ge_0_05",
    "pooled_paired_bootstrap_ci_low_gt_0",
    "at_least_two_seeds_above_d0",
    "d1_visual_gain_ge_0_10",
    "d1_mean_macro_f1_ge_0_5666020785",
    "every_seed_prompt_gap_lt_0_05",
    "every_seed_condition_syntax_schema_ge_0_99",
    "worst_seed_no_image_macro_f1_le_0_10",
    "source_and_component_overlap_eq_0",
)
```

- [ ] **Step 3: 运行测试并确认 RED**

Run: `python -m pytest tests/test_evaluate_task10c_c2.py -q`  
Expected: FAIL because the C2 evaluator does not exist.

- [ ] **Step 4: 实现严格指标、band/confusion 与非评分法医**

`condition_metrics` 固定 16 类分母并输出 syntax/schema、Accuracy、Macro-F1、head/medium/tail Macro-F1、confusion、unique IDs 和 parse failures。`forensic_metrics` 仅输出 fence rate、单层 fence stripped schema、canonical mention 和错误类别计数；不得返回替代 prediction。学习曲线必须验证每 seed 四 checkpoint、每 checkpoint 64 unique IDs、四条件各16，且所有 checkpoint 来自同一连续训练 summary。

评测器必须验证 Task 10B v2 `evaluation/completion.sha256`、`run_summary.state=completed`、`decision=PASS`、同一 protocol manifest，以及 `metrics.decision.mean_macro_f1=0.8094315406815407`；将其作为只读 `D2_reference` 写入报告，不重新训练或选择 D2。

- [ ] **Step 5: 实现 source-image paired bootstrap**

每次从排序后的 80 个 source SHA 有放回抽取 80 个；同一 source 的两个 image prompt 行绑定。per-seed delta 为该重采样下 D1 image Macro-F1 减 D0；pooled delta 先平均三个 seed D1 Macro-F1 再减 D0。固定 bootstrap seed=`20260717`，percentile 2.5/97.5，禁止把 prompt/seed 行作为独立 unit。

- [ ] **Step 6: 实现最终决策优先级**

```python
def decide_c2(evidence: Mapping[str, Any]) -> dict[str, Any]:
    if not evidence["engineering_complete"]:
        status = "ENGINEERING_FAILURE"
    elif all(evidence["gates"].values()):
        status = "PASS"
    elif evidence["learning_signal"]["passed_seed_count"] >= 2:
        status = "LEARNING_SIGNAL_ONLY"
    else:
        status = "STRUCTURAL_FAILURE"
    return {
        "status": status,
        "scientific_pass": status == "PASS",
        "authorize_larger_training": False,
        "authorize_next_experiment": False,
        "requires_user_review": True,
    }
```

九项 gate 逐项保存布尔值、观测值和阈值。缺少 candidate Top-k、completion、任一 seed/condition 或 bootstrap 时 `engineering_complete=false`，不得降级为可用科学结论。

- [ ] **Step 7: 运行 Task 5 与统计回归测试**

Run: `python -m pytest tests/test_evaluate_task10c_c2.py tests/test_evaluate_task10c_smoke.py tests/test_task8_statistics.py -q`  
Expected: all tests PASS.

- [ ] **Step 8: 提交 Task 5**

```bash
git add server/evaluate_task10c_c2.py tests/test_evaluate_task10c_c2.py
git commit -m "feat(task10): evaluate C2 learning curve"
```

---

### Task 6: 实现一次性 C2 编排与越界门控

**Files:**
- Create: `server/run_task10c_c2.sh`
- Create: `tests/test_task10c_c2_shell.py`

**Interfaces:**
- Consumes: C1 signed protocol 与 `PASS_C1_ENGINEERING` 报告、Tasks 1–5 CLI。
- Produces: 全新 `task10c_c2_learning_curve` 实验目录；不产生后续任务。

- [ ] **Step 1: 写 shell 矩阵与禁止项失败测试**

```python
def test_c2_shell_runs_frozen_matrix_and_stops_after_evaluation():
    text = Path("server/run_task10c_c2.sh").read_text(encoding="utf-8")
    assert "for seed in 17 29 43" in text
    assert "for step in 8 16 32 64" in text
    assert "train_task10c_c2.py" in text
    assert "run_task10c_c2_inference.py" in text
    assert "score_task10c_c2_candidates.py" in text
    assert "evaluate_task10c_c2.py" in text
    forbidden = ("task8", "official_test", "shutdown", "poweroff", "7b", "sam2", "task10d")
    assert not any(token in text.lower() for token in forbidden)
```

- [ ] **Step 2: 运行测试并确认 RED**

Run: `python -m pytest tests/test_task10c_c2_shell.py -q`  
Expected: FAIL because the shell script does not exist.

- [ ] **Step 3: 实现 fail-fast 一次性编排**

```bash
#!/usr/bin/env bash
set -euo pipefail
PY=/root/miniconda3/envs/eviagri/bin/python
CODE=/root/EviAgri-VL/server
C1=/root/autodl-tmp/EviAgriDiag/experiments/task10_micro_first/2026-07-17/task10c_c0_c1
ROOT=/root/autodl-tmp/EviAgriDiag/experiments/task10_micro_first/2026-07-17/task10c_c2_learning_curve
PROTOCOL="$C1/protocol"
MODEL=/root/autodl-tmp/EviAgriDiag/models/Qwen/Qwen2___5-VL-3B-Instruct
D2=/root/autodl-tmp/EviAgriDiag/experiments/task10_micro_first/2026-07-17/task10b_v2/evaluation
test ! -e "$ROOT"
test "$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' "$C1/evaluation/task10c_c1_decision_report.json")" = PASS_C1_ENGINEERING
mkdir -p "$ROOT"
for seed in 17 29 43; do
  "$PY" "$CODE/train_task10c_c2.py" --protocol-root "$PROTOCOL" --model-path "$MODEL" --experiment-root "$ROOT" --seed "$seed"
  for step in 8 16 32 64; do
    "$PY" "$CODE/run_task10c_c2_inference.py" --protocol-root "$PROTOCOL" --model-path "$MODEL" --split smoke_dev --model-kind adapter --seed "$seed" --checkpoint-step "$step" --adapter-root "$ROOT/training/seed_$seed/checkpoints/step_$(printf '%03d' "$step")" --output-root "$ROOT/inference/smoke/seed_$seed/step_$(printf '%03d' "$step")"
  done
  "$PY" "$CODE/run_task10c_c2_inference.py" --protocol-root "$PROTOCOL" --model-path "$MODEL" --split dev --model-kind adapter --seed "$seed" --checkpoint-step 64 --adapter-root "$ROOT/training/seed_$seed/checkpoints/step_064" --output-root "$ROOT/inference/dev/seed_$seed"
done
"$PY" "$CODE/run_task10c_c2_inference.py" --protocol-root "$PROTOCOL" --model-path "$MODEL" --split dev --model-kind base --output-root "$ROOT/inference/dev/base"

"$PY" "$CODE/score_task10c_c2_candidates.py" --protocol-root "$PROTOCOL" --model-path "$MODEL" --model-kind base --output-root "$ROOT/candidates/base"
for seed in 17 29 43; do
  "$PY" "$CODE/score_task10c_c2_candidates.py" --protocol-root "$PROTOCOL" --model-path "$MODEL" --model-kind adapter --seed "$seed" --checkpoint-step 64 --adapter-root "$ROOT/training/seed_$seed/checkpoints/step_064" --output-root "$ROOT/candidates/seed_$seed"
done

"$PY" "$CODE/evaluate_task10c_c2.py" --protocol-root "$PROTOCOL" --experiment-root "$ROOT" --task10b-evaluation-root "$D2" --output-root "$ROOT/evaluation" --repetitions 1000 --bootstrap-seed 20260717
```

实际 shell 在每个训练、checkpoint、推理、candidate 和 evaluation 命令后立即进入对应目录运行 `sha256sum -c completion.sha256`。任一失败由 `set -e` 停止，禁止循环重试、覆盖、删除或进入其他任务。

- [ ] **Step 4: 运行 shell 测试和全仓测试**

Run: `python -m pytest tests/test_task10c_c2_shell.py -q`  
Expected: PASS.  
Run: `python -m pytest -q`  
Expected: all repository tests PASS.

- [ ] **Step 5: 提交 Task 6**

```bash
git add server/run_task10c_c2.sh tests/test_task10c_c2_shell.py
git update-index --chmod=+x server/run_task10c_c2.sh
git commit -m "feat(task10): orchestrate C2 micro experiment"
```

---

### Task 7: 部署、执行、核验、归档与强制停止

**Files:**
- Create after result: `关键记忆/对话信息_2026_7_17/12_Task10C_C2结果.md`
- Download ignored artifacts to: `artifacts/2026-07-17_task10/10C_c2_learning_curve/`

**Interfaces:**
- Consumes: Tasks 1–6 的测试通过代码、现有 SSH key、用户已批准的 C2 规格。
- Produces: 服务器与本地双份校验结果、简短关键记忆、GitHub 提交；不启动任何后续实验。

- [ ] **Step 1: 在自建 worktree 做部署前验证**

Run: `python -m pytest -q`  
Expected: all tests PASS.  
Run: `git diff --check && git status --short`  
Expected: diff check exit 0；所有预期改动已提交。

- [ ] **Step 2: 同步新增 server 文件并逐文件核验 SHA256**

只同步 Tasks 1–6 新增的六个 server 文件到 `/root/EviAgri-VL/server/`。本地运行 `Get-FileHash -Algorithm SHA256`，服务器运行 `sha256sum`；六组值全部一致后才允许启动。只读核验 C1 protocol completion、C1 decision、模型文件 SHA、目标实验目录不存在和 GPU 空闲状态。

- [ ] **Step 3: 单次启动 C2**

```bash
screen -dmS task10c_c2 bash -lc '/root/EviAgri-VL/server/run_task10c_c2.sh > /root/autodl-tmp/EviAgriDiag/experiments/task10_micro_first/2026-07-17/task10c_c2_learning_curve.log 2>&1'
```

启动前确认不存在同名 screen 和目标目录；只启动一次。运行中只读检查当前 seed/step、finite loss、checkpoint/inference 计数、failure、进程和 GPU；异常不自行重跑。

- [ ] **Step 4: 完整性核验**

必须核对：3 个训练根目录；12 个 checkpoint；12 份 smoke inference（每份64）；3 份 D1 full-dev inference（每份320）；1 份 D0 full-dev inference（320）；4 份 candidate score（每份160）；1 份 evaluation。逐目录验证 completion SHA、adapter SHA、同一 protocol/model/decode contract、无 Task8/official-test 引用和无 failure。

- [ ] **Step 5: 下载到全新本地归档并再次验哈希**

目标固定 `artifacts/2026-07-17_task10/10C_c2_learning_curve/`，若已存在则 BLOCK。下载完成后用 PowerShell 按每份 `completion.sha256` 重算，并单独重算 12 个 adapter safetensors；任何不一致不得写完成结论。

- [ ] **Step 6: 写简短关键记忆并提交**

关键记忆只记录：数据/协议 SHA、训练与推理完整性、四 checkpoint 学习曲线、最终严格与法医指标、D0/D1/D2、bootstrap、九项 gates、最终四分流、资源耗时、归档路径和“未授权后续实验”。

```bash
git add 关键记忆/对话信息_2026_7_17/12_Task10C_C2结果.md
git commit -m "docs(memory): record Task 10C C2 decision"
```

- [ ] **Step 7: 完成前验证与 Git 收尾**

Run: `python -m pytest -q`  
Expected: all tests PASS.  
使用 `superpowers:verification-before-completion` 核对测试、哈希、Git diff、远端 HEAD 和服务器状态；使用 `superpowers:finishing-a-development-branch` 按用户选择完成 main 合并、主分支复测、GitHub 推送和自建 worktree 清理。服务器保持开机，GPU 空闲，最终只汇报 C2，不启动下一阶段。

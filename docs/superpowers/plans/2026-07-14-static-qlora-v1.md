# Static QLoRA v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在单张 RTX 4080 SUPER 32GB 上构建并验证 Qwen2.5-VL-3B-Instruct 的静态 QLoRA Evidence-First/null-evidence 训练链路，通过冒烟门槛后启动正式 1 epoch 训练。

**Architecture:** 将数据混合、Qwen 多模态 collator、4-bit LoRA 模型构建、训练门槛和生成评估分离为可独立测试的模块。使用 Transformers Trainer + PEFT + bitsandbytes，不引入 TRL/LLaMA-Factory。

**Tech Stack:** Python 3.12 conda env `eviagri`，PyTorch 2.5.1+cu121，Transformers 4.51.3，Accelerate 1.6.0，PEFT 0.15.2，bitsandbytes 0.45.5，Pillow，qwen-vl-utils。

## Global Constraints

- 设计来源：`docs/superpowers/specs/2026-07-14-static-qlora-v1-design.md`。
- 只读 raw 与 `eviagridiag_detection_v1` 不得修改。
- LoRA 只能注入 `model.layers.*.self_attn.{q_proj,k_proj,v_proj,o_proj}`，不得出现 `visual`、merger 或 projector 可训练参数。
- 训练混合为 train 13,652 正 + 6,826 null；val/test 保持原始 1:1。
- 仅 assistant JSON token 计算 loss；序列长度超过 1,024 时预检失败。
- 冒烟训练未通过全部门槛前，不启动正式训练。
- 当前工作区不是 Git 仓库；每个任务以测试输出、SHA256 和关键记忆代替 commit 审计。

---

### Task 1: 锁定训练依赖与配置

**Files:**
- Create: `server/requirements-training.txt`
- Create: `server/configs/static_qlora_v1.json`
- Test: `tests/test_static_qlora_config.py`

**Interfaces:**
- Consumes: 设计文档中的固定超参数。
- Produces: `load_training_config(path: Path) -> dict`，供数据、训练和评估脚本共用。

- [ ] **Step 1: 写失败测试**

```python
def test_static_config_freezes_approved_values():
    config = load_training_config(CONFIG_PATH)
    assert config["lora"] == {"r": 16, "alpha": 32, "dropout": 0.05}
    assert config["data"]["train_positive"] == 13652
    assert config["data"]["train_null"] == 6826
    assert config["training"]["gradient_accumulation_steps"] == 16
    assert config["training"]["num_train_epochs"] == 1
```

- [ ] **Step 2: 验证 RED**

Run: `python -m pytest tests/test_static_qlora_config.py -q`

Expected: FAIL，因 `server/static_qlora_config.py` 或配置文件不存在。

- [ ] **Step 3: 写最小实现**

`server/requirements-training.txt`:

```text
peft==0.15.2
bitsandbytes==0.45.5
```

`server/configs/static_qlora_v1.json` 必须包含以下核心值：

```json
{
  "seed": 20260714,
  "model_path": "/root/autodl-tmp/EviAgriDiag/models/Qwen/Qwen2___5-VL-3B-Instruct",
  "source_data_root": "/root/autodl-tmp/EviAgriDiag/datasets/derived/eviagridiag_detection_v1",
  "mixed_data_root": "/root/autodl-tmp/EviAgriDiag/datasets/derived/static_qlora_v1",
  "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
  "quantization": {"type": "nf4", "double_quant": true, "compute_dtype": "bfloat16"},
  "vision": {"min_pixels": 200704, "max_pixels": 401408},
  "data": {"train_positive": 13652, "train_null": 6826},
  "training": {
    "max_length": 1024,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "num_train_epochs": 1,
    "learning_rate": 0.0002,
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "logging_steps": 10,
    "eval_steps": 250,
    "save_steps": 250,
    "save_total_limit": 2
  }
}
```

`load_training_config` 检查必需键、数值范围与 2:1 训练比例，失败时抛 `ValueError`。

- [ ] **Step 4: 验证 GREEN**

Run: `python -m pytest tests/test_static_qlora_config.py -q`

Expected: PASS。

- [ ] **Step 5: 服务器依赖验收**

Run:

```bash
source /etc/network_turbo
/root/miniconda3/envs/eviagri/bin/pip install -r /root/EviAgri-VL/server/requirements-training.txt
/root/miniconda3/envs/eviagri/bin/python -c "import peft,bitsandbytes; print(peft.__version__, bitsandbytes.__version__)"
```

Expected: `0.15.2 0.45.5`，且 `transformers` 仍为 4.51.3。

---

### Task 2: 确定性构建训练混合数据

**Files:**
- Create: `server/build_static_qlora_mix.py`
- Test: `tests/test_build_static_qlora_mix.py`

**Interfaces:**
- Consumes: `load_training_config(path) -> dict`，源 JSONL 记录。
- Produces: `build_mix(config: dict, output_root: Path) -> dict`；输出 `train.jsonl`、`val.jsonl`、`test.jsonl`、`manifest.json`、`sha256sum.txt`。

- [ ] **Step 1: 写失败测试**

```python
def test_build_mix_selects_stable_two_to_one_train_ratio(tmp_path):
    source = make_fixture_source(tmp_path, train_positive=4, train_null=4)
    summary = build_mix(fixture_config(source, positive=4, null=2), tmp_path / "out")
    assert summary["counts"]["train"] == {"positive": 4, "null": 2, "total": 6}
    first = (tmp_path / "out" / "train.jsonl").read_bytes()
    shutil.rmtree(tmp_path / "out")
    build_mix(fixture_config(source, positive=4, null=2), tmp_path / "out")
    assert (tmp_path / "out" / "train.jsonl").read_bytes() == first
```

还必须覆盖：重复 ID 失败、图像不存在失败、目标目录非空拒绝覆盖。

- [ ] **Step 2: 验证 RED**

Run: `python -m pytest tests/test_build_static_qlora_mix.py -q`

Expected: FAIL with `ModuleNotFoundError: build_static_qlora_mix`。

- [ ] **Step 3: 写最小实现**

```python
def stable_rank(seed: int, record_id: str) -> str:
    return hashlib.sha256(f"{seed}:{record_id}".encode()).hexdigest()

def choose_null(records: list[dict], count: int, seed: int) -> list[dict]:
    return sorted(records, key=lambda row: stable_rank(seed, row["id"]))[:count]

def build_mix(config: dict, output_root: Path) -> dict:
    source = Path(config["source_data_root"])
    positive = read_jsonl(source / "vlm_sft/train_evidence_positive.jsonl")
    null = read_jsonl(source / "hallucination/train_prompt_conflict.jsonl")
    train = positive + choose_null(null, config["data"]["train_null"], config["seed"])
    train.sort(key=lambda row: stable_rank(config["seed"] + 1, row["id"]))
    val = read_jsonl(source / "vlm_sft/val_evidence_positive.jsonl") + read_jsonl(
        source / "hallucination/val_prompt_conflict.jsonl"
    )
    test = read_jsonl(source / "vlm_sft/test_evidence_positive.jsonl") + read_jsonl(
        source / "hallucination/test_prompt_conflict.jsonl"
    )
    validate_unique_existing_images(train + val + test)
    summary = write_atomic_bundle(
        output_root,
        {"train": train, "val": val, "test": test},
        source_files=list(source.rglob("*.jsonl")),
    )
    return summary
```

写文件前完成所有校验；使用临时目录构建，成功后原子改名为目标目录。

- [ ] **Step 4: 验证 GREEN 与全量构建**

Run:

```bash
python -m pytest tests/test_build_static_qlora_mix.py -q
/root/miniconda3/envs/eviagri/bin/python /root/EviAgri-VL/server/build_static_qlora_mix.py \
  --config /root/EviAgri-VL/server/configs/static_qlora_v1.json
```

Expected: train=20,478，val=3,052，test=7,596，所有 ID 唯一，图像路径全存在。

---

### Task 3: Qwen 多模态 Dataset、collator 与预检

**Files:**
- Create: `server/static_qlora_data.py`
- Test: `tests/test_static_qlora_data.py`

**Interfaces:**
- Consumes: 混合 JSONL，`AutoProcessor`。
- Produces: `JsonlDataset`，`AssistantOnlyVisionCollator`，`preflight_dataset(dataset, collator, max_length) -> dict`。

- [ ] **Step 1: 写失败测试**

```python
def test_collator_masks_user_padding_and_visual_tokens():
    batch = collator([fixture_record()])
    labels = batch["labels"][0]
    assert torch.all(labels[: collator.last_prefix_length] == -100)
    assert torch.any(labels[collator.last_prefix_length :] != -100)
    assert torch.all(labels[batch["attention_mask"][0] == 0] == -100)

def test_preflight_rejects_overlength_sample():
    with pytest.raises(ValueError, match="max_length"):
        preflight_dataset(dataset, overlength_collator, max_length=16)
```

测试使用假 processor，不加载真模型；还覆盖 assistant mask 全空时失败。

- [ ] **Step 2: 验证 RED**

Run: `python -m pytest tests/test_static_qlora_data.py -q`

Expected: FAIL with `ModuleNotFoundError: static_qlora_data`。

- [ ] **Step 3: 写最小实现**

```python
class JsonlDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path):
        self.records = [json.loads(line) for line in path.open(encoding="utf-8")]
    def __len__(self): return len(self.records)
    def __getitem__(self, index): return self.records[index]

class AssistantOnlyVisionCollator:
    def __call__(self, records):
        if len(records) != 1:
            raise ValueError("static_qlora_v1 requires per-device batch size 1")
        messages = records[0]["messages"]
        prefix = self.processor.apply_chat_template(
            messages[:1], tokenize=False, add_generation_prompt=True
        )
        full = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(messages)
        common = {
            "images": image_inputs,
            "videos": video_inputs,
            "padding": True,
            "return_tensors": "pt",
        }
        model_inputs = self.processor(text=[full], **common)
        prefix_inputs = self.processor(text=[prefix], **common)
        labels = model_inputs["input_ids"].clone()
        prefix_length = prefix_inputs["input_ids"].shape[1]
        labels[:, :prefix_length] = -100
        labels[model_inputs["attention_mask"] == 0] = -100
        for token_id in self.visual_token_ids:
            labels[labels == token_id] = -100
        if not torch.any(labels != -100):
            raise ValueError(f"empty assistant loss mask: {records[0]['id']}")
        model_inputs["labels"] = labels
        self.last_prefix_length = prefix_length
        return model_inputs
```

`preflight_dataset` 遍历全部 train/val，返回 max/mean token length、assistant token 范围与任何失败 ID；有失败则不返回成功状态。

- [ ] **Step 4: 验证 GREEN**

Run: `python -m pytest tests/test_static_qlora_data.py -q`

Expected: PASS。

---

### Task 4: 4-bit 模型、精确 LoRA 注入与 Trainer

**Files:**
- Create: `server/static_qlora_model.py`
- Create: `server/train_static_qlora.py`
- Test: `tests/test_static_qlora_model.py`
- Test: `tests/test_train_static_qlora.py`

**Interfaces:**
- Consumes: validated config，Dataset/collator。
- Produces: `language_attention_targets(model) -> list[str]`，`build_qlora_model(config) -> PeftModel`，`run_training(config, mode) -> dict`。

- [ ] **Step 1: 写 LoRA 目标失败测试**

```python
def test_language_attention_targets_exclude_visual_modules():
    names = language_attention_targets(FakeQwen())
    assert names == [
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.v_proj",
    ]
    assert all("visual" not in name for name in names)
```

还覆盖“未找到语言目标时失败”。

- [ ] **Step 2: 验证 RED**

Run: `python -m pytest tests/test_static_qlora_model.py tests/test_train_static_qlora.py -q`

Expected: FAIL，目标模块不存在。

- [ ] **Step 3: 实现精确模型构建**

```python
LANGUAGE_ATTN = re.compile(
    r"^model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$"
)

def language_attention_targets(model):
    targets = sorted(name for name, _ in model.named_modules() if LANGUAGE_ATTN.fullmatch(name))
    if not targets or any("visual" in name for name in targets):
        raise RuntimeError("unsafe or empty LoRA target set")
    return targets
```

`build_qlora_model` 使用：

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

然后 `prepare_model_for_kbit_training`、精确 target list 的 `LoraConfig`、`get_peft_model`。建模后遍历 `named_parameters()`，断言所有 `requires_grad=True` 参数名均包含 `lora_`，且不含 `visual`。

- [ ] **Step 4: 实现 Trainer CLI**

```python
def run_training(config: dict, mode: str) -> dict:
    paths = resolve_mode_paths(config, mode)
    ensure_empty_output(paths.output_dir)
    processor = build_processor(config)
    collator = AssistantOnlyVisionCollator(processor, max_length=config["training"]["max_length"])
    train_dataset = JsonlDataset(paths.train_jsonl)
    eval_dataset = JsonlDataset(paths.val_jsonl)
    if mode == "smoke":
        train_dataset = deterministic_smoke_subset(train_dataset, positive=24, null=8)
    preflight = preflight_dataset(train_dataset, collator, config["training"]["max_length"])
    model, trainable = build_qlora_model(config)
    arguments = build_training_arguments(config, mode, paths.output_dir)
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    trainer.save_model(paths.output_dir / "adapter")
    run_summary = write_run_evidence(
        paths.output_dir,
        result=result,
        preflight=preflight,
        trainable=trainable,
        peak_vram_bytes=torch.cuda.max_memory_allocated(),
    )
    return run_summary
```

输出目录安全约束：非空则失败；只有 `--resume-from-checkpoint` 显式指定时才恢复。

- [ ] **Step 5: 验证 GREEN**

Run:

```bash
python -m pytest tests/test_static_qlora_model.py tests/test_train_static_qlora.py -q
python -m pytest -q
```

Expected: 新测试与现有全部测试 PASS。

---

### Task 5: 冒烟训练与六项门槛

**Files:**
- Create: `server/validate_static_qlora_smoke.py`
- Test: `tests/test_validate_static_qlora_smoke.py`

**Interfaces:**
- Consumes: smoke 输出目录、adapter、日志和显存报告。
- Produces: `smoke_gate.json`，具有 `passed: bool` 与六个独立 gate 结果。

- [ ] **Step 1: 写失败测试**

```python
def test_gate_rejects_nonfinite_loss_and_excess_vram(tmp_path):
    fixture = smoke_fixture(tmp_path, losses=[2.1, float("nan")], peak_vram_gb=30.1)
    report = validate_smoke(fixture)
    assert report["passed"] is False
    assert report["gates"]["finite_loss"] is False
    assert report["gates"]["peak_vram_below_30gb"] is False
```

- [ ] **Step 2: 验证 RED**

Run: `python -m pytest tests/test_validate_static_qlora_smoke.py -q`

Expected: FAIL with `ModuleNotFoundError`。

- [ ] **Step 3: 实现 gate 验证器**

```python
def validate_smoke(output_dir: Path) -> dict:
    gates = {
        "dependencies_and_4bit_load": check_environment(output_dir),
        "language_only_lora": check_trainable_parameters(output_dir),
        "finite_loss": check_losses(output_dir, expected_steps=2),
        "peak_vram_below_30gb": read_peak_vram(output_dir) < 30.0,
        "adapter_reload": check_adapter_reload(output_dir),
        "positive_and_null_json_generation": check_generation(output_dir),
    }
    return {"passed": all(gates.values()), "gates": gates}
```

- [ ] **Step 4: 验证 GREEN**

Run: `python -m pytest tests/test_validate_static_qlora_smoke.py -q`

Expected: PASS。

- [ ] **Step 5: 在服务器执行真实冒烟**

Run:

```bash
/root/miniconda3/envs/eviagri/bin/python /root/EviAgri-VL/server/train_static_qlora.py \
  --config /root/EviAgri-VL/server/configs/static_qlora_v1.json \
  --mode smoke
/root/miniconda3/envs/eviagri/bin/python /root/EviAgri-VL/server/validate_static_qlora_smoke.py \
  --output-dir /root/autodl-tmp/EviAgriDiag/experiments/static_qlora_v1/smoke
```

Expected: `smoke_gate.json` 中 `passed=true`。

---

### Task 6: 启动正式训练并建立零 Token 监控

**Files:**
- Create: `server/run_static_qlora_formal.sh`
- Create: `server/check_static_qlora_status.sh`
- Test: `tests/test_static_qlora_shell.py`

**Interfaces:**
- Consumes: `smoke_gate.json` 和正式训练 CLI。
- Produces: `screen: static_qlora_v1`、`formal/train.log`、`formal/status.json`、checkpoints。

- [ ] **Step 1: 写失败测试**

```python
def test_formal_launcher_requires_passed_smoke_gate():
    text = LAUNCHER.read_text()
    assert "smoke_gate.json" in text
    assert "\"passed\": true" in text
    assert "screen -dmS static_qlora_v1" in text
```

- [ ] **Step 2: 验证 RED**

Run: `python -m pytest tests/test_static_qlora_shell.py -q`

Expected: FAIL，脚本不存在。

- [ ] **Step 3: 实现启动器与状态检查**

`run_static_qlora_formal.sh` 使用 `set -euo pipefail`，先检查 smoke gate、磁盘空间≥20GB、GPU 空闲、formal 目录为空，再启动：

```bash
screen -L -Logfile "$FORMAL_DIR/train.log" -dmS static_qlora_v1 \
  /root/miniconda3/envs/eviagri/bin/python /root/EviAgri-VL/server/train_static_qlora.py \
  --config /root/EviAgri-VL/server/configs/static_qlora_v1.json --mode formal
```

`check_static_qlora_status.sh` 只读输出 screen 存活、最新 step/loss、GPU 显存、checkpoint 和完成/失败标记。

- [ ] **Step 4: 验证 GREEN**

Run:

```bash
python -m pytest tests/test_static_qlora_shell.py -q
bash -n server/run_static_qlora_formal.sh
bash -n server/check_static_qlora_status.sh
```

Expected: PASS，两个 shell 脚本语法正确。

- [ ] **Step 5: 启动正式训练**

Run: `bash /root/EviAgri-VL/server/run_static_qlora_formal.sh`

Expected: `screen -ls` 显示 `static_qlora_v1 (Detached)`，日志中开始产生有限 loss。

---

### Task 7: 生成评估与实验归档

**Files:**
- Create: `server/evaluate_static_qlora.py`
- Test: `tests/test_evaluate_static_qlora.py`
- Create: `关键记忆/对话信息_2026_7_14/02_Static_QLoRA_v1启动.md`

**Interfaces:**
- Consumes: formal adapter/checkpoint 和 val/test JSONL。
- Produces: `predictions.jsonl`、`metrics.json`、`failures.jsonl`、本地归档与简短关键记忆。

- [ ] **Step 1: 写指标失败测试**

```python
def test_metrics_separate_positive_and_null_behavior():
    metrics = compute_metrics(fixture_predictions())
    assert metrics["schema_valid_rate"] == 0.75
    assert metrics["evidence_presence_f1"] == pytest.approx(2 / 3)
    assert metrics["positive"]["diagnosis_accuracy"] == 0.5
    assert metrics["null"]["false_positive_rate"] == 0.5
    assert "mean_iou" in metrics["positive"]
    assert "pointing_game" in metrics["positive"]
```

- [ ] **Step 2: 验证 RED**

Run: `python -m pytest tests/test_evaluate_static_qlora.py -q`

Expected: FAIL with `ModuleNotFoundError`。

- [ ] **Step 3: 实现指标与生成器**

```python
def bbox_iou(pred, truth):
    x1, y1 = max(pred[0], truth[0]), max(pred[1], truth[1])
    x2, y2 = min(pred[2], truth[2]), min(pred[3], truth[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    pred_area = max(0, pred[2] - pred[0]) * max(0, pred[3] - pred[1])
    truth_area = max(0, truth[2] - truth[0]) * max(0, truth[3] - truth[1])
    union = pred_area + truth_area - intersection
    return intersection / union if union else 0.0

def pointing_game(pred, truth):
    cx, cy = (pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2
    return float(truth[0] <= cx <= truth[2] and truth[1] <= cy <= truth[3])

def parse_structured_json(text):
    cleaned = text.strip().removeprefix("```json").removesuffix("```").strip()
    value = json.loads(cleaned)
    expected = ["evidence_present", "evidence_bbox", "visible_attributes", "diagnosis", "reliability"]
    if not isinstance(value, dict) or list(value) != expected:
        raise ValueError("invalid Evidence-First schema")
    return value

def compute_metrics(rows):
    parsed, failures = [], []
    for row in rows:
        try:
            prediction = parse_structured_json(row["prediction"])
            parsed.append((row, prediction))
        except (ValueError, json.JSONDecodeError) as error:
            failures.append({"id": row["id"], "error": str(error)})
    presence = binary_presence_counts(parsed)
    positive = positive_metrics(parsed, bbox_iou=bbox_iou, pointing_game=pointing_game)
    null = null_metrics(parsed)
    return {
        "schema_valid_rate": len(parsed) / len(rows) if rows else 0.0,
        "evidence_presence_precision": presence.precision,
        "evidence_presence_recall": presence.recall,
        "evidence_presence_f1": presence.f1,
        "positive": positive,
        "null": null,
        "reliability_accuracy": reliability_accuracy(parsed),
        "parse_failures": failures,
    }
```

生成使用确定性解码 `do_sample=false`，保存原始文本与解析错误；不用 LLM-as-judge。

- [ ] **Step 4: 验证 GREEN 与归档**

Run:

```bash
python -m pytest tests/test_evaluate_static_qlora.py -q
python -m pytest -q
```

Expected: 全部测试 PASS。正式训练完成后执行 val/test 评估，将配置、日志、指标、预测和环境版本下载至本地 `artifacts/2026-07-14_static_qlora_v1/`。

---

## Execution Order and Checkpoints

1. Task 1–2 完成后：检查依赖未升级 Transformers，混合数据计数/SHA256 正确。
2. Task 3–4 完成后：全测试通过，全量 train/val 预检通过。
3. Task 5 完成后：六项 smoke gate 全为 true，才进入 Task 6。
4. Task 6 启动后：记录启动时间、PID/screen、首个有限 loss 和显存。
5. Task 7 仅在正式 adapter 成功保存后执行，不提前宣称性能改善。

# Static QLoRA v1 实验设计

## 1. 目标

在单张 RTX 4080 SUPER 32GB 上，对 Qwen2.5-VL-3B-Instruct 进行首轮静态 QLoRA，验证 `Evidence-First + null-evidence` 结构化训练链路。该轮是后续 Evidence Adapter、Verifier 和条件化 LoRA gate 的静态基线。

## 2. 范围边界

- 使用现有 `eviagridiag-detection-v1`，不修改只读 raw 数据。
- 仅适配语言模型自注意力层；冻结视觉塔、multimodal merger/projector 和其他基座权重。
- 不引入 SAM2、Grad-CAM、Evidence Adapter、Verifier、动态 gate 或 Age 虫态辅助数据。
- 先通过冒烟门槛，再启动正式 1 epoch 训练；任一门槛失败时停止。

## 3. 数据设计

### 3.1 源数据

- 正样本：`vlm_sft/{split}_evidence_positive.jsonl`
- null 样本：`hallucination/{split}_prompt_conflict.jsonl`
- 服务器根目录：`/root/autodl-tmp/EviAgriDiag/datasets/derived/eviagridiag_detection_v1`

### 3.2 训练混合

- train：保留全部 13,652 条正样本，使用稳定 SHA256 排序从 13,652 条 null 中选取 6,826 条，共 20,478 条，正/null=2:1。
- val：1,526 正 + 1,526 null，保持 1:1，用于可靠性评估。
- test：3,798 正 + 3,798 null，保持 1:1，仅用于最终评估。
- 样本顺序使用固定 seed `20260714` 打乱；不跨原生 split 移动图像。
- 混合结果写入新的 derived 目录，并生成计数、唯一 ID、源文件 SHA256 和输出 SHA256 清单。

## 4. 模型与 QLoRA

- 基座模型：`/root/autodl-tmp/EviAgriDiag/models/Qwen/Qwen2___5-VL-3B-Instruct`
- 量化：4-bit NF4，double quantization，BF16 compute dtype。
- LoRA：`r=16`，`alpha=32`，`dropout=0.05`，bias=`none`。
- 精确目标：`model.layers.*.self_attn.{q_proj,k_proj,v_proj,o_proj}`。必须在训练前打印所有可训练参数名，并断言不包含 `visual` 或 multimodal merger/projector。
- 使用 `prepare_model_for_kbit_training`，启用 gradient checkpointing，关闭 `use_cache`。
- 只新增并锁定与当前 Transformers 4.51.3 兼容的 `peft` 和 `bitsandbytes`；不引入 TRL/LLaMA-Factory 这类额外训练框架。

## 5. 预处理与 loss

- Qwen 视觉处理范围：`min_pixels=256×28×28`，`max_pixels=512×28×28`。
- 最大序列长度：1,024 tokens；超限样本记录并在预检阶段失败，不静默截断证据输出。
- 仅对 assistant JSON 计算 language-model loss；user prompt、padding 和视觉特殊 token 全设为 `-100`。
- 一个 batch 中图像与文本由 `AutoProcessor` 和 `qwen_vl_utils.process_vision_info` 统一处理。
- 预检必须确认：JSON 可解析、图像存在、assistant mask 非空、每条序列不超限。

## 6. 正式训练超参数

- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=16`，有效 batch=16
- `num_train_epochs=1`
- `learning_rate=2e-4`
- `lr_scheduler_type=cosine`
- `warmup_ratio=0.03`
- `weight_decay=0.01`
- `max_grad_norm=1.0`
- optimizer：`paged_adamw_8bit`
- BF16=true，TF32=true
- `logging_steps=10`，`eval_steps=250`，`save_steps=250`，`save_total_limit=2`
- seed/data_seed=`20260714`
- 数据加载工作线程=2

## 7. 冒烟门槛

冒烟训练使用 24 正 + 8 null 的 32 条固定样本，`max_steps=2`，`gradient_accumulation_steps=2`。只有同时满足以下条件才可启动正式训练：

1. 依赖导入与 4-bit 模型加载成功；
2. 可训练参数仅来自指定的语言自注意力 LoRA；
3. 两个 optimizer step 的 loss 均为有限数；
4. 峰值 GPU 显存低于 30GB；
5. adapter 可保存、重新加载；
6. 至少 1 条正样本和 1 条 null 样本可生成并解析为固定 JSON schema。

## 8. 评估协议

- JSON Schema Valid Rate
- Evidence Presence precision/recall/F1
- 正样本害虫诊断 accuracy 和 macro-F1
- bbox Valid Rate、mean IoU、IoU@0.5、Pointing Game
- null 样本 false-positive rate / Evidence-Bound Hallucination Rate
- reliability accuracy
- 报告正样本与 null 样本的分组指标，不以 exact-string match 作为主指标。

## 9. 输出与恢复

- 代码：`/root/EviAgri-VL/server/`
- 混合数据：`/root/autodl-tmp/EviAgriDiag/datasets/derived/static_qlora_v1/`
- 冒烟：`/root/autodl-tmp/EviAgriDiag/experiments/static_qlora_v1/smoke/`
- 正式：`/root/autodl-tmp/EviAgriDiag/experiments/static_qlora_v1/formal/`
- 启动前若目标目录非空则拒绝覆盖；恢复时必须显式指定 checkpoint。
- 正式训练使用独立 `screen` 会话，保存训练日志、环境版本、数据清单、配置、GPU 峰值和 checkpoint。

## 10. 成功边界

本设计的当日成功条件是：数据混合与预检通过，冒烟训练通过全部门槛，然后可审计地启动正式 static QLoRA v1。未完成正式评估前，不宣称模型性能改善。

# Task 9D A/B/C 多种子小规模消融设计

## 目标与边界

在不读取 Task 8 `locked_confirmatory_set` 的前提下，仅使用 v2.1 的 train、val 和 `task9_dev_audit`，选择一个可进入 Task 9E 的 Static QLoRA 数据协议。9D 不运行全量训练，不实现动态 LoRA/Gating、SAM2、7B 或新 backbone。

## A/B/C 唯一变量

| 组 | 训练角色 | 训练提示 |
|---|---|---|
| A | positive + 每 family 一个精确匹配的原有 semantic null | 单一统一中性模板，正负完全一致 |
| B | A 的相同 positive/semantic 配对 + 每 family 一个轮换视觉反事实 | 与 A 相同的单一中性模板 |
| C | 与 B 完全相同的三角色 | 三个训练模板按 family 均衡轮换；同一 family 三角色使用同一模板，正负模板分布严格相同 |

“B + 成对 semantic wrong-query”不生成第二条 semantic-negative；它把 A 中同一条精确匹配 semantic null 作为 paired wrong-query 沿用，并显式记录 real/synthetic null。由此 A→B 只增加视觉反事实监督，B→C 只增加正负对称多模板。

## 冻结数据与预算

- 从 v2.1 train 中按训练类频次分层确定同一组 512 families，A/B/C 和所有 seeds 共用。
- 从 val 中固定 192 families，仅用于训练损失监控；不据 val 或 dev 选择 seed checkpoint。
- 从 `task9_dev_audit` 固定 512 families，A/B/C、Base 和所有提示条件共用。
- 另从上述 dev families 固定 128 families，构造 9D-only blank/blur/shuffle 条件：blank 颜色避开 Task 8 的 RGB(127,127,127)，blur 半径避开训练和 Task 8 参数，shuffle 使用 held-out patch grid；不使用 Task 8 图片或预测。
- Head/medium/tail 按完整 v2.1 train 的 positive 类频次三分位冻结，不依据 dev 结果修改。
- 三个训练 seeds 固定为 `17, 29, 43`。

每 run 使用 192 optimizer steps、batch size 1、gradient accumulation 8。B/C 恰好覆盖 512 families 的三角色各一次；A 的 1,024 条记录按固定 seed 循环至相同的 1,536 个 micro-samples。所有 run 的 optimizer steps 和 micro-sample 总量一致，并报告各角色实际曝光次数。

## 模型与训练协议

- Base：本地 Qwen2.5-VL-3B-Instruct，同一冻结模型 SHA。
- Static QLoRA：NF4 double quantization、bf16，仅语言 self-attention `q_proj/v_proj`；视觉模块及其他模块保持冻结。
- LoRA：r=16、alpha=32、dropout=0.05。
- 优化：保守 learning rate 1e-4、paged AdamW 8-bit、cosine、warmup 0.03、weight decay 0.01、max grad norm 1.0。
- 图像：min_pixels=200704、max_pixels=401408；max_length=1024。
- Early stopping 对所有组统一禁用；固定 step 192 的最终 checkpoint 是唯一评估 checkpoint。val loss 仅作非有限值/异常发散健康检查，不进行 seed 内 checkpoint 挑选。
- 每个 seed 保存 adapter、adapter SHA256、config、环境、训练日志、loss、peak VRAM、时长、实际样本/角色曝光和 checkpoint 依据。

## 推理与评测协议

- Base 只运行一次；九个 adapters 分别在完全相同的 dev 图片、分辨率、解码参数、JSON parser 和评测脚本上运行。
- 每个 adapter 同时运行：统一中性提示、该组训练提示、两个从未用于训练的语义等价提示改写、positive、semantic-null、rotating visual-null，以及固定 blank/blur/shuffle 条件。
- 评测输出保存 predictions、metrics 和 SHA256。
- Positive：Accuracy、Macro-F1、head/medium/tail、Mean IoU、Pointing Game、Supported Diagnosis Rate。
- Null：overall/real semantic/synthetic visual Null FPR、blank/blur/shuffle refusal、concrete diagnosis under null、False Localization Rate。
- JSON：syntax validity、schema validity、semantic consistency、task compliance 分开报告。
- 图像依赖：对同一 family 报告原图→blank、原图→blur、原图→shuffle 的具体诊断、拒答和定位配对变化，不只报告全局平均。
- 稳定性：训练提示/统一中性提示/未见改写之间的差距；每 seed 原始结果；三 seeds 均值、样本标准差、最差 seed；1,000 次 bootstrap 95% CI；adapter/Base paired bootstrap；正确性二分类采用 McNemar。

## 预先冻结淘汰与选择

任一 seed 出现以下任一项即淘汰该组：

- positive Accuracy 相对相同协议 Base 低于 3pp；
- Macro-F1 差值的 paired-bootstrap 95% CI 上界小于 0；
- blank 或 blur Null FPR ≥10%；
- 统一中性提示与训练提示的 positive Accuracy 绝对差 ≥5pp；
- text-only gate 任一 BA/AUROC >0.55，或 syntax/schema/semantic/task compliance 管线重新出现标签泄漏；
- syntax/schema/semantic/task compliance 任一低于 Base，或 semantic consistency/task compliance 低于 99%。

通过组先要求诊断性能不劣于 Base，再按以下字典序选择唯一协议：Macro-F1 更高 → mean/worst-seed Null FPR 更低 → Supported Diagnosis Rate 更高 → seed 方差更小。不得压成训练后临时设计的单一总分。若无组通过，Task 9D 决策为 BLOCK，禁止 9E；若至少一组通过，报告唯一最优协议并明确是否建议授权 9E，但不会自动启动 9E。

## 工程安全与完整性

- 所有输入 manifest 和脚本在训练前冻结 SHA256；输出目录非空时拒绝覆盖。
- 启动前对 A/B/C 分别重跑 shortcut gate；任一 BA/AUROC >0.55 时禁止对应组训练。
- 不读取 Task 8 family predictions、metrics 或 locked 图像内容；仅保留既有盲化排除证明。
- 先执行数据/单 batch/smoke gate；全部通过后才顺序运行九个小规模 run，任一 run 失败不自动删除或重启。
- 最终生成 Task 9D 决策报告和当日关键记忆，服务器保持开机。

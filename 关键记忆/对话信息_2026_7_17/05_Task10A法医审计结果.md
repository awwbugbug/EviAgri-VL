# Task 10A法医审计结果（2026-07-17）

## 范围与完整性

- 只读v2.2 `task9_dev_audit` 32 families；未读取Task 8 locked 186 families。
- 六组历史推理各352条，manifest SHA256：`1619320439116113b26329e0656dda4e85bc63983ea18b156ea58e78484aad6b`。
- 先做Control29×1 family smoke；首轮暴露slow tokenizer无offset，修为fast-offset/slow-input token ID逐样本完全一致后，smoke质量门通过。
- 正式审计288个PDM单元全部完成；服务器与本地`completion.sha256`均通过。无训练、无权重更新。

## 关键结果

- Bbox坐标链：`PASSED_COORDINATE_PROTOCOL`；三Control均32/32有效，最大round-trip误差`1.14e-13 px`，最小synthetic IoU约`1.0`。历史低mIoU不能归因于坐标缩放链错误。
- PDM-H质量：覆盖率100%，全部有限，最大归一化误差`1.31e-6`。
- 视觉依赖成立：
  - taxonomy：original-blank `0.0449`，95% CI `[0.0372,0.0536]`；original-shuffle `0.0189`，CI `[0.0110,0.0285]`。
  - evidence_present：original-blank `0.1715`，CI `[0.1127,0.2332]`；original-shuffle `0.1222`，CI `[0.0676,0.1767]`。
- Pair/输出契约失败：Control17/29/43的“正确pest_id+五类null均拒答”strict family success均为`0`；正确类别original TPR均为`0`；语义/Schema无效数分别为`15/7/10`。模型看图，但会输出错误taxonomy，不能视为可直接复用的可靠verifier。

## 决策

- v1仅按PDM授权复用过于宽松，已保留原签名并用无GPU重跑的v2勘误取代。
- 最终v2：bbox通过、视觉依赖通过、pair contract失败；`authorize_existing_verifier_for_task10d=false`。
- `authorize_task10b_planning=true`，但`authorize_task10b_execution=false`、`authorize_training=false`。
- 下一步仅可规划Task 10B冻结视觉特征线性探针；执行前仍需用户确认。不得直接进入10C/10D或大型实验。

## 产物

- 服务器：`/root/autodl-tmp/EviAgriDiag/experiments/task10_micro_first/2026-07-17/`
- 本地：`artifacts/2026-07-17_task10/10A_forensics/`
- 决策v2：`task10a_decision_v2_pair_contract/task10a_decision_report_v2.json`

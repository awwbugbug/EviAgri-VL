# Task 9D 输出法医与 v2.2 设计

- 法医产物：`artifacts/2026-07-16_task9/9D_output_forensics/`，本地/服务器 SHA256 一致。
- 输入限制：只读 Task9D frozen outputs；Task8 locked 186 families 未读取；未启动训练。
- 标签映射干净：96 names ↔ 96 IDs，无冲突；adapter schema validity 99.69%–100%。
- 主病因：复合任务/评测错配。prompt 已给类别名，模型接受时约 99.7% 复述该名称；平均 exact ID accuracy 仅 0.91%，名称↔ID 内部一致率 44.38%。
- 隐藏能力：按 `evidence_present` 重算的 verification Balanced Accuracy 为 63.72%–77.94%，故 Task9D 的约 1% Accuracy 不能等价为方向失败。
- 科学边界：当前结果只支持“有一定 query-conditioned evidence verification”，不支持“已具备开放农业病虫诊断”。
- v2.2 唯一变量：正样本 `pest_id/pest_name` value-token loss weight，Control=1.0，TaxMask=0.0；其余数据、prompt、schema、LoRA、训练与评测完全一致。
- 微实验：B 协议，32 类、64 train families、32 dev families、64 steps、seeds 17/29/43；2 arms×3 seeds。未启动。
- 通过核心：Verification Balanced Accuracy 平均 +8pp 且 bootstrap CI>0，Positive TPR +10pp，Null FPR 不恶化>3pp，schema/evidence compliance≥99%。
- 通过也只授权后续 evidence/taxonomy 解耦小消融；不授权 9E、Task8 confirmatory、动态 LoRA/Gating、SAM2、7B。
- 设计：`docs/task9d_v22_single_variable_design_2026-07-16.md`

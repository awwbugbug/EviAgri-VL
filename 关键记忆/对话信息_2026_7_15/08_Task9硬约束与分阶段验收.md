# Task 9硬约束与分阶段验收（权威版，2026-07-15）

## 总原则

- Task 8 的 `186 families` 立即锁定为 `locked_confirmatory_set`；9A–9E 禁止读取其预测结果进行方案、阈值或超参数选择。
- 仅在 v2 数据协议、模型方案、超参数、解码和评测全部冻结并生成哈希后，于 9F 运行一次最终确认审计。
- A/B/C 与全部开发决策只使用独立 `task9_dev_audit`；它与训练集、验证集和 locked set 按原图/近重复簇隔离。
- 当前禁止：下载新 backbone、动态 LoRA/Gating、SAM2、7B；保持 Qwen2.5-VL-3B Static QLoRA 路线。

## 9A：v1 数据法医审计

- 检查 prompt、system prompt、非图像元数据、JSON 字段/顺序、答案长度、固定短语、task type、文件名、目录和类别路径泄漏。
- 分别审计 real null 与 synthetic null；检查正负图像复用、类别长尾、近重复簇跨 split。
- 输出事实报告，不使用 locked confirmatory set 调参。

## 9B：冻结 v2 数据协议

- 正负样本使用相同模板分布、相同 JSON 字段和字段顺序；禁止标签相关句式、长度、固定拒答语及任何路径/元数据泄漏。
- 每个 family 初始配比：`1 positive + 1 semantic-negative query + 1 rotating visual counterfactual`；不得固定生成 5 个负样本。
- visual counterfactual 在 families 间轮换并平衡；real null 与 synthetic null 分开采样、训练记录和评测，监控过度拒答。
- 训练反事实与最终测试反事实使用互斥模板 ID、扰动类型/参数区间和随机种子；不得完全相同，防止学习扰动伪影。
- v2 JSON 语义约束：`evidence_present=false` 时必须 `diagnosis=uncertain/abstain`、`evidence_region=null`，且不得输出具体 species/stage。
- 9B 冻结：数据 manifest、split/cluster 清单、模板清单、扰动参数、随机种子、训练超参数、解码参数、评测脚本及 SHA256。

## 9C：训练前 shortcut gate

- 三种输入分别训练/评估 text-only probe：① user prompt only；② system+user prompt；③ prompt+非图像元数据。
- 每种输入必须同时满足：Balanced Accuracy `<=0.55`、AUROC `<=0.55`。
- 任一输入、任一指标超阈值即禁止训练，返回 9B 修协议；不得用普通 Accuracy 替代。

## 9D：A/B/C 多种子小规模消融

- A/B/C 只在 `task9_dev_audit` 选择方案；每组至少 3 个预先冻结的随机种子。
- 报告单种子结果、均值/离散度、bootstrap CI 与配对检验；禁止查看 locked set 后回调方案。
- JSON 分开报告：syntax validity、schema validity、semantic consistency、task compliance。
- null 指标按 real null / synthetic null / 各反事实条件分别报告，不仅报总平均。

## 9E：全量 Static QLoRA v2

- 仅使用 9D 冻结的唯一方案和超参数训练全量 v2；不得中途依据 locked set 修改。
- 完成后锁定 adapter、配置、数据 manifest、推理与评测代码哈希，才可进入 9F。

## 9F：一次性 confirmatory audit

- 只运行一次 Task 8 的 186 locked families；运行前不得用于开发选择。
- 一级“协议修复”：相对 Base Accuracy `>=-3pp`；Macro-F1 不显著更差；blank/blur Null FPR `<10%`；模板差距 `<5pp`。
- 二级“科学通过”：诊断性能不劣于 Base，且可靠性或定位至少一项显著优于 Base。
- 仅一级通过：继续 Static QLoRA/数据协议研究，禁止动态 LoRA/Gating。
- 二级通过：才允许提出下一阶段复杂模块消融，但仍需单独审批。

## 执行顺序

`9A → 9B冻结 → 9C门禁 → 9D多种子开发消融 → 9E全量v2与冻结 → 9F一次性确认审计`

- 任一阶段未通过，只能回到允许的上游阶段修订并重新冻结；不得跳级。
- 本文件覆盖 `07_Task8失败归因与Task9路线.md` 中与上述硬约束不一致的旧表述。

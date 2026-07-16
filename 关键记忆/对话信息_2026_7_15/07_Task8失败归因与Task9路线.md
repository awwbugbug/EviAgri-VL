# Task 8失败归因与Task 9路线（2026-07-15）

- 结论：方向未失败；Static QLoRA v1 的同协议因果可靠性主张失败，冻结 v1，不上动态 LoRA/Gating。
- 证据：B1→B2 中性同协议 Acc `0.645→0.306`（差 `-0.339`，95% CI `[-0.425,-0.247]`，McNemar `p=1.94e-11`）；mIoU `0.275→0.134`；B2 blank Null FPR `0.989`、blur `0.930`。
- 可保留信号：B1 Acc/Macro-F1 `0.645/0.695`，说明任务可解；B3 mIoU/Pointing `0.562/0.914`，说明定位信号和工程闭环有效。
- 已证实首因：v1 正样本固定 `Identify...`，负样本固定 `Is [class]...`；正负复用同一图像，负样本只换错误类别查询；文本句式可直接预测 presence，形成模板捷径。
- 次级风险：null 目标高度固定；仅训练语言 self-attention LoRA、视觉模块冻结；正:负约 2:1；缺少成对图像反事实和一致性约束。它们的独立因果贡献尚未拆分。
- 数据风险：Task 8 已排除 673 张跨 split 高相似 test 图；v2 必须按近重复簇整体划分，不能只事后删 test。
- Task 9：先做 v1 训练集取证与 prompt-only shortcut probe；再构建共享中性模板、对称改写、paired wrong-query/shuffle/blur/blank/no-target 的 v2 数据。
- 训练门槛：prompt-only presence 准确率应接近 50%，`>55%` 拒绝训练；先小规模逐项消融，再全量 v2。
- v2 smoke 门槛：相对 B1 中性 Acc 不低于 `-3pp`；blank/blur Null FPR `<10%`；B2/B3 模板差 `<5pp`；schema 合规率 `>99%`。
- 实例保持开机，未执行关机或重启。

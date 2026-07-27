# Task 12A 条件互补性微实验

- 目的：不预设双视图正确，用新鲜样本比较 `G/L/GG/GL`；主比较 `GL-GG` 控制维度影响。Qwen全冻结，仅固定线性探针；null不参与拟合。
- 数据：train 64、val 16、fresh positive test 32、fresh PlantSeg null 32；全部排除Task11C样本，未读Task8。
- `GG→GL`：Accuracy `0.75→0.84375`（+9.375pp，bootstrap CI `[0,+21.875pp]`）；Macro-F1 `0.70417→0.83542`（+13.125pp）。
- 真实类别概率 +0.002398，95% CI `[0.001039,0.004009]`；null置信度 -0.003933，CI `[-0.005665,-0.002161]`；confidence AUROC +0.10645，CI `[0.03904,0.19824]`。
- 决策：`H1_PRIORITY` 且 `reliability_safe=true`，但只是一批探索证据；不授权训练/Task8。必须用完全独立第二批、原样协议复现，禁止基于本批调整结构或阈值。
- 证据：`artifacts/2026-07-27_task12/12A_conditional_complementarity_result`；代码提交 `fc1bd51`。

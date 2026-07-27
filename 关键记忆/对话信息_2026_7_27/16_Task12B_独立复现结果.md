# Task 12B H1独立复现

- 与Task12A协议完全相同，仅换为排除Task11C/12A全部ID后的独立train/val/test/null批次；Qwen冻结，null不拟合，未读Task8。
- `GG→GL`：Accuracy `0.8125→0.84375`（+3.125pp，CI `[0,+9.375pp]`）；Macro-F1 `0.7625→0.81667`（+5.417pp）。
- 真实类别概率 +0.000474（CI跨0）；null置信度 -0.000270（CI跨0）；confidence AUROC +0.004883（CI跨0）。方向与12A一致但证据较弱。
- 决策：`H1_PRIORITY`、point-estimate reliability safe。两批方向一致，允许进入“上下文保持的证据机制”设计；不等于确认，不授权大训练/Task8。
- 机制边界：全局必须保留为anchor，局部只能作为条件残差；H2 token选择与H3 presence/taxonomy分离继续作为竞争解释。
- 证据：`artifacts/2026-07-27_task12/12B_independent_replication_result`；代码提交 `980fa1d`。

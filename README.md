# EviAgri-VL 研究工程

轻量、证据绑定、低幻觉的农业害虫视觉语言诊断研究工程。当前主干采用Qwen2.5-VL-3B-Instruct，并遵循Micro-First原则：任何全量训练前先运行小规模、可证伪、带停止门槛的可信度实验。

## 仓库内容

- `server/`：数据协议、训练、推理、评估和审计代码。
- `tests/`：协议与实验代码测试。
- `scripts/`：本地/服务器辅助脚本。
- `docs/`：冻结设计规格和执行计划。
- `关键记忆/`：精简研究决策、关键结果与路线记录；论文PDF目录不入Git。

## 不进入Git的内容

- 原始/派生数据集、传输归档；
- 模型权重、adapter、checkpoint；
- predictions、训练日志和大体积实验产物；
- SSH密钥、密码、token和本地环境文件；
- 本地论文PDF。

正式实验产物保存在本地`artifacts/`与服务器实验目录；可复现结论通过哈希、紧凑报告和关键记忆进入仓库。

## 当前路线

当前冻结设计为Task 10 Micro-First：

`Open Diagnosis → Candidate Verification → Evidence Grounding → Abstention`

详细规格见：`docs/superpowers/specs/2026-07-17-task10-micro-first-design.md`。


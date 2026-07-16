# Task 10 Micro-First设计确认

- 用户批准将新文献启发落实为强制小规模可信度门控，任何冗长训练前先做微实验。
- 路线：10A无训练bbox/PDM-H/pair法医 → 10B线性探针 → 10C diagnosis-only QLoRA → 10D两阶段闭环。
- 当前不启动Task9E、官方test、AGE、外部数据、VCD扩展、动态LoRA、SAM2、7B或新backbone。
- 大型实验阈值：>1000 train families、>3 GPU小时、全量三seed、官方test或新模块；必须微实验通过并再次批准。
- 完整规格：`docs/superpowers/specs/2026-07-17-task10-micro-first-design.md`。
- 书面规格SHA256：`b1702695c958e83cb4ed5d8cf5a547a90615a13df42fa0034f91c974416e2868`；后续若修改必须生成新版本与新哈希。

# 新增论文侦察与 Task11B 边界

- 定向审阅 HALP、GroundingME、VEP、ContextualLens、ClearSight、VIB-Probe、DLCAF、PlantSeg，并核对公开源码。
- 共识：Qwen 视觉信号可能已存在，主要瓶颈是生成/证据桥接与不可定位拒答；与 Task10B 有信号、Task10C 结构失败一致。
- 最优先候选：PlantSeg 审计完成后，做 vision-only vs 精确 3L/4 query-token 的同协议单变量 RepProbe；family/source split，real-null 只测试。
- HALP 代码的图像 token 边界估计和随机拆分不符合本项目要求；VEP/ClearSight 绑定 LLaVA；DLCAF/ClearSight 无清晰根许可证，均不直接复制。
- PlantSeg 仅用于 damage-null 与局部病斑证据，不转作害虫 taxonomy 训练集。
- 详细记录：`关键记忆/项目关键聊天信息/8_2026新增顶会顶刊启发与源码审计.md`。

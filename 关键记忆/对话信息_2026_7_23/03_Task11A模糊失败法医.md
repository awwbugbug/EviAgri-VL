# Task 11A blur 失败法医（2026-07-23）

- 三 seed 的 blur FPR 均为 8/80，且 8 个 source 完全相同；类别分布为 10x1、16x1、71x2、82x2、87x1、99x1。
- 8 个样本原图均被 router 正确、高置信接受；blur confidence=`0.7265–0.9761`。
- blur/原图特征余弦：8 个接受样本均值=`0.8031`，其余拒答 blur 均值=`0.4979`，说明接受组确实保留更多原图表征。
- 视觉复核：radius=10 后仍可辨虫体轮廓、颜色、体节、长吻、群聚形态或蛾翅；将所有 blur 标为零证据不成立。
- 结论分层：预注册 Task11A 仍为 `FAIL`，不得事后改 gate；但病因是 blur synthetic-null construct validity 有问题，不能据此断言 router 在真正无目标图上幻觉。
- 下一步：先用 IP102 GT bbox 构造成对 background-only/target-removed 反事实并做协议可行性 smoke；val/dev 生成规则分离。通过人工/几何无目标审计后，才允许再次测试 confidence 或训练极小 evidence/OOD head。

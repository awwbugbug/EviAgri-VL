# Task 7 全量评估完成与新实例验收

- 新克隆实例已用既有 SSH 密钥连接：RTX 4090 48GB；项目与数据盘完整。
- 全量评估完成：val `3052/3052`，test `7596/7596`；parse failure=0。
- SHA256 清单八项全部通过；adapter 与 checkpoint-1279 SHA256 一致：`162237f0...25284`。
- test：诊断 accuracy `0.6309`，macro-F1 `0.3397`，mean IoU `0.6050`，IoU@0.5 `0.7309`，pointing game `0.9255`。
- test null：FPR `0`、evidence-bound hallucination `0`；schema/reliability/evidence-presence 均为 `1.0`。
- 中断续跑记录：val existing=2639、generated=413，总数与最终指标完整一致。
- 当前新实例 GPU 空闲但处于开机计费状态。

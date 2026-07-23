# Task 11A.1 背景 Null 设计（2026-07-23）

- blur 仍含可辨虫体，改为 IP102 GT bbox 外的 paired background-only crop；先验证 null 合法性，不训练。
- 几何预审计：val 规则 64px/margin5%/17-grid 可用 25/48；dev 规则 72px/margin8%/19-grid 可用 34/80；均覆盖 head/medium/tail 与 15 类。
- smoke：val/dev 各12；几何与视觉必须 24/24 无虫体，否则 `BLOCK_INVALID_NULL`，不得按模型结果删样本。
- smoke 通过后才生成全量 59 个 crop；全量视觉复核通过后，复用 Task11A 固定 temperature=`0.18887`、tau=`0.63` 做一次特征评估。
- 不读取 Task8 locked set；不训练 Qwen/evidence head；不扩大模型。

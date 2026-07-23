# Task 11A.1 v2 BLOCK 与数据扩展（2026-07-23）

- v2 smoke 消除了底部文字泄漏，24/24 几何与 SHA256 通过。
- overlay 发现 `val_IP069000382` 长触角超出 VOC bbox+margin 并进入 crop；bbox 几何不能保证无局部虫体。
- 决策：v2=`BLOCK_INVALID_NULL`；不继续缩小 crop 或针对 smoke 调 margin，不运行模型。
- 现有数据无合格视觉 no-pest：IP102 detection 无零-object XML；AGE 全为害虫虫态；Task9 real_null 是错类别查询而非无虫图。
- 必须最小扩展：PlantDoc 官方 test 的 10 个健康叶片类，每类确定性抽4张，共40张；CC BY 4.0；人工无虫审计后仅作外部 real-null 微评估，不进入训练。

